"""
Fitted Q-Iteration (FQI) 智能体 —— 以 XGBoost 作为 Q 函数近似器

算法原理：
  FQI 是经典的批量/离线强化学习算法，将 Q-learning 的 Bellman 迭代转化为
  一系列监督回归问题，天然兼容任意回归器（XGBoost、随机森林等）。

为何选择 FQI + XGBoost：
  1. 金融时序数据量有限（数百个调仓期），深度网络样本效率低，树模型更鲁棒；
  2. XGBoost 对特征尺度不敏感、自动处理非线性交互、不易过拟合（有正则）；
  3. FQI 是离线算法，可反复利用历史经验，避免在真实市场中大量探索；
  4. 与用户熟悉的 XGBoost 框架无缝衔接，可解释性强。

Q 函数近似方式：
  每个动作独立训练一个 XGBoost 回归器：Q_a(s) -> expected_discounted_return
  预测时取各动作Q值最大者：π(s) = argmax_a Q_a(s)
  （也可拼接 one-hot(action) 用单一模型，此处分开以减少动作干扰）
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import List, Tuple, Optional, Callable

N_ACTIONS = 5
GAMMA     = 0.95   # 折扣因子
EPSILON_0 = 1.0    # 初始探索率
EPSILON_MIN = 0.05
DECAY_RATE  = 0.90

# XGBoost 超参数
XGB_PARAMS = dict(
    n_estimators    = 200,
    max_depth       = 4,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 1.0,
    random_state    = 42,
    n_jobs          = -1,
    verbosity       = 0,
)


class FQIAgent:
    """
    Fitted Q-Iteration Agent

    训练流程：
      1. 用随机/探索策略在环境中收集 N 条转换元组 (s,a,r,s',done)
      2. 初始化 Q_a(s) = 0（每个动作一个 XGBoost 模型）
      3. 迭代 K 次：
           for each (s,a,r,s',done):
               y = r + γ * max_{a'} Q_{a'}(s')  （若 done 则 y=r）
           拟合 Q_a_new = XGBRegressor.fit(states_with_a, targets_a)
      4. 更新策略 π(s) = argmax_a Q_a(s)
      5. （可选）用更新后的 ε-greedy 策略再收集数据，扩充经验池
    """

    def __init__(self):
        self.models: List[Optional[xgb.XGBRegressor]] = [None] * N_ACTIONS
        self.epsilon = EPSILON_0
        self.replay_buffer: List[Tuple] = []
        self.trained = False

    # ── 经验管理 ──────────────────────────────────
    def add_transitions(self, transitions: List[Tuple]):
        self.replay_buffer.extend(transitions)

    def _unpack_buffer(self):
        states, actions, rewards, next_states, dones = zip(*self.replay_buffer)
        return (
            np.array(states),
            np.array(actions, dtype=int),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=bool),
        )

    # ── 当前Q值估计 ────────────────────────────────
    def predict_q(self, states: np.ndarray) -> np.ndarray:
        """返回 Q 矩阵 [N, N_ACTIONS]"""
        N = len(states)
        q_mat = np.zeros((N, N_ACTIONS), dtype=np.float32)
        for a, model in enumerate(self.models):
            if model is not None:
                q_mat[:, a] = model.predict(states)
        return q_mat

    def predict_q_single(self, state: np.ndarray) -> np.ndarray:
        return self.predict_q(state[np.newaxis, :])[0]

    # ── FQI 训练 ──────────────────────────────────
    def fit(self, n_iter: int = 20, verbose: bool = True):
        """
        执行 FQI 迭代

        伪代码：
            D = replay_buffer   # {(s_i, a_i, r_i, s'_i, done_i)}
            Q_a(·) = 0          # 初始化（全零）
            for k = 1..n_iter:
                for each action a:
                    # 找出选了动作a的转换
                    idx_a = [i | a_i == a]
                    # Bellman目标
                    y[i] = r_i + γ * max_{a'} Q_{a'}(s'_i)  if not done_i
                           r_i                               if done_i
                    # 监督拟合
                    Q_a = XGBRegressor.fit(S[idx_a], y[idx_a])
                if converged: break
        """
        states, actions, rewards, next_states, dones = self._unpack_buffer()
        N = len(states)

        if verbose:
            print(f"\n[FQI] 样本量={N}，迭代{n_iter}次，折扣γ={GAMMA}")

        for k in range(n_iter):
            # Step 1: 计算 Bellman 目标 y_i
            q_next = self.predict_q(next_states)         # [N, A]
            max_q_next = q_next.max(axis=1)              # [N]
            y_all = rewards + GAMMA * max_q_next * (~dones)

            # Step 2: 按动作分组训练各 XGBRegressor
            for a in range(N_ACTIONS):
                mask = actions == a
                if mask.sum() < 5:
                    continue
                X_a = states[mask]
                y_a = y_all[mask]
                model = xgb.XGBRegressor(**XGB_PARAMS)
                model.fit(X_a, y_a)
                self.models[a] = model

            if verbose and (k + 1) % 5 == 0:
                # 计算 Bellman 残差（诊断收敛性）
                q_pred = np.array([
                    self.models[a].predict(states[actions == a])
                    if self.models[a] else np.zeros((actions == a).sum())
                    for a in range(N_ACTIONS)
                ], dtype=object)
                y_pred = np.concatenate([
                    self.models[a].predict(states[actions == a])
                    if self.models[a] else np.zeros((actions == a).sum())
                    for a in range(N_ACTIONS)
                ])
                residual = np.mean((y_all - y_pred[:N]) ** 2) ** 0.5
                print(f"  iter {k+1:3d}/{n_iter}  Bellman残差(RMSE)={residual:.5f}")

        self.trained = True
        print("[FQI] 训练完成。\n")

    # ── 策略 ──────────────────────────────────────
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy 策略"""
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        q_vals = self.predict_q_single(state)
        return int(np.argmax(q_vals))

    def greedy_action(self, state: np.ndarray) -> int:
        return self.act(state, greedy=True)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * DECAY_RATE)

    # ── 特征重要性（可解释性）────────────────────
    def feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        if not self.trained:
            raise RuntimeError("请先调用 fit()")
        rows = []
        for a, model in enumerate(self.models):
            if model is None:
                continue
            imp = model.feature_importances_
            for f, name in enumerate(feature_names):
                rows.append({"action": a, "feature": name, "importance": imp[f]})
        return pd.DataFrame(rows)

    # ── 保存/加载 ─────────────────────────────────
    def save(self, path: Path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"模型已保存: {path}")

    @classmethod
    def load(cls, path: Path) -> "FQIAgent":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
