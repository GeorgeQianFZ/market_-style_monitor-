"""
强化学习环境：动态因子选择

交互逻辑：
  每隔 HOLD_DAYS(=20) 个交易日，智能体观察当前状态后选择一种因子权重策略，
  持有该策略直到下次调仓，以实现的组合 RankIC 作为奖励。

状态空间（13维连续向量）：
  - 各因子过去 IC_WINDOW(=20) 日滚动均值 IC    [10维]
  - 市场截面收益率标准差（波动率代理）          [1维]
  - 市场整体近5日平均收益（动量代理）           [1维]
  - 因子间平均相关性（分散度代理）              [1维]

动作空间（5个离散动作）：
  0: 等权  — 10个因子各 1/10 权重
  1: 动量型 — F0/F1/F2/F5 各 1/4（趋势行情偏好）
  2: 反转型 — F3/F4/F6 各 1/3（震荡行情偏好）
  3: 波动型 — F7/F8/F9 各 1/3
  4: IC自适应 — 按滚动IC绝对值加权，符号取IC方向（每期动态）

奖励函数：
  R = 多空组合 RankIC（持仓期内每日 IC 的均值）
  最终线性合成因子得分 = factors @ weights；
  多=前20%，空=后20%，每日IC = spearmanr(score, next_day_return)。
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
from typing import Tuple, Optional

DATA_DIR   = Path(__file__).parent
HOLD_DAYS  = 20   # 调仓周期（月频）
IC_WINDOW  = 20   # 滚动IC计算窗口
N_FACTORS  = 10
N_ACTIONS  = 5

# 各动作的固定因子权重（IC自适应型在运行时动态计算）
FIXED_WEIGHTS = {
    0: np.array([1/10]*10),                            # 等权
    1: np.array([1/4,1/4,1/4,0,0,1/4,0,0,0,0]),       # 动量型
    2: np.array([0,0,0,1/3,1/3,0,1/3,0,0,0]),          # 反转型
    3: np.array([0,0,0,0,0,0,0,1/3,1/3,1/3]),          # 波动型
    # 动作4在运行时按滚动IC计算
}
ACTION_NAMES = ["等权", "动量型", "反转型", "波动型", "IC自适应"]


class FactorEnv:
    """动态因子选择强化学习环境（类Gym接口）"""

    def __init__(self, mode: str = "train"):
        """
        mode: 'train' 使用前70%数据，'test' 使用后30%数据
        """
        self._load_data()
        n = len(self.dates)
        split = int(n * 0.70)

        if mode == "train":
            self.start_day = IC_WINDOW + HOLD_DAYS  # 留出预热期
            self.end_day   = split
        else:
            self.start_day = split
            self.end_day   = n - HOLD_DAYS - 1

        self.current_day: int = self.start_day
        self._precompute_daily_ic()

    # ── 数据加载 ──────────────────────────────────
    def _load_data(self):
        factor_df = pd.read_parquet(DATA_DIR / "factors.parquet")
        return_df = pd.read_parquet(DATA_DIR / "returns.parquet")

        self.dates        = return_df.index.values
        self.stock_ids    = return_df.columns.values
        self.factor_names = [c for c in factor_df.columns if c.startswith("F")]

        # 转为 numpy [T, S, F] 和 [T, S]
        n_days, n_stocks = len(self.dates), len(self.stock_ids)
        self.factors_np = np.zeros((n_days, n_stocks, N_FACTORS))
        self.returns_np = return_df.values  # [T, S]

        for t, date in enumerate(self.dates):
            try:
                self.factors_np[t] = factor_df.loc[date][self.factor_names].values
            except KeyError:
                pass

    # ── 预计算每日截面IC [T, F] ──────────────────
    def _precompute_daily_ic(self):
        T = len(self.dates)
        self.daily_ic = np.full((T, N_FACTORS), np.nan)
        for t in range(T - 1):
            ret_t1 = self.returns_np[t]       # 未来1日收益
            for f in range(N_FACTORS):
                fac = self.factors_np[t, :, f]
                mask = np.isfinite(fac) & np.isfinite(ret_t1)
                if mask.sum() > 30:
                    self.daily_ic[t, f] = spearmanr(fac[mask], ret_t1[mask])[0]

    # ── 状态计算 ──────────────────────────────────
    def _get_state(self, t: int) -> np.ndarray:
        """计算时刻t的13维状态向量"""
        # 1. 滚动IC均值 [10维]
        ic_window = self.daily_ic[max(0, t - IC_WINDOW):t]
        rolling_ic = np.nanmean(ic_window, axis=0)  # [F]

        # 2. 市场波动率：截面收益率std，20日均值
        vol_series = [
            np.nanstd(self.returns_np[s])
            for s in range(max(0, t - IC_WINDOW), t)
        ]
        mkt_vol = np.mean(vol_series) if vol_series else 0.0

        # 3. 市场动量：近5日市场平均收益
        mkt_mom = np.nanmean(self.returns_np[max(0, t-5):t])

        # 4. 因子间平均相关性
        fac_t = self.factors_np[t]  # [S, F]
        corr_mat = np.corrcoef(fac_t.T)  # [F, F]
        triu_idx = np.triu_indices(N_FACTORS, k=1)
        avg_corr = np.nanmean(np.abs(corr_mat[triu_idx]))

        state = np.concatenate([rolling_ic, [mkt_vol, mkt_mom, avg_corr]])
        state = np.nan_to_num(state, nan=0.0)
        return state.astype(np.float32)

    # ── 动作 → 权重 ──────────────────────────────
    def _action_to_weights(self, action: int, t: int) -> np.ndarray:
        if action < 4:
            return FIXED_WEIGHTS[action]
        # 动作4：IC自适应权重
        ic = np.nanmean(self.daily_ic[max(0, t - IC_WINDOW):t], axis=0)
        ic = np.nan_to_num(ic, nan=0.0)
        abs_ic = np.abs(ic)
        denom = abs_ic.sum()
        if denom < 1e-8:
            return FIXED_WEIGHTS[0]
        weights = (np.sign(ic) * abs_ic) / denom   # 方向 * 幅度归一化
        return weights.astype(np.float32)

    # ── 奖励计算 ──────────────────────────────────
    def _compute_reward(self, t_start: int, weights: np.ndarray) -> float:
        """计算持仓期 [t_start, t_start+HOLD_DAYS) 内多空组合的平均RankIC"""
        ics = []
        for t in range(t_start, min(t_start + HOLD_DAYS, len(self.dates) - 1)):
            fac_t = self.factors_np[t]     # [S, F]
            ret_t = self.returns_np[t]     # [S]
            score = fac_t @ weights        # 合成因子得分 [S]
            mask  = np.isfinite(score) & np.isfinite(ret_t)
            if mask.sum() > 30:
                ic, _ = spearmanr(score[mask], ret_t[mask])
                ics.append(ic)
        return float(np.mean(ics)) if ics else 0.0

    # ── Gym 接口 ──────────────────────────────────
    def reset(self) -> np.ndarray:
        self.current_day = self.start_day
        return self._get_state(self.current_day)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        t = self.current_day
        weights = self._action_to_weights(action, t)
        reward  = self._compute_reward(t, weights)

        self.current_day += HOLD_DAYS
        done = self.current_day >= self.end_day

        next_state = self._get_state(self.current_day) if not done else np.zeros(13, dtype=np.float32)
        info = {"t": t, "date": self.dates[t], "weights": weights}
        return next_state, reward, done, info

    def collect_all_transitions(self, policy=None):
        """
        离线收集所有转换元组 (s, a, r, s', done)
        policy(s) -> action；若为 None 使用随机策略
        """
        transitions = []
        state = self.reset()
        done  = False
        while not done:
            if policy is None:
                action = np.random.randint(N_ACTIONS)
            else:
                action = policy(state)
            next_state, reward, done, info = self.step(action)
            transitions.append((state.copy(), action, reward, next_state.copy(), done))
            state = next_state
        return transitions

    def step_with_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """第二层专用：直接传入权重向量，跳过动作→权重转换"""
        t = self.current_day
        reward = self._compute_reward(t, weights)
        self.current_day += HOLD_DAYS
        done = self.current_day >= self.end_day
        next_state = self._get_state(self.current_day) if not done else np.zeros(13, dtype=np.float32)
        info = {"t": t, "date": self.dates[t]}
        return next_state, reward, done, info

    def get_refiner_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        为 WeightRefiner 准备训练数据
        返回：
          states      [N, 13]  各调仓期状态
          future_ics  [N, 10]  各因子下一持仓期平均 IC
        """
        states, future_ics = [], []
        t = self.start_day
        while t < self.end_day:
            states.append(self._get_state(t))
            fut_ic = np.nanmean(self.daily_ic[t:t + HOLD_DAYS], axis=0)
            future_ics.append(fut_ic)
            t += HOLD_DAYS
        return np.array(states), np.array(future_ics)

    @property
    def state_dim(self) -> int:
        return 13

    @property
    def n_actions(self) -> int:
        return N_ACTIONS
