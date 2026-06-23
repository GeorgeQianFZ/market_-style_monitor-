"""
第二层：配方内部权重微调

对每种配方，用 XGBoost 预测配方内各因子的下期 IC，以预测 IC 作为权重。

逻辑：
  已知当前状态（市场特征 + 各因子滚动IC）
  → 预测各因子下期 IC（普通 XGBoost 回归）
  → 权重 = 预测IC / sum(|预测IC|)，方向随IC符号

这与用户熟悉的"IC加权"完全一致，只是用 XGBoost 做了非线性拟合。
"""

import numpy as np
import xgboost as xgb
from typing import Dict

N_FACTORS = 10

# 各配方包含的因子索引
RECIPE_FACTORS: Dict[int, list] = {
    0: list(range(10)),   # 等权：全部
    1: [0, 1, 2, 5],      # 动量型
    2: [3, 4, 6],         # 反转型
    3: [7, 8, 9],         # 波动型
    4: list(range(10)),   # IC自适应：全部
}

XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


class WeightRefiner:
    """配方内 XGBoost IC 预测器 → 精确权重"""

    def __init__(self, recipe_id: int):
        self.recipe_id   = recipe_id
        self.factor_idxs = RECIPE_FACTORS[recipe_id]
        self.models: Dict[int, xgb.XGBRegressor] = {}
        self.trained = False

    def fit(self, states: np.ndarray, future_ics: np.ndarray):
        """
        states:      [N, 13]  每个调仓期的状态向量
        future_ics:  [N, 10]  各因子下一持仓期的实际平均 IC
        """
        for f in self.factor_idxs:
            y    = future_ics[:, f]
            mask = np.isfinite(y)
            if mask.sum() < 5:
                continue
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(states[mask], y[mask])
            self.models[f] = m
        self.trained = True

    def predict_weights(self, state: np.ndarray) -> np.ndarray:
        """
        输入：13维状态
        输出：N_FACTORS 维权重（配方外因子权重为0）
        """
        weights = np.zeros(N_FACTORS, dtype=np.float32)

        if not self.trained or not self.models:
            # 兜底：配方内等权
            for f in self.factor_idxs:
                weights[f] = 1.0 / len(self.factor_idxs)
            return weights

        # 预测各因子 IC
        pred_ic = {
            f: float(m.predict(state[np.newaxis, :])[0])
            for f, m in self.models.items()
        }

        total = sum(abs(v) for v in pred_ic.values())
        if total < 1e-8:
            for f in self.factor_idxs:
                weights[f] = 1.0 / len(self.factor_idxs)
        else:
            for f, ic in pred_ic.items():
                weights[f] = ic / total   # 方向 × 幅度归一化

        return weights
