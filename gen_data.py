"""
模拟数据生成脚本

生成逻辑说明：
  - 2个潜在市场状态（Regime 0 = 趋势行情，Regime 1 = 震荡/反转行情），Markov切换
  - 10个候选因子，分3类：动量类(0-2)、反转类(3-5)、波动/混合类(6-9)
  - 各因子在不同市场状态下 IC 符号/大小不同，体现时变有效性
  - 截面收益 = 因子信号 * 状态相关载荷 + 公共市场噪声 + 个股噪声

输出文件：
  - factors.parquet    [N_DAYS, N_STOCKS, N_FACTORS] -> 展开为 MultiIndex DataFrame
  - returns.parquet    [N_DAYS, N_STOCKS] 未来1日收益率
  - regime.parquet     [N_DAYS] 真实状态标签（仅用于分析，实盘不可观测）
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────── 参数 ──────────────────────────
N_STOCKS  = 300
N_DAYS    = 1500         # ~6年日频
N_FACTORS = 10
SEED      = 42
OUTPUT_DIR = Path(__file__).parent

# Markov转移矩阵: P[i,j] = 从状态i转移到状态j的概率
TRANSITION = np.array([
    [0.96, 0.04],   # Regime 0 (趋势) 持续性强
    [0.06, 0.94],   # Regime 1 (震荡) 持续性强
])

# 各状态下10个因子的截面预测能力 (近似RankIC)
# Regime0=趋势: 动量因子有效，反转因子失效
# Regime1=震荡: 反转因子有效，动量因子失效
FACTOR_IC = np.array([
    # F0    F1    F2    F3     F4     F5    F6    F7    F8    F9
    [+0.09, +0.07, +0.06, -0.05, -0.04, +0.05, +0.02, -0.03, +0.01, +0.03],  # Regime 0
    [-0.06, -0.04, -0.05, +0.08, +0.07, -0.04, +0.09, +0.05, +0.02, -0.02],  # Regime 1
])

FACTOR_NAMES = [f"F{i}" for i in range(N_FACTORS)]
STOCK_IDS    = [f"S{i:04d}" for i in range(N_STOCKS)]


# ─────────────────── 生成函数 ──────────────────────
def simulate_regime(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Markov Chain 模拟市场状态序列"""
    regime = np.zeros(n_days, dtype=int)
    regime[0] = 0
    for t in range(1, n_days):
        regime[t] = rng.choice(2, p=TRANSITION[regime[t - 1]])
    return regime


def generate_factors(n_days: int, n_stocks: int, rng: np.random.Generator) -> np.ndarray:
    """
    生成因子矩阵 [N_DAYS, N_STOCKS, N_FACTORS]
    因子之间存在公共成分（AR(1)时序自相关 + 截面相关结构）
    """
    # 截面相关结构：因子共享公共波动项
    factor_cov = np.eye(N_FACTORS) * 0.6 + np.ones((N_FACTORS, N_FACTORS)) * 0.4
    L = np.linalg.cholesky(factor_cov)

    raw = rng.standard_normal((n_days, n_stocks, N_FACTORS))
    # 引入截面相关
    factors = raw @ L.T   # [T, S, F]

    # 引入时序自相关 (AR coeff = 0.3)
    for t in range(1, n_days):
        factors[t] = 0.3 * factors[t - 1] + 0.954 * factors[t]  # 保持方差≈1

    return factors


def generate_returns(
    factors: np.ndarray,
    regime: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    生成未来1日收益率 [N_DAYS, N_STOCKS]
    Return[t] = 公共市场收益 + sum(因子载荷[regime[t]] * 因子[t]) + 个股噪声
    """
    n_days, n_stocks, _ = factors.shape
    returns = np.zeros((n_days, n_stocks))

    # 公共市场收益 (AR(1) with vol clustering)
    mkt = rng.standard_normal(n_days) * 0.01

    for t in range(n_days - 1):
        r = regime[t]
        alpha = factors[t] @ FACTOR_IC[r]          # 截面期望超额收益 [N_STOCKS]
        idio  = rng.standard_normal(n_stocks) * 0.03  # 个股噪声
        returns[t] = mkt[t] + alpha + idio

    return returns


# ─────────────────── 主流程 ──────────────────────
def main():
    rng = np.random.default_rng(SEED)

    print("正在生成模拟数据...")
    regime  = simulate_regime(N_DAYS, rng)
    factors = generate_factors(N_DAYS, N_STOCKS, rng)
    returns = generate_returns(factors, regime, rng)

    # ── 整理成 DataFrame ──
    dates = pd.date_range("2018-01-01", periods=N_DAYS, freq="B")

    # factors: MultiIndex(date, stock) × factor_columns
    factor_records = []
    for t, date in enumerate(dates):
        df_t = pd.DataFrame(
            factors[t], index=STOCK_IDS, columns=FACTOR_NAMES
        )
        df_t.index.name = "stock"
        df_t["date"] = date
        factor_records.append(df_t)

    factor_df = pd.concat(factor_records).reset_index()
    factor_df = factor_df.set_index(["date", "stock"])

    # returns
    return_df = pd.DataFrame(
        returns, index=dates, columns=STOCK_IDS
    )
    return_df.index.name = "date"

    # regime
    regime_df = pd.Series(regime, index=dates, name="regime")
    regime_df.index.name = "date"

    # ── 保存 ──
    factor_df.to_parquet(OUTPUT_DIR / "factors.parquet")
    return_df.to_parquet(OUTPUT_DIR / "returns.parquet")
    regime_df.to_frame().to_parquet(OUTPUT_DIR / "regime.parquet")

    print(f"因子数据: {factor_df.shape}  -> factors.parquet")
    print(f"收益数据: {return_df.shape} -> returns.parquet")
    print(f"状态序列: {regime_df.shape}  -> regime.parquet")

    # 验证：各状态下因子IC
    print("\n── 各 Regime 下因子平均 RankIC（验证数据生成正确性）──")
    from scipy.stats import spearmanr

    for r_val in [0, 1]:
        days_r = dates[regime == r_val][:200]  # 取前200天
        ics = []
        for date in days_r:
            f_t = factor_df.loc[date]
            if date in return_df.index:
                ret_t = return_df.loc[date]
                row_ic = [
                    spearmanr(f_t[col].values, ret_t.loc[f_t.index].values)[0]
                    for col in FACTOR_NAMES
                ]
                ics.append(row_ic)
        mean_ic = np.nanmean(ics, axis=0)
        print(f"  Regime {r_val}: " + " ".join(f"{v:+.3f}" for v in mean_ic))

    print("\n数据生成完成。")


if __name__ == "__main__":
    main()
