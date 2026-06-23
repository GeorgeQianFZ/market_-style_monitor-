"""
主程序：数据生成 → FQI训练 → 走样外评估 → 对比基准 → 出图

运行顺序：
  python gen_data.py   （首次生成数据，约10秒）
  python run_rl.py     （训练+评估，约2-5分钟）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

from rl_env   import FactorEnv, N_ACTIONS, ACTION_NAMES, HOLD_DAYS
from fqi_agent import FQIAgent

OUTPUT_DIR = Path(__file__).parent
matplotlib.rcParams["font.family"]      = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ─────────────────── 辅助函数 ─────────────────────
def evaluate_policy(env: FactorEnv, policy_fn, label: str) -> pd.DataFrame:
    """在测试环境上运行策略，收集每期 (date, action, reward)"""
    records = []
    state = env.reset()
    done  = False
    while not done:
        action = policy_fn(state)
        next_state, reward, done, info = env.step(action)
        records.append({
            "date":   str(info["date"])[:10],
            "action": action,
            "action_name": ACTION_NAMES[action],
            "reward": reward,
        })
        state = next_state
    df = pd.DataFrame(records)
    df["cumulative"] = df["reward"].cumsum()
    df["label"] = label
    print(f"  [{label}]  期数={len(df)}  平均IC={df['reward'].mean():.4f}"
          f"  累积IC={df['reward'].sum():.4f}")
    return df


def sharpe(rewards: pd.Series) -> float:
    m, s = rewards.mean(), rewards.std()
    return m / s * (252 / HOLD_DAYS) ** 0.5 if s > 1e-8 else 0.0


# ─────────────────── Step 1: 生成数据 ────────────
def ensure_data():
    if not (OUTPUT_DIR / "factors.parquet").exists():
        print("=== 生成模拟数据 ===")
        import gen_data
        gen_data.main()
    else:
        print("已存在数据文件，跳过生成。")


# ─────────────────── Step 2: 离线FQI训练 ─────────
def train_agent() -> FQIAgent:
    print("\n=== 训练 FQI 智能体 ===")
    train_env = FactorEnv(mode="train")

    agent = FQIAgent()

    # ── 阶段1：用随机策略收集初始经验 ──
    print("收集随机策略经验（探索）...")
    random_transitions = train_env.collect_all_transitions(policy=None)
    agent.add_transitions(random_transitions)
    print(f"  随机策略转换数: {len(random_transitions)}")

    # ── 阶段2：FQI迭代，多轮 on-policy 数据扩充 ──
    for round_idx in range(3):
        print(f"\nFQI 训练轮次 {round_idx + 1}/3")
        agent.fit(n_iter=15, verbose=True)
        agent.decay_epsilon()

        # 用当前策略收集更多数据
        def current_policy(s):
            return agent.act(s, greedy=False)

        new_trans = train_env.collect_all_transitions(policy=current_policy)
        agent.add_transitions(new_trans)
        print(f"  新增转换: {len(new_trans)}，缓冲区总计: {len(agent.replay_buffer)}")

    # 最终一次 greedy 训练
    print("\nFQI 最终训练轮次（greedy）")
    agent.fit(n_iter=20, verbose=True)

    return agent


# ─────────────────── Step 3: 测试集评估 ──────────
def evaluate(agent: FQIAgent):
    print("\n=== 测试集评估（样本外）===")
    test_env = FactorEnv(mode="test")

    # 基准1：随机策略
    np.random.seed(0)
    df_random = evaluate_policy(
        test_env, lambda s: np.random.randint(N_ACTIONS), "随机策略"
    )

    # 基准2：固定等权策略
    test_env2 = FactorEnv(mode="test")
    df_equal = evaluate_policy(test_env2, lambda s: 0, "等权策略")

    # 基准3：固定动量策略
    test_env3 = FactorEnv(mode="test")
    df_momentum = evaluate_policy(test_env3, lambda s: 1, "动量型")

    # 基准4：IC自适应策略
    test_env4 = FactorEnv(mode="test")
    df_ic = evaluate_policy(test_env4, lambda s: 4, "IC自适应")

    # FQI 智能体
    test_env5 = FactorEnv(mode="test")
    df_fqi = evaluate_policy(test_env5, agent.greedy_action, "FQI智能体")

    all_results = [df_random, df_equal, df_momentum, df_ic, df_fqi]

    # ── 汇总表 ──
    print("\n─── 绩效汇总 ───")
    summary_rows = []
    for df in all_results:
        label = df["label"].iloc[0]
        r = df["reward"]
        summary_rows.append({
            "策略": label,
            "平均IC":  round(r.mean(), 4),
            "IC>0%":   round((r > 0).mean() * 100, 1),
            "IC_ICIR": round(sharpe(r), 3),
            "累积IC":  round(r.sum(), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "rl_summary.csv", index=False, encoding="utf-8-sig")

    # ── 动作分布 ──
    print("\nFQI 选择的动作分布：")
    print(df_fqi["action_name"].value_counts().to_string())

    return all_results, summary_df


# ─────────────────── Step 4: 可视化 ──────────────
def plot_results(all_results, agent: FQIAgent):
    print("\n=== 生成图表 ===")
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    colors = {"随机策略": "gray", "等权策略": "steelblue",
              "动量型": "orange", "IC自适应": "green", "FQI智能体": "crimson"}

    # ── 子图1：累积IC曲线对比 ──
    ax1 = fig.add_subplot(gs[0, :])
    for df in all_results:
        lbl = df["label"].iloc[0]
        ax1.plot(range(len(df)), df["cumulative"], label=lbl,
                 color=colors.get(lbl, "black"),
                 linewidth=2.5 if lbl == "FQI智能体" else 1.2,
                 linestyle="-" if lbl == "FQI智能体" else "--")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("各策略累积 RankIC（测试集）", fontsize=13)
    ax1.set_xlabel("调仓期数")
    ax1.set_ylabel("累积 RankIC")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # ── 子图2：每期IC分布箱线图 ──
    ax2 = fig.add_subplot(gs[1, 0])
    box_data  = [df["reward"].values for df in all_results]
    box_labels = [df["label"].iloc[0] for df in all_results]
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, notch=True)
    for patch, lbl in zip(bp["boxes"], box_labels):
        patch.set_facecolor(colors.get(lbl, "gray"))
        patch.set_alpha(0.7)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_title("每期 RankIC 分布", fontsize=13)
    ax2.set_ylabel("RankIC")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    # ── 子图3：FQI动作选择时序 ──
    ax3 = fig.add_subplot(gs[1, 1])
    fqi_df = next(df for df in all_results if df["label"].iloc[0] == "FQI智能体")
    action_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    for a in range(N_ACTIONS):
        mask = fqi_df["action"] == a
        if mask.any():
            idx = np.where(mask)[0]
            ax3.scatter(idx, fqi_df.loc[mask, "reward"],
                        color=action_colors[a], label=ACTION_NAMES[a], s=40, alpha=0.8)
    ax3.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax3.set_title("FQI 动作选择（测试集每期）", fontsize=13)
    ax3.set_xlabel("调仓期")
    ax3.set_ylabel("RankIC")
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(alpha=0.3)

    plt.suptitle("强化学习动态因子选择 —— 评估结果", fontsize=14, fontweight="bold")
    save_path = OUTPUT_DIR / "rl_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {save_path}")

    # ── 特征重要性图 ──
    feature_names = [f"IC_F{i}" for i in range(10)] + ["市场波动率", "市场动量", "因子平均相关性"]
    imp_df = agent.feature_importance(feature_names)
    avg_imp = imp_df.groupby("feature")["importance"].mean().sort_values(ascending=True)

    fig2, ax = plt.subplots(figsize=(8, 6))
    avg_imp.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("FQI Q函数特征重要性（各动作XGBoost均值）", fontsize=12)
    ax.set_xlabel("重要性得分")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    imp_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"特征重要性图已保存: {imp_path}")


# ─────────────────── Step 5: 两层联合评估 ────────
from weight_refiner import WeightRefiner, RECIPE_FACTORS


def train_refiners() -> dict:
    """第二层：为每种配方训练 WeightRefiner（XGBoost IC预测）"""
    print("\n=== 训练 WeightRefiner（第二层：配方内部权重微调）===")
    train_env = FactorEnv(mode="train")
    states, future_ics = train_env.get_refiner_training_data()
    print(f"训练样本: {len(states)} 个调仓期")

    refiners = {}
    for recipe_id in range(N_ACTIONS):
        r = WeightRefiner(recipe_id)
        r.fit(states, future_ics)
        refiners[recipe_id] = r
        factors = RECIPE_FACTORS[recipe_id]
        print(f"  配方{recipe_id} ({ACTION_NAMES[recipe_id]}): "
              f"训练了 {len(r.models)} 个因子IC预测器，因子={factors}")

    return refiners


def evaluate_two_layer(agent: FQIAgent, refiners: dict) -> pd.DataFrame:
    """
    两层联合评估：
      第一层 FQI → 选配方
      第二层 WeightRefiner → 配方内XGBoost微调权重
    """
    print("\n=== 两层系统评估（样本外）===")
    test_env = FactorEnv(mode="test")

    records = []
    state = test_env.reset()
    done  = False

    while not done:
        # 第一层：FQI 选配方
        recipe = agent.greedy_action(state)

        # 第二层：WeightRefiner 根据当前状态微调该配方内的权重
        weights = refiners[recipe].predict_weights(state)

        # 用精调权重执行
        next_state, reward, done, info = test_env.step_with_weights(weights)

        records.append({
            "date":        str(info["date"])[:10],
            "recipe":      recipe,
            "recipe_name": ACTION_NAMES[recipe],
            "reward":      reward,
        })
        state = next_state

    df = pd.DataFrame(records)
    df["cumulative"] = df["reward"].cumsum()
    print(f"  [两层系统]  期数={len(df)}  平均IC={df['reward'].mean():.4f}"
          f"  ICIR={sharpe(df['reward']):.3f}  累积IC={df['reward'].sum():.4f}")
    return df


def plot_two_layer(df_two_layer: pd.DataFrame, df_fqi: pd.DataFrame, df_equal: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 累积IC对比
    ax = axes[0]
    ax.plot(df_equal["cumulative"].values,     label="等权基准",   color="steelblue", linestyle="--")
    ax.plot(df_fqi["cumulative"].values,        label="FQI单层",    color="orange",    linestyle="--")
    ax.plot(df_two_layer["cumulative"].values,  label="两层系统",   color="crimson",   linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("累积 RankIC：单层 vs 两层系统", fontsize=12)
    ax.set_xlabel("调仓期数")
    ax.set_ylabel("累积 RankIC")
    ax.legend()
    ax.grid(alpha=0.3)

    # 每期IC散点
    ax2 = axes[1]
    x = range(len(df_two_layer))
    ax2.bar(x, df_two_layer["reward"], color="crimson", alpha=0.6, label="两层系统")
    ax2.plot(x, df_fqi["reward"].values, color="orange", linestyle="--", alpha=0.8, label="FQI单层")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("每期 RankIC 对比", fontsize=12)
    ax2.set_xlabel("调仓期")
    ax2.set_ylabel("RankIC")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "two_layer_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"两层系统图已保存: {save_path}")


# ─────────────────── 主入口 ───────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    ensure_data()
    agent     = train_agent()
    results, summary = evaluate(agent)
    plot_results(results, agent)
    agent.save(OUTPUT_DIR / "fqi_agent.pkl")

    # 两层系统
    refiners     = train_refiners()
    df_two_layer = evaluate_two_layer(agent, refiners)

    df_fqi   = next(d for d in results if d["label"].iloc[0] == "FQI智能体")
    df_equal = next(d for d in results if d["label"].iloc[0] == "等权策略")
    plot_two_layer(df_two_layer, df_fqi, df_equal)

    # 汇总对比
    print("\n─── 最终绩效对比 ───")
    rows = []
    for label, df in [("等权基准", df_equal), ("FQI单层", df_fqi), ("两层系统", df_two_layer)]:
        r = df["reward"]
        rows.append({"策略": label, "平均IC": round(r.mean(), 4),
                     "ICIR": round(sharpe(r), 3), "累积IC": round(r.sum(), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 完成 ===")
    print("输出文件:")
    for p in sorted(OUTPUT_DIR.glob("*.{csv,png,pkl,parquet}")):
        print(f"  {p.name}")
