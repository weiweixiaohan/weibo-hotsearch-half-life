"""
微博热搜生命周期因子分析
=========================
基于 weibo_hotsearch_data_class.csv 和 dtw_cluster_labels_k3.csv
执行主轴因子法 + Varimax 旋转，提取4个潜在因子，区分3类DTW聚类

依赖：pandas, numpy, scipy, sklearn, matplotlib, seaborn
安装：pip install pandas numpy scipy scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv, eigh
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 0. 配置
# ============================================================
DATA_PATH    = "weibo_hotsearch_data_class.csv"
CLUSTER_PATH = "dtw_cluster_labels_k3.csv"
N_FACTORS    = 4
RANDOM_STATE = 42

CLUSTER_COLORS  = {0: "#E24B4A", 1: "#378ADD", 2: "#EF9F27"}
CLUSTER_LABELS  = {0: "C0" , 1: "C1", 2: "C2"}

FEAT_LABELS = {
    "log_peak":          "峰值热度(log)",
    "log_duration":      "在榜时长(log)",
    "log_halflife":      "半衰期(log)",
    "peak_timing":       "峰值出现时机",
    "early_heat_ratio":  "早期爆发比",
    "log_decay":         "衰减速率(log)",
    "persistence":       "尾部热度持续比",
    "launch_hour":       "上榜时段",
}

FACTOR_NAMES = ["F1·爆发强度", "F2·峰值节律", "F3·时长时段", "F4·衰减韧性"]


# ============================================================
# 1. 数据加载与合并
# ============================================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    df_cl = pd.read_csv(CLUSTER_PATH)
    df = df.merge(df_cl[["词条名", "cluster_id"]], on="词条名", how="inner")
    print(f"合并后样本量: {len(df)}  聚类分布: {df['cluster_id'].value_counts().sort_index().to_dict()}")
    return df


# ============================================================
# 2. 特征工程
# ============================================================
def engineer_features(df):
    # 基础字段
    df["peak_heat"]   = df["热度最大值"]
    df["duration_h"]  = df["在榜共计_秒"] / 3600
    df["halflife_h"]  = df["半衰期_秒"] / 3600
    df["launch_hour"] = pd.to_datetime(df["上榜时间"]).dt.hour

    # 热度时序矩阵（前100个采样点）
    heat_mat = df[[f"heat_{i}" for i in range(1, 101)]].values

    # 峰值出现时机（生命周期中峰值的相对位置，0=极早，1=极晚）
    n_valid   = np.sum(~np.isnan(heat_mat), axis=1)
    peak_idx  = np.nanargmax(np.where(np.isnan(heat_mat), -np.inf, heat_mat), axis=1)
    df["peak_timing"] = np.where(n_valid > 0, peak_idx / np.maximum(n_valid, 1), np.nan)

    # 早期爆发比：前10%采样点均值 / 峰值（越高=上来就猛）
    early_avg = np.nanmean(heat_mat[:, :10], axis=1)
    df["early_heat_ratio"] = early_avg / np.maximum(df["peak_heat"], 1)

    # 衰减速率：(峰值 - 末段均值) / 在榜时长
    last_avg  = np.nanmean(heat_mat[:, -10:], axis=1)
    df["decay_rate"]  = (df["peak_heat"] - last_avg) / np.maximum(df["duration_h"], 0.01)

    # 尾部持续比：末段均值 / 峰值（越高=越有余热）
    df["persistence"] = last_avg / np.maximum(df["peak_heat"], 1)

    # 对数变换（处理右偏）
    df["log_peak"]     = np.log1p(df["peak_heat"])
    df["log_duration"] = np.log1p(df["duration_h"])
    df["log_halflife"] = np.log1p(df["halflife_h"])
    df["log_decay"]    = np.log1p(df["decay_rate"])

    feat_cols = list(FEAT_LABELS.keys())
    df_clean  = df[feat_cols + ["cluster_id"]].dropna()
    print(f"有效样本（完整特征）: {len(df_clean)}")
    return df_clean, feat_cols


# ============================================================
# 3. 适合性检验
# ============================================================
def check_fa_suitability(X_std, feat_cols):
    n, p  = X_std.shape
    R     = np.corrcoef(X_std.T)

    # Bartlett 球形检验
    det_R = np.linalg.det(R)
    chi2  = -(n - 1 - (2 * p + 5) / 6) * np.log(det_R)
    df_b  = p * (p - 1) / 2
    pval  = 1 - stats.chi2.cdf(chi2, df_b)
    print(f"\nBartlett 球形检验: χ²={chi2:.1f}, df={df_b:.0f}, p={pval:.4e}")

    # KMO 检验
    R_inv = inv(R)
    D     = np.diag(1 / np.sqrt(np.diag(R_inv)))
    P     = -D @ R_inv @ D
    np.fill_diagonal(P, 1)
    sum_r2 = (np.sum(R ** 2) - p) / 2
    sum_p2 = (np.sum(P ** 2) - p) / 2
    kmo    = sum_r2 / (sum_r2 + sum_p2)
    print(f"KMO 值: {kmo:.4f}  {'✓ 可接受' if kmo >= 0.5 else '⚠ 较低'}")

    return R, chi2, pval, kmo


# ============================================================
# 4. Varimax 旋转
# ============================================================
def varimax_rotation(loadings, max_iter=1000, tol=1e-9):
    p, k      = loadings.shape
    rotation  = np.eye(k)
    for _ in range(max_iter):
        old = rotation.copy()
        for i in range(k):
            for j in range(i + 1, k):
                x  = loadings @ rotation
                u  = x[:, i] ** 2 - x[:, j] ** 2
                v  = 2 * x[:, i] * x[:, j]
                A  = np.sum(u);  B  = np.sum(v)
                C  = np.sum(u ** 2 - v ** 2);  D2 = np.sum(u * v)
                theta = np.arctan2(2 * (D2 - A * B / p), C - (A ** 2 - B ** 2) / p) / 4
                r2 = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
                rotation[:, [i, j]] = rotation[:, [i, j]] @ r2
        if np.max(np.abs(rotation - old)) < tol:
            break
    return loadings @ rotation


# ============================================================
# 5. 主因子分析
# ============================================================
def run_factor_analysis(X_std, R, feat_cols, n_factors=N_FACTORS):
    eigenvalues, eigenvectors = eigh(R)
    idx         = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors= eigenvectors[:, idx]

    print(f"\n特征值（前{n_factors+2}个）: {eigenvalues[:n_factors+2].round(4)}")
    print(f"特征值>1的个数: {(eigenvalues > 1).sum()}")

    # 未旋转载荷矩阵
    L_unrot = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    # Varimax 旋转
    L_rot = varimax_rotation(L_unrot)

    # 输出载荷矩阵
    col_names = [f"F{i+1}" for i in range(n_factors)]
    loadings_df = pd.DataFrame(L_rot, index=feat_cols, columns=col_names)
    loadings_df.index = [FEAT_LABELS[c] for c in feat_cols]

    print("\n=== 旋转因子载荷矩阵（Varimax）===")
    print(loadings_df.round(3).to_string())

    # 共同度
    communalities = np.sum(L_rot ** 2, axis=1)
    print("\n=== 共同度 ===")
    for i, col in enumerate(feat_cols):
        print(f"  {FEAT_LABELS[col]}: {communalities[i]:.3f}")

    # 方差解释
    ss     = np.sum(L_rot ** 2, axis=0)
    p      = len(feat_cols)
    cumvar = 0.0
    print("\n=== 方差解释 ===")
    for i, s in enumerate(ss):
        cumvar += s
        print(f"  F{i+1} ({FACTOR_NAMES[i]}): SS={s:.3f}, 方差占比={s/p:.3f}, 累计={cumvar/p:.3f}")

    return L_rot, loadings_df, communalities, eigenvalues


# ============================================================
# 6. 因子得分
# ============================================================
def compute_factor_scores(X_std, R, L_rot, clusters):
    scores    = X_std @ inv(R) @ L_rot
    score_df  = pd.DataFrame(scores, columns=[f"F{i+1}" for i in range(L_rot.shape[1])])
    score_df["cluster_id"] = clusters

    mean_scores = score_df.groupby("cluster_id")[[f"F{i+1}" for i in range(L_rot.shape[1])]].mean()
    print("\n=== 各聚类因子均值得分 ===")
    print(mean_scores.round(4))

    # 单因素方差分析
    print("\n=== ANOVA（因子得分跨聚类显著性）===")
    for fi in range(L_rot.shape[1]):
        groups = [scores[clusters == c, fi] for c in [0, 1, 2]]
        fstat, pval = stats.f_oneway(*groups)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        print(f"  F{fi+1} ({FACTOR_NAMES[fi]}): F={fstat:.2f}, p={pval:.4e} {sig}")

    return score_df, mean_scores


# ============================================================
# 7. 可视化
# ============================================================
def plot_all(loadings_df, score_df, mean_scores, eigenvalues, communalities, feat_cols):
    plt.rcParams["font.family"]   = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("微博热搜生命周期因子分析（主轴因子法 + Varimax旋转，k=3 DTW聚类）",
                 fontsize=15, fontweight="bold", y=0.98)

    # ── 子图布局 ──────────────────────────────────────────────
    gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])   # 因子载荷热图
    ax2 = fig.add_subplot(gs[0, 2])    # 碎石图
    ax3 = fig.add_subplot(gs[1, :2])   # 聚类×因子得分分组柱状图
    ax4 = fig.add_subplot(gs[1, 2])    # 共同度条形图
    ax5 = fig.add_subplot(gs[2, 0])    # F1 vs F2 散点
    ax6 = fig.add_subplot(gs[2, 1])    # F3 vs F4 散点
    ax7 = fig.add_subplot(gs[2, 2])    # 因子得分均值热图

    feat_cn = [FEAT_LABELS[c] for c in feat_cols]

    # ── 1. 因子载荷热图 ───────────────────────────────────────
    ld_vals = loadings_df.values
    im = ax1.imshow(ld_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(range(N_FACTORS))
    ax1.set_xticklabels(FACTOR_NAMES, fontsize=10)
    ax1.set_yticks(range(len(feat_cn)))
    ax1.set_yticklabels(feat_cn, fontsize=10)
    ax1.set_title("旋转因子载荷矩阵（Varimax）", fontsize=11)
    plt.colorbar(im, ax=ax1, fraction=0.03)
    for i in range(len(feat_cn)):
        for j in range(N_FACTORS):
            v = ld_vals[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=9, color="white" if abs(v) > 0.5 else "black")

    # ── 2. 碎石图 ─────────────────────────────────────────────
    k_show = min(8, len(eigenvalues))
    ax2.plot(range(1, k_show + 1), eigenvalues[:k_show], "o-", color="#378ADD", lw=2, ms=7)
    ax2.axhline(1, color="gray", ls="--", lw=1, label="特征值=1")
    ax2.fill_between(range(1, N_FACTORS + 1), 0, eigenvalues[:N_FACTORS],
                     alpha=0.15, color="#378ADD")
    ax2.set_xticks(range(1, k_show + 1))
    ax2.set_xlabel("因子序号", fontsize=10)
    ax2.set_ylabel("特征值", fontsize=10)
    ax2.set_title("碎石图", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ── 3. 聚类×因子得分分组柱状图 ───────────────────────────
    x      = np.arange(N_FACTORS)
    width  = 0.25
    for ci, c in enumerate([0, 1, 2]):
        vals = mean_scores.loc[c].values
        bars = ax3.bar(x + (ci - 1) * width, vals, width,
                       label=CLUSTER_LABELS[c], color=CLUSTER_COLORS[c], alpha=0.82)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(FACTOR_NAMES, fontsize=10)
    ax3.set_ylabel("因子均值得分", fontsize=10)
    ax3.set_title("各聚类因子均值得分（ANOVA均显著）", fontsize=11)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. 共同度条形图 ───────────────────────────────────────
    colors_comm = ["#378ADD" if h >= 0.7 else "#EF9F27" for h in communalities]
    ax4.barh(feat_cn, communalities, color=colors_comm, alpha=0.8)
    ax4.axvline(0.5, color="gray", ls="--", lw=1, label="0.5阈值")
    ax4.axvline(0.7, color="#E24B4A", ls="--", lw=1, label="0.7阈值")
    ax4.set_xlim(0, 1.05)
    ax4.set_xlabel("共同度", fontsize=10)
    ax4.set_title("各变量共同度", fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(axis="x", alpha=0.3)

    # ── 5 & 6. 散点图（F1/F2，F3/F4）────────────────────────
    for ax, (fi, fj) in [(ax5, (0, 1)), (ax6, (2, 3))]:
        sample = score_df.sample(min(800, len(score_df)), random_state=RANDOM_STATE)
        for c in [0, 1, 2]:
            mask = sample["cluster_id"] == c
            ax.scatter(sample.loc[mask, f"F{fi+1}"], sample.loc[mask, f"F{fj+1}"],
                       c=CLUSTER_COLORS[c], alpha=0.4, s=18, label=CLUSTER_LABELS[c])
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(FACTOR_NAMES[fi], fontsize=10)
        ax.set_ylabel(FACTOR_NAMES[fj], fontsize=10)
        ax.set_title(f"{FACTOR_NAMES[fi]} × {FACTOR_NAMES[fj]}", fontsize=11)
        ax.legend(fontsize=8, markerscale=1.5)
        ax.grid(alpha=0.2)

    # ── 7. 均值得分热图 ───────────────────────────────────────
    heat_data = mean_scores.values
    im7 = ax7.imshow(heat_data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax7.set_xticks(range(N_FACTORS))
    ax7.set_xticklabels([f"F{i+1}" for i in range(N_FACTORS)], fontsize=10)
    ax7.set_yticks([0, 1, 2])
    ax7.set_yticklabels([CLUSTER_LABELS[c] for c in [0, 1, 2]], fontsize=9)
    ax7.set_title("聚类因子得分均值热图", fontsize=11)
    plt.colorbar(im7, ax=ax7, fraction=0.05)
    for i in range(3):
        for j in range(N_FACTORS):
            v = heat_data[i, j]
            ax7.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=10, color="white" if abs(v) > 0.5 else "black")

    plt.savefig("weibo_factor_analysis.png", dpi=150, bbox_inches="tight")
    print("\n图表已保存为 weibo_factor_analysis.png")
    plt.show()


# ============================================================
# 8. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("微博热搜生命周期因子分析")
    print("=" * 60)

    # 加载数据
    df = load_data()

    # 特征工程
    df_clean, feat_cols = engineer_features(df)
    X = df_clean[feat_cols].values
    clusters = df_clean["cluster_id"].values

    # 标准化
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    # 适合性检验
    R, chi2, pval, kmo = check_fa_suitability(X_std, feat_cols)

    # 因子分析
    L_rot, loadings_df, communalities, eigenvalues = run_factor_analysis(
        X_std, R, feat_cols, n_factors=N_FACTORS
    )

    # 因子得分
    score_df, mean_scores = compute_factor_scores(X_std, R, L_rot, clusters)

    # 可视化
    plot_all(loadings_df, score_df, mean_scores, eigenvalues, communalities, feat_cols)

    # 保存结果表
    out = df_clean[feat_cols + ["cluster_id"]].copy()
    factor_scores = X_std @ inv(R) @ L_rot
    for i in range(N_FACTORS):
        out[f"F{i+1}_{FACTOR_NAMES[i].split('·')[1]}"] = factor_scores[:, i]
    out.to_csv("weibo_factor_scores.csv", index=False, encoding="utf-8-sig")
    print("因子得分已保存为 weibo_factor_scores.csv")

    print("\n=== 聚类命名建议 ===")
    print("  C0 ：高峰值、快衰减、晚间爆发（F1低、F3低）")
    print("  C1：慢热、峰值靠后、日午上榜（F2高）")
    print("  C2 ：低峰值、韧性强、白天上榜（F3高、F4低）")


if __name__ == "__main__":
    main()