"""
在榜时长离群值检测 — 三图面板
"""
import csv, numpy as np, matplotlib.pyplot as plt, matplotlib, warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

MASTER = 'weibo_hotsearch_dynamics_master.csv'
THRESHOLD_H = 64

def parse_duration(text):
    try:
        h = int(text.split('时')[0])
        m = int(text.split('时')[1].split('分')[0])
        s = int(text.split('分')[1].split('秒')[0])
        return h * 3600 + m * 60 + s
    except: return 0

# ============================================================
# 1. 加载
# ============================================================
all_durs = []
with open(MASTER, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        dur_sec = int(float(row.get('在榜共计_秒', 0) or 0))
        if dur_sec == 0:
            dur_sec = parse_duration(row.get('在榜共计', ''))
        if dur_sec > 0:
            all_durs.append(dur_sec / 3600)

all_durs = np.array(all_durs)
print(f"全量: {len(all_durs)} 条")
print(f"在榜时长: min={all_durs.min():.1f}h, median={np.median(all_durs):.1f}h, "
      f"max={all_durs.max():.0f}h")
print(f">64h: {np.sum(all_durs > THRESHOLD_H)} 条 ({np.mean(all_durs > THRESHOLD_H)*100:.1f}%)")

# ============================================================
# 2. 三图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- (a) 对数直方图 ---
ax = axes[0]
log_durs = np.log10(np.maximum(all_durs, 1/60))
ax.hist(log_durs, bins=80, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(np.log10(THRESHOLD_H), color='red', linestyle='--', linewidth=2,
           label=f'{THRESHOLD_H}h 阈值')
# 分位数标注
for pct, color in [(50, 'darkgreen'), (90, 'darkorange'), (99, 'darkred')]:
    v = np.percentile(all_durs, pct)
    ax.axvline(np.log10(v), color=color, linestyle=':', linewidth=1.2, alpha=0.6)
    ax.text(np.log10(v), ax.get_ylim()[1]*0.92, f'P{pct}={v:.0f}h',
            fontsize=7, color=color, ha='center')

# X 轴刻度
ticks_h = [1/60, 1, 24, THRESHOLD_H, 7*24, 30*24, 365*24]
tick_labels = ['1分钟', '1h', '1天', f'{THRESHOLD_H}h', '1周', '1月', '1年']
ax.set_xticks([np.log10(t) for t in ticks_h])
ax.set_xticklabels(tick_labels, fontsize=8)
ax.set_xlabel('在榜时长')
ax.set_ylabel('词条数')
ax.set_title(f'在榜时长分布（对数轴, n={len(all_durs)}）')
ax.legend(fontsize=8)

# --- (b) 箱线图 + 散点 ---
ax = axes[1]
# 箱线图（滤掉 >64h）
durs_clean = all_durs[all_durs <= THRESHOLD_H]
bp = ax.boxplot(durs_clean, positions=[0], widths=0.4, patch_artist=True, showfliers=False)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][0].set_alpha(0.5)

# 散点（jitter, log y）
np.random.seed(42)
jitter = np.random.uniform(-0.15, 0.15, len(all_durs))
colors = ['coral' if d > THRESHOLD_H else 'steelblue' for d in all_durs]
alphas = [0.7 if d > THRESHOLD_H else 0.08 for d in all_durs]
ax.scatter(jitter, all_durs, c=colors, alpha=alphas, s=12, edgecolors='none')

ax.axhline(THRESHOLD_H, color='red', linestyle='--', linewidth=2, label=f'{THRESHOLD_H}h 阈值')
ax.set_yscale('log')
ax.set_ylabel('在榜时长（小时，对数轴）')
ax.set_xticks([])
ax.set_title(f'箱线图 + 全量散点（红点={np.sum(all_durs > THRESHOLD_H)} 条离群）')
ax.legend(fontsize=8)

# 标注
ax.text(0.5, THRESHOLD_H * 1.5, f'剔除 {np.sum(all_durs > THRESHOLD_H)} 条\n({np.mean(all_durs > THRESHOLD_H)*100:.1f}%)',
        fontsize=9, color='red', ha='center', fontweight='bold')

# --- (c) CDF + 筛选效率 ---
ax = axes[3] if len(axes) > 3 else axes[2]
# 实际用 axes[2]
sorted_durs = np.sort(all_durs)
y_cdf = np.arange(1, len(sorted_durs)+1) / len(sorted_durs)

ax_cdf = axes[2]
ax_cdf.plot(sorted_durs, y_cdf * 100, 'steelblue', linewidth=2.2, label='CDF')
ax_cdf.set_xscale('log')
ax_cdf.set_xlabel('在榜时长（小时，对数轴）')
ax_cdf.set_ylabel('累积占比 (%)', color='steelblue')
ax_cdf.tick_params(axis='y', labelcolor='steelblue')

# 标注阈值
idx_64 = np.searchsorted(sorted_durs, THRESHOLD_H)
pct_64 = y_cdf[idx_64] * 100
ax_cdf.axvline(THRESHOLD_H, color='red', linestyle='--', linewidth=2, alpha=0.6)
ax_cdf.axhline(pct_64, color='red', linestyle='--', linewidth=1, alpha=0.4)
ax_cdf.annotate(f'{THRESHOLD_H}h → 保留 {pct_64:.1f}%\n剔除 {np.sum(all_durs > THRESHOLD_H)} 条极端长尾',
                (THRESHOLD_H, pct_64), fontsize=9, fontweight='bold',
                xytext=(15, -25), textcoords='offset points', color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

# 右轴: 筛选效率（保留比例 × 数据稳定性 = 中位数/均值比）
ax_eff = ax_cdf.twiny()
# 计算不同阈值下的「保留比例」和「中位数/均值比」（均值被长尾拉偏程度）
thresholds = np.logspace(-1, 4, 200)
pct_retained = [np.mean(all_durs <= t) for t in thresholds]
median_mean_ratio = [np.median(all_durs[all_durs <= t]) / np.mean(all_durs[all_durs <= t])
                     if np.sum(all_durs <= t) > 10 else np.nan
                     for t in thresholds]
ax_eff.plot(thresholds, median_mean_ratio, 'darkorange', linewidth=1.5, alpha=0.7,
            label='中位数/均值比')
ax_eff.set_xscale('log')
ax_eff.set_xlabel('阈值（小时）', color='darkorange')
ax_eff.tick_params(axis='x', labelcolor='darkorange')
ax_eff.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
ax_eff.axvline(THRESHOLD_H, color='red', linestyle='--', linewidth=1, alpha=0.3)

ax_cdf.set_title(f'CDF + 筛选效率（{THRESHOLD_H}h 保留 {pct_64:.1f}% 数据）')

# 合并图例
lines1, labels1 = ax_cdf.get_legend_handles_labels()
lines2, labels2 = ax_eff.get_legend_handles_labels()
ax_cdf.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower right')

plt.suptitle(f'在榜时长离群值检测（n={len(all_durs)}, 阈值={THRESHOLD_H}h）',
             fontsize=14, y=1.01)
plt.tight_layout()
out = 'duration_outlier_detection.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\n图表: {out}')
plt.close()
