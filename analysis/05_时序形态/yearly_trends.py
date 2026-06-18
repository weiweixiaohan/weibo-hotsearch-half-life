"""年份趋势：在榜时长 & 热度峰值 & 半衰期"""
import csv, numpy as np, matplotlib.pyplot as plt, matplotlib, warnings
from scipy.stats import kruskal, mannwhitneyu

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

MASTER = 'weibo_hotsearch_dynamics_master.csv'
MAX_DUR_H = 64

def parse_duration(text):
    try:
        h = int(text.split('时')[0])
        m = int(text.split('时')[1].split('分')[0])
        s = int(text.split('分')[1].split('秒')[0])
        return h * 3600 + m * 60 + s
    except: return 0

# 加载
YEAR_ORDER = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']
data = {y: {'dur': [], 'heat': [], 'hl': []} for y in YEAR_ORDER}

with open(MASTER, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        year = row['年份']
        if year not in data: continue

        dur_sec = int(float(row.get('在榜共计_秒', 0) or 0))
        if dur_sec == 0:
            dur_sec = parse_duration(row.get('在榜共计', ''))
        if dur_sec > MAX_DUR_H * 3600 or dur_sec == 0:
            continue

        hl_sec = float(row.get('半衰期_秒', 0) or 0)
        heat = float(row.get('热度最大值', 0) or 0)

        data[year]['dur'].append(dur_sec / 3600)
        data[year]['heat'].append(heat / 1e6)
        if hl_sec > 0:
            data[year]['hl'].append(hl_sec / 3600)

# 统计
def fmt_med(vals): return f'{np.median(vals):.1f}' if vals else 'N/A'

print(f"{'年份':>6s}  {'n':>5s}  {'在榜中位':>8s}  {'热度中位':>8s}  {'半衰中位':>8s}")
print('-' * 42)
for y in YEAR_ORDER:
    d, h, hl = data[y]['dur'], data[y]['heat'], data[y]['hl']
    print(f"{y:>6s}  {len(d):>5d}  {fmt_med(d):>8s}h  {fmt_med(h):>8s}M  {fmt_med(hl):>8s}h")

# Kruskal-Wallis + 两两 Mann-Whitney U (Bonferroni)
alpha_corr = 0.05 / 21  # 7年 × 6对/2 = 21 对比较
print(f'\n{"="*60}')
print(f'显著性检验 (Bonferroni α={alpha_corr:.5f})')
print(f'{"="*60}')

pairwise_sig = {}  # (metric, i, j) -> p-value
for metric, label in [('dur', '在榜时长'), ('heat', '热度峰值'), ('hl', '半衰期')]:
    groups = [data[y][metric] for y in YEAR_ORDER if data[y][metric]]
    stat, p_kw = kruskal(*groups)
    sig_kw = '***' if p_kw < 0.001 else '**' if p_kw < 0.01 else '*' if p_kw < 0.05 else 'ns'
    print(f'\n{label}: Kruskal-Wallis H={stat:.1f}, p={p_kw:.4f} {sig_kw}')

    # 相邻年份两两比较
    for i in range(len(YEAR_ORDER) - 1):
        gi, gj = data[YEAR_ORDER[i]][metric], data[YEAR_ORDER[i+1]][metric]
        if gi and gj:
            u, p = mannwhitneyu(gi, gj, alternative='two-sided')
            sig = '***' if p < alpha_corr else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            pairwise_sig[(metric, i, i+1)] = (p, sig)
            if sig != 'ns':
                print(f'  {YEAR_ORDER[i]}→{YEAR_ORDER[i+1]}: U={u:.0f}, p={p:.4f} {sig}')

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 7))

for ax, metric, ylabel, title in [
    (axes[0], 'dur', '在榜时长 (h)', '在榜时长'),
    (axes[1], 'heat', '热度峰值 (M)', '热度峰值'),
    (axes[2], 'hl', '半衰期 (h)', '半衰期'),
]:
    positions = range(7)
    for i, y in enumerate(YEAR_ORDER):
        vals = data[y][metric]
        if vals:
            bp = ax.boxplot(vals, positions=[i], widths=0.55, patch_artist=True,
                            showfliers=False, manage_ticks=False)
            bp['boxes'][0].set_facecolor(colors[i])
            bp['boxes'][0].set_alpha(0.85)
            ax.text(i, np.median(vals), f'n={len(vals)}\n{np.median(vals):.2f}',
                    ha='center', fontsize=7, va='bottom')

    # 叠加中位数趋势线
    meds = [np.median(data[y][metric]) if data[y][metric] else np.nan for y in YEAR_ORDER]
    ax.plot(positions, meds, 'ro-', linewidth=2, markersize=8, zorder=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(YEAR_ORDER, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # 相邻年份显著性标注
    y_max = max(np.median(data[y][metric]) if data[y][metric] else 0 for y in YEAR_ORDER)
    for i in range(len(YEAR_ORDER) - 1):
        if (metric, i, i+1) in pairwise_sig:
            p, sig = pairwise_sig[(metric, i, i+1)]
            if sig != 'ns':
                y_line = y_max * 1.05 + (i % 3) * y_max * 0.08
                ax.plot([i, i+1], [y_line, y_line], 'k-', linewidth=0.8)
                ax.text(i + 0.5, y_line + y_max*0.01, sig, ha='center', fontsize=7, color='red')

plt.suptitle(f'微博热搜年份趋势（在榜<{MAX_DUR_H}h, Bonferroni 校正两两检验）', fontsize=14, y=1.01)
plt.tight_layout()
out = 'yearly_trends.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\n图表: {out}')
plt.close()
