"""
事件类型维度分析：半衰期 & 启动时段
"""
import csv, numpy as np, matplotlib.pyplot as plt, matplotlib, warnings
from scipy.stats import kruskal, mannwhitneyu
from collections import Counter

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

MASTER_CSV = 'weibo_hotsearch_dynamics_master.csv'
MAX_DUR_H = 64
SLOT_NAMES = ['凌晨', '上午', '下午', '晚上']


def classify(hour):
    if 6 <= hour <= 11: return 1
    if 12 <= hour <= 17: return 2
    if 18 <= hour <= 23: return 3
    return 0



# 1. 加载数据
print("加载 master CSV ...")
records = []
with open(MASTER_CSV, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        dur_sec = int(float(row.get('在榜共计_秒', 0) or 0))
        if dur_sec > MAX_DUR_H * 3600:
            continue
        hl_sec = float(row.get('半衰期_秒', 0) or 0)
        if not hl_sec:
            continue

        uptime = row.get('上榜时间', '')
        up_hour = -1
        if uptime:
            try: up_hour = int(uptime[11:13])
            except: pass
        up_slot = classify(up_hour) if 0 <= up_hour <= 23 else -1

        records.append({
            'name': row['词条名'],
            'year': row.get('年份', ''),
            'etype': row.get('事件类型', ''),
            'hl_hours': hl_sec / 3600,
            'dur_hours': dur_sec / 3600,
            'max_heat': float(row.get('热度最大值', 0) or 0),
            'up_slot': up_slot,
        })

print(f"有效记录: {len(records)}")

# Top 事件类型
type_counts = Counter(r['etype'] for r in records)
top_types = [t for t, _ in type_counts.most_common(10)]
print(f"\nTop 10 事件类型: {[(t, type_counts[t]) for t in top_types]}")


# 2. 半衰期 × 事件类型
print(f"\n{'='*60}")
print("半衰期 × 事件类型 统计分析")
print(f"{'='*60}")

# 分组数据
groups_hl = {t: [r['hl_hours'] for r in records if r['etype'] == t] for t in top_types}

# Kruskal-Wallis
stat, p = kruskal(*groups_hl.values())
print(f"\nKruskal-Wallis: H={stat:.1f}, p={p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")

# 两两比较 (Bonferroni)
print(f"\n两两比较 (Mann-Whitney U + Bonferroni, α={0.05/45:.5f}):")
n_comparisons = 45  # 10 choose 2
alpha_c = 0.05 / n_comparisons
pairs_hl = []
for i in range(len(top_types)):
    for j in range(i + 1, len(top_types)):
        gi = groups_hl[top_types[i]]
        gj = groups_hl[top_types[j]]
        u, pmw = mannwhitneyu(gi, gj, alternative='two-sided')
        sig = '***' if pmw < alpha_c else 'ns'
        pairs_hl.append((top_types[i], top_types[j], np.median(gi), np.median(gj), pmw, sig))
        if sig == '***':
            print(f"  {top_types[i]}({np.median(gi):.1f}h) vs {top_types[j]}({np.median(gj):.1f}h): p={pmw:.4f} {sig}")

print(f"\n半衰期中位数 (h):")
for t in top_types:
    vals = groups_hl[t]
    print(f"  {t}: median={np.median(vals):.1f}, mean={np.mean(vals):.1f}, n={len(vals)}")


# 3. 启动时段 × 事件类型
print(f"\n{'='*60}")
print("启动时段 × 事件类型")
print(f"{'='*60}")

# 交叉表
slot_etype = np.zeros((4, len(top_types)))
for r in records:
    if r['up_slot'] >= 0 and r['etype'] in top_types:
        slot_etype[r['up_slot'], top_types.index(r['etype'])] += 1
slot_pct = slot_etype / slot_etype.sum(axis=0, keepdims=True) * 100

print(f"\n启动时段占比 (%):")
print(f"{'类型':<12s} {'凌晨':>6s} {'上午':>6s} {'下午':>6s} {'晚上':>6s}")
for ci, t in enumerate(top_types):
    print(f"  {t:<10s} {slot_pct[0,ci]:5.1f}% {slot_pct[1,ci]:5.1f}% "
          f"{slot_pct[2,ci]:5.1f}% {slot_pct[3,ci]:5.1f}%")


# 4. 可视化 (1×2)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors_types = plt.cm.tab10(np.linspace(0, 1, len(top_types)))

# (a) 半衰期箱线图 × 事件类型
ax = axes[0]
data_hl = [groups_hl[t] for t in top_types]
bp = ax.boxplot(data_hl, patch_artist=True, showfliers=False, widths=0.6)
for patch, color in zip(bp['boxes'], colors_types):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 显著性标注
max_med = max(np.median(g) for g in data_hl)
for (t1, t2, med1, med2, pmw, sig), yi in zip(pairs_hl, range(len(pairs_hl))):
    if sig == '***':
        i1, i2 = top_types.index(t1), top_types.index(t2)
        ax.plot([i1 + 1, i2 + 1], [max_med * 1.1, max_med * 1.1], 'k-', linewidth=0.5, alpha=0.3)
        ax.text((i1 + i2) / 2 + 1, max_med * 1.12, '*', ha='center', fontsize=6, color='red')

ax.set_xticklabels(top_types, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('半衰期（小时）')
ax.set_title(f'半衰期 × 事件类型（Kruskal-Wallis p={p:.4f}）')
# 显示样本量
for i, d in enumerate(data_hl):
    ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(d)}', ha='center', fontsize=7, color='gray')

# (b) 启动时段 × 事件类型 堆叠柱状图
ax = axes[1]
bottom = np.zeros(len(top_types))
bar_colors = ['#6b7280', '#fbbf24', '#fb923c', '#1e40af']
for slot in range(4):
    ax.bar(range(len(top_types)), slot_pct[slot], bottom=bottom,
           color=bar_colors[slot], label=SLOT_NAMES[slot], edgecolor='white', width=0.6)
    bottom += slot_pct[slot]

ax.set_xticks(range(len(top_types)))
ax.set_xticklabels(top_types, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('占比 (%)')
ax.set_title('启动时段 × 事件类型（堆叠 100%）')
ax.legend(fontsize=8, loc='upper right')


plt.suptitle(f'事件类型维度分析（在榜<{MAX_DUR_H}h, n={len(records)}）', fontsize=14, y=1.01)
plt.tight_layout()
out = 'event_type_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {out}")
plt.close()
