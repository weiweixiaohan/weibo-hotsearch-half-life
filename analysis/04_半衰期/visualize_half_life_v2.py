"""
半衰期数据可视化诊断 — 剔除半衰期 >120h 的极端长尾
"""
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

CSV_PATH = "weibo_hotsearch_2019.csv"
MAX_HL_HOURS = 120   # 半衰期上限
MAX_DUR_HOURS = 300  # 在榜时长上限


# 1. 加载 + 过滤

all_rows = []
with open(CSV_PATH, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if r['半衰期_秒'] and r['热度最大值']:
            hl_h = float(r['半衰期_秒']) / 3600
            all_rows.append({
                'name': r['词条名'],
                'duration_sec': int(r.get('在榜共计_秒', 0) or 0),
                'max_heat': float(r['热度最大值']),
                'half_life': float(r['半衰期_秒']),
                'hl_hours': hl_h,
            })

rows = [r for r in all_rows
        if r['hl_hours'] <= MAX_HL_HOURS
        and r['duration_sec'] / 3600 <= MAX_DUR_HOURS]
dropped_hl = [r for r in all_rows if r['hl_hours'] > MAX_HL_HOURS]
dropped_dur = [r for r in all_rows
               if r['hl_hours'] <= MAX_HL_HOURS
               and r['duration_sec'] / 3600 > MAX_DUR_HOURS]

print(f"原始有效记录: {len(all_rows)}")
print(f"剔除 半衰期 >{MAX_HL_HOURS}h: {len(dropped_hl)} 条")
for r in dropped_hl:
    print(f"  ✗ {r['name'][:30]}: 半衰期 {r['hl_hours']:.0f}h")
print(f"剔除 在榜时长 >{MAX_DUR_HOURS}h: {len(dropped_dur)} 条")
for r in dropped_dur:
    print(f"  ✗ {r['name'][:30]}: 在榜 {r['duration_sec']/3600:.0f}h")
print(f"纳入分析: {len(rows)} 条")

hl = [r['half_life'] for r in rows]
hl_hours = [r['hl_hours'] for r in rows]


# 2. 基础统计

med = np.median(hl_hours)
print(f"\n半衰期统计 (n={len(hl)}):")
print(f"  min={min(hl)/60:.0f}分钟  "
      f"P25={np.percentile(hl,25)/3600:.1f}h  "
      f"median={med:.1f}h  "
      f"mean={np.mean(hl)/3600:.1f}h  "
      f"P75={np.percentile(hl,75)/3600:.1f}h  "
      f"P90={np.percentile(hl,90)/3600:.1f}h  "
      f"max={max(hl)/3600:.1f}h")


# 3. 可视化

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

#  (a) 线性直方图 
ax = axes[0, 0]
ax.hist(hl_hours, bins=80, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(med, color='red', linestyle='--', linewidth=2, label=f'中位数={med:.1f}h')
ax.axvline(np.percentile(hl_hours, 90), color='orange', linestyle='--', linewidth=1.5,
           label=f'P90={np.percentile(hl_hours,90):.1f}h')
ax.set_xlabel('半衰期（小时）')
ax.set_ylabel('词条数')
ax.set_title(f'半衰期分布（剔除 >{MAX_HL_HOURS}h, n={len(hl)}）')
ax.legend(fontsize=8)

#  (b) 对数直方图 
ax = axes[0, 1]
log_hl = np.log10(np.maximum(hl_hours, 1 / 3600))
ax.hist(log_hl, bins=50, color='darkorange', edgecolor='white', alpha=0.85)
ax.axvline(np.log10(med), color='red', linestyle='--', linewidth=2, label=f'中位数={med:.1f}h')
ticks_hours = [1/3600, 1/60, 1, 12, 24, 120]
tick_labels = ['1秒', '1分钟', '1小时', '12小时', '1天', '5天']
ax.set_xticks([np.log10(t) for t in ticks_hours])
ax.set_xticklabels(tick_labels, fontsize=8)
ax.set_xlabel('半衰期')
ax.set_ylabel('词条数')
ax.set_title(f'半衰期对数分布（n={len(hl)}）')
ax.legend()

#  (c) 半衰期 vs 热度峰值 
ax = axes[0, 2]
ax.scatter([r['max_heat'] for r in rows], hl_hours,
           alpha=0.45, s=14, c='steelblue', edgecolors='none')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('热度最大值')
ax.set_ylabel('半衰期（小时）')
ax.set_title('半衰期 vs 热度峰值')
ax.axhline(med, color='red', linestyle='--', alpha=0.5, linewidth=1, label=f'中位数={med:.1f}h')
ax.legend(fontsize=8)

#  (d) 半衰期 vs 在榜时长 
ax = axes[1, 0]
dur_hours = [r['duration_sec'] / 3600 for r in rows]
ax.scatter(dur_hours, hl_hours, alpha=0.45, s=14, c='darkorange', edgecolors='none')
ax.set_xlim(0, 24)
ax.set_ylim(0, max(hl_hours) * 1.05)
ax.set_xlabel('在榜时长（小时，>24h 截断）')
ax.set_ylabel('半衰期（小时）')
ax.set_title('半衰期 vs 在榜时长')

#  (e) 分段箱线图 
ax = axes[1, 1]
time_groups = [
    ("<1分钟", [h for h in hl if h < 60]),
    ("1-5分钟", [h for h in hl if 60 <= h < 300]),
    ("5-10分钟", [h for h in hl if 300 <= h < 600]),
    ("10-30分钟", [h for h in hl if 600 <= h < 1800]),
    ("30-60分钟", [h for h in hl if 1800 <= h < 3600]),
    ("1-2小时", [h for h in hl if 3600 <= h < 7200]),
    ("2-6小时", [h for h in hl if 7200 <= h < 21600]),
    ("6-12小时", [h for h in hl if 21600 <= h < 43200]),
    ("12-24小时", [h for h in hl if 43200 <= h < 86400]),
    ("1-5天", [h for h in hl if 86400 <= h <= MAX_HL_HOURS * 3600]),
]
labels = [g[0] for g in time_groups]
data = [np.array(d) / 3600 for d in [g[1] for g in time_groups]]
bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False, vert=False)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_xlabel('半衰期（小时）')
ax.set_title(f'半衰期分段箱线图（n={len(hl)}）')
for i, (label, d) in enumerate(zip(labels, data)):
    ax.text(ax.get_xlim()[1], i + 1, f' n={len(d)}', va='center', fontsize=7)

#  (f) CDF 
ax = axes[1, 2]
sorted_hl = sorted(hl_hours)
y_arr = np.arange(1, len(sorted_hl) + 1) / len(sorted_hl)
ax.plot(sorted_hl, y_arr, 'steelblue', linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('半衰期（小时）对数值')
ax.set_ylabel('累计比例')
ax.set_title(f'半衰期累积分布 CDF（n={len(hl)}）')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax.axhline(0.9, color='orange', linestyle='--', alpha=0.5)
p50 = np.percentile(hl_hours, 50)
p90 = np.percentile(hl_hours, 90)
ax.annotate(f'P50={p50:.1f}h', (p50, 0.5), fontsize=8,
            xytext=(15, -10), textcoords='offset points', color='red')
ax.annotate(f'P90={p90:.1f}h', (p90, 0.9), fontsize=8,
            xytext=(15, -10), textcoords='offset points', color='orange')
ax.set_xlim(left=1 / 60)

plt.tight_layout()
out = "half_life_diagnosis_v2.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {out}")
plt.close()
