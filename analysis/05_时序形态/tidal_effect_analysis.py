"""
微博热搜潮汐效应分析 — 大盘 24 小时热度曲线
"""
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

CSV_PATH = "weibo_hotsearch_2019.csv"


# 1. 展开全量 (hour, heat) 对
TIME_FMT = "%Y-%m-%d %H:%M:%S"
hour_heats = defaultdict(list)   # hour -> [heat, ...]
hour_names = defaultdict(set)    # hour -> {unique topic names}

total_pairs = 0
with open(CSV_PATH, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['词条名']
        i = 1
        while True:
            t_key = f't_{i}'
            h_key = f'heat_{i}'
            if t_key not in row or not row[t_key]:
                break
            try:
                t = row[t_key]
                h = float(row[h_key])
                hour = int(t[11:13])
                if 0 <= hour <= 23:
                    hour_heats[hour].append(h)
                    hour_names[hour].add(name)
                    total_pairs += 1
            except (ValueError, IndexError):
                pass
            i += 1

print(f"全量时间-热度对: {total_pairs}")
print(f"覆盖小时: {sorted(hour_heats.keys())}")


# 2. 按小时聚合
hours = np.arange(24)
avg_heat = np.zeros(24)
total_heat = np.zeros(24)
count = np.zeros(24, dtype=int)

for h in range(24):
    vals = hour_heats.get(h, [])
    if vals:
        avg_heat[h] = np.mean(vals)
        total_heat[h] = np.sum(vals)
        count[h] = len(hour_names.get(h, set()))   # 去重：唯一词条数
    else:
        avg_heat[h] = np.nan
        total_heat[h] = 0
        count[h] = 0

# 打印 24 小时数据
print(f"\n{'小时':>5s}  {'平均热度':>12s}  {'总热度(M)':>10s}  {'去重词条数':>8s}")
print("-" * 40)
for h in range(24):
    print(f"  {h:02d}:00  {avg_heat[h]:>10.0f}  {total_heat[h]/1e6:>8.1f}M  {count[h]:>8,d}")


# 3. 双轴图
fig, ax1 = plt.subplots(figsize=(14, 6))

color1 = 'steelblue'
ax1.set_xlabel('物理时间（小时）')
ax1.set_ylabel('平均热度', color=color1)
line1, = ax1.plot(hours, avg_heat, color=color1, linewidth=2.2, marker='o', markersize=6, label='平均热度')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(hours)
ax1.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45)

# 波峰标注
peak_idx = np.argmax(avg_heat)
ax1.annotate(f'峰值 {avg_heat[peak_idx]:.0f}\n{peak_idx}:00', (peak_idx, avg_heat[peak_idx]),
             fontsize=10, fontweight='bold',
             xytext=(0, -30), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle='->', color='red'), color='red')

# 次峰检测（排除峰值 ±2h 邻域）
exclude = set(range(max(0, peak_idx-2), min(24, peak_idx+3)))
candidates = [(avg_heat[h], h) for h in range(24) if h not in exclude and not np.isnan(avg_heat[h])]
if candidates:
    second_val, second_h = max(candidates)
    ax1.annotate(f'次峰 {second_val:.0f}\n{second_h}:00', (second_h, second_val),
                 fontsize=9, xytext=(0, -25), textcoords='offset points', ha='center',
                 arrowprops=dict(arrowstyle='->', color='darkorange'), color='darkorange')

# 右轴: 活跃词条数
ax2 = ax1.twinx()
color2 = 'darkorange'
ax2.set_ylabel('去重词条数', color=color2)
ax2.fill_between(hours, 0, count, alpha=0.2, color=color2)
line2, = ax2.plot(hours, count, color=color2, linewidth=1.8, marker='s', markersize=5, label='去重词条数')
ax2.tick_params(axis='y', labelcolor=color2)

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=9)

ax1.set_title(f'微博热搜 24 小时大盘热度（潮汐效应诊断, n={total_pairs:,} 个采样点）')
ax1.set_xlim(-0.5, 23.5)

# 标注典型时段
for h, label, color in [(8, '早间', 'gray'), (12, '午间', 'gray'), (20, '晚间', 'gray')]:
    ax1.axvline(h, color=color, linestyle=':', alpha=0.4, linewidth=1)
    ax1.text(h, ax1.get_ylim()[1] * 0.95, label, fontsize=8, ha='center', color='gray')

plt.tight_layout()
out = "tidal_effect.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {out}")
plt.close()
