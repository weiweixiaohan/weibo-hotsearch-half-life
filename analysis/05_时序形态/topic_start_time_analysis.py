"""
词条启动时间探索 v2 — 全七年 + 半衰期趋势
"""
import csv, numpy as np, matplotlib.pyplot as plt, matplotlib
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import kruskal
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.unicode_minus'] = False

MASTER_CSV = 'weibo_hotsearch_dynamics_master.csv'
YEAR_GROUPS = ['2019', '2020-22', '2023', '2024', '2025']
SLOT_NAMES = ['凌晨', '上午', '下午', '晚上']
SLOT_COLORS = ['#6b7280', '#fbbf24', '#fb923c', '#1e40af']
MAX_DUR_H = 64


def classify(hour):
    if 6 <= hour <= 11:   return 1
    if 12 <= hour <= 17:  return 2
    if 18 <= hour <= 23:  return 3
    return 0


def year_group(y):
    if y in ('2020', '2021', '2022'): return '2020-22'
    return y



# 1. 加载全七年
def parse_duration(text):
    """ '3时15分2秒' → 秒数 """
    try:
        h = int(text.split('时')[0])
        m = int(text.split('时')[1].split('分')[0])
        s = int(text.split('分')[1].split('秒')[0])
        return h * 3600 + m * 60 + s
    except:
        return 0

all_topics = []
with open(MASTER_CSV, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        dur_sec = int(float(row.get('在榜共计_秒', 0) or 0))
        if dur_sec == 0:  # 尝试从文本列回退
            dur_text = row.get('在榜共计', '')
            if dur_text:
                dur_sec = parse_duration(dur_text)
        if dur_sec > MAX_DUR_H * 3600 or dur_sec == 0:
            continue

        uptime = row.get('上榜时间', '')
        up_hour = -1
        if uptime:
            try: up_hour = int(uptime[11:13])
            except: pass
        if not (0 <= up_hour <= 23):
            continue

        hl_sec = float(row.get('半衰期_秒', 0) or 0)
        i = 1; max_h = 0; pk_hour = -1
        while True:
            tk = f't_{i}'; hk = f'heat_{i}'
            if tk not in row or not row[tk]: break
            try:
                h = float(row[hk])
                if h > max_h: max_h = h; pk_hour = int(row[tk][11:13])
            except: pass
            i += 1

        all_topics.append({
            'name': row['词条名'],
            'year': row['年份'],
            'ygroup': year_group(row['年份']),
            'up_slot': classify(up_hour),
            'dur_sec': dur_sec,
            'max_heat': max_h,
            'hl_hours': hl_sec / 3600 if hl_sec else 0,
        })

print(f"加载: {len(all_topics)} 条 (在榜<{MAX_DUR_H}h)")

# 统计各年
for yg in YEAR_GROUPS:
    n = sum(1 for t in all_topics if t['ygroup'] == yg)
    print(f"  {yg}: {n} 条")


# 2. 显著性检验（函数）
def run_stats(groups, year_label):
    """groups: {slot: [values]}"""
    slots_present = [s for s in range(4) if len(groups.get(s, [])) >= 5]
    if len(slots_present) >= 2:
        try:
            stat, p = kruskal(*[groups[s] for s in slots_present])
        except ValueError:
            return f"{year_label}: n/a (all identical)"
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        return f"{year_label}: H={stat:.1f}, p={p:.4f} {sig}"
    return ""



# 3. 可视化 (3x3)
fig, axes = plt.subplots(3, 3, figsize=(18, 16))

# --- Row 1: 在榜时长---
# (a) 在榜时长 × 启动时段（七年汇总）
ax = axes[0, 0]
for slot in range(4):
    durs = [t['dur_sec']/3600 for t in all_topics if t['up_slot'] == slot]
    bp = ax.boxplot(durs, positions=[slot], widths=0.55, patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(SLOT_COLORS[slot])
    ax.text(slot, np.median(durs), f'n={len(durs)}\n{np.median(durs):.1f}h',
            ha='center', fontsize=7, va='bottom')
ax.set_xticks(range(4)); ax.set_xticklabels(SLOT_NAMES)
ax.set_ylabel('在榜时长 (h)')
ax.set_title(f'在榜时长 × 启动时段 (n={len(all_topics)})')

# (b) 在榜时长 × 年份趋势
ax = axes[0, 1]
for slot in range(4):
    meds = []
    for yg in YEAR_GROUPS:
        d = [t['dur_sec']/3600 for t in all_topics if t['ygroup'] == yg and t['up_slot'] == slot]
        meds.append(np.median(d) if d else np.nan)
    ax.plot(range(5), meds, 'o-', color=SLOT_COLORS[slot], linewidth=1.8, markersize=8, label=SLOT_NAMES[slot])
ax.set_xticks(range(5)); ax.set_xticklabels(YEAR_GROUPS)
ax.set_ylabel('在榜时长中位数 (h)')
ax.set_title('在榜时长 × 年份趋势')
ax.legend(fontsize=8)

# (c) 在榜时长 显著性
ax = axes[0, 2]
ax.axis('off')
stats_lines = []
for yg in YEAR_GROUPS + ['七年汇总']:
    if yg == '七年汇总':
        ts = all_topics
    else:
        ts = [t for t in all_topics if t['ygroup'] == yg]
    groups = {s: [t['dur_sec']/3600 for t in ts if t['up_slot'] == s] for s in range(4)}
    line = run_stats(groups, yg)
    if line: stats_lines.append(line)
ax.text(0.05, 0.95, '在榜时长 显著性\n(Kruskal-Wallis)\n\n' + '\n'.join(stats_lines),
        transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace')
ax.set_title('显著性检验')

# --- Row 2: 热度峰值---
# (d) 热度峰值 × 启动时段
ax = axes[1, 0]
for slot in range(4):
    heats = [t['max_heat']/1e6 for t in all_topics if t['up_slot'] == slot]
    bp = ax.boxplot(heats, positions=[slot], widths=0.55, patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(SLOT_COLORS[slot])
    ax.text(slot, np.median(heats), f'n={len(heats)}\n{np.median(heats):.2f}M',
            ha='center', fontsize=7, va='bottom')
ax.set_xticks(range(4)); ax.set_xticklabels(SLOT_NAMES)
ax.set_ylabel('热度峰值 (M)')
ax.set_title(f'热度峰值 × 启动时段 (n={len(all_topics)})')

# (e) 热度峰值 × 年份趋势
ax = axes[1, 1]
for slot in range(4):
    meds = []
    for yg in YEAR_GROUPS:
        d = [t['max_heat']/1e6 for t in all_topics if t['ygroup'] == yg and t['up_slot'] == slot]
        meds.append(np.median(d) if d else np.nan)
    ax.plot(range(5), meds, 'o-', color=SLOT_COLORS[slot], linewidth=1.8, markersize=8, label=SLOT_NAMES[slot])
ax.set_xticks(range(5)); ax.set_xticklabels(YEAR_GROUPS)
ax.set_ylabel('热度峰值中位数 (M)')
ax.set_title('热度峰值 × 年份趋势')
ax.legend(fontsize=8)

# (f) 热度峰值 显著性
ax = axes[1, 2]
ax.axis('off')
stats_lines2 = []
for yg in YEAR_GROUPS + ['七年汇总']:
    if yg == '七年汇总':
        ts = all_topics
    else:
        ts = [t for t in all_topics if t['ygroup'] == yg]
    groups = {s: [t['max_heat']/1e6 for t in ts if t['up_slot'] == s] for s in range(4)}
    line = run_stats(groups, yg)
    if line: stats_lines2.append(line)
ax.text(0.05, 0.95, '热度峰值 显著性\n(Kruskal-Wallis)\n\n' + '\n'.join(stats_lines2),
        transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace')
ax.set_title('显著性检验')

# --- Row 3: 半衰期趋势---
# (g) 半衰期 × 启动时段
ax = axes[2, 0]
for slot in range(4):
    hls = [t['hl_hours'] for t in all_topics if t['up_slot'] == slot and t['hl_hours'] > 0]
    bp = ax.boxplot(hls, positions=[slot], widths=0.55, patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(SLOT_COLORS[slot])
    ax.text(slot, np.median(hls), f'n={len(hls)}\n{np.median(hls):.1f}h',
            ha='center', fontsize=7, va='bottom')
ax.set_xticks(range(4)); ax.set_xticklabels(SLOT_NAMES)
ax.set_ylabel('半衰期 (h)')
ax.set_title(f'半衰期 × 启动时段 (n={sum(1 for t in all_topics if t["hl_hours"]>0)})')

# (h) 半衰期 × 年份趋势
ax = axes[2, 1]
for slot in range(4):
    meds = []
    for yg in YEAR_GROUPS:
        d = [t['hl_hours'] for t in all_topics if t['ygroup'] == yg and t['up_slot'] == slot and t['hl_hours'] > 0]
        meds.append(np.median(d) if d else np.nan)
    ax.plot(range(5), meds, 'o-', color=SLOT_COLORS[slot], linewidth=1.8, markersize=8, label=SLOT_NAMES[slot])
ax.set_xticks(range(5)); ax.set_xticklabels(YEAR_GROUPS)
ax.set_ylabel('半衰期中位数 (h)')
ax.set_title('半衰期 × 年份趋势')
ax.legend(fontsize=8)

# (i) 半衰期 显著性
ax = axes[2, 2]
ax.axis('off')
stats_lines3 = []
for yg in YEAR_GROUPS + ['七年汇总']:
    if yg == '七年汇总':
        ts = all_topics
    else:
        ts = [t for t in all_topics if t['ygroup'] == yg]
    groups = {s: [t['hl_hours'] for t in ts if t['up_slot'] == s and t['hl_hours'] > 0] for s in range(4)}
    line = run_stats(groups, yg)
    if line: stats_lines3.append(line)
ax.text(0.05, 0.95, '半衰期 显著性\n(Kruskal-Wallis)\n\n' + '\n'.join(stats_lines3),
        transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace')
ax.set_title('显著性检验')


plt.suptitle(f'词条启动时段探索 v2 — 全七年 — 在榜时长 & 热度峰值 & 半衰期 (在榜<{MAX_DUR_H}h)',
             fontsize=14, y=1.01)
plt.tight_layout()
out = 'topic_start_exploration.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {out}")


# 打印数值表
print(f"\n{'='*70}")
print("在榜时长中位数 (h)")
print(f"{'年份':>10s}  {'凌晨':>6s}  {'上午':>6s}  {'下午':>6s}  {'晚上':>6s}")
for yg in YEAR_GROUPS:
    ts = [t for t in all_topics if t['ygroup'] == yg]
    vals = [f"{np.median([t['dur_sec']/3600 for t in ts if t['up_slot']==s]):5.1f}" if ts else '   N/A' for s in range(4)]
    print(f"{yg:>10s}  {vals[0]:>6s}  {vals[1]:>6s}  {vals[2]:>6s}  {vals[3]:>6s}")

print(f"\n热度峰值中位数 (M)")
for yg in YEAR_GROUPS:
    ts = [t for t in all_topics if t['ygroup'] == yg]
    vals = [f"{np.median([t['max_heat']/1e6 for t in ts if t['up_slot']==s]):5.2f}" if ts else '   N/A' for s in range(4)]
    print(f"{yg:>10s}  {vals[0]:>6s}  {vals[1]:>6s}  {vals[2]:>6s}  {vals[3]:>6s}")

print(f"\n半衰期中位数 (h)")
for yg in YEAR_GROUPS:
    ts = [t for t in all_topics if t['ygroup'] == yg]
    vals = [f"{np.median([t['hl_hours'] for t in ts if t['up_slot']==s and t['hl_hours']>0]):5.1f}" if ts else '   N/A' for s in range(4)]
    print(f"{yg:>10s}  {vals[0]:>6s}  {vals[1]:>6s}  {vals[2]:>6s}  {vals[3]:>6s}")

plt.close()
