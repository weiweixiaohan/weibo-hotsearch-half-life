"""数据点数阈值证据图"""
import csv, numpy as np, matplotlib.pyplot as plt, matplotlib, warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

MASTER = 'weibo_hotsearch_dynamics_master.csv'

def parse_duration(text):
    try:
        h = int(text.split('时')[0])
        m = int(text.split('时')[1].split('分')[0])
        s = int(text.split('分')[1].split('秒')[0])
        return h*3600 + m*60 + s
    except: return 0

# 加载
counts, durs = [], []
with open(MASTER, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        i = 1
        while f'heat_{i}' in row and row[f'heat_{i}']: i += 1
        n = i - 1
        if n <= 0: continue
        dur_sec = int(float(row.get('在榜共计_秒',0) or 0))
        if dur_sec == 0: dur_sec = parse_duration(row.get('在榜共计',''))
        counts.append(n)
        durs.append(dur_sec / 3600)

counts = np.array(counts)
durs = np.array(durs)
print(f"n={len(counts)}, 点数中位数={np.median(counts):.0f}, "
      f"<6点: {np.sum(counts<6)} 条 ({np.mean(counts<6)*100:.1f}%)")

# 三图
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) 点数分布直方图
ax = axes[0]
bins = np.concatenate([np.arange(0.5, 20.5, 1), np.logspace(np.log10(21), np.log10(1000), 50)])
ax.hist(np.clip(counts, 1, 1000), bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(6, color='red', linestyle='--', linewidth=2, label='6 点阈值')
ax.set_xscale('log')
ax.set_xlabel('数据点数（对数轴）')
ax.set_ylabel('词条数')
ax.set_title(f'热度数据点数分布（n={len(counts)}）')
ax.legend(fontsize=8)
# 标注
ax.annotate(f'<6 点\n{np.sum(counts<6)} 条\n({np.mean(counts<6)*100:.1f}%)',
            (2, ax.get_ylim()[1]*0.7), fontsize=9, color='red', fontweight='bold', ha='center')

# (b) 点数 vs 在榜时长散点
ax = axes[1]
mask_low = counts < 6
ax.scatter(counts[~mask_low], durs[~mask_low], alpha=0.15, s=8, c='steelblue', edgecolors='none', label=f'≥6 点')
ax.scatter(counts[mask_low], durs[mask_low], alpha=0.9, s=30, c='coral', edgecolors='white', linewidth=0.5, label=f'<6 点 ({mask_low.sum()})')
ax.axvline(6, color='red', linestyle='--', linewidth=2)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('数据点数（对数轴）'); ax.set_ylabel('在榜时长（小时，对数轴）')
ax.set_title('数据点数 vs 在榜时长')
ax.legend(fontsize=8)
# 标注区域
ax.axhspan(0, 2, xmin=0, xmax=0.05, alpha=0.08, color='red')
ax.text(2, 0.5, f'快闪区\n在榜<2h', fontsize=8, color='red', alpha=0.7)

# (c) 阈值 vs 保留比例 & 可分析性
ax = axes[2]
thresholds = np.arange(1, 30)
pct_retained = [np.mean(counts >= t) * 100 for t in thresholds]
pct_retained = [np.mean(counts >= t)*100 for t in thresholds]
# 可分析性定义: 至少足够做 basic stats (median), peak detection, half-life
ax.plot(thresholds, pct_retained, 'steelblue', linewidth=2.5, label='保留比例')
ax.fill_between(thresholds, pct_retained, 100, alpha=0.1, color='steelblue')

# 标注各级阈值含义
for t, label, color in [
    (2, '能算中位数\n(>2点)', 'gray'),
    (6, '能算半衰期\n(≥6点)', 'red'),
    (10, '能做平滑\n(≥10点)', 'darkorange'),
]:
    pct = np.mean(counts >= t) * 100
    ax.axvline(t, color=color, linestyle='--', alpha=0.5)
    ax.text(t, pct + 3, label, fontsize=7, color=color, ha='center')
    ax.plot(t, pct, 'o', color=color, markersize=6)

ax.set_xlabel('数据点数阈值')
ax.set_ylabel('保留比例 (%)')
ax.set_title('不同阈值下的数据保留率 & 可分析性')
ax.set_ylim(80, 101)
ax.legend(fontsize=8)
ax.axhline(98.6, color='gray', linestyle=':', alpha=0.3)
ax.text(25, 98.7, '98.6%', fontsize=7, color='gray')

plt.suptitle('数据点数阈值选择依据（<6 点 = 在榜 <30 分钟的快闪词条，无法拟合衰减曲线）',
             fontsize=14, y=1.01)
plt.tight_layout()
out = 'datapoint_threshold.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'图表: {out}')
plt.close()
