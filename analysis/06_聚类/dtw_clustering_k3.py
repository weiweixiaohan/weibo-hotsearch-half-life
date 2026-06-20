"""
DTW 聚类 k=3
"""
import csv, numpy as np, matplotlib, matplotlib.pyplot as plt, warnings, os
from scipy.interpolate import interp1d
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from collections import Counter
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = f'{BASE}/weibo_hotsearch_dynamics_master.csv'
LABELS_CSV = f'{BASE}/dtw_cluster_labels_k3.csv'
MAX_DUR_H, N_POINTS, N_CLUSTERS, RANDOM_SEED = 64, 100, 3, 42

def parse_duration(text):
    try:
        h=int(text.split('时')[0]); m=int(text.split('时')[1].split('分')[0])
        s=int(text.split('分')[1].split('秒')[0]); return h*3600+m*60+s
    except: return 0


# Phase 1: 加载
print("加载数据 ...", flush=True)
records = []
with open(MASTER_CSV, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        dur_sec = int(float(row.get('在榜共计_秒',0) or 0))
        if dur_sec==0: dur_sec = parse_duration(row.get('在榜共计',''))
        if dur_sec > MAX_DUR_H*3600 or dur_sec==0: continue
        heats=[]; i=1
        while f'heat_{i}' in row and row[f'heat_{i}']:
            heats.append(float(row[f'heat_{i}'])); i+=1
        if len(heats)<6: continue
        up_hour = -1
        ut = row.get('上榜时间','')
        if ut:
            try:
                up_hour = int(ut[11:13])
            except:
                pass
        up_slot = -1
        if 0<=up_hour<=5: up_slot=0
        elif 6<=up_hour<=11: up_slot=1
        elif 12<=up_hour<=17: up_slot=2
        elif 18<=up_hour<=23: up_slot=3
        records.append((row['词条名'], row.get('年份',''), row.get('事件类型',''), up_slot, np.array(heats)))

print(f"保留: {len(records)} 条", flush=True)


# Phase 2: 归一化

print("归一化 ...", flush=True)
grid = np.linspace(0,100,N_POINTS)
matrix_raw = np.zeros((len(records), N_POINTS))
for idx,(_,_,_,_,heats) in enumerate(records):
    n=len(heats)
    matrix_raw[idx]=interp1d(np.linspace(0,100,n),heats,kind='linear',
                              bounds_error=False,fill_value=(heats[0],heats[-1]))(grid)

scaler = TimeSeriesScalerMeanVariance()
matrix_scaled = scaler.fit_transform(matrix_raw).squeeze()


# Phase 3: k=3 聚类

print(f"DTW KMeans k={N_CLUSTERS} ...", flush=True)
if os.path.exists(LABELS_CSV):
    print("  → 从已有 CSV 加载标签", flush=True)
    with open(LABELS_CSV, encoding='utf-8-sig') as f:
        labels = np.array([int(r['cluster_id']) for r in csv.DictReader(f)])
    centroids = np.array([matrix_scaled[labels==c].mean(axis=0) for c in range(N_CLUSTERS)])
else:
    km = TimeSeriesKMeans(n_clusters=N_CLUSTERS, metric="dtw", max_iter=50,
                           random_state=RANDOM_SEED, n_jobs=-1, verbose=0)
    labels = km.fit_predict(matrix_scaled)
    centroids = km.cluster_centers_.squeeze()
    with open(LABELS_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['词条名','年份','事件类型','cluster_id'])
        for idx,(name,year,etype,_,_) in enumerate(records):
            writer.writerow([name,year,etype,labels[idx]])

counts = Counter(labels)
print(f"分布: {dict(sorted(counts.items()))}", flush=True)
print(f"标签已保存: {LABELS_CSV}", flush=True)


# Phase 4: 可视化

centroids_smooth = np.array([savgol_filter(c, window_length=11, polyorder=3) for c in centroids])
centroids_raw = centroids_smooth * scaler.std + scaler.mu
centroids_scaled = centroids_smooth
CLUSTER_COLORS = ['#e74c3c', '#3498db', '#f39c12']
SLOT_NAMES = ['凌晨','上午','下午','晚上']

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

# (a) 质心叠加
ax = fig.add_subplot(gs[0, 0])
for c in range(N_CLUSTERS):
    ax.plot(grid, centroids_raw[c], color=CLUSTER_COLORS[c], linewidth=2.5,
            label=f'C{c} (n={counts[c]})')
ax.set_xlabel('生命周期进度 (%)'); ax.set_ylabel('热度值')
ax.set_title('三类质心叠加对比'); ax.legend(fontsize=8)

# (b)(c)(d) 个体 + 质心 — C0/C1 在 row0, C2 在 row1
pos = [(0,1),(0,2),(1,0)]
for c in range(N_CLUSTERS):
    ax = fig.add_subplot(gs[pos[c][0], pos[c][1]])
    mask = labels == c; nc = mask.sum()
    sidx = np.where(mask)[0]
    if len(sidx) > 200:
        sidx = np.random.default_rng(RANDOM_SEED).choice(sidx, 200, replace=False)
    for idx in sidx:
        ax.plot(grid, matrix_scaled[idx], color='gray', alpha=0.05, linewidth=0.5)
    ax.plot(grid, centroids_scaled[c], color='red', linewidth=2.5)
    ax.set_xlabel('生命周期进度 (%)'); ax.set_ylabel('Z-Score')
    ax.set_title(f'Cluster {c} (n={nc})')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.3)

w = 0.25

# (e) 聚类 × 年份 (row1, col1)
ax = fig.add_subplot(gs[1, 1])
YEAR_GROUPS = ['2019','2020-22','2023','2024','2025']
ym = np.zeros((N_CLUSTERS, len(YEAR_GROUPS)))
ymap = {y:i for i,y in enumerate(YEAR_GROUPS)}
for idx,(_,year,_,_,_) in enumerate(records):
    if year in ('2020','2021','2022'): ym[labels[idx],ymap['2020-22']]+=1
    elif year in ymap: ym[labels[idx],ymap[year]]+=1
yp = ym/ym.sum(axis=1,keepdims=True)*100
x=np.arange(len(YEAR_GROUPS))
for c in range(N_CLUSTERS):
    ax.bar(x+c*w, yp[c], w, color=CLUSTER_COLORS[c], label=f'C{c}', edgecolor='white')
ax.set_xticks(x+w); ax.set_xticklabels(YEAR_GROUPS, fontsize=8)
ax.set_ylabel('占比 (%)'); ax.set_title('聚类 × 年份'); ax.legend(fontsize=7)

# (f) 聚类 × 事件类型 (row1, col2)
ax = fig.add_subplot(gs[1, 2])
tc = Counter(r[2] for r in records if r[2])
tt = [t for t,_ in tc.most_common(8)]
tm = np.zeros((N_CLUSTERS, len(tt)))
tmap = {t:i for i,t in enumerate(tt)}
for idx,(_,_,etype,_,_) in enumerate(records):
    if etype in tmap: tm[labels[idx],tmap[etype]]+=1
tp = tm/tm.sum(axis=0,keepdims=True)*100
xt=np.arange(len(tt))
for c in range(N_CLUSTERS):
    ax.bar(xt+c*w, tp[c], w, color=CLUSTER_COLORS[c], label=f'C{c}', edgecolor='white')
    if c==0:
        for ti,t in enumerate(tt): ax.text(ti,102,f'n={tc[t]}',ha='center',fontsize=6,color='gray')
ax.set_xticks(xt+w); ax.set_xticklabels(tt, fontsize=7, rotation=30, ha='right')
ax.set_ylabel('占比 (%)'); ax.set_title('聚类 × 事件类型 (列归一化)')
ax.set_ylim(0,115); ax.axhline(100/N_CLUSTERS, color='gray', linestyle=':', alpha=0.3)
ax.legend(fontsize=7)

# (g) 聚类 × 启动时段 (row2, col0) — 绝对值
ax = fig.add_subplot(gs[2, 0])
sm = np.zeros((N_CLUSTERS, 4))
for idx,(_,_,_,up_slot,_) in enumerate(records):
    if up_slot >= 0: sm[labels[idx], up_slot] += 1
xs = np.arange(4)
for c in range(N_CLUSTERS):
    ax.bar(xs+c*w, sm[c], w, color=CLUSTER_COLORS[c], label=f'C{c}', edgecolor='white')
ax.set_xticks(xs+w); ax.set_xticklabels(SLOT_NAMES)
ax.set_ylabel('词条数'); ax.set_title('聚类 × 启动时段 (绝对值)')
ax.legend(fontsize=7)

# (h) 每个聚类内启动时段占比 (row2, col1) — 堆叠柱状
ax = fig.add_subplot(gs[2, 1])
sp = sm / sm.sum(axis=1, keepdims=True) * 100
bottom = np.zeros(N_CLUSTERS)
stack_order = [1, 2, 3, 0]  # 底→顶: 上午、下午、晚上、凌晨
for slot in stack_order:
    ax.bar(range(N_CLUSTERS), sp[:, slot], bottom=bottom,
           color=['#6b7280','#fbbf24','#fb923c','#1e40af'][slot],
           label=SLOT_NAMES[slot], edgecolor='white', width=0.5)
    bottom += sp[:, slot]
ax.set_xticks(range(N_CLUSTERS))
ax.set_xticklabels([f'C{c}' for c in range(N_CLUSTERS)])
ax.set_ylabel('占比 (%)')
ax.set_title('每类内部启动时段分布 (堆叠100%)')
ax.legend(fontsize=7)

# 隐藏空位
fig.add_subplot(gs[2, 2]).axis('off')

plt.suptitle(f'微博热搜热度曲线 DTW 聚类 k=3 (n={len(records)}, 在榜<{MAX_DUR_H}h)',
             fontsize=14, y=1.01)
plt.tight_layout()
out = f'{BASE}/dtw_clustering_k3.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'图表: {out}', flush=True)
plt.close()

# 质心特征
print(f"\n{'='*60}")
for c in range(N_CLUSTERS):
    ct = centroids_scaled[c]
    pi, pv = np.argmax(ct), ct.max()
    ev = ct[-1]
    after = ct[pi:]; b50 = np.where(after<=pv*0.5)[0]
    t50 = b50[0]/N_POINTS*100 if len(b50)>0 else -1
    print(f"C{c} (n={counts[c]}): 峰值@{pi/N_POINTS*100:.0f}%={pv:.2f}, 终点={ev:.2f}, 半衰@{t50:.0f}%")
