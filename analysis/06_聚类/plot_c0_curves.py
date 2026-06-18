"""C0 随机 10 条衰减曲线"""
import csv, numpy as np, matplotlib, matplotlib.pyplot as plt, warnings
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

MASTER = 'weibo_hotsearch_dynamics_master.csv'
MAX_DUR_H, N_POINTS, SEED = 64, 100, 42

def parse_dur(text):
    try:
        h=int(text.split('时')[0]); m=int(text.split('时')[1].split('分')[0])
        s=int(text.split('分')[1].split('秒')[0]); return h*3600+m*60+s
    except: return 0

records = []
with open(MASTER, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        dur_sec = int(float(row.get('在榜共计_秒',0) or 0))
        if dur_sec==0: dur_sec = parse_dur(row.get('在榜共计',''))
        if dur_sec > MAX_DUR_H*3600 or dur_sec==0: continue
        heats=[]; i=1
        while f'heat_{i}' in row and row[f'heat_{i}']:
            heats.append(float(row[f'heat_{i}'])); i+=1
        if len(heats)<6: continue
        records.append((row['词条名'], np.array(heats)))

grid = np.linspace(0,100,N_POINTS)
matrix = np.zeros((len(records), N_POINTS))
for idx,(_,heats) in enumerate(records):
    n=len(heats)
    matrix[idx]=interp1d(np.linspace(0,100,n),heats,kind='linear',
                          bounds_error=False,fill_value=(heats[0],heats[-1]))(grid)

with open('dtw_cluster_labels_v2.csv', encoding='utf-8-sig') as f:
    labels = np.array([int(r['cluster_id']) for r in csv.DictReader(f)][:len(records)])

rng = np.random.default_rng(SEED)
c0_idx = np.where(labels==0)[0]
chosen = rng.choice(c0_idx, min(10,len(c0_idx)), replace=False)
colors = plt.cm.tab10(np.linspace(0,1,10))

fig, ax = plt.subplots(figsize=(14,7))
for i,idx in enumerate(chosen):
    name,_ = records[idx]
    curve = matrix[idx]
    curve_norm = (curve-curve.min())/(curve.max()-curve.min())
    ax.plot(grid, curve_norm, color=colors[i], linewidth=2, alpha=0.85, label=name[:22])

c0_c = matrix[labels==0].mean(axis=0)
c0_n = (c0_c-c0_c.min())/(c0_c.max()-c0_c.min())
ax.plot(grid, c0_n, 'black', linewidth=3.5, linestyle='--', label='C0 质心', zorder=10)

ax.set_xlabel('生命周期进度 (%)', fontsize=12)
ax.set_ylabel('归一化热度 (0-1)', fontsize=12)
ax.set_title(f'Cluster 0 随机 10 条衰减曲线（n={len(c0_idx)}）', fontsize=14)
ax.legend(fontsize=8, loc='upper right', ncol=2)
ax.set_xlim(0,100); ax.set_ylim(-0.05,1.05)
plt.tight_layout()
out = 'c0_sample_curves.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'图表: {out}')
for i,idx in enumerate(chosen):
    name,_ = records[idx]
    print(f'{i+1}. {name}')
plt.close()
