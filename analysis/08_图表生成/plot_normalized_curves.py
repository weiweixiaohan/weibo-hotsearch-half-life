"""
归一化热度曲线可视化
"""
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

CSV_PATH = "heat_curves_normalized.csv"

# ============================================================
# 1. 加载数据
# ============================================================
with open(CSV_PATH, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 时间列（分钟）
time_cols = [k for k in rows[0].keys() if k.endswith('min')]
grid = np.array([int(c.replace('min', '')) for c in time_cols])

# 热度矩阵: [词条数 × 时间点数]
names = []
matrix = []
for r in rows:
    vals = []
    for c in time_cols:
        v = r[c]
        vals.append(float(v) if v else np.nan)
    names.append(r['词条名'])
    matrix.append(vals)
matrix = np.array(matrix)

print(f"词条数: {matrix.shape[0]}, 时间点数: {matrix.shape[1]}")
print(f"时间范围: {grid[0]}~{grid[-1]} 分钟")

# ============================================================
# 2. 统计曲线
# ============================================================
# 每个时间点的中位数、P25、P75（忽略 NaN）
def masked_percentile(arr, p):
    return np.array([np.nanpercentile(arr[:, i], p)
                     for i in range(arr.shape[1])])

med = masked_percentile(matrix, 50)
p25 = masked_percentile(matrix, 25)
p75 = masked_percentile(matrix, 75)
p10 = masked_percentile(matrix, 10)
p90 = masked_percentile(matrix, 90)

# 每个时间点的有效词条数（还在榜的）
alive = np.sum(~np.isnan(matrix), axis=0)

# ============================================================
# 3. 画图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# --- (a) 中位数 + IQR 包络线 ---
ax = axes[0, 0]
ax.fill_between(grid, p25, p75, alpha=0.25, color='steelblue', label='P25–P75')
ax.fill_between(grid, p10, p90, alpha=0.12, color='steelblue', label='P10–P90')
ax.plot(grid, med, 'steelblue', linewidth=2, label='中位数')
ax.set_xlabel('归一化时间（分钟）')
ax.set_ylabel('热度值')
ax.set_title(f'热度衰减曲线（中位数 ± 分位数带, n={matrix.shape[0]}）')
ax.legend(fontsize=8)
ax.set_xlim(0, grid[-1])

# --- (b) 选 30 条个体曲线覆盖在统计曲线上 ---
ax = axes[0, 1]
rng = np.random.default_rng(42)
idx = rng.choice(matrix.shape[0], min(30, matrix.shape[0]), replace=False)
for i in idx:
    ax.plot(grid, matrix[i], color='gray', alpha=0.25, linewidth=0.6)
ax.plot(grid, med, 'red', linewidth=2.2, label='中位数')
ax.fill_between(grid, p25, p75, alpha=0.2, color='red', label='P25–P75')
ax.set_xlabel('归一化时间（分钟）')
ax.set_ylabel('热度值')
ax.set_title(f'个体曲线 + 中位数（随机 30 条）')
ax.legend(fontsize=8)
ax.set_xlim(0, grid[-1])

# --- (c) 存活率曲线 ---
ax = axes[1, 0]
alive_pct = alive / matrix.shape[0] * 100
ax.plot(grid, alive_pct, 'darkgreen', linewidth=2)
ax.fill_between(grid, 0, alive_pct, alpha=0.15, color='darkgreen')
ax.axhline(50, color='red', linestyle='--', alpha=0.5)
# 找 50% 存活的时间点
half_alive = grid[np.argmax(alive_pct <= 50)] if np.any(alive_pct <= 50) else grid[-1]
ax.annotate(f'50% 在榜 ≈ {half_alive}min', (half_alive, 50),
            fontsize=9, xytext=(10, -15), textcoords='offset points', color='red')
ax.set_xlabel('归一化时间（分钟）')
ax.set_ylabel('仍在榜的词条比例 (%)')
ax.set_title('词条在榜存活曲线')
ax.set_xlim(0, grid[-1])
ax.set_ylim(0, 105)

# --- (d) 热度热力图（前 60 条，按在榜时长排序）---
ax = axes[1, 1]
n_show = min(60, matrix.shape[0])
# 计算每条词条的在榜时长（最后一个非 NaN 的时间点）
last_valid = np.array([grid[np.where(~np.isnan(matrix[i]))[0][-1]]
                       if np.any(~np.isnan(matrix[i])) else 0
                       for i in range(matrix.shape[0])])
top_idx = np.argsort(last_valid)[-n_show:]
heatmap_data = matrix[top_idx]
# 归一化到 0-1 以便着色（每条词条独立归一化到自己的峰值）
heatmap_norm = heatmap_data / np.nanmax(heatmap_data, axis=1, keepdims=True)
im = ax.imshow(heatmap_norm, aspect='auto', cmap='YlOrRd',
               extent=[grid[0], grid[-1], n_show, 0], vmin=0, vmax=1)
ax.set_xlabel('归一化时间（分钟）')
ax.set_ylabel(f'词条（Top {n_show} 按峰值排）')
ax.set_title(f'热度热力图（按在榜时长排, {n_show} 条样本）')
plt.colorbar(im, ax=ax, label='相对热度 (0–1)')

plt.tight_layout()
out = "normalized_curves.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {out}")
plt.close()
