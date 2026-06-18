"""
热度曲线时间归一化脚本
- 时间对齐到 0 分钟起点
- 统一 30 分钟间隔网格，上限 24h
- 线性插值填充，下榜后填 NaN
- 自动剔除长尾
"""
import csv
import numpy as np
from datetime import datetime

CSV_PATH = "weibo_hotsearch_2019.csv"
OUT_PATH = "heat_curves_normalized.csv"

MAX_HL_HOURS = 120
MAX_DUR_HOURS = 300
GRID_STEP_MIN = 30      # 网格间隔（分钟）
GRID_END_MIN = 24 * 60  # 网格终点（分钟）
TIME_FMT = "%Y-%m-%d %H:%M:%S"


# 1. 解析每条词条的时间-热度序列
curves = []  
dropped_hl = []
dropped_dur = []
total = 0

with open(CSV_PATH, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['词条名']
        hl_sec = float(row['半衰期_秒'] or 0)
        dur_sec = int(row.get('在榜共计_秒', 0) or 0)
        if not hl_sec:
            continue
        total += 1

        hl_h = hl_sec / 3600
        dur_h = dur_sec / 3600
        if hl_h > MAX_HL_HOURS:
            dropped_hl.append((name, hl_h))
            continue
        if dur_h > MAX_DUR_HOURS:
            dropped_dur.append((name, dur_h))
            continue

        # 提取时间-热度对
        points = []
        i = 1
        while True:
            t_key = f't_{i}'
            h_key = f'heat_{i}'
            if t_key not in row or not row[t_key]:
                break
            try:
                t = datetime.strptime(row[t_key], TIME_FMT)
                heat = float(row[h_key])
                points.append((t, heat))
            except (ValueError, TypeError):
                break
            i += 1

        if len(points) >= 2:
            curves.append((name, points))

print(f"原始有效词条: {total}")
print(f"剔除 半衰期 >{MAX_HL_HOURS}h: {len(dropped_hl)}")
print(f"剔除 在榜 >{MAX_DUR_HOURS}h: {len(dropped_dur)}")
print(f"纳入归一化: {len(curves)} 条")


# 2. 时间归一化 & 统一网格插值

grid_minutes = np.arange(0, GRID_END_MIN + GRID_STEP_MIN, GRID_STEP_MIN)
print(f"网格: {len(grid_minutes)} 个点, 间隔 {GRID_STEP_MIN}min, 终点 {GRID_END_MIN}min")

# 构建 CSV 列头
cols = ['词条名'] + [f'{m}min' for m in grid_minutes]

rows = []
for name, points in curves:
    t0 = points[0][0]
    # 转为相对分钟
    rel_min = np.array([(p[0] - t0).total_seconds() / 60 for p in points])
    heats = np.array([float(p[1]) for p in points])

    # 在网格上线性插值，超出范围填 NaN
    interp = np.interp(grid_minutes, rel_min, heats, left=np.nan, right=np.nan)

    row = {'词条名': name}
    for i, m in enumerate(grid_minutes):
        val = interp[i]
        row[f'{m}min'] = '' if np.isnan(val) else f'{val:.0f}'
    rows.append(row)


# 3. 输出 CSV
with open(OUT_PATH, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n输出: {OUT_PATH}")
print(f"  行数: {len(rows)}")
print(f"  列数: {len(cols)} (1 词条名 + {len(grid_minutes)} 时间点)")

# 统计有效覆盖
coverage = []
for row in rows:
    last_valid = max((i for i, m in enumerate(grid_minutes)
                      if row[f'{m}min'] != ''), default=-1)
    coverage.append(last_valid)
med_cov = grid_minutes[int(np.median([c for c in coverage if c >= 0]))] if coverage else 0
p90_cov = grid_minutes[int(np.percentile([c for c in coverage if c >= 0], 90))] if coverage else 0
print(f"  NaN 出现位置 中位数: {med_cov}min, P90: {p90_cov}min")
