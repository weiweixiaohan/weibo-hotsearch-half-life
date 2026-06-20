"""四年二次爆发分析"""
import csv, numpy as np, pandas as pd
from datetime import datetime

TIME_FMT = "%Y-%m-%d %H:%M:%S"
ROLLING_WINDOW = 12
GAP_MIN, GAP_MAX = 120, 7 * 24 * 60
COOLING_RATIO, REBOUND_MAIN, REBOUND_VALLEY = 0.50, 0.30, 1.50


def find_peaks_after_smoothing(timestamps, heats):
    n = len(heats)
    if n < ROLLING_WINDOW:
        return [], None
    s = pd.Series(heats)
    smoothed = s.rolling(window=ROLLING_WINDOW, center=True).mean().values
    smoothed = np.where(np.isnan(smoothed), heats, smoothed)
    peaks = []
    for i in range(2, n - 2):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            local_min = min(smoothed[i - 2], smoothed[i + 2])
            if smoothed[i] > local_min * 1.10:
                peaks.append({
                    'idx': i, 'heat': float(smoothed[i]),
                    'timestamp': timestamps[i],
                    'hour': int(timestamps[i][11:13]),
                })
    return peaks, smoothed


def extract_second_wave(name, timestamps, heats):
    peaks, _ = find_peaks_after_smoothing(timestamps, heats)
    if len(peaks) < 2:
        return None
    main_peak = max(peaks, key=lambda x: x['heat'])
    after = [p for p in peaks if p['idx'] > main_peak['idx']]
    if not after:
        return None
    valid = []
    for p in after:
        try:
            t_main = datetime.strptime(main_peak['timestamp'], TIME_FMT)
            t_cand = datetime.strptime(p['timestamp'], TIME_FMT)
            gap = (t_cand - t_main).total_seconds() / 60
        except:
            continue
        if gap < GAP_MIN or gap > GAP_MAX:
            continue
        valley = np.min(heats[main_peak['idx']:p['idx']])
        if valley >= main_peak['heat'] * COOLING_RATIO:
            continue
        if p['heat'] <= main_peak['heat'] * REBOUND_MAIN:
            continue
        if p['heat'] <= valley * REBOUND_VALLEY:
            continue
        valid.append((p, gap, valley))
    if not valid:
        return None
    best, gap, valley = max(valid, key=lambda x: x[0]['heat'])
    return {
        'name': name,
        'main_ts': main_peak['timestamp'],
        'main_hour': main_peak['hour'],
        'main_heat': main_peak['heat'],
        'second_ts': best['timestamp'],
        'second_hour': best['hour'],
        'second_heat': best['heat'],
        'gap_min': gap,
        'valley': valley,
        'ratio': best['heat'] / main_peak['heat'] if main_peak['heat'] > 0 else 0,
    }


BASE = os.path.dirname(os.path.abspath(__file__))
all_results = []  # 汇总四年数据

for year in ['2019', '2023', '2024', '2025']:
    path = f'{BASE}/weibo_hotsearch_{year}.csv'
    results = []
    total = 0
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            ts, hs = [], []
            i = 1
            while True:
                tk, hk = f't_{i}', f'heat_{i}'
                if tk not in row or not row[tk]:
                    break
                ts.append(row[tk])
                hs.append(float(row[hk]))
                i += 1
            if len(hs) < ROLLING_WINDOW:
                continue
            total += 1
            r = extract_second_wave(row['词条名'], ts, np.array(hs))
            if r:
                results.append(r)

    results.sort(key=lambda x: -x['ratio'])
    # 标注年份
    for r in results:
        r['year'] = year
    all_results.extend(results)

    hour_dist = np.zeros(24, dtype=int)
    for r in results:
        hour_dist[r['second_hour']] += 1
    peak_hours = sorted(np.argsort(-hour_dist)[:3])

    print(f"\n{'='*85}")
    print(f"  {year} 年: {len(results)}/{total} 条二次爆发")
    print(f"  二波高峰小时: {', '.join(f'{h:02d}:00({hour_dist[h]}条)' for h in peak_hours)}")
    print(f"{'='*85}")

    for r in results:
        print(f"  {r['name'][:22]:<22s}  "
              f"主{r['main_ts'][5:16]} {r['main_heat']:>8.0f} → "
              f"谷{r['valley']:>7.0f} → "
              f"次{r['second_ts'][5:16]} {r['second_heat']:>8.0f}  "
              f"{r['gap_min']:>5.0f}分 {r['ratio']:.2f}")

# 输出汇总 CSV
out_path = f'{BASE}/second_wave_all_years.csv'
with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'year', 'name', 'main_ts', 'main_hour', 'main_heat',
        'valley', 'second_ts', 'second_hour', 'second_heat',
        'gap_min', 'ratio'
    ])
    writer.writeheader()
    for r in sorted(all_results, key=lambda x: (x['year'], -x['ratio'])):
        writer.writerow({
            'year': r['year'],
            'name': r['name'],
            'main_ts': r['main_ts'],
            'main_hour': r['main_hour'],
            'main_heat': f"{r['main_heat']:.0f}",
            'valley': f"{r['valley']:.0f}",
            'second_ts': r['second_ts'],
            'second_hour': r['second_hour'],
            'second_heat': f"{r['second_heat']:.0f}",
            'gap_min': f"{r['gap_min']:.0f}",
            'ratio': f"{r['ratio']:.3f}",
        })

print(f"\n汇总 CSV 已保存: {out_path} ({len(all_results)} 条)")
