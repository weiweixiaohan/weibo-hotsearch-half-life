"""
异常数据词云图 — 被清洗掉的离群词条
"""
import csv, numpy as np, matplotlib.pyplot as plt, warnings
import jieba
from wordcloud import WordCloud
from collections import Counter

warnings.filterwarnings('ignore')

MASTER = 'weibo_hotsearch_dynamics_master.csv'
MAX_DUR_H = 64
MIN_POINTS = 6
MAX_HL_H = 120

def parse_duration(text):
    try:
        h = int(text.split('时')[0])
        m = int(text.split('时')[1].split('分')[0])
        s = int(text.split('分')[1].split('秒')[0])
        return h*3600 + m*60 + s
    except: return 0

# 收集异常词条
anomalies = {'dur_long': [], 'points_few': [], 'hl_extreme': []}
valid = 0
all_durs = []

with open(MASTER, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        dur_sec = int(float(row.get('在榜共计_秒',0) or 0))
        if dur_sec == 0: dur_sec = parse_duration(row.get('在榜共计',''))
        if dur_sec <= 0: continue
        all_durs.append(dur_sec / 3600)

        # 数据点数
        i = 1
        while f'heat_{i}' in row and row[f'heat_{i}']: i += 1
        n_points = i - 1

        hl_sec = float(row.get('半衰期_秒',0) or 0)
        name = row['词条名']

        # 三轮判定
        flags = []
        if dur_sec > MAX_DUR_H * 3600:
            anomalies['dur_long'].append((name, dur_sec/3600))
            flags.append(f'在榜{dur_sec/3600:.0f}h')
        if n_points < MIN_POINTS:
            anomalies['points_few'].append((name, n_points))
            flags.append(f'{n_points}点')
        if hl_sec / 3600 > MAX_HL_H:
            anomalies['hl_extreme'].append((name, hl_sec/3600))
            flags.append(f'HL={hl_sec/3600:.0f}h')

        if not flags:
            valid += 1

print(f"有效: {valid}")
print(f"在榜>{MAX_DUR_H}h: {len(anomalies['dur_long'])} 条")
print(f"数据点<{MIN_POINTS}: {len(anomalies['points_few'])} 条")
print(f"半衰期>{MAX_HL_H}h: {len(anomalies['hl_extreme'])} 条")

# 合并所有异常词条 → 词频
all_anomaly_names = []
for cat in anomalies:
    for name, _ in anomalies[cat]:
        all_anomaly_names.append(name)

freq = Counter(all_anomaly_names)

# 去重后总数
unique_anomalies = set(all_anomaly_names)
print(f"\n去重异常词条: {len(unique_anomalies)} 个")
print(f"总共清洗掉: {len(all_durs) - valid} 条 ({(len(all_durs)-valid)/len(all_durs)*100:.1f}%)")

# 词云
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, (cat, cat_name, color) in zip(axes, [
    ('dur_long', f'在榜 > {MAX_DUR_H}h\n（周期回归型长尾）', 'Reds'),
    ('points_few', f'数据点 < {MIN_POINTS}\n（快闪词条）', 'Blues'),
    ('hl_extreme', f'半衰期 > {MAX_HL_H}h\n（极端缓慢衰减）', 'Purples'),
]):
    # jieba 分词 → 频率字典
    names = [name for name, _ in anomalies[cat]]
    words_list = []
    for name in names:
        words_list.extend(jieba.lcut(name))
    freq = Counter(words_list)
    # 过滤单字和标点
    freq = {k: v for k, v in freq.items() if len(k) >= 2 and k.strip()}

    if not freq:
        ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14, color='gray')
        ax.set_title(cat_name, fontsize=10)
        continue

    wc = WordCloud(
        width=400, height=300,
        background_color='white',
        colormap=color,
        max_words=80,
        font_path='/System/Library/Fonts/Hiragino Sans GB.ttc',
        relative_scaling=0.5,
        min_font_size=8,
        collocations=False,
    ).generate_from_frequencies(freq)

    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{cat_name}\nn={len(anomalies[cat])}', fontsize=10)

plt.suptitle(f'数据清洗：被剔除的异常词条词云（共 {len(unique_anomalies)} 个不同词条，{len(all_durs) - valid} 条记录）',
             fontsize=13, y=1.01)
plt.tight_layout()
out = 'anomaly_wordcloud.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\n图表: {out}')
plt.close()
