"""
数据爬取 Pipeline 图 — PPT 展示用
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

# ============================================================
# 六步卡片
# ============================================================
steps = [
    {
        'title': '① API 逆向',
        'icon': '🔑',
        'items': ['分析 weibotop.cn', '提取 AES 密钥', '解密通信协议'],
        'color': '#2c3e50',
    },
    {
        'title': '② 日期采样',
        'icon': '📅',
        'items': ['2019-2025 七年', '每年随机选 25 天', '每天早中晚 ×3 时段'],
        'color': '#2980b9',
    },
    {
        'title': '③ AES 加密',
        'icon': '🔒',
        'items': ['AES-128-ECB', 'PKCS7 填充', '参数加密后请求'],
        'color': '#8e44ad',
    },
    {
        'title': '④ 异步并发',
        'icon': '⚡',
        'items': ['aiohttp 异步', 'Semaphore 限速', '指数退避重试'],
        'color': '#e67e22',
    },
    {
        'title': '⑤ 响应解密',
        'icon': '🔓',
        'items': ['Base64 解码', 'AES 解密响应', 'JSON 结构化'],
        'color': '#27ae60',
    },
    {
        'title': '⑥ 原始数据',
        'icon': '📊',
        'items': ['年均 500 条词条', '时间-热度序列', '7 个年度 CSV'],
        'color': '#c0392b',
    },
]

x_start = 0.3
card_w = 2.4
gap = 0.15

for i, step in enumerate(steps):
    x = x_start + i * (card_w + gap)
    y_top = 3.5
    box_h = 1.4

    # 圆角矩形
    rect = FancyBboxPatch(
        (x, y_top - box_h), card_w, box_h,
        boxstyle="round,pad=0.1",
        facecolor=step['color'], edgecolor='white', linewidth=2, alpha=0.9
    )
    ax.add_patch(rect)

    # 标题
    ax.text(x + card_w / 2, y_top - 0.3, step['title'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # 要点
    for j, item in enumerate(step['items']):
        ax.text(x + card_w / 2, y_top - 0.7 - j * 0.3, f'• {item}',
                ha='center', va='center', fontsize=8, color='white', alpha=0.95)

    # 箭头（除最后一步）
    if i < len(steps) - 1:
        arrow_x = x + card_w + 0.02
        arrow = FancyArrowPatch(
            (arrow_x, y_top - box_h / 2), (arrow_x + gap - 0.04, y_top - box_h / 2),
            arrowstyle='->', mutation_scale=15, color='#7f8c8d', linewidth=2
        )
        ax.add_patch(arrow)

# ============================================================
# 底部技术栈
# ============================================================
tech_y = 0.9
ax.text(8, tech_y, '技术栈：Python 3 ｜ aiohttp ｜ asyncio ｜ pycryptodome ｜ CryptoJS ｜ AES-128-ECB ｜ Base64',
        ha='center', fontsize=9, color='#7f8c8d',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.8))

# ============================================================
# 顶部统计
# ============================================================
stats_y = 4.4
ax.text(8, stats_y, '数据规模：7 年 × 500 条/年 = ~3500 条原始词条 → 2478 条清洗后有效',
        ha='center', fontsize=10, color='#2c3e50', fontweight='bold')

# 标题
ax.text(8, 4.8, '微博热搜数据采集 Pipeline', ha='center', fontsize=16, fontweight='bold', color='#2c3e50')

plt.tight_layout(pad=0.5)
out = 'scraping_pipeline.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'图表: {out}')
plt.close()
