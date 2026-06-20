"""微博热搜 LLM 事件分类 — 合并多年CSV、调用大模型打标签、输出 master 总表"""
import os, time, json, pandas as pd
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

# ---- 配置 ----
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
CANDIDATE_CATEGORIES = "【社会民生, 娱乐八卦, 时政要闻, 科技数码, 体育赛事, 医疗健康, 文化内容，教育问题, 影视综艺, 游戏电竞, 商业财经, 时尚美妆】"
BATCH_SIZE = 50
MAX_RETRIES = 3

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---- Prompt ----
def get_classification_prompt(batch_topics, candidate_categories):
    return f"""你是一个新闻文本分类专家。请将以下微博热搜词条精准归类到13个标准标签。

【标准标签列表（含"其他"兜底）】
{candidate_categories}

【分类映射规则】
1. 明星、网红、八卦、恋情、出轨、分手、情感 → 【娱乐八卦】
2. 电影、电视剧、综艺、票房、舞台、春晚 → 【影视综艺】
3. 游戏、手游、电竞、LOL、王者荣耀、皮肤、战队 → 【游戏电竞】
4. 生活日常、美食、旅行、天气、宠物 → 【社会民生】
5. 学校、教师、考试、高考、学历、保研 → 【教育问题】
6. 非遗、传统文化、节日、博物馆、文物、诗词 → 【文化内容】
7. 企业、股市、破产、资本、收购 → 【商业财经】
8. 国际关系、外交部、贪污、省委书记 → 【时政要闻】
9. 手机、芯片、AI、航天、火箭、数码 → 【科技数码】
10. 奥运会、世界杯、国乒、足球、篮球、金牌 → 【体育赛事】
11. 疫情、疫苗、医生、健康、药品、医保 → 【医疗健康】
12. 穿搭、口红、美妆、护肤、奢侈品、时尚 → 【时尚美妆】
13. 仅乱码或无法解读的极少数词条允许归为【其他】，禁止滥用。

【约束】
- 分类结果必须且只能是上述13个标签之一，禁止自创标签。
- 返回标准JSON对象，键为词条名，值为标签。
- 禁止Markdown标记、换行符、前言后记。
- 所有输入词条必须出现在返回的JSON键中，不得遗漏。

【待分类词条】
{json.dumps(batch_topics, ensure_ascii=False)}"""

# ---- 1. 合并多年 CSV ----
target_files = [
    "weibo_hotsearch_2019.csv", "weibo_hotsearch_2020_2022.csv",
    "weibo_hotsearch_2023.csv", "weibo_hotsearch_2024.csv", "weibo_hotsearch_2025.csv"
]
df_list = []
for file_name in target_files:
    full_path = os.path.join(script_dir, file_name)
    if os.path.exists(full_path):
        df_list.append(pd.read_csv(full_path))
        print(f"  ✓ {file_name} ({len(df_list[-1])} 行)")

df_combined = pd.concat(df_list, axis=0, ignore_index=True)
df_combined['上榜时间'] = pd.to_datetime(df_combined['上榜时间'], errors='coerce')
df_combined['年份'] = df_combined['上榜时间'].dt.year.fillna("未知").astype(str)
df_cleaned = df_combined.copy()
print(f"合并完成: {len(df_cleaned)} 行")

# ---- 2. 批量调用 LLM 分类（断点续传） ----
unique_topics = df_cleaned['词条名'].dropna().unique().tolist()
print(f"待分类词条: {len(unique_topics)}")

cache_file = os.path.join(script_dir, "classification_cache.json")
name_to_category = {}
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        name_to_category = json.load(f)
    print(f"缓存已加载: {len(name_to_category)} 个")

for i in range(0, len(unique_topics), BATCH_SIZE):
    batch = unique_topics[i:i + BATCH_SIZE]
    if all(t in name_to_category and name_to_category[t] != "其他" for t in batch):
        continue

    print(f"[{i+1}/{len(unique_topics)}] 分类中...")
    prompt = get_classification_prompt(batch, CANDIDATE_CATEGORIES)
    success = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "Qwen2.5-72B-Instruct"),
                messages=[{"role": "system", "content": "你只输出合法JSON，不含任何废话。"},
                           {"role": "user", "content": prompt}],
                temperature=0.1, response_format={"type": "json_object"})
            raw = resp.choices[0].message.content
            try:
                name_to_category.update(json.loads(raw))
                success = True
                break
            except json.JSONDecodeError:
                if attempt == MAX_RETRIES:
                    break
        except Exception as e:
            print(f"  网络异常 (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(3)

    if not success:
        for t in batch:
            if t not in name_to_category:
                name_to_category[t] = "其他"

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(name_to_category, f, ensure_ascii=False, indent=4)
    time.sleep(0.4)

# ---- 3. 注入标签并输出总表 ----
df_cleaned['事件类型'] = df_cleaned['词条名'].map(name_to_category).fillna("其他")
front_cols = ['词条名', '年份', '事件类型', '上榜时间', '最后在榜时间', '在榜共计', '热度最大值', '半衰期_秒']
remain_cols = [c for c in df_cleaned.columns if c not in front_cols]
df_final = df_cleaned[front_cols + remain_cols]

output = "weibo_hotsearch_dynamics_master.csv"
df_final.to_csv(output, index=False, encoding='utf-8-sig')
print(f"完成: {output} ({len(df_final)} 行 × {len(df_final.columns)} 列)")
print(df_final['事件类型'].value_counts())
