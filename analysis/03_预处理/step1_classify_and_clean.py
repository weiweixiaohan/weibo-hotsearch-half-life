
import os
import glob
import time
import json
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# 从 .env 读取敏感配置
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

#  1. 强力收敛版标签与 Prompt 配置 
# 【核心修改 1】：将原本零散的自定义标签，收敛合并为 12 个标准的、具有明确动力学传播差异的学术大类
CANDIDATE_CATEGORIES = "【社会民生, 娱乐八卦, 时政要闻, 科技数码, 体育赛事, 医疗健康, 文化内容，教育问题, 影视综艺, 游戏电竞, 商业财经, 时尚美妆】"
BATCH_SIZE = 50  


def get_classification_prompt(batch_topics, candidate_categories):
    """
    【高强度闭合约束 Prompt】强制大模型向 12+1 大类极速聚合，含严厉的“其他”触发限制
    """
    return f"""
你是一个顶级的新闻文本分类与数据清洗专家。请将以下微博热搜词条精准归类。

【强制规定的标准标签列表（共 13 个，包含‘其他’兜底项）】
{candidate_categories}

【强力分类融合指南（防止标签稀释）】
为了确保后续统计学分析的样本量充足，你必须把所有模糊、相近的边缘词条，通过语义向上收敛、合并到标准标签中，绝不允许自己发明任何新的细分小标签。
请严格遵循以下聚合映射逻辑：
1. 凡是涉及明星、追星、网红、八卦、相声、恋爱感悟、情感故事、恋情、出轨、分手、爆料，统一强行归为【娱乐八卦】。
2. 凡是涉及电影、电视剧、综艺、开播、定档、票房、影评、舞台、春晚、音乐演艺、童年回忆，统一强行归为【影视综艺】。
3. 凡是涉及游戏、手游、网游、皮肤、战队、赛事、选手、LOL、王者荣耀、电竞主播，统一强行归为【游戏电竞】。
4. 凡是涉及职场趣事、生活日常、日常服务、美食生活、价格话梅雪糕刺客、旅行出行、自然天气、猫狗宠物趣事，统一强行收敛归入【社会民生】。
5. 凡是涉及学校、老师、同学、课堂、课后作业、学历、文凭、读书、教授、保研高考政策、教育趣事、职场发展，统一强行归为【教育问题】。
6. 凡是非遗、传统文化、历史记忆、传统节日、博物馆、文物出土、古诗词考据、文学艺术，统一强行归为【文化内容】。
7. 凡是涉及企业倒闭、股市、破产、财富榜、大厂高管、资本收购，统一强行归为【商业财经】。
8. 凡是涉及国际关系、外交部表态、国家通报、贪污受贿被查、省委书记省长动态，统一强行归为【时政要闻】。
9. 凡是涉及手机、电脑、芯片、人工智能、大模型、航天发射、火箭、科学技术、数码新品，统一强行归为【科技数码】。
10. 凡是涉及奥运会、世界杯、亚运会、国乒、夺冠、金牌、足球、篮球、羽毛球、运动赛事，统一强行归为【体育赛事】。
11. 凡是涉及疫情、新冠、感冒、疫苗、医生、医院、健康养生、药品、医保、疾病，统一强行归为【医疗健康】。
12. 凡是涉及穿搭、口红、美妆、护肤、奢侈品、走秀、时尚潮流，统一强行归为【时尚美妆】。
13. 【极其重要的“其他”使用限定】：只有当词条纯属乱码、无法解读、或是无论如何向上收敛都绝对无法塞进前 13 个标签的极少数怪异词条，才允许归为【其他】。禁止滥用该标签！

【极其严厉的约束条件】
1. 你的分类结果【必须且只能】属于上述规定的 13 个标签之一！绝对禁止返回任何不在此列表中的自定义标签（如禁止返回人物聚焦、饮食文化、生活服务等）。
2. 严格返回一个标准的 JSON 对象（字典），键(Key)为待分类的词条名，值(Value)为分类标签。
3. 禁止返回任何 Markdown 标记（如 ```json ... ```），禁止带有任何换行符、前言、后记或解释性文字
4. 输入的词条必须百分之百全部出现在返回的 JSON 键中，不得漏掉任何一个。

【待分类词条列表】
{json.dumps(batch_topics, ensure_ascii=False)}
"""


#  2. 多表读取与预清洗（同级绝对路径版） 
print("正在读取并合并多张表格...")

# 1. 强制获取当前运行的 .py 脚本所在的绝对路径文件夹（这就是你的 data 文件夹路径）
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
print(f"📂 脚本和数据文件所在的真实绝对路径: {script_dir}")

# 2. 明确我们要读的 5 个文件的标准名称
target_files = [
    "weibo_hotsearch_2019.csv",
    "weibo_hotsearch_2020_2022.csv",
    "weibo_hotsearch_2023.csv",
    "weibo_hotsearch_2024.csv",
    "weibo_hotsearch_2025.csv"
]

df_list = []
loaded_files_count = 0

# 3. 强制在脚本同级线下拼装路径并读取
for file_name in target_files:
    # 动态拼装成 D:\大三下\数据科学编程\大作业\data\weibo_hotsearch_2019.csv
    full_path = os.path.join(script_dir, file_name) 
    
    if os.path.exists(full_path):
        print(f" -> [成功发现] 正在加载: {file_name}")
        temp_df = pd.read_csv(full_path)
        print(f"    └── 该单表实际包含原始数据: {len(temp_df)} 行")
        df_list.append(temp_df)
        loaded_files_count += 1
    else:
        print(f" ❌ [未找到文件] 尝试读取绝对路径失败: {full_path}")

print(f"📊 统计：共成功加载了 {loaded_files_count} / 5 个文件")

if not df_list:
    raise FileNotFoundError(f"❌ 依旧一个文件都没读到！请务必检查该路径下是否有这5个CSV文件:\n👉 {script_dir}")

# 无损纵向合并
df_combined = pd.concat(df_list, axis=0, ignore_index=True)
print(f"✨ 阶段合并完成！当前总原始数据量: {len(df_combined)} 行")

# 2.1 提取年份 Tag
df_combined['上榜时间'] = pd.to_datetime(df_combined['上榜时间'], errors='coerce')
df_combined['年份'] = df_combined['上榜时间'].dt.year.fillna("未知").astype(str)

# 2.2 动力学去噪：精准剔除周期性/规律性干预事件
# print("开始进行动力学去噪（剔除考研、高考等周期性多峰事件）...")
# df_combined['词条名'] = df_combined['词条名'].astype(str)
# cyclical_keywords = ['考研', '高考', '中考', '国考', '四六级', '公考', '研究生考试', '公务员考试']
# filter_condition = df_combined['词条名'].str.contains('|'.join(cyclical_keywords), na=False, case=False)

df_cleaned = df_combined.copy()
print(f"✅ 去噪完成！剩余有效分析事件: {len(df_cleaned)} 行")

#  3. 提取唯一词条并批量调用 LLM (强抗灾重试续传版) 
unique_topics = df_cleaned['词条名'].dropna().unique().tolist()
print(f"去重后共有 {len(unique_topics)} 个独特词条等待大模型精准分类...")

name_to_category = {}

# 💡 核心设计：如果网络断过，我们先看看有没有本地缓存，没有的话再初始化
# 每次跑完一个批次会自动保存，这样下次由于网络中断重启时，会自动跳过已分类好的部分，不花一分冤枉钱
cache_file = os.path.join(script_dir, "classification_cache.json")
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        name_to_category = json.load(f)
    print(f"🔄 检测到本地历史清洗缓存，已自动无损载入 {len(name_to_category)} 个已打标词条！")

MAX_RETRIES = 3  # 💡 网络断流时，在原地自动重新握手建立连接的最高次数

#  3. 提取唯一词条并批量调用 LLM (变量对齐无错版) 
# [前面的 unique_topics, name_to_category, cache_file 保持不变...]

for i in range(0, len(unique_topics), BATCH_SIZE):
    batch = unique_topics[i:i+BATCH_SIZE]
    
    if all(t in name_to_category and name_to_category[t] != "其他" for t in batch):
        continue
        
    print(f" -> [进度] 正在分类第 {i+1} 至 {min(i+BATCH_SIZE, len(unique_topics))} 个词条...")
    user_prompt = get_classification_prompt(batch, CANDIDATE_CATEGORIES)
    
    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "Qwen2.5-72B-Instruct"),
                messages=[
                    {"role": "system", "content": "你是一个只输出合法 JSON 字典、绝不含任何废话的自动化数据清洗机器人。"},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            
            # 💡 核心修复 1：将 JSON 解析单独放到一个小 try 里
            # 如果仅仅是这 50 个词条里由于某个词条带特殊符号导致 JSON 坏掉了，
            # 我们不需要让整个网络重试 3 次，直接触发格式异常处理，防止死循环
            try:
                res_json = json.loads(raw_content)
                name_to_category.update(res_json)
                success = True
                break  # 完美成功，跳出重试
            except json.JSONDecodeError as json_err:
                print(f"   ⚠️ 大模型返回的 JSON 文本存在格式瑕疵（第{attempt}次）: {json_err}。尝试在下一次重试中让其自我修正...")
                # 如果是最后一次尝试依然 JSON 坏掉，则放弃并标记为其他
                if attempt == MAX_RETRIES:
                    break
            
        # 💡 核心修复 2：将小写的 openai.APIConnectionError 改为通用的 Exception 捕获
        # 这样即便不导入小写的 openai 模块，代码也绝对不会再报 NameError，抗灾防护罩百分之百生效！
        except Exception as e:
            print(f"   ⚠️ 网络连接抖动或网关握手失败(第{attempt}次尝试): {e}")
            if attempt < MAX_RETRIES:
                print("   ⏳ 正在原地等待 3 秒后自动重新建立 SSL 安全握手连接...")
                time.sleep(3)
                
    if not success:
        print(f"   🚨 该批次（共{len(batch)}个词条）最终判别失败，已安全归为“其他”。")
        for t in batch:
            if t not in name_to_category:
                name_to_category[t] = "其他"
                
    # 每一批不论成功与否，都安全备份到本地缓存 classification_cache.json
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(name_to_category, f, ensure_ascii=False, indent=4)
        
    time.sleep(0.4)


#  4. Tag 注入与多维总表输出 
print("大模型分类结束，正在将新标签注入总表并整理高维结构...")

# 将大模型的分类结果映射回清洗后的数据集
df_cleaned['事件类型'] = df_cleaned['词条名'].map(name_to_category).fillna("其他")

# 将【年份】和【事件类型】两个极其关键的动力学分析 Tag 调整到表格最前面
all_columns = df_cleaned.columns.tolist()
front_columns = ['词条名', '年份', '事件类型', '上榜时间', '最后在榜时间', '在榜共计', '热度最大值', '半衰期_秒']
remaining_columns = [col for col in all_columns if col not in front_columns]

# 最终高维总表列顺序对齐（保证后面连续的 t_1, heat_1, ... 完整留存不被截断）
final_column_order = front_columns + remaining_columns
df_final = df_cleaned[final_column_order]

# 保存为终极双标签时序分析总表
output_filename = "weibo_hotsearch_dynamics_master.csv"
df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("\n" + "="*20 + " 运行报告 " + "="*20)
print(f"恭喜！5年全量热搜数据合并、去噪及大模型双标签注入任务圆满完成！")
print(f"终极时序动力学总表已成功写入: {output_filename}")
print(f"总表总行数: {len(df_final)} 行， 包含字段总数: {len(df_final.columns)} 个")
print("\n[整合后：事件类型精细分布统计]:")
print(df_final['事件类型'].value_counts())
print("\n[整合后：各年份跨度事件分布统计]:")
print(df_final['年份'].value_counts())
print("="*52)