import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import warnings # 【新增】导入警告控制模块

# ============================================================
# 1. 环境与中文字体配置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 环境防乱码
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

INPUT_PURE_CSV = "weibo_hotsearch_data_class.csv"

# ============================================================
# 2. 定义双指数衰减数学模型
# ============================================================
def bi_exponential_model(t, A, alpha, B, beta):
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

# ============================================================
# 3. 核心统计与拟合引擎
# ============================================================
def analyze_pure_data():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    full_input_path = os.path.join(script_dir, INPUT_PURE_CSV)
    
    if not os.path.exists(full_input_path):
        print(f"❌ 找不到新总表文件，请检查是否生成在: {full_input_path}")
        return

    df = pd.read_csv(full_input_path, encoding="utf-8-sig")
    heat_cols = [col for col in df.columns if col.startswith("heat_")]
    
    fit_results = []    
    print("🚀 正在对合并去重后的数据进行逐条双指数动力学拟合（数值鲁棒性升级版）...")
    
    # 【核心防御】将 scipy 弹出的协方差无法估计警告转化为异常，从而用 try-except 完美捕捉并跳过
    warnings.filterwarnings('error', category=RuntimeWarning)
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings('error', category=OptimizeWarning)
    
    for idx, row in df.iterrows():
        word = row["词条名"]
        category = row["事件类型"]
        
        raw_heats = pd.to_numeric(pd.Series(row[heat_cols]), errors='coerce').dropna().values
        if len(raw_heats) < 5:
            continue
            
        # 动力学截取：从单事件热度绝对巅峰开始计算单调衰减
        peak_idx = np.argmax(raw_heats)
        decay_heats = raw_heats[peak_idx:]
        if len(decay_heats) < 4:
            continue
            
        t_steps = np.arange(len(decay_heats))
        
        try:
            # 1. 自适应规模初值估计
            init_guess = [decay_heats[0] * 0.7, 0.6, decay_heats[0] * 0.3, 0.05]
            
            # 2. 【核心优化】设置合理的物理边界 (Bounds)
            # 限制 alpha 最大不能超过 10（1小时衰减到 e^-10 已经无限趋近于0，足够描述最残酷的速冷了）
            # 限制 beta 最大不能超过 1.0，防止长尾机制与速冷机制发生参数混淆
            lower_bounds = [0, 0, 0, 0]
            upper_bounds = [np.inf, 10.0, np.inf, 1.0] 
            
            popt, pcov = curve_fit(
                bi_exponential_model, t_steps, decay_heats, p0=init_guess,
                bounds=(lower_bounds, upper_bounds), maxfev=3000
            )
            A, alpha, B, beta = popt
            
            # 计算拟合优度判定系数 R²
            residuals = decay_heats - bi_exponential_model(t_steps, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((decay_heats - np.mean(decay_heats))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # 3. 学术级阈值过滤
            if r_squared > 0.65:
                total_coef = A + B
                b_ratio = B / total_coef if total_coef > 0 else 0
                
                fit_results.append({
                    "词条名": word,
                    "事件类型": category,
                    "系数_A": A,
                    "速冷系数_alpha": alpha,
                    "系数_B": B,
                    "长尾系数_beta": beta,
                    "长尾基因占比_B_ratio": b_ratio,
                    "拟合优度_R2": r_squared
                })
        except (Exception, RuntimeWarning, OptimizeWarning):
            # 一旦发生无法收敛、奇异矩阵、或触发警告的非自然噪声词条，直接温柔跳过，绝不卡死
            continue

    # 恢复正常的警告设置，避免影响后续绘图
    warnings.resetwarnings()

    features_df = pd.DataFrame(fit_results)
    if features_df.empty:
        print("❌ 糟糕，没有有效的词条通过拟合过滤，请检查输入数据！")
        return
        
    output_detail_path = os.path.join(script_dir, "weibo_bi_exponential_features.csv")
    features_df.to_csv(output_detail_path, index=False, encoding="utf-8-sig")
    print(f"💾 已将全量词条的[完全体双指数特征表]导出至：{output_detail_path}")
    
    # ============================================================
    # 4. 展示基于全新精细化分类的动力学控制特征表格
    # ============================================================
    print("\n📊 " + "="*35 + " 动力学对比分析矩阵（完全体） " + "="*35)
    print("📈 [大作业核心数据表：基于新大模型的事件类型双指数控制参数及权重统计]")
    print("="*110)
    
    summary_table = features_df.groupby("事件类型").agg(
        拟合有效事件数=("词条名", "count"),
        速冷机制_A_中位数=("系数_A", "median"),
        速冷机制_alpha_中位数=("速冷系数_alpha", "median"),
        长尾熬时间_B_中位数=("系数_B", "median"),
        长尾熬时间_beta_中位数=("长尾系数_beta", "median"),
        长尾基因占比_B_ratio_中位数=("长尾基因占比_B_ratio", "median"),
        平均拟合优度_R2=("拟合优度_R2", "mean")
    ).sort_values(by="速冷机制_alpha_中位数", ascending=False)
    
    print(summary_table.to_string(formatters={
        '速冷机制_A_中位数': '{:,.1f}'.format,
        '速冷机制_alpha_中位数': '{:,.4f}'.format,
        '长尾熬时间_B_中位数': '{:,.1f}'.format,
        '长尾熬时间_beta_中位数': '{:,.4f}'.format,
        '长尾基因占比_B_ratio_中位数': '{:.1%}'.format,
        '平均拟合优度_R2': '{:.1%}'.format
    }))
    print("="*110)
    
    # ============================================================
    # 5. 绘制精细标签交叉对比交叉箱线图（Boxplot）
    # ============================================================
    print("🎨 正在绘制多标签动力学特征学术对比图表...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：Alpha 箱线图
    sns.boxplot(data=features_df, x="事件类型", y="速冷系数_alpha", ax=ax1, hue="事件类型", palette="Set3", legend=False)
    ax1.set_title("不同事件类型的『速冷系数 $\\alpha$』分布对比\n($\\alpha$ 越大代表最初 1-2 小时降温腰斩越猛烈)")
    ax1.set_xlabel("事件类型（大模型清洗分类）")
    ax1.set_ylabel("速冷系数 $\\alpha$")
    ax1.tick_params(axis='x', rotation=35)
    
    # 右图：长尾基因占比 B_ratio 箱线图
    sns.boxplot(data=features_df, x="事件类型", y="长尾基因占比_B_ratio", ax=ax2, hue="事件类型", palette="Pastel1", legend=False)
    ax2.set_title("不同事件类型的『长尾基因初始占比 $B/(A+B)$』分布对比\n(占比越高说明该事件原生具备的社会公共讨论底噪越强)")
    ax2.set_xlabel("事件类型（大模型清洗分类）")
    ax2.set_ylabel("长尾基因占比 (B_ratio)")
    ax2.tick_params(axis='x', rotation=35)
    
    plt.tight_layout()
    plot_path = os.path.join(script_dir, "事件类型_动力学特征完全体对比图.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🎉 鲁棒版双动力学交叉箱线图绘制成功！已妥善输出至：\n👉 {plot_path}")

if __name__ == "__main__":
    analyze_pure_data()