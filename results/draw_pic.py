'''
将实验结果绘制散点图
就是出现阈值进行判断那个
'''
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# —— 1. 设置中文字体 —— #
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei']        # 或者 ['Microsoft YaHei']
# —— 2. 解决坐标轴负号 '-' 显示成方块的问题 —— #
matplotlib.rcParams['axes.unicode_minus'] = False

# 文件路径（根据实际位置调整）
human_tsv_path    = '../datasets/CSL/csl_abstract_40k_sample_random_500.tsv'
human_xlsx_path   = './results/scores_human_mode1.xlsx'
machine_tsv_path  = '../datasets/CSL/csl_abstract_to_combine_LLM_ds_db_500.tsv'
machine_xlsx_path = './results/scores_machine_mode1.xlsx'

# 加载数据
human_df      = pd.read_csv(human_tsv_path, sep='\t', encoding='utf-8')
human_scores  = pd.read_excel(human_xlsx_path, engine='openpyxl')
machine_df    = pd.read_csv(machine_tsv_path, sep='\t', encoding='utf-8')
machine_scores= pd.read_excel(machine_xlsx_path, engine='openpyxl')

# 计算文本长度
human_lengths   = human_df['human'].apply(len)
machine_lengths= machine_df['machine'].apply(len)

# 绘制散点图
plt.figure(figsize=(8,6))
plt.scatter(human_lengths, human_scores['human'], color='blue', label='human')
plt.scatter(machine_lengths, machine_scores['machine'], color='red', label='machine')
plt.xlabel('文本长度（字符数）')
plt.ylabel('分数')
plt.title('文本长度与分数散点图')
plt.legend()
plt.tight_layout()

# 保存到图片文件
output_path = './results/mode1.png'  # 可以根据需要调整路径
plt.savefig(output_path, dpi=300)
print(f"散点图已保存到: {output_path}")