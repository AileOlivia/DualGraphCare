# 没有根据PCA结果进行排序。直接使用PCA后的位置编码结果。
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from FlagEmbedding import FlagModel
from transformers import AutoModel
import os

# 加载嵌入模型
model_path = '/data/BAAI/bge-large-zh-v1.5'
model_instance = AutoModel.from_pretrained(
    model_path,
    ignore_mismatched_sizes=True
)
model = FlagModel(
    model_name_or_path=model_path,
    model=model_instance,
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True
)

# 抗生素列表 - 根据您提供的映射表
antibiotics = [
    "AMOXicillin Oral Susp.", "Amikacin", "Amoxicillin", "Amoxicillin-Clavulanate Susp.", 
    "Amoxicillin-Clavulanic Acid", "Ampicillin", "Ampicillin Sodium", "Ampicillin-Sulbactam",
    "Augmentin Suspension", "Azithromycin", "Azithromycin", "Aztreonam", "Bactrim", "CeFAZolin",
    "CefTAZidime", "CefTAZidime Graded Challenge", "CefTAZidime-Avibactam (Avycaz)", "CefTRIAXone",
    "CefTRIAXone Graded Challenge", "CefazoLIN", "Cefazolin", "CefePIME", "Cefepime", "Cefpodoxime Proxetil",
    "CeftAZIDime", "Ceftaroline", "CeftazIDIME", "Ceftazidime", "Ceftolozane-Tazobactam", "Ceftolozane-Tazobactam *NF*",
    "CeftriaXONE", "Cephalexin", "Ciprofloxacin", "Ciprofloxacin HCl", "Ciprofloxacin IV", "Clarithromycin",
    "Clindamycin", "Clindamycin HCl", "Clindamycin Phosphate", "Clindamycin Solution", "Clindamycin Suspension",
    "DiCLOXacillin", "Dicloxacillin", "Doxycycline Hyclate", "Erythromycin", "Erythromycin Ethylsuccinate Suspension",
    "Erythromycin Lactobionate", "Gentamicin", "Gentamicin Sulfate", "LevoFLOXacin", "Levofloxacin", "Linezolid",
    "Meropenem", "MetRONIDAZOLE (FLagyl)", "MetroNIDAZOLE", "Metronidazole", "Minocycline", "Mupirocin",
    "Mupirocin Nasal Ointment 2%", "Mupirocin Ointment 2%", "Nafcillin", "Neomycin-Polymyxin B GU",
    "Neomycin/Polymyxin B Sulfate", "Nitrofurantoin (Macrodantin)", "Nitrofurantoin Monohyd (MacroBID)", "Oxacillin",
    "Penicillin G Potassium", "Penicillin V Potassium", "Piperacillin", "Piperacillin-Tazo Graded Challenge",
    "Piperacillin-Tazobactam", "Piperacillin-Tazobactam Na", "Rifampin", "Streptomycin Sulfate", "Sulfameth/Trimethoprim",
    "Sulfameth/Trimethoprim DS", "Sulfameth/Trimethoprim SS", "Sulfameth/Trimethoprim Suspension", "Sulfamethoxazole-Trimethoprim",
    "Tetracycline", "Tobramycin Inhalation Soln", "Tobramycin Sulfate", "Trimethoprim", "Unasyn", "Vancomycin",
    "Vancomycin", "Vancomycin Antibiotic Lock", "Vancomycin Enema", "Vancomycin HCl", "Vancomycin Oral Liquid",
    "ceFAZolin", "moxifloxacin", "vancomycin"
]

# 生成抗生素嵌入向量
embeddings = model.encode(antibiotics)

# 使用PCA降维到1维
pca = PCA(n_components=1)
antibiotic_positions = pca.fit_transform(embeddings).flatten()

# 归一化位置到0-1范围
scaler = MinMaxScaler()
antibiotic_positions = scaler.fit_transform(antibiotic_positions.reshape(-1, 1)).flatten()

# 创建抗生素位置映射
antibiotic_position_map = {abx: pos for abx, pos in zip(antibiotics, antibiotic_positions)}

# 打印示例抗生素及其位置
print("抗生素位置示例:")
for abx in antibiotics[:5] + antibiotics[-5:]:
    print(f"- {abx}: {antibiotic_position_map[abx]:.6f}")

# 保存编码映射
code_mapping = pd.DataFrame({
    'antibiotic_nm': antibiotics,
    'antibiotic_position': [antibiotic_position_map[abx] for abx in antibiotics]
})
code_mapping = code_mapping.sort_values('antibiotic_position')
mapping_path = "/autodl-fs/data/SRD-main/data/antibiotic_position_codes.csv"
code_mapping.to_csv(mapping_path, index=False)

print(f"\n抗生素位置编码映射已保存到: {mapping_path}")
print(f"抗生素总数: {len(antibiotics)}")
print(f"位置范围: {np.min(antibiotic_positions):.6f} - {np.max(antibiotic_positions):.6f}")

# 现在将位置编码应用到原始数据中
# 读取抗生素数据
antibiotic_df = pd.read_csv(
    "/autodl-fs/data/SRD-main/data/antibiotic_zigong.csv",
    sep=',',  # 根据您的数据使用适当的分隔符
    header=0
)

# 添加位置编码
antibiotic_df['antibiotic_position'] = antibiotic_df['antibiotic_nm'].map(antibiotic_position_map)

# 按患者ID和时间点分组，计算平均位置
grouped_antibiotics = antibiotic_df.groupby(['id', 'hr'])['antibiotic_position'] \
    .mean() \
    .reset_index(name='avg_antibiotic_position')

# 读取动态数据
dynamic_df = pd.read_csv(
    "/autodl-fs/data/SRD-main/data/dynamic_zigong_with_label.csv",
    sep=',',
    header=0  # 根据您的数据使用适当的分隔符
)

# 合并抗生素位置信息
merged_df = pd.merge(
    dynamic_df,
    grouped_antibiotics,
    on=['id', 'hr'],
    how='left'
)

# 填充缺失值（无抗生素时）
merged_df['avg_antibiotic_position'] = merged_df['avg_antibiotic_position'].fillna(0.0)

# 确定标签列位置并插入抗生素位置
label_col = 'label'  # 替换为实际标签列名
if label_col in merged_df.columns:
    label_index = merged_df.columns.get_loc(label_col)
    cols = merged_df.columns.tolist()
    
    # 将抗生素位置移到标签前
    if 'avg_antibiotic_position' in cols:
        cols.remove('avg_antibiotic_position')
        cols.insert(label_index, 'avg_antibiotic_position')
        merged_df = merged_df[cols]

# 保存最终结果
output_path = "/root/GraphCare-main/data/dynamic_with_antibiotic_position.csv"
merged_df.to_csv(output_path, index=False, sep=',')

print(f"\n合并完成！结果已保存到: {output_path}")
print("前5行数据示例:")
print(merged_df.head())


# git clone https://github.com/FlagOpen/FlagEmbedding.git
