import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import numpy as np
import numpy.core.multiarray
import torch.serialization
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 添加安全全局变量
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

# 添加项目根目录到系统路径
project_root = '/root/GraphCare-main-text'
sys.path.append(project_root)

# 使用绝对导入路径
from process.network.graphcare import GraphCare

def get_embedding_dim(dyna_gc):
    """获取动态图构造器的嵌入维度"""
    # 方法1: 检查是否有 node_dim 属性
    if hasattr(dyna_gc, 'node_dim'):
        print(f"找到 node_dim: {dyna_gc.node_dim}")
        return dyna_gc.node_dim
    
    # 方法2: 检查是否有 embedding_dim 属性
    if hasattr(dyna_gc, 'embedding_dim'):
        print(f"找到 embedding_dim: {dyna_gc.embedding_dim}")  # 打印嵌入维度]
        return dyna_gc.embedding_dim
    
    # 方法3: 从线性层获取输入特征数
    for name, module in dyna_gc.named_modules():
        if isinstance(module, nn.Linear):
            print(f"找到线性层: {name}, 输入维度={module.in_features}")
            return module.in_features
    
    # 方法4: 从参数形状推断
    for name, param in dyna_gc.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            print(f"找到权重参数: {name}, 形状={param.shape}")
            return param.shape[1]
    
    # 默认值
    return 16

# 1. 构造"空"模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 关键修改：设置正确的 node_dim
model = GraphCare(
    input_size=53,      # 节点数=53
    max_length=12,      # 时间步长=12 (根据配置文件 window=12)
    node_dim=16,        # 嵌入维度=16 (根据打印信息)
    weight_file=None    # 先不加载
).to(device)

# 2. 加载权重 - 使用 weights_only=True 和安全全局变量
resume_path = Path('/root/GraphCare-main-text/outputs_mimic/checkpoints/weight.pth')
state_dict = torch.load(resume_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict, strict=False)
model.eval()

# 3. 从CSV文件读取数据
csv_path = '/root/GraphCare-main-text/data/dynamic_mimic_with_antibiotic_position.csv'  # 替换为实际CSV文件路径
df = pd.read_csv(csv_path)

# 选择特征列 - 根据提供的格式，排除id、hr和label列
feature_columns = [
    'heart_rate', 'sbp', 'mbp', 'resp_rate', 'temperature', 'spo2', 'glucose_lab_fingerstick',
    'urineoutput', 'cvp', 'albumin', 'aniongap', 'bicarbonate', 'bun', 'calcium', 'chloride',
    'creatinine', 'glucose', 'sodium', 'potassium', 'fibrinogen', 'inr', 'pt', 'ptt',
    'hematocrit', 'hemoglobin', 'platelet', 'wbc', 'alt', 'ast', 'bilirubin', 'pao2', 'paco2',
    'fio2', 'pao2fio2ratio', 'ph', 'baseexcess', 'lactate', 'troponin', 'magnesium', 'bnp',
    'lymphocytes', 'neutrophils', 'alkaline_phosphatase', 'gnri', 'vasopressor', 'ventilation',
    'sofa', 'avg_antibiotic_position'
]

# 检查特征数量
if len(feature_columns) != 48:
    print(f"警告: 期望48个特征，实际找到{len(feature_columns)}个特征。可能需要调整特征列表。")

# 提取特征数据并填充缺失值
features = df[feature_columns].fillna(0).values  # 用0填充缺失值

# 标准化数据
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 4. 提取邻接矩阵
with torch.no_grad():
    # 节点索引 (0到52)
    idx = torch.arange(53).to(device)
    
    # 静态图邻接矩阵
    Ap = model.gc.fullA(idx)
    
    # 动态图邻接矩阵 - 创建嵌入向量
    # 使用自定义函数获取嵌入维度
    embedding_dim = get_embedding_dim(model.dyna_gc)
    print(f"使用嵌入维度: {embedding_dim}")
    
    # 从CSV数据创建嵌入向量
    # 选择第一行数据并调整形状 (1, 53, embedding_dim)
    # 注意: 我们有48个特征，但模型期望53个节点，所以需要扩展或截断
    if scaled_features.shape[1] < 53:
        # 如果特征少于53个，填充0
        padding = np.zeros((scaled_features.shape[0], 53 - scaled_features.shape[1]))
        padded_features = np.hstack((scaled_features, padding))
        emb = torch.tensor(padded_features[0:1, :], dtype=torch.float32).view(1, 53, -1).to(device)
    elif scaled_features.shape[1] > 53:
        # 如果特征多于53个，截断
        emb = torch.tensor(scaled_features[0:1, :53], dtype=torch.float32).view(1, 53, -1).to(device)
    else:
        # 正好53个特征
        emb = torch.tensor(scaled_features[0:1, :], dtype=torch.float32).view(1, 53, -1).to(device)
    
    print(f"嵌入向量形状: {emb.shape}")
    
    # 获取动态邻接矩阵
    Ad = model.dyna_gc.fullA(idx, emb)  # (1, 53, 53)
    
    # 取第一个样本的动态邻接矩阵
    Ad_sample = Ad[0].cpu().numpy()

# 5. 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(Ap.cpu().numpy(), cmap='Blues', square=True, cbar=True)
plt.title('Prior Graph A_p (53x53)')
plt.tight_layout()
plt.savefig('/root/autodl-fs/SRD-main-text/outputs_mimic/Ap_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(Ad_sample, cmap='Reds', square=True, cbar=True)
plt.title('Dynamic Graph A_d (Sample 0, 53x53)')
plt.tight_layout()
plt.savefig('/root/autodl-fs/SRD-main-text/outputs_mimic/Ad_heatmap_sample0.png', dpi=300, bbox_inches='tight')
plt.close()

print("成功生成邻接矩阵热力图！")

# 6. 绘制特征对随时间变化曲线
# 设置滑动窗口大小
win = 12  # 滑动窗口大小为12个时间步

# 计算总窗口数
windows = scaled_features.shape[0] - win + 1
print(f"总数据点: {scaled_features.shape[0]}, 滑动窗口大小={win}, 总窗口数={windows}")

# 选择4对特征
pairs = [
    (11, 48),  
    (1, 48), 
    (28, 48), 
    (4, 48)    
]

plt.figure(figsize=(12, 6))
for i, j in pairs:
    rho2_series = []
    for w in range(windows):
        # 提取当前窗口数据
        a = scaled_features[w:w+win, i]
        b = scaled_features[w:w+win, j]
        
        # 中心化数据
        a -= a.mean()
        b -= b.mean()
        
        # 计算相关系数的平方
        denom = np.mean(a**2) * np.mean(b**2)
        rho2 = (np.mean(a*b)**2) / denom if denom != 0 else 0
        rho2_series.append(rho2)
    
    # 绘制曲线
    plt.plot(range(windows), rho2_series, label=f"特征对 ({i},{j})", linewidth=2)

plt.xlabel("时间窗口索引", fontsize=12)
plt.ylabel("相关系数平方 (ρ²)", fontsize=12)
plt.ylim(0, 1.1)
plt.xlim(0, windows-1)
plt.legend(fontsize=10)
plt.title(f"特征相关系数随时间变化 (窗口大小={win})", fontsize=14)
plt.tight_layout()
plt.grid(True, linestyle='-', alpha=0.3)
plt.savefig('/root/GraphCare-main-text/outputs_mimic/feature_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 保存特征名称和对应索引
feature_info = pd.DataFrame({
    'index': range(len(feature_columns)),
    'feature_name': feature_columns
})
feature_info.to_csv('/root/GraphCare-main-text/outputs_mimic/feature_indices.csv', index=False)
print(f"保存特征索引信息到: /root/GraphCare-main-text/outputs_mimic/feature_indices.csv")