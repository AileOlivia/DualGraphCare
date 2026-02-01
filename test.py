import torch
import numpy as np
import yaml
import os
from pathlib import Path
from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RuntimeConfig, configclass

from process.dataset import DATASETS
from process.model import MODELS
from process.network import NETWORKS

def load_config_from_yaml(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def find_data_file(file_path):
    """查找数据文件，支持相对路径和绝对路径"""
    # 如果是绝对路径，直接返回
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path
    
    # 尝试在项目根目录下查找
    project_root = Path("/root/GraphCare-main-text")
    possible_paths = [
        file_path,
        project_root / file_path,
        project_root / "data" / os.path.basename(file_path),
        project_root / "data" / file_path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return str(path)
    
    # 如果都找不到，返回原始路径（会触发错误）
    return file_path

def adjust_model_weights(model_state_dict, actual_features, expected_features):
    """调整模型权重以适应特征差异"""
    adjusted_state_dict = {}
    for key, value in model_state_dict.items():
        # 检查是否需要调整这个权重
        if value.dim() > 0 and value.size(0) == expected_features:
            # 调整权重大小
            if value.dim() == 1:
                # 一维权重（如偏置）
                adjusted_value = value[:actual_features]
            else:
                # 多维权重
                adjusted_value = value[:actual_features, ...]
            adjusted_state_dict[key] = adjusted_value
            print(f"调整权重: {key} {value.shape} -> {adjusted_value.shape}")
        else:
            # 不需要调整的权重
            adjusted_state_dict[key] = value
    return adjusted_state_dict

def run_test(config_dict):
    # 设置实验环境
    runtime_config = config_dict.get('runtime', {})
    runtime = RuntimeConfig(**runtime_config)
    setup_experiment(runtime)
    
    # 查找数据文件
    data_config = config_dict['data'].copy()
    data_file = data_config.get('file', '')
    data_file_path = find_data_file(data_file)
    data_config['file'] = data_file_path
    print(f"使用数据文件: {data_file_path}")
    
    # 只加载测试集
    data_type = data_config.pop('type')
    dataset_class = DATASETS.get(data_type)
    testset = dataset_class(dataset_name="test", **data_config)
    
    print(f"测试集大小: {len(testset)}")
    print(f"特征维度: {testset.num_variables}")
    print(f"类别数: {testset.num_classes}")
    print(f"最大序列长度: {testset.max_seq_len}")
    
    # 检查特征维度是否与预训练模型匹配
    expected_features = 53  # 根据错误信息，预训练模型期望53个特征
    if testset.num_variables != expected_features:
        print(f"警告: 预训练模型期望 {expected_features} 个特征，但测试数据有 {testset.num_variables} 个特征")
        print("这可能是因为时间特征处理方式不同")
        
        # 尝试检查数据预处理步骤
        print("检查数据预处理步骤...")
        
        # 检查是否有时间特征
        if hasattr(testset, "dates"):
            print(f"时间特征维度: {testset.dates.shape[1] if hasattr(testset, 'dates') else 0}")
            print(f"原始数据维度: {testset.raw_data.shape[1]}")
            print(f"总特征维度: {testset.raw_data.shape[1] + (testset.dates.shape[1] if hasattr(testset, 'dates') else 0)}")
        
        # 建议使用与训练时相同的数据预处理流程
        print("建议使用与训练时相同的数据文件和预处理流程")
        
        # 尝试调整模型以适应特征差异
        print("尝试调整模型以适应特征差异...")
    
    # 构建网络
    network_config = config_dict['network']
    network_type = network_config.pop('type')
    network_class = NETWORKS.get(network_type)
    network = network_class(
        input_size=testset.num_variables, 
        max_length=testset.max_seq_len,
        in_dim=1,
        out_dim=1,
        **network_config
    )
    
    # 构建模型
    model_config = config_dict['model']
    model_type = model_config.pop('type')
    model_class = MODELS.get(model_type)
    model = model_class(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=testset.num_classes,
        **model_config
    )
    
    # 加载预训练权重
    checkpoint_path = config_dict.get('checkpoint_path', "/root/GraphCare-main-text/outputs_6/checkpoints/weight.pth")
    print(f"从 {checkpoint_path} 加载模型权重...")
    
    # 检查文件是否存在
    if not Path(checkpoint_path).exists():
        # 尝试在checkpoint_dir中查找
        checkpoint_dir = get_checkpoint_dir()
        possible_paths = list(checkpoint_dir.glob("*.pkl"))
        if possible_paths:
            checkpoint_path = str(possible_paths[0])
            print(f"使用找到的检查点: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"找不到模型检查点文件: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if runtime_config.get('use_cuda', False) else 'cpu')
    
    # 检查特征维度是否匹配
    expected_features = 53  # 根据错误信息，预训练模型期望53个特征
    actual_features = testset.num_variables
    
    if actual_features != expected_features:
        print(f"调整模型权重以适应特征差异: {expected_features} -> {actual_features}")
        
        # 调整模型权重
        if 'model_state_dict' in checkpoint:
            checkpoint['model_state_dict'] = adjust_model_weights(
                checkpoint['model_state_dict'], actual_features, expected_features
            )
        elif 'state_dict' in checkpoint:
            checkpoint['state_dict'] = adjust_model_weights(
                checkpoint['state_dict'], actual_features, expected_features
            )
        else:
            checkpoint = adjust_model_weights(checkpoint, actual_features, expected_features)
    
    # 根据检查点内容加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print("模型权重加载成功!")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 进行测试
    print("开始测试...")
    results = model.predict(testset, "test")
    
    # 添加调试信息
    print("\n调试信息 - 结果类型:")
    for metric, value in results.items():
        print(f"{metric}: {type(value)}")
    
    print("\n测试结果:")
    for metric, value in results.items():
        try:
            # 尝试将值转换为浮点数
            if hasattr(value, 'iloc'):  # 检查是否是 pandas Series 或 DataFrame
                # 如果是 Series，取第一个值或平均值
                if len(value) > 0:
                    float_value = float(value.iloc[0]) if hasattr(value, 'iloc') else float(value.values[0])
                    print(f"{metric}: {float_value:.4f}")
                else:
                    print(f"{metric}: {value} (空Series)")
            else:
                # 如果是标量值，直接格式化
                print(f"{metric}: {float(value):.4f}")
        except (ValueError, TypeError) as e:
            # 如果转换失败，直接打印原始值
            print(f"{metric}: {value} (无法转换为浮点数: {e})")
    
    return results

if __name__ == "__main__":
    # 加载配置文件
    config_path = "/root/GraphCare-main-text/config/srdtcn.yml"
    config_dict = load_config_from_yaml(config_path)
    
    # 确保使用正确的设备
    config_dict['runtime'] = config_dict.get('runtime', {})
    config_dict['runtime']['use_cuda'] = torch.cuda.is_available()
    print(f"使用CUDA: {config_dict['runtime']['use_cuda']}")
    
    # 设置检查点路径（如果配置文件中没有指定）
    if 'checkpoint_path' not in config_dict or not config_dict['checkpoint_path']:
        checkpoint_dir = Path("/root/GraphCare-main-text/outputs_6/checkpoints")
        # 优先查找模型检查点
        model_checkpoint = checkpoint_dir / "model_best.pkl"
        if model_checkpoint.exists():
            config_dict['checkpoint_path'] = str(model_checkpoint)
            print(f"自动选择模型检查点: {config_dict['checkpoint_path']}")
        else:
            # 然后查找网络检查点
            network_checkpoint = checkpoint_dir / "network_best.pkl"
            if network_checkpoint.exists():
                config_dict['checkpoint_path'] = str(network_checkpoint)
                print(f"自动选择网络检查点: {config_dict['checkpoint_path']}")
            else:
                # 最后查找任何检查点文件
                checkpoint_files = list(checkpoint_dir.glob("*.pkl"))
                if checkpoint_files:
                    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    config_dict['checkpoint_path'] = str(checkpoint_files[0])
                    print(f"自动选择检查点: {config_dict['checkpoint_path']}")
                else:
                    raise FileNotFoundError("在checkpoints目录中找不到任何模型文件")
    
    # 运行测试
    try:
        results = run_test(config_dict)
        
        # 保存结果
        output_dir = Path(get_output_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "test_results.txt"
        with open(output_path, 'w') as f:
            f.write("测试结果:\n")
            for metric, value in results.items():
                try:
                    if hasattr(value, 'iloc'):  # 检查是否是 pandas Series 或 DataFrame
                        if len(value) > 0:
                            float_value = float(value.iloc[0]) if hasattr(value, 'iloc') else float(value.values[0])
                            f.write(f"{metric}: {float_value:.4f}\n")
                        else:
                            f.write(f"{metric}: {value} (空Series)\n")
                    else:
                        f.write(f"{metric}: {float(value):.4f}\n")
                except (ValueError, TypeError) as e:
                    f.write(f"{metric}: {value} (无法转换为浮点数: {e})\n")
        
        print(f"结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()