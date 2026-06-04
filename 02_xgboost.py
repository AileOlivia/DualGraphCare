import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntibioticDataset(Dataset):
    """自定义抗生素数据集"""
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        chinese = str(row['chinese_name']).strip()
        english = str(row['english_name']).strip()
        
        # 对中英文文本分别编码
        chinese_encoding = self.tokenizer(
            chinese, 
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        english_encoding = self.tokenizer(
            english,
            padding='max_length', 
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'chinese_input_ids': chinese_encoding['input_ids'].squeeze(),
            'chinese_attention_mask': chinese_encoding['attention_mask'].squeeze(),
            'english_input_ids': english_encoding['input_ids'].squeeze(), 
            'english_attention_mask': english_encoding['attention_mask'].squeeze()
        }

class BGEEncoder(nn.Module):
    """BGE编码器封装"""
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 使用平均池化获取句子嵌入
        embeddings = self.mean_pooling(outputs, attention_mask)
        return embeddings
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def train_bge_model():
    """训练BGE模型"""
    
    # 配置
    model_name = "BAAI/bge-large-zh-v1.5"
    data_path = "/root/GraphCare-main-text-zigong-48h/data/antibiotics_raw.csv"
    output_dir = "/root/GraphCare-main-text-zigong-48h/data/bge_final"
    
    # 检查文件
    if not os.path.exists(data_path):
        logger.error(f"文件不存在: {data_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载模型和tokenizer
    logger.info("加载BGE模型...")
    model = BGEEncoder(model_name)
    tokenizer = model.tokenizer
    
    # 移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 2. 准备数据
    logger.info("准备数据集...")
    dataset = AntibioticDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    cosine_loss = nn.CosineEmbeddingLoss()
    
    # 4. 训练循环
    model.train()
    num_epochs = 3
    
    logger.info("开始训练...")
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            # 移动数据到设备
            chinese_input_ids = batch['chinese_input_ids'].to(device)
            chinese_attention_mask = batch['chinese_attention_mask'].to(device)
            english_input_ids = batch['english_input_ids'].to(device) 
            english_attention_mask = batch['english_attention_mask'].to(device)
            
            # 前向传播
            chinese_embeddings = model(chinese_input_ids, chinese_attention_mask)
            english_embeddings = model(english_input_ids, english_attention_mask)
            
            # 计算损失 - 我们希望中英文嵌入相似
            target = torch.ones(chinese_embeddings.size(0)).to(device)  # 目标相似度为1
            loss = cosine_loss(chinese_embeddings, english_embeddings, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 5 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
    
    # 5. 保存模型
    logger.info(f"保存模型到: {output_dir}")
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("训练完成!")

def test_model():
    """测试训练好的模型"""
    data_path = "/root/GraphCare-main-text-zigong-48h/data/antibiotics_raw.csv"
    model_dir = "/root/GraphCare-main-text-zigong-48h/data/bge_final"
    
    if not os.path.exists(model_dir):
        logger.error("模型目录不存在，请先训练模型")
        return
    
    # 加载训练好的模型
    model = BGEEncoder(model_dir)
    model.eval()
    
    # 测试几个样本
    test_pairs = [
        ("阿莫西林", "Amoxicillin"),
        ("青霉素", "Penicillin"), 
        ("头孢曲松", "Ceftriaxone")
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info("测试模型相似度:")
    with torch.no_grad():
        for chinese, english in test_pairs:
            # 编码中文
            chinese_encoding = model.tokenizer(chinese, return_tensors='pt', padding=True, truncation=True)
            chinese_input_ids = chinese_encoding['input_ids'].to(device)
            chinese_attention_mask = chinese_encoding['attention_mask'].to(device)
            
            # 编码英文  
            english_encoding = model.tokenizer(english, return_tensors='pt', padding=True, truncation=True)
            english_input_ids = english_encoding['input_ids'].to(device)
            english_attention_mask = english_encoding['attention_mask'].to(device)
            
            # 获取嵌入
            chinese_embedding = model(chinese_input_ids, chinese_attention_mask)
            english_embedding = model(english_input_ids, english_attention_mask)
            
            # 计算余弦相似度
            cos_sim = nn.CosineSimilarity(dim=1)
            similarity = cos_sim(chinese_embedding, english_embedding)
            
            logger.info(f"  {chinese} - {english}: {similarity.item():.4f}")

if __name__ == "__main__":
    # 先训练模型
    train_bge_model()
    
    # 然后测试模型
    test_model()

#   warnings.warn(_BETA_TRANSFORMS_WARNING)
# INFO:__main__:准备数据集...
# INFO:__main__:开始训练...
# INFO:__main__:Epoch 1, Batch 5, Loss: 0.0307
# INFO:__main__:Epoch 1, Batch 10, Loss: 0.0134
# INFO:__main__:Epoch 1 完成, 平均损失: 0.0961
# INFO:__main__:Epoch 2, Batch 5, Loss: 0.0092
# INFO:__main__:Epoch 2, Batch 10, Loss: 0.0088
# INFO:__main__:Epoch 2 完成, 平均损失: 0.0092
# INFO:__main__:Epoch 3, Batch 5, Loss: 0.0077
# INFO:__main__:Epoch 3, Batch 10, Loss: 0.0073
# INFO:__main__:Epoch 3 完成, 平均损失: 0.0075
# INFO:__main__:保存模型到: /root/autodl-fs/SRD-main-text-zigong-48h/data/bge_final
# INFO:__main__:训练完成!
# INFO:__main__:测试模型相似度:
# Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
# INFO:__main__:  阿莫西林 - Amoxicillin: 1.0000
# INFO:__main__:  青霉素 - Penicillin: 0.9999
# INFO:__main__:  头孢曲松 - Ceftriaxone: 1.0000
