import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import random

"""
    数据集
    比原版多了一个"交叉特征"：user_id % 10 * 500 + item_id
    这个组合特征会被喂给 Wide 侧做记忆
"""
class ToyCTRDataset(Dataset):
    def __init__(self, n_samples=10000):
        self.data = []
        for _ in range(n_samples):
            user_id = random.randint(0, 999)
            item_id = random.randint(0, 499)
            gender = random.randint(0, 1)
            hour = random.randint(0, 23)

        # Wide 侧的"交叉特征"：把 user_id bucket × item_id 编码成一个整数
        # 现实里通常用 feature hashing 把高维稀疏特征压到固定桶数
        # 这里简化：user bucket(0-9) × 500 + item_id → 最大 9×500+499 = 4999
        cross = (user_id % 10) * 500 + item_id

        # 可学习模式：user 和 item 的 bucket 相同时点击
        click = 1 if (user_id % 10 == item_id % 10) else 0

        self.data.append((user_id, item_id, gender, hour, cross, click))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

"""
Wide & Deep 模型
 
   ┌─────────────────────────────────────────┐
   │              输入特征                    │
   │  user_id  item_id  gender  hour  cross  │
   └────┬──────────┬──────────────────┬──────┘
        │ Deep 侧  │                  │ Wide 侧
        ▼          ▼                  ▼
    Embedding   Embedding        Linear(cross)
        └──────────┘                  │
           Concat (40-dim)            │
               │                      │
            MLP × 3                   │
               │                      │
          deep_logit             wide_logit
               └──────── Add ─────────┘
                          │
                       Sigmoid
                          │
                     CTR 预估值
"""   
class WideAndDeepCTRModel(nn.Model):
    """
    Wide & Deep Learning for Recommender Systems (Google, 2016)
 
    Wide 侧：线性模型，直接作用于稀疏交叉特征
             擅长"记忆"：历史上共现过的特征组合
             缺点：无法泛化到未见过的组合
 
    Deep 侧：Embedding + MLP，作用于稠密特征
             擅长"泛化"：通过低维向量空间找到相似性
             缺点：容易过度泛化，丢失精确规则
 
    联合训练：两侧 logit 相加后一起过 sigmoid，梯度同时回传
    """
    def __init__(
        self,
        # Deep 侧超参数
        n_users=1000, n_items=500, n_genders=2, n_hours=24,
        user_emb_dim=16, item_emb_dim=16, gender_emb_dim=4, hour_emb_dim=4,
        deep_hidden=[128, 64],   # MLP 每层的宽度
        dropout=0.3,             # Dropout 防过拟合
        # Wide 侧超参数
        cross_feature_dim=5000,  # 交叉特征的桶数（feature hashing 后的维度）
    ):
        super.__init__()

        # ── Deep 侧：Embedding 层 ──────────────────────────────────────────
        # 每个离散特征 → 低维稠密向量
        # Embedding(词表大小, 向量维度)
        self.user_emb = nn.embedding(n_users, user_emb_dim)         # (1000, 16)
        self.item_emb = nn.embedding(n_items, item_emb_dim)         # (500, 16)
        self.gender_emb = nn.embedding(n_genders, gender_emb_dim)   # (2, 4)
        self.hour_emb = nn.embedding(n_hours, hour_emb_dim)         # (24, 4)

        # ── Deep 侧：MLP ──────────────────────────────────────────────────
        # 输入维度 = 所有 embedding 拼接后的总维度
        deep_input_dim = user_emb_dim + item_emb_dim + gender_emb_dim + hour_emb_dim
        # 16 + 16 + 4 + 4 = 40

        layers = []
        in_dim = deep_input_dim
        for out_dim in deep_hidden:
            layers += [
                nn.linear(in_dim, out_dim),
                # BatchNorm：对每个 batch 内的特征做归一化
                # 稳定训练、允许更大学习率、有轻微正则化效果
                # 注意：BN 在 eval() 模式下使用全局均值/方差，train() 模式用 batch 统计
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                # Dropout：随机丢弃神经元，防止 co-adaptation（协同适应）
                # 只在 train() 模式激活，eval() 模式自动关闭
                nn.Dropout(dropout),
            ]
            in_dim = out_dim

        # 最后一层：输出 1 个标量（Deep 侧 logit）
        layers.append(nn.linear(in_dim, 1))
        self.deep_mlp = nn.Sequential(*layers)

        # ── Wide 侧：线性层 ───────────────────────────────────────────────
        # 输入：cross 特征的 one-hot 向量（维度=cross_feature_dim）
        # 这里用 Embedding 模拟稀疏线性层：
        #   Embedding(5000, 1) 等价于 对 one-hot 向量做 Linear(5000, 1) 但更高效
        #   因为不需要真正构造 5000 维向量，直接查表取对应权重即可
        # 输出：1 个标量（Wide 侧 logit）
        self.wide_linear = nn.embedding(cross_feature_dim, 1)

        # Wide 侧 bias（全局截距项）
        self.wide_bias = nn.Parameter(torch.zero(1))

    def forward(self, user_id, item_id, gender, hour, cross):
        # ── Deep 侧前向 ───────────────────────────────────────────────────
        # 1. 各特征查 Embedding 表，得到稠密向量
        user_vec   = self.user_emb(user_id)     # [B, 16]