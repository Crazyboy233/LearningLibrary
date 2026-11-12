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
class WideAndDeepCTRModel(nn.Module):
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
        super().__init__()

        # ── Deep 侧：Embedding 层 ──────────────────────────────────────────
        # 每个离散特征 → 低维稠密向量
        # Embedding(词表大小, 向量维度)
        self.user_emb = nn.Embedding(n_users, user_emb_dim)         # (1000, 16)
        self.item_emb = nn.Embedding(n_items, item_emb_dim)         # (500, 16)
        self.gender_emb = nn.Embedding(n_genders, gender_emb_dim)   # (2, 4)
        self.hour_emb = nn.Embedding(n_hours, hour_emb_dim)         # (24, 4)

        # ── Deep 侧：MLP ──────────────────────────────────────────────────
        # 输入维度 = 所有 embedding 拼接后的总维度
        deep_input_dim = user_emb_dim + item_emb_dim + gender_emb_dim + hour_emb_dim
        # 16 + 16 + 4 + 4 = 40

        layers = []
        in_dim = deep_input_dim
        for out_dim in deep_hidden:
            layers += [
                nn.Linear(in_dim, out_dim),
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
        layers.append(nn.Linear(in_dim, 1))
        self.deep_mlp = nn.Sequential(*layers)

        # ── Wide 侧：线性层 ───────────────────────────────────────────────
        # 输入：cross 特征的 one-hot 向量（维度=cross_feature_dim）
        # 这里用 Embedding 模拟稀疏线性层：
        #   Embedding(5000, 1) 等价于 对 one-hot 向量做 Linear(5000, 1) 但更高效
        #   因为不需要真正构造 5000 维向量，直接查表取对应权重即可
        # 输出：1 个标量（Wide 侧 logit）
        self.wide_linear = nn.Embedding(cross_feature_dim, 1)

        # Wide 侧 bias（全局截距项）
        self.wide_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_id, item_id, gender, hour, cross):
        # ── Deep 侧前向 ───────────────────────────────────────────────────
        # 1. 各特征查 Embedding 表，得到稠密向量
        user_vec   = self.user_emb(user_id)     # [B, 16]
        item_vec   = self.item_emb(item_id)     # [B, 16] 
        gender_vec = self.gender_emb(gender)    # [B, 4]
        hour_vec   = self.hour_emb(hour)        # [B, 4]

        # 2. 拼接所有向量
        # torch.cat([...], dim=1) → 在特征维度(dim=1)拼接
        deep_input = torch.cat([user_vec, item_vec, gender_vec, hour_vec], dim=1)
        # shape: [B, 40]

        # 3. 过 MLP，输出 Deep logit
        deep_logit = self.deep_mlp(deep_input)  # [B, 1]

        # ── Wide 侧前向 ───────────────────────────────────────────────────
        # 直接用交叉特征 ID 查权重表（等价于稀疏线性层）
        # wide_linear(cross) 的结果是 [B, 1]
        wide_logit = self.wide_linear(cross) + self.wide_bias   # [B, 1]

        # ── 联合输出 ──────────────────────────────────────────────────────
        # 两侧 logit 直接相加（这是 Wide&Deep 的核心设计）
        # 不是加权平均，而是直接相加让两侧互补
        logit = deep_logit + wide_logit # [B, 1]

        return logit

# ------------------------    
# 训练
# ------------------------
dataset = ToyCTRDataset()
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = WideAndDeepCTRModel()
criterion = nn.BCEWithLogitsLoss()  # 内部包含 sigmoid，数值更稳定

# Wide 和 Deep 侧用不同学习率（工程实践）
# Wide 侧：用 FTRL（这里用 SGD 近似），学习率较大，快速记忆
# Deep 侧：用 Adam，学习率较小，稳定学习 embedding
wide_params = list(model.wide_linear.parameters()) + [model.wide_bias]
deep_params = (
    list(model.user_emb.parameters()) +
    list(model.item_emb.parameters()) +
    list(model.gender_emb.parameters()) +
    list(model.hour_emb.parameters()) +
    list(model.deep_mlp.parameters())
)

optimizer = optim.AdamW([
    {"params":wide_params, "lr":0.01, "weight_decay":0.0},      # wide 侧
    {"params":deep_params, "lr":0.001, "weight_decay":1e-5},    # deep 侧
])
 
# 学习率调度：每 5 epoch 乘以 0.5（模拟退火）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print("开始训练 Wide & Deep CTR 模型...")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}\n")

for epoch in range(1, 11):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for batch in loader:
        user_id, item_id, gender, hour, cross, label = batch
        label = label.float().unsqueeze(1)  # [B] -> [B, 1]

        logit = model(user_id, item_id, gender, hour, cross)    # 这里其实是在调用 forward()

        loss = criterion(logit, label)

        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪：防止梯度爆炸（对 Wide 侧特别重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计准确率（threshold=0.5）
        pred = (torch.sigmoid(logit) > 0.5).float()
        total_correct += (pred == label).sum().item()
        total_loss    += loss.item() * label.size(0)
        total         += label.size(0)

    scheduler.step()

    avg_loss = total_loss / total
    accuracy = total_correct / total
    print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
          f"LR(wide): {optimizer.param_groups[0]['lr']:.5f}")
    

# ─────────────────────────────────────────────
#  推理示例
# ─────────────────────────────────────────────
model.eval()
# ↑ 切换到推理模式：
#   - BatchNorm 从"用当前 batch 统计"切换到"用训练期间积累的全局均值/方差"
#   - Dropout 从"随机丢弃"切换到"全部保留（乘以保留概率缩放）"

with torch.no_grad():
    # ↑ 关闭梯度计算，节省显存和计算，推理时不需要反向传播
    # 构造一条样本：user=5, item=5（同 bucket，预期点击）
    user_id = torch.tensor([5])
    item_id = torch.tensor([5])
    gender  = torch.tensor([1])
    hour    = torch.tensor([12])
    cross   = torch.tensor([(5 % 10) * 500 + 5])  # = 5*500+5 = 2505

    logit = model(user_id, item_id, gender, hour, cross)
    prob = torch.sigmoid(logit).item()
    print(f"\n[推理] user=5, item=5（同 bucket）→ CTR 预估: {prob:.4f}")

    # 构造一条样本：user=5, item=6（不同 bucket，预期不点击）
    user_id = torch.tensor([5])
    item_id = torch.tensor([6])
    cross   = torch.tensor([(5 % 10) * 500 + 6])
 
    logit = model(user_id, item_id, gender, hour, cross)
    prob  = torch.sigmoid(logit).item()
    print(f"[推理] user=5, item=6（不同bucket）→ CTR 预估: {prob:.4f}")