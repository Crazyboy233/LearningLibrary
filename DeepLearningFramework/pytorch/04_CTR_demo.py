import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import random

class ToyCTRDataset(Dataset):
    def __init__(self, n_samples=10000):
        self.data = []
        for _ in range(n_samples):
            user_id = random.randint(0, 999)    # 1000 users
            item_id = random.randint(0, 499)    # 500 items
            gender = random.randint(0, 1)       # 0/1
            hour = random.randint(0, 23)
            # print(f"{user_id}, {item_id}, {gender}, {hour}")
            # 人为制造一点“可学习模式”
            click = 1 if (user_id % 10 == item_id % 10) else 0

            self.data.append((user_id, item_id, gender, hour, click))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CTRModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 本质是一个 可学习的查表矩阵
        # 等价于：user_emb.weight.shape == (1000, 16)
        # 输入：user_id（LongTensor）
        # 输出：对应行的向量
        # 这就是搜广推里的 sparse feature → dense vector
        """
            这里的1000, 500, 2, 24 是指有这么多用户，分类。
            16, 4, 4 是人为设置的超参数

            nn.Embedding(1000, 16) 相当于创建了一个1000 * 16的矩阵
            从此刻开始，一个 user_id 才变成 1 * 16 的 Tensor
        """
        self.user_emb = nn.Embedding(1000, 16)        
        self.item_emb = nn.Embedding(500, 16)
        self.gender_emb = nn.Embedding(2, 4)
        self.hour_emb = nn.Embedding(24, 4)
        
        """
            这里 nn.Linear() 输入是 16 + 16 + 4 + 4。原因：前向里做了拼接 x = torch.cat().
            所以每个样本最终特征长度是 16 + 16 + 4 + 4 = 40
            即：x.shape = [batch_size, 40]

            32 表示是把 40 维输入映射成 32 维隐藏特征。
            也就是：
            输入 x: [B,40]
                ↓
                W.shape = (32,40)
                ↓
                输出 h: [B,32]
            这里的 32 不是固定的，这是自己设定的 隐藏层宽度（hidden size）超参数。
        """
        self.mlp = nn.Sequential(
            nn.Linear(16 + 16 + 4 + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_id, item_id, gender, hour):
        x = torch.cat([
            self.user_emb(user_id),
            self.item_emb(item_id),
            self.gender_emb(gender),
            self.hour_emb(hour)
        ], dim=1)
        # print("============emb begin=============")
        # print(self.user_emb(user_id))
        # print(self.item_emb(item_id))
        # print(self.gender_emb(gender))
        # print(self.hour_emb(hour))
        # print("============emb end, x begin=============")
        # print(x)
        # print("=============x end============")
        logit = self.mlp(x)
        return logit


# 训练
dataset = ToyCTRDataset()
# print(dataset.data)

"""
    这里解释下 batch_size
    假设 batch_size=4
    原始样本 [
            (123,88,1,21,0),
            (777,17,0,10,1),
            (456,302,1,23,0),
            (9,15,0,8,1)
            ]
    DataLoader 自动整理后：
    user_id = tensor([123,777,456,9])
    item_id = tensor([88,17,302,15])
    gender  = tensor([1,0,1,0])
    hour    = tensor([21,10,23,8])
    label   = tensor([0,1,0,1])
"""
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = CTRModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for step, batch in enumerate(loader):
    """
        这里 label 就是 click 
        也就是监督学习里的 真实答案
    """
    user_id, item_id, gender, hour, label = batch

    label = label.float().unsqueeze(1)

    """
        这里模型内部：
            embedding 查表
            → 拼接特征
            → MLP
            → 输出一个分数
        例如：tensor([
                    [ 1.23],
                    [-0.52],
                    [ 0.88],
                    ...
                    ])
        这叫 logit（未过 sigmoid）。
    """
    logit = model(user_id, item_id, gender, hour)

    loss = criterion(logit, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss {loss.item():.4f}")
