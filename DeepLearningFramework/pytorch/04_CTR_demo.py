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
        self.user_emb = nn.Embedding(1000, 16)        
        self.item_emb = nn.Embedding(500, 16)
        self.gender_emb = nn.Embedding(2, 4)
        self.hour_emb = nn.Embedding(24, 4)
        
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
        # print("=========================")
        # print(self.user_emb(user_id))
        # print(self.item_emb(item_id))
        # print(self.gender_emb(gender))
        # print(self.hour_emb(hour))
        # print("=========================")
        # print(x)
        # print("=========================")
        logit = self.mlp(x)
        return logit


# 训练
dataset = ToyCTRDataset()
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = CTRModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for step, batch in enumerate(loader):
    user_id, item_id, gender, hour, label = batch

    label = label.float().unsqueeze(1)

    logit = model(user_id, item_id, gender, hour)
    loss = criterion(logit, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss {loss.item():.4f}")
