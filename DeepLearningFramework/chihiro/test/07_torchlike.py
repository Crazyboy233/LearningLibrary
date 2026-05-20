import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "build"))

import core
import core.ops as ops


# ─────────────────────────────────────────────
#  Dataset
#  对应 PyTorch 的 Dataset + DataLoader
#  这里简化：直接把整个 batch 一次性准备好
# ─────────────────────────────────────────────
class ToyCTRDataset:
    """
    玩具 CTR 数据集
    规律：user_id % 10 == item_id % 10 时 click=1，否则 click=0
    """
    def __init__(self, n_samples=200, n_users=10, n_items=20, seed=42):
        import random
        random.seed(seed)
        self.user_ids = []
        self.item_ids = []
        self.labels   = []

        for _ in range(n_samples):
            u = random.randint(0, n_users - 1)
            i = random.randint(0, n_items - 1)
            y = 1.0 if (u % 10 == i % 10) else 0.0
            self.user_ids.append(u)
            self.item_ids.append(i)
            self.labels.append(y)

    def get_batch(self, indices):
        """返回指定下标的 user_ids, item_ids, labels（core.Tensor）"""
        u = [self.user_ids[i] for i in indices]
        v = [self.item_ids[i] for i in indices]
        y = [self.labels[i]   for i in indices]
        label_tensor = core.Tensor([len(y), 1], y)
        return u, v, label_tensor

    def __len__(self):
        return len(self.labels)


# ─────────────────────────────────────────────
#  Model
#  对应 PyTorch 的 nn.Module
#  Wide & Deep 的简化版：只保留 Deep 侧
# ─────────────────────────────────────────────
class DeepCTRModel:
    """
    Deep 侧：Embedding → cat → Linear → ReLU → Linear → logit

    user_id ──▶ Embedding(N_USERS, emb_dim) ──┐
                                               cat ──▶ fc ──▶ ReLU ──▶ out ──▶ logit
    item_id ──▶ Embedding(N_ITEMS, emb_dim) ──┘
    """
    def __init__(self, n_users, n_items, emb_dim, hidden_dim, seed=42):
        self.user_emb  = core.Embedding(n_users, emb_dim,       seed=seed)
        self.item_emb  = core.Embedding(n_items, emb_dim,       seed=seed + 1)
        self.fc        = core.Linear(emb_dim * 2, hidden_dim,   seed + 2)
        self.out_layer = core.Linear(hidden_dim,  1,            seed + 3)

    def forward(self, user_ids, item_ids):
        u_emb  = self.user_emb(user_ids)           # [B, emb_dim]
        i_emb  = self.item_emb(item_ids)           # [B, emb_dim]
        concat = ops.cat([u_emb, i_emb])           # [B, emb_dim*2]
        h      = ops.relu(self.fc(concat))         # [B, hidden_dim]
        logits = self.out_layer(h)                 # [B, 1]
        return logits

    def parameters(self):
        return (
            self.user_emb.parameters() +
            self.item_emb.parameters() +
            self.fc.parameters() +
            self.out_layer.parameters()
        )

    def __call__(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)


# ─────────────────────────────────────────────
#  训练
# ─────────────────────────────────────────────
def train(dataset, model, sgd, epochs, batch_size):
    import random
    indices = list(range(len(dataset)))

    for epoch in range(1, epochs + 1):
        random.shuffle(indices)
        epoch_loss = 0.0
        n_batches  = 0

        # mini-batch 循环（对应 PyTorch 的 DataLoader 迭代）
        for start in range(0, len(dataset), batch_size):
            batch_idx = indices[start : start + batch_size]
            user_ids, item_ids, label = dataset.get_batch(batch_idx)

            sgd.zero_grad()

            logits = model(user_ids, item_ids)
            loss   = ops.bce_with_logits_loss(logits, label)

            loss.backward()
            sgd.step()

            epoch_loss += loss.data[0]
            n_batches  += 1

            if epoch % 10 == 0:
                prob = ops.sigmoid(logits)
                print("================================")
                print(f"epoch = {epoch}, step = {start // batch_size + 1}")
                print(f"loss:            {loss.data}")
                print(f"sigmoid(logits): {[round(v, 4) for v in prob.data]}")
                print(f"label:          {label.data}")
                print("================================")

    print("\n训练完成")


if __name__ == "__main__":
    N_USERS = 10
    N_ITEMS = 20
    EMB_DIM = 4
    HIDDEN = 8
    EPOCHS    = 100
    BATCH_SIZE = 16

    dataset = ToyCTRDataset(n_samples=208, n_users=N_USERS, n_items=N_ITEMS)
    model   = DeepCTRModel(N_USERS, N_ITEMS, EMB_DIM, HIDDEN)
    sgd     = core.SGD(model.parameters(), lr=0.05, momentum=0.9)

    train(dataset, model, sgd, EPOCHS, BATCH_SIZE)