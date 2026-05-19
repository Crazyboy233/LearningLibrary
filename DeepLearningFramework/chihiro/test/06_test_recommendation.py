import sys
from pathlib import Path
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "build")
)

import core
import core.ops as ops

def test_recommendation():
    # 超参
    EMB_DIM  = 4
    HIDDEN   = 8
    N_USERS  = 10
    N_ITEMS  = 20
    LR       = 0.05
    EPOCHS   = 100

    user_emb = core.Embedding(N_USERS, EMB_DIM, seed=1)
    item_emb = core.Embedding(N_ITEMS, EMB_DIM, seed=2)
    fc = core.Linear(EMB_DIM * 2, HIDDEN, 3)
    out_layer = core.Linear(HIDDEN, 1, 4)

    # 收集所有参数
    params = (
        user_emb.parameters() +
        item_emb.parameters() +
        fc.parameters() +
        out_layer.parameters()
    )

    sgd = core.SGD(params, LR, momentum=0.9)

    # 固定 batch：4 条样本，label=1 表示用户-商品有正向交互
    user_ids = [0, 2, 5, 7]
    item_ids = [3, 1, 8, 5]
    # target: [4, 1]
    target = core.Tensor([4, 1], [1.0, 1.0, 0.0, 1.0])

    # 训练
    for epoch in range(EPOCHS):
        sgd.zero_grad()

        # forward
        u_emb   = user_emb(user_ids)
        i_emb   = item_emb(item_ids)
        concat  = ops.cat([u_emb, i_emb])
        h       = ops.relu(fc(concat))
        logits  = out_layer(h)
        loss    = ops.bce_with_logits_loss(logits, target)
 
        loss.backward()

        # 
        if epoch % 10 == 0:
            prob = ops.sigmoid(logits)
            print("================================")
            print(f"epoch = {epoch}")
            print(f"loss:            {loss}")
            print(f"sigmoid(logits): {[round(v, 4) for v in prob.data]}")
            print(f"target:          {target.data}")
            print("================================")
 
        sgd.step()


if __name__ == "__main__":
    test_recommendation()