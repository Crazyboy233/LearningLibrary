"""
    该文件对比 01_train.py 包含如下内容:
    
    1.打印并可视化每层激活的形状/分布（对单张样本和整批数据）

    2.可配置隐藏层大小（256 / 512 / 1024）

    3.可配置是否增加一层隐藏层

    4.可切换激活函数（ReLU / Sigmoid）

    5.包含一个单独的 CNN 实现，用来对比 MLP

    6.训练/验证曲线绘图（loss & accuracy）
"""
import torch
# nn 神经网络模块（包含常用层，如 Linear、ReLU）
# optim 优化器（SGD、Adam 等）
from torch import nn, optim
# datasets 内置数据集，这里用 MNIST
# transforms 数据预处理，例如把图片转成 tensor
from torchvision import datasets, transforms
# DataLoader 批量加载数据，训练循环必备
from torch.utils.data import DataLoader

# profiler
from torch.profiler import profile, record_function, ProfilerActivity

import matplotlib.pyplot as plt
import numpy as np
import os

# ========== 配置 ==========
device = torch.device("cpu")
batch_size = 64
epochs = 3
data_root = "./data"

# 实验配置
experiments = [
    {"name": "mlp_base", "hidden_sizes": [256], "activation": "relu"},
    {"name": "mlp_512", "hidden_sizes": [512], "activation": "relu"},
    {"name": "mlp_1024", "hidden_sizes": [1024], "activation": "relu"},
    {"name": "mlp_two_hidden", "hidden_sizes": [512, 256], "activation": "relu"},
    {"name": "mlp_sigmoid", "hidden_sizes": [256], "activation": "sigmoid"},
    {"name": "cnn_simple", "cnn": True}
]

os.makedirs("results", exist_ok=True)

# 数据
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========= 模型定义 =========
class MLP(nn.Module):
    def __init__(self, hidden_sizes=[256], activation="relu"):
        super().__init__()
        layer = []
        layer.append(nn.Flatten())
        input_dim = 28*28
        for h in hidden_sizes:
            layer.append(nn.Linear(input_dim, h))
            if activation == "relu":
                layer.append(nn.ReLU())
            elif activation == "sigmoid":
                layer.append(nn.Sigmoid())
            else:
                raise ValueError("activation must be 'relu' or 'sigmoid'")
            input_dim = h
        layer.append(nn.Linear(input_dim, 10))
        self.net = nn.Sequential(*layer)    # 这里返回的是一个 OrderedDict （有序字典）
        # nn.Sequential() 返回的是一个 nn.Sequential 实例（类对象）

    def forward(self, x, return_activations=False):
        activation = []
        out = x
        for layer in self.net:  # layer 是一个对象
            out = layer(out)    # 这里等同于 layer.forward(out)
            # 记录 Activation (记录线性/激活输出)
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.Sigmoid)):
                activation.append(out.detach().cpu().clone())
        if return_activations:
            return out, activation
        return out

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 提取特征
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), # -> 16*28*28
            nn.ReLU(),
            nn.MaxPool2d(2),                            # -> 16*14*14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),# -> 32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2),                            # -> 32*7*7
            nn.Flatten()
        )
        # 分类头，把提取出来的特征“读懂并分类”
        self.fc = nn.Sequential(
            nn.Linear(32*7*7, 128), # 这里的 128 是一个超参数，不是写死的。64 → 参数更少，模型变小，可能稍微掉一点准确率；256 → 表达力更强，但更容易过拟合；1024 → 太大了，MNIST 这种任务没必要，浪费算力。
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x, return_activations=False):
        acts = []
        out = x
        for layer in self.conv:
            out = layer(out)
            if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
                acts.append(out.detach().cpu().clone())
        
        """
        这里没有执行循环，而是直接调用了 out = self.fc(out)。并不影响结果。
        nn.Sequential 本身就是把内部层按顺序执行一次 forward。
        out = self.fc(out) 等价于：
        out = self.fc[0](out)
        out = self.fc[1](out)
        out = self.fc[2](out)
        """
        out = self.fc(out)
        if return_activations:
            return out, acts
        return out

# ========= 训练/验证/工具函数 =========
def evaluate(model, loader, criterion):
    # 切换到评估模式
    model.eval()

    # 初始化统计变量
    total = 0 # 总的测试样本数
    correct = 0 # 预测正确的个数
    losses = 0.0 # 累计损失（注意，是加权的，按 batch_size）

    # 评估阶段不需要梯度
    # 这里告诉 pytorch：不要记录梯度，不要构建计算图，不要浪费内存。
    # 评估就是 forward，不需要 backward。
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            # 前向计算
            outputs = model(images)
            # 计算 loss 
            loss = criterion(outputs, labels)
            losses += loss.item() * images.size(0)
            # 计算预测结果
            _, pred = torch.max(outputs, 1)
            # 累计正确与样本数
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    # 返回平均 loss 和准确率
    return losses / total, correct / total

def train_one_exp(cfg):
    print(f"\n=== Running experiment: {cfg['name']} ===")
    if cfg.get("cnn", False):
        model = SimpleCNN().to(device)
    else:
        model = MLP(hidden_sizes=cfg["hidden_sizes"], activation=cfg["activation"]).to(device)

    criterion = nn.CrossEntropyLoss() # 交叉熵损失，用于分类任务
    optimizer = optim.SGD(model.parameters(), lr=0.01) # model.parameters() 会返回一系列 tensor，这些 tensor 都是需要梯度计算的参数。

    # 用于保存每个 epoch 的训练/验证损失和准确率，方便后续绘图或分析。
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:

            # 遍历训练数据
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 梯度清零。PyTorch 默认梯度会累加，需要在每次反向传播前清零。防止梯度叠加影响更新。
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 参数更新
                optimizer.step() # 使用优化器（SGD）更新模型参数：param = param - lr * grad

                # 累加 batch 指标
                """
                loss.item()：把 tensor 换成 Python float
                * images.size(0)：乘以 batch 大小，因为 loss 是平均值
                torch.max(outputs,1)：取每行最大值索引，得到预测类别
                running_correct += (pred == labels).sum().item()：统计预测正确的数量
                running_total：累加样本总数
                """
                running_loss += loss.item() * images.size(0)
                _, pred = torch.max(outputs, 1)
                running_correct += (pred == labels).sum().item()
                running_total += labels.size(0)

                if batch_idx >= 10:
                    break
        print(prof.key_averages().table(sort_by="cpu_time_total"))

        # 计算本轮 epoch 的训练和验证指标
        train_loss = running_loss / running_total # 训练损失
        train_acc = running_correct / running_total # 训练准确率
        val_loss, val_acc = evaluate(model, test_loader, criterion) # 计算验证集上的损失和准确率

        # 将本轮的训练/验证结果存到列表里，用于绘图和保存
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"[{cfg['name']}] Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")
    
    # 画训练曲线
    # plt.figure(figsize=(8,4))
    # plt.subplot(1,2,1)
    # plt.plot(train_losses, label='train_loss')
    # plt.plot(val_losses, label='val_loss')
    # plt.title(cfg['name'] + " loss")
    # plt.legend()
    # plt.subplot(1,2,2)
    # plt.plot(train_accs, label='train_acc')
    # plt.plot(val_accs, label='val_acc')
    # plt.title(cfg['name'] + " acc")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"results/{cfg['name']}_curve.png")
    # plt.close()

    # # 取一张图片，查看中间激活（示例）
    # sample_img, sample_label = test_dataset[0]
    # sample_img_batch = sample_img.unsqueeze(0).to(device)
    # if cfg.get("cnn", False):
    #     _, acts = model(sample_img_batch, return_activations=True)
    #     # acts 是 list of tensors (conv outputs). 展示第一个conv通道的前几个feature maps
    #     for i, a in enumerate(acts[:2]):
    #         # a shape (C,H,W) or (C,...)  -> convert to numpy and plot first 4 channels
    #         at = a.squeeze(0).cpu().numpy()
    #         num_ch = min(4, at.shape[0])
    #         fig, axs = plt.subplots(1, num_ch, figsize=(num_ch*2,2))
    #         for c in range(num_ch):
    #             axs[c].imshow(at[c], cmap='gray')
    #             axs[c].axis('off')
    #         plt.suptitle(f"{cfg['name']} - conv act {i}")
    #         plt.savefig(f"results/{cfg['name']}_act_conv{i}.png")
    #         plt.close()
    # else:
    #     _, acts = model(sample_img_batch, return_activations=True)
    #     # acts 是 list of tensors (linear and activation outputs); 展示前几个 activations 的形状和分布
    #     for i, a in enumerate(acts[:4]):
    #         a_np = a.cpu().numpy().flatten()
    #         plt.figure(figsize=(4,2))
    #         plt.hist(a_np, bins=50)
    #         plt.title(f"{cfg['name']} - act {i} shape {tuple(acts[i].shape)}")
    #         plt.savefig(f"results/{cfg['name']}_act_hist{i}.png")
    #         plt.close()

    print(f"[{cfg['name']}] Done. Saved model+metrics+activations in results/")

# ========== 运行所有实验 ==========
if __name__ == "__main__":
    for cfg in experiments:
        train_one_exp(cfg)

    print("\nAll experiments finished. Check the results/ folder for models, curves, activations.")
