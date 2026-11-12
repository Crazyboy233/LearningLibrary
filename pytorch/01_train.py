import torch
# nn 神经网络模块（包含常用层，如 Linear、ReLU）
# optim 优化器（SGD、Adam 等）
from torch import nn, optim
# datasets 内置数据集，这里用 MNIST
# transforms 数据预处理，例如把图片转成 tensor
from torchvision import datasets, transforms
# DataLoader 批量加载数据，训练循环必备
from torch.utils.data import DataLoader


# 数据预处理，把图片转为 tensor 并归一化到 [0,1]
transform = transforms.ToTensor()

# 下载训练集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# 下载测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# 用 dataLoader 批量加载数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 这里打印一个 batch，看看数据长什么样
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)
# 输出类似 torch.Size([64, 1, 28, 28]) torch.Size([64])
# [64,1,28,28] → batch 64 张，单通道（灰度），28×28 像素
# [64] → 每张图片对应的 label

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# 创建模型、损失函数和优化器
model = MLP()
criterion = nn.CrossEntropyLoss() # 多分类交叉熵（内置 Softmax）
optimizer = optim.SGD(model.parameters(), lr=0.01)
# model.parameters() → 告诉优化器哪些参数可以更新
# CrossEntropyLoss → 结合 softmax + log loss，常用分类损失

# 训练循环
# epochs = 3
# for epoch in range(epochs):
for batch_idx, (images, labels) in enumerate(train_loader):
    # forward
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward
    optimizer.zero_grad()   # 清空梯度
    loss.backward()         # 自动求梯度，反向传播
    optimizer.step()        # 更新参数

    if batch_idx % 100 == 0:
        # print(f"Epoch[{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

# 测试模型准确率
correct = 0
total = 0
with torch.no_grad():   # 测试不需要梯度 
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")