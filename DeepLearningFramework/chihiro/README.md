# Mini Deep Learning Framework (C++)

这是一个从 0 实现的简化版深度学习框架，目标是：

- 理解自动微分（autograd）的核心机制
- 理解计算图（computation graph）的执行流程
- 搭建一个最小可训练系统（forward + backward + optimizer）

当前支持：

- ✅ 标量计算（scalar）
- ✅ 静态计算图（DAG）
- ✅ 自动微分（backward）
- ✅ 基于计算图的 forward / backward 执行
- ✅ SGD 参数更新
- ✅ 多节点链式计算

---
# 🚀 Quick Start
```c++
// 目前参考 test/04_test.cpp
```
# 工作流程
## 1. Computation Graph（计算图）
- 整个系统基于 **DAG（有向无环图）**
- 每个 `Node` 表示一次计算
- `Tensor` 在节点之间流动
```
x ----\
    * ----> y
w ----/
```

## 2. Forward Pass
执行流程：
1. 对 Graph 进行拓扑排序
2. 按顺序执行每个 Node：
    ```c++
    op->forward(inputs, output);
    ```
3. 得到最终输出（如 loss）
      
## 3. Backward Pass（自动微分）
1. 初始化：
    ```c++
    loss.grad = 1
    ```
2. 按拓扑逆序执行：
    ```c++
    op->backward(inputs, output);
    ```
3. 梯度沿计算图反向传播

## 4. Parameter Update
使用 SGD：
```c++
w = w - lr * grad
```
---
# 🧩 Core Components

## Tensor
数据的基本单位：
- `value`：前向值
- `grad`：梯度
- `producer`：生成该 Tensor 的 Node

👉 Tensor = 数据 + 梯度 + 图信息

## Parameter
继承自 Tensor：
- 表示**可训练参数**
- 会被 Optimizer 更新

## Op（Operator）
定义计算规则：
```c++
virtual void forward(...)
virtual void backward(...)
```
当前实现：
- AddOp
- SubOp
- MulOp
- SumOp

## Node
一次具体计算：
- 持有 Op
- 持有输入 Tensor
- 产生输出 Tensor

👉 Node = Op 的一次执行实例

## Graph
计算图容器：
- 存储所有 Node
- 负责拓扑排序
- 管理依赖关系

## Executor
执行器：
- forward：执行前向计算
- backward：执行反向传播

## Optimizer
参数更新模块：
- 当前实现：SGD
- 管理 Parameter 列表

---
# 🏗️ Design Choices

## 1. 静态计算图（Static Graph）
Graph 在执行前构建完成

**优点：**
- 执行逻辑简单
- 易于理解

**缺点：**
- 不如动态图灵活（相比 PyTorch）

## 2. 显式 Backward（非自动记录）
每个 Op 手动实现 backward：
**优点：**
- 清晰理解梯度传播
- 更贴近框架底层实现

**缺点：**
- 开发成本较高

## 3. Tensor / Parameter 分离
- Tensor：中间变量
- Parameter：需要更新的变量

👉 明确优化目标

---
# 当前限制
当前版本是最小实现，存在一些限制：
- ❌ 仅支持 scalar（无 shape）
- ❌ 不支持 batch
- ❌ 不支持 broadcast
- ❌ 不支持多输出 Node
- ❌ 无 requires_grad 控制
- ❌ 无内存优化
- ❌ 无动态图机制

---
# Roadmap
未来计划：
- 支持 Tensor shape（向量 / 矩阵）
- 支持 broadcast
- 引入 requires_grad
- 支持多输入 / 多输出 Node
- 实现 Adam / Momentum
- 动态计算图（类似 PyTorch）
- C++ / Python 前端接口