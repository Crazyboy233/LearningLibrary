# Mini Deep Learning Framework (C++)

这是一个从 0 实现的简化版深度学习框架，目标是：

- 理解自动微分（autograd）的核心机制
- 理解计算图（computation graph）的执行流程
- 搭建一个最小可训练系统（forward + backward + optimizer）

当前支持：

- ✅ 二维矩阵计算
- ✅ 动态计算图（DAG）
- ✅ 自动微分（backward）
- ✅ SGD 参数更新
- ✅ 多节点链式计算
- ✅ boardcast
- ✅ python 前段接口
- ✅ 支持多维
- ✅ 支持 transformer

---
# 🚀 Quick Start
```c++
// 目前版本参考 test/08_test_N_dim.cpp
// 综合测试参考 test/08_test_N_dim.cpp

/* 
    python 前端综合测试 test/07_torchlike.py
    执行测试命令如下：
        cd DeepLearningFramework/chihiro
        mkdir -p build
        cd build
        cmake ..
        make
        cd ..
        python test/07_torchlike.py
    注意：
        保证你的python环境具有 pybind11。 
    依赖安装（Ubuntu/Debian）：
        pip install pybind11
*/
```
# 工作流程
## 1. Computation Graph（计算图）
- 整个系统基于 **DAG（有向无环图）**
- `Tensor` 是
```
x ----\
    * ----> y
w ----/
```

## 2. Forward Pass
执行流程：
1. 调用`ops`下的函数构造计算图
3. 得到最终输出（如 loss）
   
## 3. Backward Pass（自动微分）
1. 初始化：
    ```c++
    loss.grad = 1
    ```
    
2. 逆拓扑排序并完成梯度沿计算图反向传播：
    ```c++
    loss->backward();
    ```

## 4. Parameter Update
使用 SGD：
```c++
w = w - lr * grad
```
---
# 🧩 Core Components

## Tensor
数据的基本单位：

- `shape_`：形状

- `value_`：前向值
- `grad_`：梯度
- `requires_grad_`：是否需要计算梯度
- `grad_fn_`：生成该 Tensor 的 GradFn，`backward()` 做拓扑排序时需要沿着这条链往上爬，就是靠这个。

## Parameter
继承自 Tensor：
- 表示**可训练参数**
- 会被 Optimizer 更新

## Ops（Operator）
定义计算规则：
```c++
namespace ops {
    TensorPtr add(const TensorPtr& a, const TensorPtr& b);
    TensorPtr sub(const TensorPtr& a, const TensorPtr& b);
    ...
}
```
当前实现：
- add
- sub
- Mul
- matmul
- relu
- sigmoid
- sum
- bceWithLogitsLoss
- cat
- crossEntropyLoss
- transpose
- reshape
- softmax
- layerNorm

## Optimizer
参数更新模块：
- 当前实现：SGD
- 管理 Parameter 列表

## Model
目前支持：
- Linear
- Embedding

---

# 🏗️ Design Choices

## 1. 动态计算图（Dynamic Graph）
每次 forward 执行时动态构建计算图——每个 op 将对应的 GradFn 挂载到输出 Tensor 的 grad_fn_ 上，形成从 loss 到叶节点的有向链。backward() 沿此链逆拓扑遍历，完成梯度传播。
**优点**：
执行逻辑简单，forward 与图构建同步完成，无需额外 "compile" 步骤
易于调试，每步输出的 Tensor 可直接检查值与 grad_fn

**缺点**：
图结构运行时才确定，无法做静态图层面的算子融合或编译优化
每次 forward 都重新建图，存在一定内存分配开销

## 2. 显式 Backward（非自动记录）
每个 Op 手动实现对应的 GradFn::apply()，明确写出对各输入的偏导数逻辑，而非通过 tape 或符号微分自动推导。
**优点**：
梯度传播路径清晰可读，便于理解链式法则的底层机制
贴近 PyTorch 等框架的实际底层实现，适合学习参考
方便针对特定 op 做数值稳定性优化（如 SigmoidBackward 复用 forward 输出值 y，避免重算 exp）

**缺点**：
每新增一个 Op 需同步实现 GradFn 子类，开发成本较高
手写梯度易出错，需配合数值梯度验证（参见 numericalGrad()）

## 3. Tensor / Parameter 分离
- Tensor：中间变量
- Parameter：需要更新的变量

👉 明确优化目标

---

---
# 静态图 & 动态图
```
静态图：
Graph 持有 Node → Node 持有 Op + inputs + output
Executor 做拓扑排序 + 驱动 forward/backward
```
```
动态图：
Tensor 自己记录 { op, inputs }  ← 这就是 autograd tape
backward() 从 loss 出发，自动反向遍历
没有显式 Graph，没有 Executor
```