我正在从 0 实现一个简化版深度学习框架（偏 C++ 后端），目前已经完成了一个最小可训练的自动微分系统，结构如下：

【整体设计分层】
当前系统分为几个核心模块：

1. Tensor（数据层）

- 存储：

  - value（double）
  - grad（double）
  - producer（Node*） 记录“这个 Tensor 是由哪个 Node 计算出来的
- 支持：

  - value() / setValue()
  - grad() / setGrad()
- 当前是标量（scalar），尚未支持向量或多维

2. Parameter（继承自 Tensor）

- 表示“可训练参数”
- 和普通 Tensor 的区别在于：

  - 会被 Optimizer 更新
- 当前没有 requires_grad 机制（默认都参与反向传播）

3. Op（算子层）

- 抽象类，定义：

  - forward(inputs, output)
  - backward(inputs, output)
- 已实现：

  - AddOp
  - SubOp
  - MulOp

4. Node（计算节点）

- 成员：

  - Op- op_
  - std::vector<Tensor-> inputs_
  - Tensor output_
- 方法：

  - forward() → 调用 op_->forward
  - backward() → 调用 op_->backward
- 每个 Node 表示一个计算操作实例
- output Tensor 存在 Node 内部

5. Graph（计算图）

- 存储：

  - std::vector<std::unique_ptr<Node>> nodes_
- 提供：

  - addNode(std::unique_ptr<Node>)
- 当前：

  - 已实现拓扑排序（保证 DAG 顺序执行）
  - Node 之间通过 Tensor- 建立依赖关系

6. Executor（执行器）

- forward(Graph)：

  - 按拓扑顺序执行 Node::forward()
- backward(Graph, loss)：

  - 从 loss 开始反向传播
  - loss.grad = 1
  - 按拓扑逆序执行 Node 的 backward
- zeroGrad(Graph)：

  - 清空所有 Node.output 的 grad

7. Optimizer（优化器）

- 已实现 SGD：

  - 持有 std::vector<Parameter->
  - step()：w = w - lr - grad
  - zeroGrad()：清空 Parameter.grad

【当前能力】
系统已经支持：

- 构建简单计算图（DAG）
- forward 计算
- backward 自动微分
- 参数更新（SGD）
- 多节点链式计算
- 多 Op 组合

【已验证功能】
我做了如下测试：

1. 构建计算图：
   y = w - x
   loss = (y - target)^2

2. 使用 SGD 训练：

   - w 初始为 0
   - x = 2, target = 10
   - 学习率 lr = 0.1

3. 结果：

   - w 收敛到 5
   - loss 收敛到 0
   - grad 指数级衰减到 0

4. 过程中发现并修复问题：

   - 梯度未清零导致爆炸（已修复）
   - unique_ptr 拷贝问题（已修复）
   - Node 生命周期 & 指针安全问题（已理解）

【当前设计特点】

- Graph 是静态 DAG（非动态图）
- Tensor 仅支持 scalar（无 shape）
- Parameter 与 Tensor 分离（语义区分）
- Executor 和 Optimizer 解耦
- backward 是显式实现（非 tape）

【当前局限】

- Tensor 不支持 vector / shape / broadcast
- 不支持 batch
- 不支持多输出 Node
- 无 requires_grad 控制
- Op 接口较简化（未区分 forward/backward cache）
- 无内存优化（Tensor 生命周期简单）
- 无 Python 前端

【接下来想做的方向】
希望在这个基础上继续扩展，比如：

- 多参数模型（如 y = wx + b）
- Tensor 升级（支持向量/shape）
- 优化器升级（Momentum / Adam）
- 更通用的计算图（多输入多输出）
- 更贴近真实框架（如 PyTorch）的设计
