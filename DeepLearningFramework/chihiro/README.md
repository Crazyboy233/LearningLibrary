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
