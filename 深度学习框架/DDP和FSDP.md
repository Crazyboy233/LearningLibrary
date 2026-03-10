DDP 和 FSDP 都是数据并行。一个是算的快，一个是活的下去。

---
**DDP**
DDP(Distributed Data Parallel): 每张卡一份完整模型，一份参数，一份梯度。

流程：
- 每个 GPU 拿到不同的 mini-batch。
- 各自做前向、反向。
- 反向时对梯度做 **AllReduce**（全卡求和/平均）。
- 参数在所有卡上保持一致。

优点：
- 通信模型简单，几乎没有魔法。
- 计算和通信可以部分重叠。
- 对算子、autograd、优化器都非常友好。
- 工程稳定性极高，调试体验相对人类友好。

只要模型能完整塞进一张卡，DDP 是首选。缺点也很致命：
| 显存 = 模型参数 * N 卡(每张卡一份)
模型大了直接G。

---
**FSDP**
FSDP(Fully Sharded Data Parallel): 参数，梯度，优化器状态，统统分片(shard)到各张卡上。
也就是说：
- 单卡 不再拥有完整模型
- 每张卡只存自己那一小片参数
- 真正用到某层时，临时把参数 **AllGather** 过来
- 用完立刻丢掉（或重新 shard）

缺点：
- 通信 pattern 复杂，容易被 NCCL 拖后腿。
- 对模型结构有隐含要求(参数必须可重构)。
- Debug 难度高。

---
**ZeRO系列** 
ZeRO（Zero Redundancy Optimizer）是 Deepspeed 的核心创新，通过分片存储模型状态，消除显存冗余，分 3 个阶段，显存优化逐级增强：
1. ZeRO-1：优化器状态分片
    - 分片对象：优化器状态（如 Adam 的 m、v）
    - 显存节省：约 40%
    - 原理：每个 GPU 只存部分优化器状态，更新时 AllGather 收集。

2. ZeRO-2：优化器状态 + 梯度分片
    - 分片对象：优化器状态 + 梯度
    - 显存节省：约 8 倍（对比 DDP）
    - 原理：梯度也分片，反向传播后 ReduceScatter 聚合，单卡显存压力大幅降低DeepSpeed
3. ZeRO-3：参数 + 梯度 + 优化器状态全分片
    - 分片对象：参数、梯度、优化器状态全部分片
    - 显存节省：N 倍（N 为 GPU 数），支持万亿参数模型
    - 原理：前向 / 反向时动态 AllGather 收集所需参数分片，用完即释放，单卡只存自己的分片DeepSpeed
4.  ZeRO-Offload / ZeRO-Infinity：内存卸载（突破 GPU 物理限制）
    - ZeRO-Offload：把优化器状态 / 参数卸载到 CPU 内存，GPU 专注前向 / 反向计算。
    - ZeRO-Infinity：进一步卸载到 NVMe 高速存储，支持单卡训万亿参数模型DeepSpeed。