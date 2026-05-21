# 函数说明文档
## Tensor
### 1. reduceTo
```c++
inline std::vector<double> reduceTo(const std::vector<double>& grad,
                                    const std::vector<size_t>& grad_shape,
                                    const std::vector<size_t>& target_shape);
/**
 @brief 将广播后的梯度 reduce 回目标 shape

 在自动求导中，如果前向传播发生了 broadcast：

      [3,1] -> [3,4]

 那么反向传播得到的梯度 shape 会是：

      [3,4]

 但原 tensor 的 shape 是：

      [3,1]

 因此需要把 broadcast 出来的维度重新做 sum reduction，
 将梯度压缩回原始 shape。

 本函数实现的就是：

      reduce_sum_to_shape(grad, target_shape)

 具体规则：

 1. target_shape 会先左侧补 1，与 grad_shape 对齐

      grad_shape   = [2,3,4]
      target_shape = [3,1]

 对齐后：

      ts = [1,3,1]

 2. 对 grad 的每个元素：

      - flat 下标 -> 多维坐标
      - broadcast 维（ts[d] == 1）映射到 0
      - 重新计算目标 tensor 的 flat 下标
      - 累加到 result

 3. 最终得到与 target_shape 对应的 reduce 后梯度

 示例：

      grad_shape   = [2,3]
      target_shape = [2,1]

 forward:

      [2,1] broadcast -> [2,3]

 backward:

      [[a,b,c],
       [d,e,f]]

 reduce 后：

      [[a+b+c],
       [d+e+f]]

 即：

      sum(axis=1)

 @param grad          广播后得到的梯度数据（flatten）
 @param grad_shape    grad 的 shape
 @param target_shape  原 tensor 的目标 shape

 @return reduce 后、shape 为 target_shape 的梯度
/
```
**对于循环中的详细介绍**：
**第一部分：flat -> 多维坐标**
```c++
size_t tmp = flat;
for (int d = (int)ndim - 1; d >= 0; --d) {
    idx[d] = tmp % grad_shape[d];
    tmp /=  grad_shape[d];
}
```
这个过程是扁平坐标转多维坐标
例如：
```c++
grad_shape = [2,3]
```
那么：
| flat | idx   |
| ---- | ----- |
| 0    | [0,0] |
| 1    | [0,1] |
| 2    | [0,2] |
| 3    | [1,0] |
| 4    | [1,1] |
| 5    | [1,2] |

**第二部分：广播维处理**
```c++
size_t res_flat = 0;

for (size_t d = 0; d < ndim; ++d) {
    size_t i = (ts[d] == 1) ? 0 : idx[d];
    res_flat += i out_st[d];
}
```
如果：
```c++
ts[d] == 1
```
说明这个维度在 forward 的时候被 broadcast 了。
例如：
```c++
A: [3,1]
B: [3,4]
```
A 在第二维被广播：
```c++
[3,1] -> [3,4]
```
那么 backward 时：
```c++
dA = sum(dOut, axis=1)
```
所以第二维所有 index：
```c++
0 1 2 3
```
都必须映射回
```
0
```
所以：
```c++
i = 0
```
**第三部分：多维坐标 -> flat**
```c++
res_flat += i out_st[d];
```
把 reduce 后的新坐标：
```c++
[i0,i1,i2...]
```
重新转成一维下标。
**第四部分：累加**
```c++
result[res_flat] += grad[flat];
```
因为 broadcast 维会映射到同一个位置。
所以：
多个 grad 元素会累加到同一个 output 元素。
这就是：**reduce sum**

### 2. shapeStrides
```c++
inline std::vector<size_t> shapeStrides(const std::vector<size_t>& shape);
/**
 @brief 计算 tensor 的行主序（row-major）stride

 stride 表示：

      在某个维度上移动 1 个元素，
      在底层一维数组中需要跨过多少元素。

 在连续内存布局（C/C++ row-major）中：

      最后一维 stride 恒为 1
      前面的 stride 依次累乘后续维度大小

 例如：

      shape = [2,3,4]

 内存布局：

      [[[x,x,x,x],
        [x,x,x,x],
        [x,x,x,x]],

       [[x,x,x,x],
        [x,x,x,x],
        [x,x,x,x]]]

 对应 stride：

      st[2] = 1
      st[1] = 4
      st[0] = 3 4 = 12

 即：

      strides = [12,4,1]

 含义：

      第0维移动1步 -> 跳12个元素
      第1维移动1步 -> 跳4个元素
      第2维移动1步 -> 跳1个元素

 stride 常用于：

      1. 多维坐标 -> flat 下标

              flat = i0*st[0]
                   + i1*st[1]
                   + ...

      2. flat 下标 -> 多维坐标

      3. tensor indexing

      4. broadcast/view/reshape 等张量操作

 本函数生成的是：

      contiguous row-major stride

 即 NumPy / PyTorch 默认连续张量布局。

 @param shape tensor shape

 @return 每个维度对应的 stride
*/
```

### 3. normalizeDim
```c++
// 规范化负维度索引 
inline int normalizeDim(int dim, int ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::out_of_range("dim out of range");
    return dim;
}
/**
 @brief 规范化 tensor 的维度索引（支持负维）

 在 NumPy / PyTorch 等张量库中，
 dim 可以使用负数表示“从后往前数”的维度：

      -1 表示最后一维
      -2 表示倒数第二维
      ...

 本函数用于：

      1. 将负维度转换为正维度
      2. 检查维度是否合法

 例如：

      ndim = 4

 对应维度：

          0  1  2  3
         -4 -3 -2 -1

 示例：

      normalizeDim(-1, 4) -> 3
      normalizeDim(-2, 4) -> 2
      normalizeDim( 1, 4) -> 1

 若维度越界：

      normalizeDim(4, 4)
      normalizeDim(-5, 4)

 则抛出：

      std::out_of_range

 常用于：

      - sum(dim)
      - mean(dim)
      - squeeze(dim)
      - unsqueeze(dim)
      - transpose(dim0, dim1)
      - softmax(dim)

 等 tensor 维度相关操作。

 @param dim   用户输入的维度（允许负数）
 @param ndim  tensor 的总维度数

 @return 转换后的合法正维度索引
*/
```

## ops
### 1. broadcastExpand
```c++
static void broadcastExpand(const std::vector<double>& src,
                            const std::vector<size_t>& src_shape,
                            const std::vector<double>& out_shape,
                            const std::vector<size_t>& dst);
/**
@brief broadcastExpand：将输入 tensor 按 broadcast 规则扩展到 out_shape
将一个较小 shape 的 tensor（src），通过“维度复制”的方式，
扩展成目标 shape（out_shape），用于逐元素计算。

----------------------------
核心思想：
----------------------------
broadcast 规则：
  - 若某一维 size == 1，则该维上的值会被“复制”到所有位置
  - 若某一维 size > 1，则正常按索引访问

本函数不真正分配二维/多维结构，而是基于 flat index 做映射：
  out_index → src_index

----------------------------
处理流程：
----------------------------
1. 计算 out_shape 的总元素个数 total
2. 将 src_shape 左侧补 1，对齐到 out_shape 维度（ss）
   例如：
       src_shape = [3]
       out_shape = [2,3]
       ss = [1,3]

3. 计算 stride（用于 flat <-> multi-dim 转换）
   - out_st：out_shape 的 stride（用于遍历 out）
   - ss_st  ：src 对齐后的 stride（用于定位 src）

4. 遍历 out tensor 的每个 flat index：
   a. 将 flat index 转换为多维坐标 idx
   b. 根据 broadcast 规则计算 src 对应坐标：
        - 若 ss[d] == 1 → 该维广播，索引固定为 0
        - 否则使用 idx[d]
   c. 将 src 多维坐标转回 flat index
   d. 从 src 取值，写入 dst

----------------------------
示例：
----------------------------
src_shape = [2,1]
out_shape = [2,3]

src:
  [[1],
   [2]]

broadcast 后：
  [[1,1,1],
   [2,2,2]]

----------------------------
注意：
----------------------------
- 常用于 forward 计算，配合 reduceTo 做 backward

@param src        输入 tensor（flatten）
@param src_shape  输入 tensor shape
@param out_shape  目标 broadcast shape
@param dst        输出 tensor（flatten，已扩展）
*/
```
> 该函数实现类似 reduceTo，详细可看 reduceTo 介绍

### 2. ops::sum
```c++
TensorPtr ops::sum(const TensorPtr& a, int dim, bool keepdim); 
// 对 Tensor a 的第 dim 维求和
```
这个 ops::sum 是一个 沿指定维度做求和归约（reduction） 的实现，同时支持：
- dim：指定沿哪个维度求和
- keepdim：是否保留被压缩的维度
- autograd（自动求导）

整体逻辑类似于：
```python
torch.sum(a, dim=dim, keepdim=keepdim)
```
例如：
```c++
a.shape = [2, 3];
a.ndim = 2;
// 数据
a.value = [[1,2,3],
          [4,5,6]]
```
**dim=0**：按行方向压缩：

```c++
a.value = [5, 7, 9];
a.shape = [3];
a.ndim = 1;

// keepdim = true 时
a.value = [[5, 7, 9]];
a.shape = [1, 3];
a.ndim = 2;
```
**dim=1**：按列方向压缩：

```c++
a.value = [6,15];
a.shape = [2];
a.ndim = 1;
```

### 3. ops::crossEntropyLoss

```c++
TensorPtr crossEntropyLoss(const TensorPtr& logits, const vector<size_t>& target);
// 交叉熵损失
// 对 logits 做数值稳定的 softmax，取正确类别的概率求负对数均值得到 loss
```

| 参数 | 类型 | 含义 |
|------|:-----|------|
| `logits` | `TensorPtr` | 模型原始输出，shape `[N, C]`，未经 softmax |
| `target` | `vector<size_t>` | 每个样本的正确类别索引，长度为 N |
| 返回值 | `TensorPtr` | 标量 loss，shape `[1]` |

---

第一步：输入校验

```cpp
assert(logits->dim() == 2);
size_t N = logits->shape()[0];
size_t C = logits->shape()[1];
assert(target.size() == N);
```

确保 logits 是二维张量，且 target 数量和样本数一致。

---

第二步：Softmax（数值稳定版）

**朴素版本的问题：**
```cpp
// 危险！x 很大时 exp(x) 直接溢出成 inf
soft[c] = exp(logits[c]) / sum_exp;
```

**稳定版本：每行减去最大值**
```cpp
double mx = max_element(row);          // 找该行最大值
soft[c] = exp(logits[c] - mx);         // 平移后再 exp
soft[c] /= sum_exp;                    // 归一化
```

数学上完全等价：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}} = \frac{e^{x_i - m}}{\sum e^{x_j - m}}
$$
平移后最大的 exp 值是 $e^0 = 1$，完全避免溢出。

结果 `soft` 的每一行都是一个合法的概率分布，所有值在 (0,1) 之间，行和为 1。

---

第三步：NLL Loss 计算

```cpp
for (size_t n = 0; n < N; ++n) {
    loss -= std::log(soft[n * C + target[n]] + 1e-12);
}
loss /= (double)N;
```

**只看正确类别的概率：**

```
样本0  soft = [0.1, 0.7, 0.2]  target=1  → -log(0.7)  = 0.357
样本1  soft = [0.8, 0.1, 0.1]  target=0  → -log(0.8)  = 0.223
样本2  soft = [0.2, 0.2, 0.6]  target=1  → -log(0.2)  = 1.609
                                          平均 loss    = 0.730
```

`+1e-12` 是为了防止概率为 0 时 `log(0) = -inf` 导致崩溃。

公式：
$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \log \hat{p}_{n,\, y_n}$$

**直觉理解：** 正确类别的概率越接近 1，loss 越小；越接近 0，loss 越大（趋向无穷）。


第四步：构建反向传播节点

```cpp
if (!anyRequiresGrad({logits})) {
    return Tensor::create({1}, {loss});   // 推理模式，直接返回
}

auto fn = std::make_shared<CrossEntropyBackward>();
fn->softmax_val_ = soft;    // 反向需要用到 softmax 结果
fn->target_      = target;  // 反向需要知道哪个是正确类别
fn->N_ = N;  fn->C_ = C;
fn->saved_inputs_ = {logits};

return Tensor::createFromOp({1}, {loss}, fn);
```

**为什么要保存 `soft`？**

反向传播时对 logits 的梯度公式是：

$$\frac{\partial \mathcal{L}}{\partial z_{n,c}} = \frac{1}{N} \begin{cases} \hat{p}_{n,c} - 1 & \text{if } c = y_n \\ \hat{p}_{n,c} & \text{otherwise} \end{cases}$$

即：
- 正确类别：softmax 概率 **减 1**
- 其他类别：直接用 softmax 概率

这个梯度只依赖 `soft` 和 `target`，不需要重新计算，所以提前保存。

整体流程

```
logits [N×C]
   │
   │  对每行做 Softmax（减最大值，数值稳定）
   ▼
soft [N×C]  ←─────────────────────────────┐
   │                                       │ 保存，供反向传播使用
   │  取 soft[n, target[n]]，取 -log，求均值 │
   ▼                                       │
loss（标量）                                │
   │                                       │
   └──── 挂载 CrossEntropyBackward 节点 ───┘
              （记录 soft, target, N, C）
```
