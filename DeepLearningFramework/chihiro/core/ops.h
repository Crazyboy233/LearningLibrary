#pragma once
#include "Tensor.h"

/*
============================================================
    namespace ops

    所有算子以自由函数形式暴露
    每个函数：
        1、执行 forward 计算
        2、若 NoGradGuard 未激活，构造对应的 GradFn，保存所需中间值
        3、返回挂载了 GradFn 的输出 TensorPtr
============================================================
*/
namespace ops {

// -------- 基础算术（N 维 broadcast）--------
TensorPtr add(const TensorPtr& a, const TensorPtr& b);
TensorPtr sub(const TensorPtr& a, const TensorPtr& b);
TensorPtr mul(const TensorPtr& a, const TensorPtr& b);

// -------- 矩阵运算（batched）--------
// A: [..., m, k]  B: [..., k, n]  →  C: [..., m, n]
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);

// -------- 激活 --------
TensorPtr relu(const TensorPtr& a);
TensorPtr sigmoid(const TensorPtr& a);

// -------- Reduce --------
// 全局 sum → 标量 {1}
TensorPtr sum(const TensorPtr& a);
// 沿 dim 求和，keepdim 控制是否保留该维度
TensorPtr sum(const TensorPtr& a, int dim, bool keepdim = false);

// -------- Loss --------
TensorPtr bceWithLogitsLoss(const TensorPtr& logits, const TensorPtr& target);
// logits: [N, C], target: [N] (class indices as double)
TensorPtr crossEntropyLoss(const TensorPtr& logits, const std::vector<size_t>& target);

// -------- 形状变换 --------
// 沿任意 dim 拼接（默认 dim=1 保持向后兼容）
TensorPtr cat(const std::vector<TensorPtr>& inputs, int dim = 1);
// 交换两个维度
TensorPtr transpose(const TensorPtr& a, int dim0, int dim1);
// reshape，总元素数必须相同
TensorPtr reshape(const TensorPtr& a, const std::vector<size_t>& new_shape);

// -------- 归一化 / 注意力 --------
// softmax 沿 dim
TensorPtr softmax(const TensorPtr& a, int dim);
// layer norm：对最后一维归一化
// x: [..., D]  w: [D]  b: [D]
TensorPtr layerNorm(const TensorPtr& x, const TensorPtr& w, const TensorPtr& b,
                    double eps = 1e-5);

}   // namespace ops