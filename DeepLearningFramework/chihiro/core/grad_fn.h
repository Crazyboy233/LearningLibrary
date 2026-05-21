#pragma once
#include <vector>
#include <string>
#include <memory>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

/*
============================================================
    GradFn - 计算图节点的反向传播函数
    每次 forward op 都会构造一个 GradFn 子类实例，挂在
    输出 Tensor 的 grad_fn_ 上。

    backward() 遍历时：
        1、拿到输出 Tensor 的 grad
        2、调用 apply(grad) 得到各输入的梯度
        3、累加到 save_inputs_ 对应 Tensor 的 grad_ 上
============================================================
*/
class GradFn {
public:
    virtual ~GradFn() = default;

    // 返回每个 saved_input_ 对应的梯度，顺序与 saved_input_ 一致
    virtual std::vector<std::vector<double>> apply(const std::vector<double>& grad) = 0;

    virtual std::string name() const = 0;

    // 该 GradFn 依赖上游的 Tensor（即 forward 时的输入）
    std::vector<std::shared_ptr<Tensor>> saved_inputs_;
};

/*
============================================================
    各 op 对应的 GradFn 子类
============================================================
*/

class AddBackward : public GradFn {
public:
    std::vector<size_t> shapeA_, shapeB_, shapeOut_;

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "AddBackward"; }
};

class SubBackward : public GradFn {
public:
    std::vector<size_t> shapeA_, shapeB_, shapeOut_;

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "SubBackward"; }
};

class MulBackward : public GradFn {
public:
    std::vector<double> x_val_, y_val_;
    std::vector<size_t> shapeA_, shapeB_, shapeOut_;
    bool same_tensor_ = false;

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "MulBackward"; }
};

/*
    A: [..., m, k]   B: [..., k, n]   C: [..., m, n]
    batch 维度做 broadcast
*/
class MatMulBackward : public GradFn {
public:
    std::vector<double> A_val_, B_val_;
    std::vector<size_t> shapeA_, shapeB_;   // 原始shape，含batch
    size_t m_, k_, n_;                      // 最后两维

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "MatMulBackward"; }
};

class ReLUBackward : public GradFn {
public:
    std::vector<double> x_val_;

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "ReLUBackward"; }
};

class SigmoidBackward : public GradFn {
public:
    std::vector<double> y_val_; // sigmoid 反向只需要输出值

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "SigmoidBackward"; }
};

// -------- Reduce --------
/*
    SumBackward 支持两种模式：
      全局 sum：sum_dim_ = -1（旧接口）
      沿 dim   ：sum_dim_ >= 0，keepdim_ 记录是否保留维度
*/
class SumBackward : public GradFn {
public:
    std::vector<size_t> input_shape_;
    int sum_dim_ = -1;  // -1 表示全局 sum
    bool keepdim_ = false;
    // size_t input_size_;
    
    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "SumBackward"; }
};

// -------- Loss --------
class BCEWithLogitsBackward : public GradFn {
public:
    std::vector<double> sigmoid_val_, target_val_;   // 内部算好的 sigmoid(logits)，反向直接用
    size_t n_;

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "BCEWithLogitsBackward"; }
};

/*
============================================================
    EmbeddingBackward
    forward : out[i] = W[ids[i]]          行索引查表
    backward: dW[ids[i]] += grad[i, :]    梯度写回对应行
 
    saved_inputs_ = {W}   （ids 是整数，不入计算图）
 
    grad  : [batch * embedding_dim]，行主序展平
    return: {dW}，shape [num_embeddings * embedding_dim]
 
    关键细节：同一个 id 在 batch 里出现多次时，
              对应行的梯度需要多次累加（+=），不能覆盖（=）
============================================================
*/
class EmbeddingBackward : public GradFn {
public:
    std::vector<size_t> ids_;       // 前向时查了哪些行，shape [batch]
    size_t num_embeddings_;
    size_t embedding_dim_;

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "EmbeddingBackward"; }
};

/*
============================================================
    CatBackward
    forward : out = cat([a, b, c, ...], dim=1)
              沿列方向拼接，所有输入行数相同
 
              inputs[0]: [m, n0]
              inputs[1]: [m, n1]
              ...
              output   : [m, n0+n1+...]
 
    backward: 把 grad [m, N] 按各输入的列宽切回去
              d_inputs[i] = grad[:, offset_i : offset_i + n_i]
 
    saved_inputs_ = {a, b, c, ...}   （顺序与 forward 一致）
    col_widths_   = {n0, n1, ...}    （每个输入的列数，切片用）
============================================================
*/
class CatBackward : public GradFn {
public:
    std::vector<size_t> out_shape_;
    int cat_dim_;                       // 拼接维度（已规范化，非负）
    std::vector<size_t> split_sizes_;    // 每个输入在 cat_dim_ 上的大小

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "CatBackward"; }
};

// -------- Transpose（交换任意两维）--------
class TransposeBackward : public GradFn {
public:
    std::vector<size_t> in_shape_;
    int dim0_, dim1_;
    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "TransposeBackward"; }
};
 
// -------- Reshape --------
class ReshapeBackward : public GradFn {
public:
    // 梯度只需 reshape 回原 shape，shape 存在 saved_inputs_[0] 上
    // 这里直接记原 shape
    std::vector<size_t> in_shape_;
    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "ReshapeBackward"; }
};
 
// -------- Softmax --------
class SoftmaxBackward : public GradFn {
public:
    std::vector<double> y_val_;         // softmax 输出
    std::vector<size_t> shape_;
    int dim_;
    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "SoftmaxBackward"; }
};
 
// -------- LayerNorm --------
/*
    forward: y = (x - mean) / sqrt(var + eps) * w + b
    saved_inputs_ = {x, w, b}
    额外保存: x_norm（归一化后，乘 w 之前）、var + eps 的倒数根
*/
class LayerNormBackward : public GradFn {
public:
    std::vector<double> x_val_, x_norm_, w_val_;
    std::vector<double> rstd_;          // 1/sqrt(var+eps)，每个归一化组一个值
    std::vector<size_t> shape_;         // input shape
    size_t norm_size_;                  // 归一化的最后一维大小（== embedding_dim）
    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "LayerNormBackward"; }
};
 
// -------- CrossEntropyLoss --------
/*
    input : [N, C]  logits
    target: [N]     class indices (size_t)
    loss  : scalar
*/
class CrossEntropyBackward : public GradFn {
public:
    std::vector<double> softmax_val_;   // [N*C] softmax(logits)
    std::vector<size_t> target_;        // [N]  class indices
    size_t N_, C_;
    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "CrossEntropyBackward"; }
};
 