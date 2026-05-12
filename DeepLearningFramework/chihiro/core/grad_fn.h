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
    std::vector<size_t> shapeA_, shapeB_;   // 处理 broadcast

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "AddBackward"; }
};

class SubBackward : public GradFn {
public:
    std::vector<size_t> shapeA_, shapeB_;

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "SubBackward"; }
};

class MulBackward : public GradFn {
public:
    std::vector<double> x_val_, y_val_;
    bool same_tensor_ = false;

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "MulBackward"; }
};

class MatMulBackward : public GradFn {
public:
    std::vector<double> A_val_, B_val_;
    size_t m_, k_, n_;

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

class SumBackward : public GradFn {
public:
    size_t input_size_;
    
    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "SumBackward"; }
};

class BCEWithLogitsBackward : public GradFn {
public:
    std::vector<double> sigmoid_val_;   // 内部算好的 sigmoid(logits)，反向直接用
    std::vector<double> target_val_;
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
    size_t rows_;    // batch size，即 m
    std::vector<size_t> col_widths_;     // 每个输入的列数，顺序与 saved_inputs_ 一致

    std::vector<std::vector<double>> apply(const std::vector<double>& grad) override;
    std::string name() const override { return "CatBackward"; }
};