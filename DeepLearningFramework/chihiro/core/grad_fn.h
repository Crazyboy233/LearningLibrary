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