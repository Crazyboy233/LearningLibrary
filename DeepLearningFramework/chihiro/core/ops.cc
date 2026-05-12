#include "ops.h"
#include "grad_fn.h"
#include <cassert>
#include <cmath>

/*
============================================================
    辅助：判断至少一个输入需要梯度
    只要有一个输入 requires_grad，输出就需要记录计算图
============================================================
*/
static bool anyRequiresGrad(const std::vector<TensorPtr>& inputs) {
    for (auto& t : inputs) {
        if((t)->requireGrad()) {
            return true;
        }
    }
    return false;
}

/*
============================================================
    add : [m,n] + [m,n] 或 [m,n] + [1,n]
============================================================
*/
TensorPtr ops::add(const TensorPtr& a, const TensorPtr& b) {
    const auto& shapeA = a->shape();
    const auto& shapeB = b->shape();

    assert(shapeA.size() == 2 && shapeB.size() == 2);
    assert(shapeA[1] == shapeB[1]); // 列数保持一致
    assert(shapeB[0] == 1 || shapeA[0] == shapeB[0]);   // B 行数为1，或与 A 行数相同

    size_t m = shapeA[0];
    size_t n = shapeA[1];
    std::vector<double> result(m * n);
    for(size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t b_i = (shapeB[0] == 1) ? 0 : i;
            result[i * n + j] = a->value()[i * n + j] + b->value()[b_i * n + j];
        }
    }

    if(!anyRequiresGrad({a, b})) {
        return Tensor::create(shapeA, result);
    }

    auto fn = std::make_shared<AddBackward>();
    fn->shapeA_ = shapeA;
    fn->shapeB_ = shapeB;
    fn->saved_inputs_ = {a, b};

    return Tensor::createFromOp(shapeA, result, fn);
}

/*
============================================================
    sub : [m,n] - [m,n] 或 [m,n] - [1,n]
============================================================
*/
TensorPtr ops::sub(const TensorPtr& a, const TensorPtr& b) {
    const auto& shapeA = a->shape();
    const auto& shapeB = b->shape();

    assert(shapeA.size() == 2 && shapeB.size() == 2);
    assert(shapeA[1] == shapeB[1]); // 列数保持一致
    assert(shapeB[0] == 1 || shapeA[0] == shapeB[0]);   // B 行数为1，或与 A 行数相同

    size_t m = shapeA[0];
    size_t n = shapeA[1];
    std::vector<double> result(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t b_i = (shapeB[0] == 1) ? 0 : i;
            result[i * n + j] = a->value()[i * n + j] - b->value()[b_i * n + j];
        }
    }

    if(!anyRequiresGrad({a, b})) {
        return Tensor::create(shapeA, result);
    }

    auto fn = std::make_shared<SubBackward>();
    fn->shapeA_ = shapeA;
    fn->shapeB_ = shapeB;
    fn->saved_inputs_ = {a, b};

    return Tensor::createFromOp(shapeA, result, fn);
}

/*
============================================================
    mul : 逐元素乘法，shape 必须完全相同
============================================================
*/
TensorPtr ops::mul(const TensorPtr& a, const TensorPtr& b) {
    assert(a->shape() == b->shape());

    size_t n = a->size();
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = a->value()[i] * b->value()[i];
    }

    if(!anyRequiresGrad({a, b})) {
        return Tensor::create(a->shape(), result);
    }

    auto fn = std::make_shared<MulBackward>();
    fn->same_tensor_ = (a.get() == b.get());
    fn->x_val_ = a->value();
    fn->y_val_ = b->value();
    if (fn->same_tensor_) {
        fn->saved_inputs_ = {a};
    } else {
        fn->saved_inputs_ = {a, b};
    }
    
    return Tensor::createFromOp(a->shape(), result, fn);
}

/*
============================================================
    matmul : A[m, k] @ B[k, n] = C[m, n]
============================================================
*/
TensorPtr ops::matmul(const TensorPtr& a, const TensorPtr& b) {
    size_t m = a->rows();
    size_t k = a->cols();
    size_t n = b->cols();

    assert(k == b->rows());

    std::vector<double> result(m * n, 0.0);
    for(size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            for (size_t j = 0; j < n; ++j) {
                result[i * n + j] += a->value()[i * k + p] * b->value()[p * n + j];
            }
        }
    }

    if (!anyRequiresGrad({a, b})) {
        return Tensor::create({m, n}, result);
    }

    auto fn = std::make_shared<MatMulBackward>();
    fn->A_val_ = a->value();
    fn->B_val_ = b->value();
    fn->m_ = m;
    fn->k_ = k;
    fn->n_ = n;
    fn->saved_inputs_ = {a, b};

    return Tensor::createFromOp({m, n}, result, fn);
}

/*
============================================================
    relu : max(0, x)
============================================================
*/
TensorPtr ops::relu(const TensorPtr& a) {
    size_t n = a->size();
    
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = (a->value()[i] > 0.0) ? a->value()[i] : 0.0;
    }

    if (!anyRequiresGrad({a})) {
        return Tensor::create(a->shape(), result);
    }

    auto fn = std::make_shared<ReLUBackward>();
    fn->x_val_ = a->value();
    fn->saved_inputs_ = {a};

    return Tensor::createFromOp(a->shape(), result, fn);
}

/*
============================================================
    sigmoid : 1 / (1 + exp(-x))
============================================================
*/
TensorPtr ops::sigmoid(const TensorPtr& a) {
    size_t n = a->size();

    std::vector<double> result(n);

    for (size_t i = 0; i < n; ++i) {
        result[i] = 1.0 / (1.0 + std::exp(-a->value()[i]));
    }

    if(!anyRequiresGrad({a})) {
        return Tensor::create(a->shape(), result);
    }

    auto fn = std::make_shared<SigmoidBackward>();
    fn->y_val_ = result;    // 这里保存输出值，反向要使用
    fn->saved_inputs_ = {a};

    return Tensor::createFromOp(a->shape(), result, fn);
}

/*
============================================================
    sum : 所有元素求和，输出标量 shape {1}
============================================================
*/
TensorPtr ops::sum(const TensorPtr& a) {
    double s = 0.0;
    for (auto v : a->value()) {
        s += v;
    }

    if (!anyRequiresGrad({a})) {
        return Tensor::create({1}, {s});
    }

    auto fn = std::make_shared<SumBackward>();
    fn->input_size_ = a->size();
    fn->saved_inputs_ = {a};

    return Tensor::createFromOp({1}, {s}, fn);
}

/*
============================================================
    bce_with_logits_loss : 数值稳定的 BCE，接受 sigmoid 之前的 logits

    forward 用 log-sum-exp trick 避免 log(0)：
        L_i = max(x,0) - x*y + log(1 + e^{-|x|})
    mean 规约到标量 {1}

    使用时不需要提前调用 ops::sigmoid，直接传 fc 的输出
============================================================
*/
TensorPtr ops::bceWithLogitsLoss(const TensorPtr& logits, const TensorPtr& target) {
    assert(logits->shape() == target->shape());

    size_t n = logits->size();

    // forward：numerically stable BCE
    std::vector<double> sig_val(n);
    double loss_val = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double x = logits->value()[i];
        double y = target->value()[i];
        // log(1 + e^{-|x|}) 不会下溢
        loss_val += std::max(x, 0.0) - x * y + std::log(1 + std::exp(-std::abs(x)));
        sig_val[i] = 1.0 / (1.0 + std::exp(-x));    // 反向需要
    }

    loss_val /= static_cast<double>(n);

    if(!anyRequiresGrad({logits})) {
        return Tensor::create({1}, {loss_val});
    }

    auto fn = std::make_shared<BCEWithLogitsBackward>();
    fn->sigmoid_val_ = sig_val;
    fn->target_val_ = target->value();
    fn->n_ = n;
    fn->saved_inputs_ = {logits};   // target 无梯度，不入图

    return Tensor::createFromOp({1}, {loss_val}, fn);
}

/*
============================================================
    cat : 沿 dim=1（列方向）拼接任意数量的 2D Tensor
          inputs[0]: [m, n0]
          inputs[1]: [m, n1]
          ...
          output   : [m, n0+n1+...]
============================================================
*/
TensorPtr ops::cat(const std::vector<TensorPtr>& inputs) {
    assert(!inputs.empty());

    size_t m = inputs[0]->rows();
    size_t N = 0;
    std::vector<size_t> col_widths;
    
    for (auto& t : inputs) {
        assert(t->rows() == m);
        assert(t->ndim() == 2);

        col_widths.push_back(t->cols());
        N += t->cols();
    }

    // ---- forward：按行逐段拼 ----
    std::vector<double> result(m * N);
    for (size_t r = 0; r < m; ++r) {
        size_t col_offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            size_t w = col_widths[i];
            const auto& val = inputs[i]->value();
            for (size_t c = 0; c < w; ++c) {
                result[r * N + col_offset + c] = val[r * w + c];
            }
            col_offset += w;
        }
    }

    if (!anyRequiresGrad(inputs)) {
        return Tensor::create({m, N}, result);
    }

    auto fn = std::make_shared<CatBackward>();
    fn->rows_ = m;
    fn->col_widths_ = col_widths;
    fn->saved_inputs_ = inputs; // 保存所有输入，backward 按顺序分发梯度

    return Tensor::createFromOp({m, N}, result, fn);
}