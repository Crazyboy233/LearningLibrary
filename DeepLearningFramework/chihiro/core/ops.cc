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
static bool anyRequiresGrad(const std::vector<const TensorPtr*>& inputs) {
    for (auto* t : inputs) {
        if((*t)->requireGrad()) {
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

    if(!anyRequiresGrad({&a, &b})) {
        return Tensor::creat(shapeA, result);
    }

    auto fn = std::make_shared<AddBackward>();
    fn->shapeA_ = shapeA;
    fn->shapeB_ = shapeB;
    fn->saved_inputs_ = {a, b};

    return Tensor::creatFromOp(shapeA, result, fn);
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

    if(!anyRequiresGrad({&a, &b})) {
        return Tensor::creat(shapeA, result);
    }

    auto fn = std::make_shared<SubBackward>();
    fn->shapeA_ = shapeA;
    fn->shapeB_ = shapeB;
    fn->saved_inputs_ = {a, b};

    return Tensor::creatFromOp(shapeA, result, fn);
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

    if(!anyRequiresGrad({&a, &b})) {
        return Tensor::creat(a->shape(), result);
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
    
    return Tensor::creatFromOp(a->shape(), result, fn);
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

    std::vector<double> result(m * n);
    for(size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            for (size_t j = 0; j < n; ++j) {
                result[i * n + j] += a->value()[i * k + p] * b->value()[p * n + j];
            }
        }
    }

    if (!anyRequiresGrad({&a, &b})) {
        return Tensor::creat({m, n}, result);
    }

    auto fn = std::make_shared<MatMulBackward>();
    fn->A_val_ = a->value();
    fn->B_val_ = b->value();
    fn->m_ = m;
    fn->k_ = k;
    fn->n_ = n;
    fn->saved_inputs_ = {a, b};

    return Tensor::creatFromOp({m, n}, result, fn);
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

    if (!anyRequiresGrad({&a})) {
        return Tensor::creat(a->shape(), result);
    }

    auto fn = std::make_shared<ReLUBackward>();
    fn->x_val_ = a->value();
    fn->saved_inputs_ = {a};

    return Tensor::creatFromOp(a->shape(), result, fn);
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

    if(!anyRequiresGrad({&a})) {
        return Tensor::creat(a->shape(), result);
    }

    auto fn = std::make_shared<SigmoidBackward>();
    fn->y_val_ = result;    // 这里保存输出值，反向要使用
    fn->saved_inputs_ = {a};

    return Tensor::creatFromOp(a->shape(), result, fn);
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

    if (!anyRequiresGrad({&a})) {
        return Tensor::creat({1}, {s});
    }

    auto fn = std::make_shared<SumBackward>();
    fn->input_size_ = a->size();
    fn->saved_inputs_ = {a};

    return Tensor::creatFromOp({1}, {s}, fn);
}