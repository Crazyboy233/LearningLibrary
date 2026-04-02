#include "op.h"
#include <iostream>
#include <cassert>

void AddOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    assert(input[0]->shape() == input[1]->shape());

    std::vector<double> x = input[0]->value();
    std::vector<double> y = input[1]->value();
    
    std::vector<double> result;
    for (int i = 0; i < x.size() && i < y.size(); ++i) {
        result.push_back(x[i] + y[i]);
    }
    output.setValue(input[0]->shape(), result);
}

void AddOp::backward(const std::vector<Tensor*>& input, Tensor& output) {
    assert(input[0]->shape() == input[1]->shape());
    std::vector<double> grad = output.grad();

    input[0]->addGrad(grad);
    input[1]->addGrad(grad);
}


void MulOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    assert(input[0]->shape() == input[1]->shape());
    std::vector<double> x = input[0]->value();
    std::vector<double> y = input[1]->value();
    std::vector<double> result;

    for (int i = 0; i < x.size() && i < y.size(); ++i) {
        result.push_back(x[i] * y[i]);
    }
    output.setValue(input[0]->shape(), result);
}

void MulOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    assert(inputs[0]->shape() == inputs[1]->shape());

    const auto& grad = output.grad();

    const auto& x = inputs[0]->value();
    const auto& y = inputs[1]->value();

    assert(x.size() == y.size() && x.size() == grad.size());
    
    if (inputs[0] == inputs[1]) {
        // d * d 的情况，梯度是 2 * grad * x
        std::vector<double> result;
        result.reserve(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result.push_back(2.0 * grad[i] * x[i]);
        }
        inputs[0]->addGrad(result);
    } else {
        std::vector<double> result1, result2;
        result1.reserve(x.size());
        result2.reserve(x.size());

        for (int i = 0; i < x.size(); ++i) {
            assert(x.size() == y.size() && x.size() == grad.size());

            result1.push_back(grad[i] * y[i]);
            result2.push_back(grad[i] * x[i]);
        }

        inputs[0]->addGrad(result1);
        inputs[1]->addGrad(result2);
    }
}

void SubOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    assert(input[0]->shape() == input[1]->shape());
    std::vector<double> x = input[0]->value();
    std::vector<double> y = input[1]->value();

    std::vector<double> result;
    for (int i = 0; i < x.size() && i < y.size(); ++i) {
        result.push_back(x[i] - y[i]);
    }

    output.setValue(input[0]->shape(), result);
}

void SubOp::backward(const std::vector<Tensor*>& input, Tensor& output) {
    assert(input[0]->shape() == input[1]->shape());
    std::vector<double> grad = output.grad();

    std::vector<double> result;
    for (int i = 0; i < grad.size(); ++i) {
        result.push_back(grad[i] * -1);
    }

    input[0]->addGrad(grad);
    input[1]->addGrad(result);
}

void SumOp::forward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const std::vector<double>& x = inputs[0]->value();
    double sum = 0.0;
    for (double v : x) {
        sum += v;
    }
    output.setValue({1}, std::vector<double>{sum}); // scalar, 这里的 shape 本就应为 1
}

void SumOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const std::vector<double>& grad_out = output.grad();
    Tensor& x = *inputs[0];

    std::vector<double> grad_x(x.value().size(), grad_out[0]);
    x.addGrad(grad_x);
}

void MatMulOp::forward(const std::vector<Tensor*>& inputs, Tensor& output) {
    // A: [m, k],  B: [k, n]  =>  output: [m, n]
    size_t m = inputs[0]->rows();
    size_t k = inputs[0]->cols();
    assert(k == inputs[1]->rows());
    size_t n = inputs[1]->cols();

    // 矩阵乘法
    std::vector<double> result(m * n, 0.0);
    for (size_t i = 0; i < m; ++i) {    // 结果的行
        for (size_t j = 0; j < n; ++j) {    // 结果的列
            for (size_t p = 0; p < k; ++p) {    // 点积求和
                result[i * n + j] += inputs[0]->value()[i * k + p] * inputs[1]->value()[p * n + j];
            }
        }
    }
    output.setValue({m, n}, result);
}

void MatMulOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    // A: [m, k],  B: [k, n]  =>  output: [m, n]
    const auto& grad = output.grad();
    if (grad.empty() ) {
        return;
    }

    size_t m = inputs[0]->rows();
    size_t k = inputs[0]->cols();
    size_t n = inputs[1]->cols();

    const auto& A = inputs[0]->value();
    const auto& B = inputs[1]->value();

    // dA = grad * B^T，shape [m, k]
    std::vector<double> dA(m * k, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            for (size_t j = 0; j < n; ++j) {
                dA[i * k + p] += grad[i * n + j] * B[p * n + j];    // B^T[j,p] = B[p,j]
            }
        }
    }

    // dB = A^T * grad，shape [k, n]
    std::vector<double> dB(k * n, 0.0);
    for (size_t p = 0; p < k; ++p) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                dB[p * n + j] += A[i * k + p] * grad[i * n + j];    // A^T[p, i] = A[i, p]
            }
        }
    }

    inputs[0]->addGrad(dA);
    inputs[1]->addGrad(dB);
}