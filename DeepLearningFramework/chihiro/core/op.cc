#include "op.h"
#include <iostream>
#include <cassert>

void AddOp::forward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const auto& shapeA = inputs[0]->shape();
    const auto& shapeB = inputs[1]->shape();
    const auto& A = inputs[0]->value();
    const auto& B = inputs[1]->value();
    
    // 目前只支持二维，且只处理 [m,n] + [1,n] 的 broadcast
    assert(shapeA.size() == 2 && shapeB.size() == 2);
    assert(shapeA[1] == shapeB[1]); // 列数保持一致
    assert(shapeB[0] == 1 || shapeA[0] == shapeB[0]);   // B 的第 0 维是 1 或者两者的 shape 完全一致。

    size_t m = shapeA[0];
    size_t n = shapeA[1];

    std::vector<double> result(m * n);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t b_i = (shapeB[0] == 1) ? 0 : i;  // broadcast 时 B 的行索引固定为 0
            result[i * n + j] = A[i * n + j] + B[b_i * n + j];
        }
    }
    output.setValue(shapeA, result);
}

void AddOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    // 目前只支持二维，且只处理 [m,n] + [1,n] 的 broadcast
    const auto& shapeA = inputs[0]->shape();
    const auto& shapeB = inputs[1]->shape();
    const auto& grad = output.grad();

    size_t m = shapeA[0];
    size_t n = shapeA[1];

    // dA 直接透传
    inputs[0]->addGrad(grad);
    
    // dB：如果 B 被 broadcast 了，沿第0维求和折叠回 [1,n]
    if (shapeB[0] == 1) {
        std::vector<double> dB(n, 0.0);
        for(size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dB[j] += grad[i * n + j];
            }
        }
        inputs[1]->addGrad(dB);
    } else {
        inputs[1]->addGrad(grad);
    } 
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

void SubOp::forward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const auto& shapeA = inputs[0]->shape();
    const auto& shapeB = inputs[1]->shape();
    const auto& A = inputs[0]->value();
    const auto& B = inputs[1]->value();
    
    // 目前只支持二维，且只处理 [m,n] + [1,n] 的 broadcast
    assert(shapeA.size() == 2 && shapeB.size() == 2);
    assert(shapeA[1] == shapeB[1]); // 列数保持一致
    assert(shapeB[0] == 1 || shapeA[0] == shapeB[0]);   // B 的第 0 维是 1 或者两者的 shape 完全一致。

    size_t m = shapeA[0];
    size_t n = shapeA[1];

    std::vector<double> result(m * n);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t b_i = (shapeB[0] == 1) ? 0 : i;  // broadcast 时 B 的行索引固定为 0
            result[i * n + j] = A[i * n + j] - B[b_i * n + j];
        }
    }
    output.setValue(shapeA, result);
}

void SubOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    // 目前只支持二维，且只处理 [m,n] + [1,n] 的 broadcast
    const auto& shapeA = inputs[0]->shape();
    const auto& shapeB = inputs[1]->shape();
    const auto& grad = output.grad();

    size_t m = shapeA[0];
    size_t n = shapeA[1];

    // dA 直接透传
    inputs[0]->addGrad(grad);
    
    // dB：如果 B 被 broadcast 了，沿第0维求和折叠回 [1,n]
    if (shapeB[0] == 1) {
        std::vector<double> dB(n, 0.0);
        for(size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dB[j] -= grad[i * n + j];   // SubOp 对 B 的梯度是负的
            }
        }
        inputs[1]->addGrad(dB);
    } else {
        std::vector<double> dB(grad.size());
        for (size_t i = 0; i < dB.size(); ++i) {
            dB[i] = -grad[i];
        }
        inputs[1]->addGrad(dB);
    }
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

// \sigma(x) = \max(0, x)
void ReLUOp::forward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const auto& x = inputs[0]->value();
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0.0 ? x[i] : 0.0;  
    }
    output.setValue(inputs[0]->shape(), result);
}

void ReLUOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const auto& grad = output.grad();
    const auto& x = inputs[0]->value();

    std::vector<double> dx(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        dx[i] = x[i] > 0.0 ? grad[i] : 0.0;
    }

    inputs[0]->addGrad(dx);
}

// \sigma(x) = \frac{1}{1+e^{-x}}
void SigmodOp::forward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const auto& x = inputs[0]->value();
    
    std::vector<double> result(x.size());
    for(size_t i = 0; i < x.size(); ++i) {
        result[i] = 1.0 / (1.0 + std::exp(-x[i]));
    }

    output.setValue(inputs[0]->shape(), result);
}

void SigmodOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const auto& grad = output.grad();
    const auto& y = output.value(); // 这里直接取 forward 的输出值

    std::vector<double> dx(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        dx[i] = grad[i] * y[i] *(1.0 - y[i]);
    }

    inputs[0]->addGrad(dx);
}