#include "grad_fn.h"
#include <assert.h>

/*
============================================================
    AddBackward
    forward: C = A + B, 支持 [m, n] + [1, n] broaddcast
    dA = grad （直接透传）
    dB = grad （若 broadcast，沿第 0 维求和折叠回 [1, n])
============================================================
*/
std::vector<std::vector<double>> AddBackward::apply(const std::vector<double>& grad) {
    size_t m = shapeA_[0];
    size_t n = shapeA_[1];

    // dA 直接透传
    std::vector<double> dA = grad;

    // dB：若 B 被 broadcast，沿第0维求和
    std::vector<double> dB;
    if (shapeB_[0] == 1) {
        dB.assign(n, 0.0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dB[j] += grad[i * n + j];
            }
        }
    } else {
        dB = grad;
    }

    return {dA, dB};
}

/*
============================================================
    SubBackward
    forward: C = A - B
    dA = grad
    dB = -grad（若 broadcast，沿第0维求和后取负）
============================================================
*/
std::vector<std::vector<double>> SubBackward::apply(const std::vector<double>& grad) {
    size_t m = shapeA_[0];
    size_t n = shapeA_[1];

    std::vector<double> dA = grad;

    std::vector<double> dB;
    if (shapeB_[0] == 1) {
        dB.assign(n, 0.0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dB[j] -= grad[i * n + j];
            }
        }
    } else {
        dB.resize(grad.size());
        for (size_t i = 0; i < grad.size(); ++i) {
            dB[i] = -grad[i];
        }
    }

    return {dA, dB};
}

/*
============================================================
    MulBackward
    forward: C = A * B（逐元素）
    dA = grad * B
    dB = grad * A
    特殊情况：A == B（同一个 Tensor），dA += 2 * grad * A
============================================================
*/
std::vector<std::vector<double>> MulBackward::apply(const std::vector<double>& grad) {
    size_t n = x_val_.size();
    assert(grad.size() == n);

    if (same_tensor_) {
        std::vector<double> dA(n);
        for (size_t i = 0; i < n; ++i) {
            dA[i] = 2.0 * grad[i] * x_val_[i];
        }
        return {dA};    // 只有1个输入
    }

    std::vector<double> dA(n), dB(n);
    for (size_t i = 0; i < n; ++i) {
        dA[i] = grad[i] * y_val_[i];
        dB[i] = grad[i] * x_val_[i];
    }

    return {dA, dB};
}

/*
============================================================
    MatMulBackward
    forward: C = A @ B, A[m, k] B[k, n] C[m, n]
    dA = grad @ B^T, shape [m, k]
    dB = grad @ A^T, shape [k, n]
============================================================
*/
std::vector<std::vector<double>> MatMulBackward::apply(const std::vector<double>& grad) {
    assert(grad.size() == m_ * n_);

    // dA = grad @ B^T
    std::vector<double> dA(m_ * k_, 0.0);
    for (size_t i = 0; i < m_; ++i) {
        for (size_t p = 0; p < k_; ++p) {
            for (size_t j = 0; j < n_; ++j) {
                dA[i * k_ + p] += grad[i * n_ + j] * B_val_[p * n_ + j];
            }
        }
    }

    // dB = grad @ A^T
    std::vector<double> dB(k_ * n_, 0.0);
    for (size_t i = 0; i < m_; ++i) {
        for (size_t p = 0; p < k_; ++p) {
            for (size_t j = 0; j < n_; ++j) {
                dB[p * n_ + j] += A_val_[i * k_ + p] * grad[i * n_ + j];
            }
        }
    }

    return {dA, dB};
}

/*
============================================================
    ReLUBackward
    forward: y = max(0, x)
    dx = grad if x > 0 else 0
============================================================
*/
std::vector<std::vector<double>> ReLUBackward::apply(const std::vector<double>& grad) {
    size_t n = x_val_.size();
    assert(grad.size() == n);

    std::vector<double> dx(n);
    for (size_t i = 0; i < n; ++i) {
        dx[i] = x_val_[i] > 0.0 ? grad[i] : 0.0;
    }

    return {dx};
}
/*
============================================================
    SigmoidBackward
    forward: y = 1 / (1 + exp(-x))
    dx = grad * y * (1 - y)
    注：保存的是 forward 的输出 y，不是输入 x
============================================================
*/
std::vector<std::vector<double>> SigmoidBackward::apply(const std::vector<double>& grad) {
    size_t n = y_val_.size();
    assert(grad.size() == n);

    std::vector<double> dx(n);
    for (size_t i = 0; i < n; ++i) {
        dx[i] = grad[i] * y_val_[i] * (1.0 - y_val_[i]);
    }

    return {dx};
}

/*
============================================================
    SumBackward
    forward: scalar = sum(x)
    dx = grad[0] 广播到所有元素（grad 是标量，只有一个值）
============================================================
*/
std::vector<std::vector<double>> SumBackward::apply(const std::vector<double>& grad) {
    assert(grad.size() == 1);
    std::vector<double> dx(input_size_, grad[0]);

    return {dx};
}

/*
============================================================
    BCEWithLogitsBackward
    forward: loss = mean( max(x,0) - x*y + log(1 + e^{-|x|}) )
    
    ∂loss/∂x_i = (1/N) * (sigmoid(x_i) - y_i)
    
    保存的是 sigmoid(x)，反向直接 p - y，无除法，数值稳定
============================================================
*/
std::vector<std::vector<double>> BCEWithLogitsBackward::apply(const std::vector<double>& grad) {
    assert(grad.size() == 1);

    double g = grad[0];
    size_t n = sigmoid_val_.size();
    std::vector<double> dx(n);

    for (size_t i = 0; i < n; ++i) {
        dx[i] = g * (sigmoid_val_[i] - target_val_[i]) / static_cast<double>(n);
    }

    return {dx};
}