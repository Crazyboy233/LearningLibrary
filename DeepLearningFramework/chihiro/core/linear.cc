#include "linear.h"
#include "ops.h"
#include <random>

Linear::Linear(size_t in_features, size_t out_features, unsigned seed) 
    : in_features_(in_features), out_features_(out_features)
{
    // ---- 随机引擎 ----
    std::mt19937 rng;
    if (seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }

    // ---- He 初始化：W ~ N(0, sqrt(2 / in_features)) ----
    // 适合 ReLU；若用 Sigmoid/Tanh 可改为 Xavier: sqrt(1 / in_features)
    double std_dev = std::sqrt(2.0 / static_cast<double>(in_features));
    std::normal_distribution<double> dist(0.0, std_dev);

    std::vector<double> w_data(in_features * out_features);
    for (auto& v : w_data) {
        v = dist(rng);
    }

    // b 全零
    std::vector<double> b_data(out_features, 0.0);
 
    W_ = Tensor::create({in_features, out_features}, w_data, /*requires_grad=*/true);
    b_ = Tensor::create({1, out_features}, b_data, /*requires_grad=*/true);
}

TensorPtr Linear::forward(const TensorPtr& x) {
    // y = x @ w + b
    // x: [batch, in]  W: [in, out]  b: [1, out]  →  y: [batch, out]
    auto xW = ops::matmul(x, W_);
    return ops::add(xW, b_);
}