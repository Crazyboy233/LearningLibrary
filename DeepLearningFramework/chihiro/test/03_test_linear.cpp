#include "../core/Tensor.h"
#include "../core/ops.h"
#include "../core/linear.h"
#include "../core/optimizer.h"

#include <iostream>
#include <iomanip>
#include <cmath>

// 编译命令
// g++ -std=c++17 ./core/*.cc ./test/03_test_linear.cpp && ./a.out
 
// 注：目前测试并不收敛，但是没有排查到原因

int main() {
    // ── 数据 ──────────────────────────────────────────────
    auto x      = Tensor::create({4, 2}, {1,2, 2,1, 3,4, 4,3}, false);
    auto target = Tensor::create({4, 1}, {2*1 + 3*2, 2*2 + 3*1, 2*3 + 3*4, 2*4 + 3*3}, false);
 
    // ── 模型 ──────────────────────────────────────────────
    Linear fc1(2, 1, /*seed=*/1);   // 隐藏层，把 2 维输入扩展到 4 维，学习非线性特征
 
    auto params = fc1.parameters();

    SGD sgd(params, /*lr=*/0.5, /*momentum=*/0.9);
 
    // ── 训练 ──────────────────────────────────────────────
    const int EPOCHS = 10;
 
    std::cout << std::fixed << std::setprecision(6);    // 设置浮点数的输出格式
    std::cout << "epoch | loss\n";
 
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // forward（x 是 [4,2]，一次处理全部样本）
        auto pred = fc1.forward(x);

        // loss = sum((pred - target)^2)
        auto diff = ops::sub(pred, target);
        auto diff2 = ops::mul(diff, diff);
        auto loss  = ops::sum(diff2);

        // backward + update
        loss->backward();
        // if ((epoch + 1) % 100 == 0) {
            std::cout << "epoch " << (epoch + 1)
                  << " | loss = " << loss->value()[0]
                  << " | pred = [";
            for (double v : pred->value()) {
                std::cout << v << " ";
            }
            std::cout << "]" << " |fc1_W| = [";
            for(auto w : fc1.W()->value()) {
                std::cout << w << " ";
            }
            std::cout << "]" << " |fc1_W_grad| = [";
            for(auto w_grad : fc1.W()->grad()) {
                std::cout << w_grad << " ";
            }
            std::cout << "]" << " |fc1_b| = [";
            for(auto b : fc1.b()->value()) {
                std::cout << b << " ";
            }
            std::cout << "]\n\n";
        // }

        sgd.step();
        sgd.zeroGrad();
    }
 
    return 0;
}
 