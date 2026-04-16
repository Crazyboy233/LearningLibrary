#include "../core/Tensor.h"
#include "../core/ops.h"
#include "../core/linear.h"
#include "../core/optimizer.h"

#include <iostream>
#include <iomanip>
#include <cmath>

// 编译命令
// g++ -std=c++17 ./core/*.cc ./test/02_test_xor.cpp && ./a.out
 
// 注：目前测试并不收敛，但是没有排查到原因

int main() {
    // ── 数据 ──────────────────────────────────────────────
    auto x      = Tensor::create({4, 2}, {0,0, 0,1, 1,0, 1,1}, false);
    auto target = Tensor::create({4, 1}, {0, 1, 1, 0},          false);
 
    // ── 模型 ──────────────────────────────────────────────
    Linear fc1(2, 4, /*seed=*/1);   // 隐藏层，把 2 维输入扩展到 4 维，学习非线性特征
    Linear fc2(4, 1, /*seed=*/2);   // 输出层，把 4 维压缩到 1 维，输出预测概率
 
    auto params = fc1.parameters();
    auto p2     = fc2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());

    SGD sgd(params, /*lr=*/0.5, /*momentum=*/0.9);
 
    // ── 训练 ──────────────────────────────────────────────
    const int EPOCHS = 10;
 
    std::cout << std::fixed << std::setprecision(6);    // 设置浮点数的输出格式
    std::cout << "epoch | loss\n";
 
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // forward（x 是 [4,2]，一次处理全部样本）
        auto h    = ops::relu(fc1.forward(x));
        auto pred = ops::sigmoid(fc2.forward(h));

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
            std::cout << "]" << "|fc2_W| = [";
            for(auto w : fc2.W()->value()) {
                std::cout << w << " ";
            }
            std::cout << "]" << " |fc2_W_grad| = [";
            for(auto w_grad : fc2.W()->grad()) {
                std::cout << w_grad << " ";
            }
            std::cout << "]" << " |fc2_b| = [";
            for(auto b : fc2.b()->value()) {
                std::cout << b << " ";
            }
            std::cout << "]\n\n";
        // }

        sgd.step();
        sgd.zeroGrad();
    }
 
    return 0;
}
 