#include "../core/Tensor.h"
#include "../core/ops.h"
#include "../core/linear.h"
#include "../core/optimizer.h"

#include <iostream>
#include <iomanip>
#include <cmath>

// 编译命令
// g++ -std=c++17 ./core/*.cc ./test/03_test_linear.cpp && ./a.out

int main() {
    // ── 数据 ──────────────────────────────────────────────
    auto x      = Tensor::create({4, 2}, {1,2, 2,1, 3,4, 4,3}, false);
    auto target = Tensor::create({4, 1}, {8, 7, 18, 17}, false);
 
    // ── 模型 ──────────────────────────────────────────────
    Linear fc1(2, 1, /*seed=*/1); 
    auto params = fc1.parameters();

    SGD sgd(params, /*lr=*/0.01 /*momentum=*/);
 
    // ── 训练 ──────────────────────────────────────────────
    const int EPOCHS = 100;
 
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
        // ∂loss/∂pred = 2×(pred−target)∂fc1.weight/∂loss ​= ∂pred/∂loss​ × ∂weight/∂pred​
        // 这里 topo 排序，除去不需要计算 grad 的输入节点 x，target，应该有4个节点：pred,diff,diff2,loss。
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
 
/*
训练流程：

样本输入 x [4,2]
    ↓
fc1.forward(x)：线性变换 pred = x·W + b，shape [4,1]
    ↓
ops::sub(pred, target)：计算残差 diff = pred - target
    ↓
ops::mul(diff, diff)：逐元素平方 diff²
    ↓
ops::sum(diff²)：求和得标量 loss（MSE 无除N）
    ↓
loss->backward()：从 loss 出发做拓扑排序，
                  反向链式求导，把梯度写入 W->grad(), b->grad()
    ↓
sgd.step()：W = W - lr × grad，更新权重
    ↓
sgd.zeroGrad()：清空所有梯度（为下一轮做准备）
    ↓
循环 100 个 epoch
    ↓
W 和 b 收敛到能让 pred ≈ target 的值
*/