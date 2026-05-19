#include "../core/Tensor.h"
#include "../core/ops.h"
#include "../core/embeddings.h"
#include "../core/optimizer.h"
#include "../core/linear.h"

#include <iostream>

// 编译命令
// g++ -std=c++17 ./core/*.cc ./test/05_test_embedding_cat.cpp && ./a.out
 
void test_recommendation() {
    // 超参
    const size_t EMB_DIM = 4;
    const size_t HIDDEN = 8;
    const size_t N_USERS = 10;
    const size_t N_ITEMS = 20;
    const double LR = 0.05;
    const size_t EPOCHS = 100;

    Embedding user_emb(N_USERS, EMB_DIM, 1);
    Embedding item_emb(N_ITEMS, EMB_DIM, 2);
    Linear fc(EMB_DIM * 2, HIDDEN, 3);  // cat 后维度翻倍
    Linear out_layer(HIDDEN, 1, 4);

    // 收集所有参数
    std::vector<TensorPtr> params;
    for (auto& p : user_emb.parameters()) params.push_back(p);
    for (auto& p : item_emb.parameters()) params.push_back(p);
    for (auto& p : fc.parameters()) params.push_back(p);
    for (auto& p : out_layer.parameters()) params.push_back(p);

    SGD sgd(params, LR, /*momentum=*/0.9);

    // 固定 batch：4 条样本，label=1 表示用户-商品有正向交互
    std::vector<size_t> user_ids = {0, 2, 5, 7};
    std::vector<size_t> item_ids = {3, 1, 8, 5};
    // target: [4, 1]
    auto target = Tensor::create({4, 1}, {1.0, 1.0, 0.0, 1.0});

    // 训练
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        sgd.zeroGrad();

        // forward
        auto u_emb = user_emb.forward(user_ids);
        auto i_emb = item_emb.forward(item_ids);
        auto concat = ops::cat({u_emb, i_emb});
        auto h = ops::relu(fc.forward(concat));
        auto logits = out_layer.forward(h);

        auto loss = ops::bceWithLogitsLoss(logits, target);

        loss->backward();

        // 打印：
        if (epoch % 10 == 0) {
            std::cout << "================================" << std::endl;
            std::cout << "epoch = " << epoch << std::endl;
            std::cout << "loss: " << "[";
            for (auto v : loss->value()) {
                std::cout << v << ", ";
            }
            std::cout << "]" << std::endl;;
            
            std::cout << "sigmoid(logits): " << "[";
            auto prob = ops::sigmoid(logits);
            for (auto v : prob->value()) {
                std::cout << v << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "================================" << std::endl;

        }
        // end 打印

        sgd.step();
    }

}

int main() {
    test_recommendation();

    return 0;
}