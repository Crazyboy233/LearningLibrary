#include "linear.h"
#include <random>

Linear::Linear(size_t in_features, size_t out_fratures)
    :in_(in_features), out_(out_fratures)
{
    /*
        in_ 是 in_features，这一层的输入维度，也就是 W 的行数
        out_ 是 out_features，这一层的输出维度，也就是 W 的列数

        以 fc1(2, 4) 为例：
            in_ = 2：输入是2维特征，对应 XOR 的两个输入值
            out_ = 4：输出是4维，隐藏层有4个神经元
        决定了 W 的 shape 是 [in_, out_] = [2, 4]，b 的 shape 是 [1, out_] = [1, 4]。
    */

    // Xavier 初始化 W
    std::mt19937 rng(42);   // 42 是随机数种子，种子相同，生成的随机数序列完全相同（可复现）
    double bound = std::sqrt(1.0 / in_features);
    std::uniform_real_distribution<double> dist(-bound, bound);

    std::vector<double> w_data(in_features * out_fratures);
    for (auto& v : w_data) {
        v = dist(rng);
    }

    w_ = std::make_unique<Parameter>(std::vector<size_t>{in_features, out_fratures}, w_data);
    b_ = std::make_unique<Parameter>(std::vector<size_t>{1, out_fratures}, std::vector<double>(out_fratures, 0.0));
}

Tensor& Linear::forward(Tensor& input, Graph& graph) {
    // y = input * W
    auto matmul_node = std::make_unique<Node>(&mat_mul_op_, std::vector<Tensor*>{&input, w_.get()});
    Node* matmul_ptr = matmul_node.get();
    graph.addNode(std::move(matmul_node));

    // out = y + b
    auto add_node = std::make_unique<Node>(&add_op_, std::vector<Tensor*>{&matmul_ptr->output(), b_.get()});
    Node* add_ptr = add_node.get();
    graph.addNode(std::move(add_node));

    return add_ptr->output();
}