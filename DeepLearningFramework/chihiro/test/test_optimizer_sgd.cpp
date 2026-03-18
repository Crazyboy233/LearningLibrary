#include "../core/executor.h"
#include "../core/parameter.h"
#include "../core/optimizer.h"

#include <functional>
#include <vector>
#include <string>
#include <iostream>

// 编译命令
// g++ test/test_optimizer_sgd.cpp ./core/*.cc -I./core/

int main() {
    Parameter w(0.0);
    Tensor x(2.0);
    Tensor target(10.0);

    MulOp mul_op;
    SubOp sub_op;

    Graph graph;

    // y = w * x
    auto mul_node = std::make_unique<Node>(&mul_op, std::vector<Tensor*>{&w, &x});
    Node* mul_ptr = mul_node.get();
    graph.addNode(std::move(mul_node));

    // d = y - target
    auto sub_node = std::make_unique<Node>(&sub_op, std::vector<Tensor*>{&mul_ptr->output(), &target});
    Node* sub_ptr = sub_node.get();
    graph.addNode(std::move(sub_node));

    // loss = d * d
    auto loss_node = std::make_unique<Node>(&mul_op, std::vector<Tensor*>{&sub_ptr->output(), &sub_ptr->output()});
    Node* loss_ptr = loss_node.get();
    graph.addNode(std::move(loss_node));

    // 定义优化器
    SGD optimizer({&w}, 0.01);
    
    // 训练循环
    Executor executor;
    for (int i = 0; i < 100; ++i) {
        optimizer.zeroGrad();
        executor.forward(graph);
        Tensor& loss = loss_ptr->output();
        executor.backward(graph, loss);
        optimizer.step();
        std::cout << "step " << i
              << " | loss = " << loss.value()
              << " | w = " << w.value()
              << std::endl;
    }
    return 0;
}