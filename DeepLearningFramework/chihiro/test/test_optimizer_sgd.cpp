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
    // 该测试通过计算得知，lr 必须满足 0 < lr < 0.25。感兴趣可以自行计算。 
    SGD optimizer({&w}, 0.1);
    
    // 训练循环
    Executor executor;
    for (int i = 0; i < 30; ++i) {
        executor.zeroGrad(graph);
        // 这里的 zeroGrad 主要保证了 parameter 不在 graph.nodes 里的情况
        optimizer.zeroGrad();

        executor.forward(graph);
        Tensor& loss = loss_ptr->output();
        executor.backward(graph, loss);
        optimizer.step();
        std::cout << "step " << i
              << " | grad = " << w.grad()
              << " | loss = " << loss.value()
              << " | w = " << w.value()
              << std::endl;
    }
    return 0;
}