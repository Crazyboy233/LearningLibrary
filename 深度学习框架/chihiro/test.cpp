#include "executor.h"

#include <functional>
#include <vector>
#include <string>
#include <iostream>

/*
===写在前面===
现在的结构：
    Graph = 结构
    Node = 依赖 + 输出
    Op = 计算规则
    Executor = 执行策略
*/ 

// 编译命令如下：
// g++ test.cpp -I./core/

int main() {
    Tensor a{3.0};
    Tensor b{4.0};
    Tensor c{5.0};

    AddOp add_op;
    Graph graph;
    auto node = std::make_unique<Node>(&add_op, std::vector<Tensor*>{&a, &b});
    auto node2 = std::make_unique<Node>(&add_op, std::vector<Tensor*>{&node.get()->output(), &c});
    Node* node_ptr = node.get();
    Node* node2_ptr = node2.get();

    graph.addNode(std::move(node));
    graph.addNode(std::move(node2));

    Executor executor;
    executor.run(graph);

    std::cout << node_ptr->output().value() << std::endl;
    std::cout << node2_ptr->output().value() << std::endl;
}