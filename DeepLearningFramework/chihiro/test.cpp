#include "core/executor.h"

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
    Tensor a(2.0);
    Tensor b(3.0);
    Tensor c(4.0);

    AddOp add_op;
    MulOp mul_op;
    Graph graph;

    auto add_node = std::make_unique<Node>(&add_op, std::vector<Tensor*>{&a, &b});
    Node* add_ptr = add_node.get();
    graph.addNode(std::move(add_node));

    auto mul_node = std::make_unique<Node>(&mul_op, std::vector<Tensor*>{&add_ptr->output(), &c});
    Node* mul_ptr = mul_node.get();
    graph.addNode(std::move(mul_node));

    Executor executor;
    executor.forward(graph);

    Tensor& z = mul_ptr->output();
    executor.backward(graph, z);
    
    std::cout << "z = " << z.value() << std::endl;
    std::cout << "da = " << a.grad() << std::endl;
    std::cout << "db = " << b.grad() << std::endl;
    std::cout << "dc = " << c.grad() << std::endl;
    
    std::cout << add_ptr->output().value() << std::endl;
    std::cout << mul_ptr->output().value() << std::endl;
}