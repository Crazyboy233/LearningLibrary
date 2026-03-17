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
// g++ test.cpp ./core/*.cc -I./core/

int main() {
    Tensor a(2.0);
    Tensor b(3.0);
    Tensor c(4.0);
    Tensor d(5.0);

    AddOp add_op;
    MulOp mul_op;
    SubOp sub_op;
    Graph graph;

    auto add_node = std::make_unique<Node>(&add_op, std::vector<Tensor*>{&a, &b});
    Node* add_ptr = add_node.get();
    graph.addNode(std::move(add_node));

    auto mul_node = std::make_unique<Node>(&mul_op, std::vector<Tensor*>{&add_ptr->output(), &c});
    Node* mul_ptr = mul_node.get();
    graph.addNode(std::move(mul_node));

    auto sub_node = std::make_unique<Node>(&sub_op, std::vector<Tensor*>{&mul_ptr->output(), &d});
    Node* sub_ptr = sub_node.get();
    graph.addNode(std::move(sub_node));

    Executor executor;
    executor.forward(graph);

    Tensor& z = sub_ptr->output();
    executor.backward(graph, z);
    
    std::cout << "z = " << z.value() << std::endl;
    std::cout << "da = " << a.grad() << std::endl;
    std::cout << "db = " << b.grad() << std::endl;
    std::cout << "dc = " << c.grad() << std::endl;
    std::cout << "dd = " << d.grad() << std::endl;
    
    std::cout << add_ptr->output().value() << std::endl;
    std::cout << mul_ptr->output().value() << std::endl;
    std::cout << sub_ptr->output().value() << std::endl;
}