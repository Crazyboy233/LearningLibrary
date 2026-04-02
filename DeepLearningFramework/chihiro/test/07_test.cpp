#include "../core/executor.h"
#include "../core/parameter.h"
#include "../core/optimizer.h"

#include <vector>
#include <string>
#include <iostream>

// 编译命令
// g++ test/07_test.cpp ./core/*.cc -I./core/

/* 该测试基本独立，不构成完整计算图， 测试内容如下：
    修改了 AddOp 和 SubOp
    测试 AddOp 和 SubOp 的 boardcast 功能
*/ 

void test_AddOp() {
    // 只测 AddOp，不需要完整计算图
    Tensor A({2, 2}, {1.0, 2.0, 
                    3.0, 4.0});
    Tensor B({1, 2}, {10.0, 20.0});

    AddOp add_op;
    Graph graph;
    graph.addInput(&A);
    graph.addInput(&B);

    auto add_node = std::make_unique<Node>(&add_op, std::vector<Tensor*>{&A, &B});
    Node* add_ptr = add_node.get();
    graph.addNode(std::move(add_node));

    Executor executor(graph);
    executor.forward();

    std::cout << "forward result: ";
    for (auto v : add_ptr->output().value())
        std::cout << v << " ";
    std::cout << std::endl;
    // 期望: 11 22 13 24

    // 直接手动设置 grad 并调用 backward，绕过 scalar 检查
    add_ptr->output().addGrad({1.0, 2.0, 3.0, 4.0});
    add_ptr->backward();

    std::cout << "dA: ";
    for (auto v : A.grad()) std::cout << v << " ";
    std::cout << std::endl;
    // 期望: 1 2 3 4

    std::cout << "dB: ";
    for (auto v : B.grad()) std::cout << v << " ";
    std::cout << std::endl;
    // 期望: 4 6
}

void test_SubOp() {
    // 只测 AddOp，不需要完整计算图
    Tensor A({2, 2}, {1.0, 2.0, 
                    3.0, 4.0});
    Tensor B({1, 2}, {10.0, 20.0});

    SubOp sub_op;
    Graph graph;
    graph.addInput(&A);
    graph.addInput(&B);

    auto sub_node = std::make_unique<Node>(&sub_op, std::vector<Tensor*>{&A, &B});
    Node* sub_ptr = sub_node.get();
    graph.addNode(std::move(sub_node));

    Executor executor(graph);
    executor.forward();

    std::cout << "forward result: ";
    for (auto v : sub_ptr->output().value())
        std::cout << v << " ";
    std::cout << std::endl;
    // 期望: 11 22 13 24

    // 手动设置 grad，模拟从后面传来的梯度
    sub_ptr->output().addGrad({1.0, 2.0, 3.0, 4.0});
    sub_ptr->backward();

    std::cout << "dA: ";
    for (auto v : A.grad()) std::cout << v << " ";
    std::cout << std::endl;
    // 期望: 1 2 3 4

    std::cout << "dB: ";
    for (auto v : B.grad()) std::cout << v << " ";
    std::cout << std::endl;
    // 期望: 4 6
}

int main() {
    test_AddOp();
    std::cout << "========================" << std::endl;
    test_SubOp();
    return 0;
}