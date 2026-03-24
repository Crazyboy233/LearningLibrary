#include "../core/executor.h"
#include "../core/parameter.h"
#include "../core/optimizer.h"

#include <functional>
#include <vector>
#include <string>
#include <iostream>

// 编译命令
// g++ test/test_vectorTensor.cpp ./core/*.cc -I./core/

/* 该测试是基于 test_vectorTensor.cpp 增加了如下内容：
    增加 SumOp，将反向传播的起点改为标量 scalar。
*/ 

int main() {
    Parameter w({0.0, 1.0, 2.0});
    Tensor x({2.0, 3.0, 4.0});
    Tensor target({10.0, 12.0, 12.0});

    MulOp mul_op;
    SubOp sub_op;
    SumOp sum_op;

    Graph graph;

    // y = x * x
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
    
    auto sum_node = std::make_unique<Node>(&sum_op, std::vector<Tensor*>{ &loss_ptr->output() });   // ⭐关键：输入是 loss_vec
    Node* sum_ptr = sum_node.get();
    graph.addNode(std::move(sum_node));

    std::cout << "图 构造完成" << std::endl;
    // 定义优化器
    SGD optimizer({&w}, 0.01);
    Executor executor;

    for (int i = 0; i < 100; ++i) {
        executor.zeroGrad(graph);
        optimizer.zeroGrad();
        executor.forward(graph);
        Tensor& loss = sum_ptr->output();
        executor.backward(graph, loss);
        optimizer.step();

        std::cout << "step ：" << i;
        for (int i = 0; i < w.value().size(); i++) {
            std::cout << "| grad = " << w.grad()[i]
                << "| loss = " << loss_ptr->output().value()[i]
                << "| w = " << w.value()[i];
        }
        std::cout << std::endl;
    }
    

    return 0;
}