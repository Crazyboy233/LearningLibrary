#include "../core/executor.h"
#include "../core/parameter.h"
#include "../core/optimizer.h"
#include "../core/linear.h"

#include <functional>
#include <vector>
#include <string>
#include <iostream>

// 编译命令
// g++ test/08_xor_linear.cpp ./core/*.cc -I./core/

/* 该测试是基于 06_test.cpp， 重写 XOR 测试。
    相比于上个版本，增加了 linear。
*/ 

int main() {
    /*
        x 是输入，是已知数据，比如：用户特征，房屋面积，一个数字。是无法修改的数据
        target 是目标，是正确答案，比如：你想让模型输出的值，房价，标签（label）
        w 是参数(Parameter)，是模型需要学习的东西，这是唯一可以改变的。
        
        y 是预测值(Prediction)，是模型当前输出结果。y = 模型输出 = f(w, x)。这里 y = w * x 是一个最简单的线性模型。
        loss 误差衡量(核心)，loss = 衡量 y 和 target 差多远。
    */

    // 标准 XOR 输入
    Tensor X({4, 2}, {0.0, 0.0,
                      0.0, 1.0,
                      1.0, 0.0,
                      1.0, 1.0});

    Tensor target({4, 1}, {0.0, 1.0, 1.0, 0.0});

    SigmoidOp sigmod_op;
    MulOp mul_op;
    SubOp sub_op;
    SumOp sum_op;

    Graph graph;
    
    // linear 内部管理 w 和 b
    Linear fc1(2, 4);
    Linear fc2(4, 1);
    
    // 注册外部节点。避免累计脏数据
    graph.addInput(&X);
    graph.addInput(&target);

    // 构图
    // h = X * W1 + b1, shape [4, 4]
    Tensor& h = fc1.forward(X, graph);

    // a = Sigmoid(h), shape [4, 4]
    auto n_sig1 = std::make_unique<Node>(&sigmod_op, std::vector<Tensor*>{&h});
    Node* sig1_ptr = n_sig1.get();
    graph.addNode(std::move(n_sig1));

    // y = a * W2 + b2, shape [4, 1]
    Tensor& y = fc2.forward(sig1_ptr->output(), graph);

    // out = Sigmoid(y), shape [4, 1]
    auto n_sig2 = std::make_unique<Node>(&sigmod_op, std::vector<Tensor*>{&y});
    Node* sig2_ptr = n_sig2.get();
    graph.addNode(std::move(n_sig2));
    
    // d = out - target, shape [4, 1]
    auto n_sub = std::make_unique<Node>(&sub_op, std::vector<Tensor*>{&sig2_ptr->output(), &target});   
    Node* sub_ptr = n_sub.get();
    graph.addNode(std::move(n_sub));

    // loss_vec = d * d, shape [4, 1]
    auto n_mul = std::make_unique<Node>(&mul_op, std::vector<Tensor*>{&sub_ptr->output(), &sub_ptr->output()});
    Node* mul_ptr = n_mul.get();
    graph.addNode(std::move(n_mul));

    // loss = sum(loss_vec), shape [1]
    auto n_sum = std::make_unique<Node>(&sum_op, std::vector<Tensor*>{&mul_ptr->output()});
    Node* sum_ptr = n_sum.get();
    graph.addNode(std::move(n_sum));
    
    // 收集所有参数
    auto p1 = fc1.parameter();
    auto p2 = fc2.parameter();
    std::vector<Parameter*> all_params;
    all_params.insert(all_params.end(), p1.begin(), p1.end());
    all_params.insert(all_params.end(), p2.begin(), p2.end());
    
    // 定义优化器
    SGD optimizer(all_params, 1.0);
    Executor executor(graph);

    for (int step = 0; step < 2000; ++step) {
        executor.zeroGrad();
        optimizer.zeroGrad();

        executor.forward();
        Tensor& loss = sum_ptr->output();
        executor.backward(loss);
        optimizer.step();

        if (step % 100 == 0) {
            std::cout << "step " << step
                      << " | loss = " << loss.value()[0]
                      << " | output = [";
            for (auto v : sig2_ptr->output().value())
                std::cout << v << ", ";
            std::cout << "]" << std::endl;
        }
    }
    
    return 0;
}