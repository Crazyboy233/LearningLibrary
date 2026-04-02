#include "../core/executor.h"
#include "../core/parameter.h"
#include "../core/optimizer.h"

#include <functional>
#include <vector>
#include <string>
#include <iostream>

// 编译命令
// g++ test/05_test3.cpp ./core/*.cc -I./core/

/* 该测试是基于 05_test2.cpp 增加了如下内容：
    测试 MatMulOp。
*/ 

int main() {
    /*
        x 是输入，是已知数据，比如：用户特征，房屋面积，一个数字。是无法修改的数据
        target 是目标，是正确答案，比如：你想让模型输出的值，房价，标签（label）
        w 是参数(Parameter)，是模型需要学习的东西，这是唯一可以改变的。
        
        y 是预测值(Prediction)，是模型当前输出结果。y = 模型输出 = f(w, x)。这里 y = w * x 是一个最简单的线性模型。
        loss 误差衡量(核心)，loss = 衡量 y 和 target 差多远。

            这里解释：为什么 loss 要这样算？
                误差 = y - target 其实就是计算模型当前输出值与目标值之间的差距。
                loss就是给这个误差值做一个正确的衡量标准。这里之所以loss是误差的平方，是因为如下：
                    如果 loss = y - target, 对于 batch，样本1: y-target = +10，样本2: y-target = -10，loss=0（明显不对）
                这里平方是经典的 L2 loss(MSE),当然，平方不是唯一选择，L1 loss ：|y - target|，特点：1、不怕 outlier。2、不怕梯度不连续（0点）

        整个测试过程是在构图，执行图的过程，对于框架来说，构图的公式可以随意写，但是对于训练不是，要有合适的loss计算公式，并且保证结果可以收敛。
    */

    // X 固定不训练
    Tensor x({2, 3}, {1.0, 2.0, 3.0, 
                    4.0, 5.0, 6.0}); // shape={2, 3}, 3 * 2 的矩阵

    // target
    Tensor target({2, 2}, {6.0, 6.0, 
                        15.0, 15.0});

    // W 是要训练的参数，初始全0
    Parameter w({3, 2}, {0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0});

    MulOp mul_op;
    SubOp sub_op;
    SumOp sum_op;
    MatMulOp matmul_op;

    Graph graph;
    
    // 将外部节点注册进graph，以便清理外部节点的grad，从而避免累计脏数据。
    graph.addInput(&x);
    graph.addInput(&target);

   // y = X * W, shape [2, 2], 这里是矩阵相乘
    auto matmul_node = std::make_unique<Node>(&matmul_op, std::vector<Tensor*>{&x, &w});
    Node* matmul_ptr = matmul_node.get();
    graph.addNode(std::move(matmul_node));

    // d = y - target, shape [2, 2]
    auto sub_node = std::make_unique<Node>(&sub_op, std::vector<Tensor*>{&matmul_ptr->output(), &target});
    Node* sub_ptr = sub_node.get();
    graph.addNode(std::move(sub_node));

    // loss_vec = d * d, shape [2, 2]
    auto loss_node = std::make_unique<Node>(&mul_op, std::vector<Tensor*>{&sub_ptr->output(), &sub_ptr->output()});
    Node* loss_ptr = loss_node.get();
    graph.addNode(std::move(loss_node));
    
    // loss = sum(loss_vec), shape [1]
    auto sum_node = std::make_unique<Node>(&sum_op, std::vector<Tensor*>{ &loss_ptr->output() });   // ⭐关键：输入是 loss_vec
    Node* sum_ptr = sum_node.get();
    graph.addNode(std::move(sum_node));

    std::cout << "图 构造完成" << std::endl;
    // 定义优化器
    SGD optimizer({&w}, 0.01);
    Executor executor(graph);

    for (int i = 0; i < 1000; ++i) { // 这里修改为1000步，不然无法收敛
        executor.zeroGrad();
        optimizer.zeroGrad();

        executor.forward();
        Tensor& loss = sum_ptr->output();
        executor.backward(loss);
        optimizer.step();

        std::cout << "step ：" << i;
        std::cout << "| grad = [ ";
        for (int i = 0; i < w.value().size(); i++) {
            std::cout << w.grad()[i] << ",";
                
        }
        std::cout << "]" << "| loss = [ " ;
        for (int i = 0; i < loss_ptr->output().value().size(); i++) {
            std::cout <<loss_ptr->output().value()[i] << ",";
                
        }
        std::cout << "]" << "| w = [ " ;
        for (int i = 0; i < w.value().size(); i++) {
            std::cout << w.value()[i] << ",";
                
        }
        std::cout << "]" << std::endl;
    }
    
    return 0;
}