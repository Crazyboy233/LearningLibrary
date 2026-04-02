#include "../core/executor.h"
#include "../core/parameter.h"
#include "../core/optimizer.h"

#include <functional>
#include <vector>
#include <string>
#include <iostream>

// 编译命令
// g++ test/06_test.cpp ./core/*.cc -I./core/

/* 该测试是基于 05_test3.cpp 增加了如下内容：
    增加了 sigmod 和 ReLU，
    测试解决 XOR 问题：
    输入    输出
    0,0  →  0
    0,1  →  1
    1,0  →  1
    1,1  →  0

    收敛结果：output 输出为[0, 1, 1, 0]
*/ 

int main() {
    /*
        x 是输入，是已知数据，比如：用户特征，房屋面积，一个数字。是无法修改的数据
        target 是目标，是正确答案，比如：你想让模型输出的值，房价，标签（label）
        w 是参数(Parameter)，是模型需要学习的东西，这是唯一可以改变的。

        整个测试过程是在构图，执行图的过程，对于框架来说，构图的公式可以随意写，但是对于训练不是，要有合适的loss计算公式，并且保证结果可以收敛。
    */

    /* 注：该测试并不完美，所有数值均经过调试，切勿修改，否则可能无法收敛，n2 由原来的 ReLU 转为 sigmod。
        由于XOR 数据本身的特性：
        XOR 的第一个输入是 {0, 0}，乘以任何 W1 结果都是 0，
        经过 ReLU 还是 0，经过 W2 还是 0，
        Sigmoid(0) = 0.5，永远是 0.5，
        梯度也永远是 0，这个样本根本学不动。

        这里增加 bias，由于框架目前没有，先用一个 trick：给输入X加一列常数1, W1 增加一行：
    */
    Tensor X({4, 3}, {0.0, 0.0, 1.0,
                    0.0, 1.0, 1.0,
                    1.0, 0.0, 1.0,
                    1.0, 1.0, 1.0});

    Tensor target({4, 1}, {0.0, 1.0, 1.0, 0.0});
    
    // 参数初始化，不能全0，否则对称性导致所有神经元学到一样的东西
    Parameter W1({3, 4}, { 0.3,  0.7, -0.4,  0.6,
                       0.8, -0.3,  0.5, -0.7,
                      -0.6,  0.4,  0.2, -0.5});

    Parameter W2({4, 1}, { 0.8, -0.6,  0.5, -0.7});
    
    MatMulOp matmul_op;
    ReLUOp relu_op;
    SigmodOp sigmod_op;
    SubOp sub_op;
    MulOp mul_op;
    SumOp sum_op;

    Graph graph;
    graph.addInput(&X);
    graph.addInput(&target);

    // h = X * W1, shape [4, 4]
    auto n1 = std::make_unique<Node>(&matmul_op, std::vector<Tensor*>{&X, &W1});
    Node* n1_ptr = n1.get();
    graph.addNode(std::move(n1));

    // a = Sigmoid(h), shape [4, 4]
    auto n2 = std::make_unique<Node>(&sigmod_op, std::vector<Tensor*>{&n1_ptr->output()});
    Node* n2_ptr = n2.get();
    graph.addNode(std::move(n2));

    // y = a * W2, shape [4, 1]
    auto n3 = std::make_unique<Node>(&matmul_op, std::vector<Tensor*>{&n2_ptr->output(), &W2});
    Node* n3_ptr = n3.get();
    graph.addNode(std::move(n3));

    // out = Sigmoid(y), shape [4, 1]
    auto n4 = std::make_unique<Node>(&sigmod_op, std::vector<Tensor*>{&n3_ptr->output()});
    Node* n4_ptr = n4.get();
    graph.addNode(std::move(n4));

    // d = out - target, shape [4, 1]
    auto n5 = std::make_unique<Node>(&sub_op, std::vector<Tensor*>{&n4_ptr->output(), &target});
    Node* n5_ptr = n5.get();
    graph.addNode(std::move(n5));
    
    // loss_vec = d * d, shape [4, 1]
    auto n6 = std::make_unique<Node>(&mul_op, std::vector<Tensor*>{&n5_ptr->output(), &n5_ptr->output()});
    Node* n6_ptr = n6.get();
    graph.addNode(std::move(n6));

    // loss = sum(loss_vec), shape [1]
    auto n7 = std::make_unique<Node>(&sum_op, std::vector<Tensor*>{&n6_ptr->output()});
    Node* n7_ptr = n7.get();
    graph.addNode(std::move(n7));

    SGD optimizer({&W1, &W2}, 1.0);
    Executor executor(graph);
    
    for (int step = 0; step < 2000; ++step) {
        executor.zeroGrad();
        optimizer.zeroGrad();

        executor.forward();
        Tensor& loss = n7_ptr->output();
        executor.backward(loss);

        optimizer.step();

        if (step % 100 == 0) {
            std::cout << "step " << step << " | loss = " << loss.value()[0] << " | output = [";
            for (auto v : n4_ptr->output().value())
                std::cout << v << ", ";
            std::cout << "]" << std::endl;
        }
    }



    return 0;
}

/*
该测试的图如下：
    h = X · W1                  # MatMul,  [4,2] · [2,4] = [4,4]
    a = ReLU(h)                 # 逐元素,  [4,4]
    y = a · W2                  # MatMul,  [4,4] · [4,1] = [4,1]
    out = σ(y)                  # 逐元素,  [4,1]
    d = out - target            # 逐元素,  [4,1]
    loss_vec = d * d            # 逐元素,  [4,1]
    loss = sum(loss_vec)        # 标量,    [1]

    backward 反向流程
    ∂loss/∂loss_vec = 1                         # SumOp backward
    ∂loss/∂d        = 2 * d                     # MulOp backward
    ∂loss/∂out      = 2 * d                     # SubOp backward
    ∂loss/∂y        = 2 * d * out * (1 - out)   # Sigmoid backward
    ∂loss/∂W2       = aᵀ · ∂loss/∂y            # MatMul backward
    ∂loss/∂a        = ∂loss/∂y · W2ᵀ           # MatMul backward
    ∂loss/∂h        = ∂loss/∂a * 1(h>0)        # ReLU backward
    ∂loss/∂W1       = Xᵀ · ∂loss/∂h            # MatMul backward
*/