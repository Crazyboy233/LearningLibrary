#include "op.h"
#include <iostream>

void AddOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    std::vector<double> x = input[0]->value();
    std::vector<double> y = input[1]->value();
    
    std::vector<double> result;
    for (int i = 0; i < x.size() && i < y.size(); ++i) {
        result.push_back(x[i] + y[i]);
    }
    output.setValue(result);
}

void AddOp::backward(const std::vector<Tensor*>& input, Tensor& output) {
    std::vector<double> grad = output.grad();

    input[0]->addGrad(grad);
    input[1]->addGrad(grad);
}


void MulOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    // std::cout << "begin MulOp forward" << std::endl;
    std::vector<double> x = input[0]->value();
    std::vector<double> y = input[1]->value();
    // std::cout << "获取 mulop 的二元操作数" << std::endl;
    std::vector<double> result;
    for (int i = 0; i < x.size() && i < y.size(); ++i) {
        // std::cout << "mulop forward 循环计算值" << std::endl;
        result.push_back(x[i] * y[i]);
        // std::cout << result[i] << std::endl;
    }
    output.setValue(result);
    // std::cout << "MulOp forward success" << std::endl;
}

void MulOp::backward(const std::vector<Tensor*>& input, Tensor& output) {
    // std::cout << "begin MulOp backward" << std::endl;
    std::vector<double> grad = output.grad();

    std::vector<double> x = input[0]->value();
    std::vector<double> y = input[1]->value();
    
    std::vector<double> result1;
    std::vector<double> result2;

    // std::cout << "x 的大小" << x.size() 
    //         << ", y 的大小" << y.size() 
    //         << ", grad 的大小" << grad.size() << std::endl;

    // 【优化】更清晰的空梯度判断
    if (grad.empty()) {
        // std::cout << "MulOp 空梯度，跳过计算" << std::endl;
        return;
    }
    
    for (int i = 0; i < x.size() && i < y.size() && grad.size(); ++i) {
        // std::cout << "mulop backward循环计算值" << std::endl;
        // std::cout << grad[i] << " " << y[i] << " " << x[i] << std::endl;
        result1.push_back(grad[i] * y[i]);
        result2.push_back(grad[i] * x[i]);
        // std::cout << result1[i] << std::endl;
        // std::cout << result2[i] << std::endl;
    }

    input[0]->addGrad(result1);
    input[1]->addGrad(result2);
    // std::cout << "MulOp backward success" << std::endl;
}

void SubOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    std::vector<double> x = input[0]->value();
    std::vector<double> y = input[1]->value();

    std::vector<double> result;
    for (int i = 0; i < x.size() && i < y.size(); ++i) {
        result.push_back(x[i] - y[i]);
    }

    output.setValue(result);
}

void SubOp::backward(const std::vector<Tensor*>& input, Tensor& output) {
    // std::cout << "begin SubOp backward" << std::endl;
    std::vector<double> grad = output.grad();
    
    // 【修复】梯度为空直接返回，不执行后续逻辑
    if (grad.empty()) {
        return;
    }

    std::vector<double> result;
    for (int i = 0; i < grad.size(); ++i) {
        result.push_back(grad[i] * -1);
    }

    input[0]->addGrad(grad);
    input[1]->addGrad(result);
    // std::cout << "SubOp backward success" << std::endl;
}

void SumOp::forward(const std::vector<Tensor*>& inputs, Tensor& output) {
    // std::cout << "begin SumOp::forward " << std::endl;
    const std::vector<double>& x = inputs[0]->value();
    double sum = 0.0;
    for (double v : x) {
        sum += v;
    }
    // std::cout << "SumOp::forward sum = " << std::endl;
    output.setValue(std::vector<double>{sum}); // scalar
}

void SumOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const std::vector<double>& grad_out = output.grad();
    Tensor& x = *inputs[0];

    std::vector<double> grad_x(x.value().size(), grad_out[0]);
    x.addGrad(grad_x);
}