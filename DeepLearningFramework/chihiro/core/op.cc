#include "op.h"
#include <iostream>
#include <cassert>

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
    std::vector<double> x = input[0]->value();
    std::vector<double> y = input[1]->value();
    std::vector<double> result;

    for (int i = 0; i < x.size() && i < y.size(); ++i) {
        result.push_back(x[i] * y[i]);
    }
    output.setValue(result);
}

void MulOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const auto& grad = output.grad();

    const auto& x = inputs[0]->value();
    const auto& y = inputs[1]->value();

    assert(x.size() == y.size() && x.size() == grad.size());
    
    if (inputs[0] == inputs[1]) {
        // d * d 的情况，梯度是 2 * grad * x
        std::vector<double> result;
        result.reserve(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result.push_back(2.0 * grad[i] * x[i]);
        }
        inputs[0]->addGrad(result);
    } else {
        std::vector<double> result1, result2;
        result1.reserve(x.size());
        result2.reserve(x.size());
        
        for (int i = 0; i < x.size(); ++i) {
            assert(x.size() == y.size() && x.size() == grad.size());

            result1.push_back(grad[i] * y[i]);
            result2.push_back(grad[i] * x[i]);
        }

        inputs[0]->addGrad(result1);
        inputs[1]->addGrad(result2);
    }
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
    std::vector<double> grad = output.grad();

    std::vector<double> result;
    for (int i = 0; i < grad.size(); ++i) {
        result.push_back(grad[i] * -1);
    }

    input[0]->addGrad(grad);
    input[1]->addGrad(result);
}

void SumOp::forward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const std::vector<double>& x = inputs[0]->value();
    double sum = 0.0;
    for (double v : x) {
        sum += v;
    }
    output.setValue(std::vector<double>{sum}); // scalar
}

void SumOp::backward(const std::vector<Tensor*>& inputs, Tensor& output) {
    const std::vector<double>& grad_out = output.grad();
    Tensor& x = *inputs[0];

    std::vector<double> grad_x(x.value().size(), grad_out[0]);
    x.addGrad(grad_x);
}