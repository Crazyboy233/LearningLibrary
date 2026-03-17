#include "op.h"

void AddOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    output.setValue(input[0]->value() + input[1]->value());
}

void AddOp::backward(const std::vector<Tensor*>& input, Tensor& output) {
    double grad = output.grad();

    input[0]->addGrad(grad);
    input[1]->addGrad(grad);
}


void MulOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    double x = input[0]->value();
    double y = input[1]->value();
    output.setValue(x * y);
}

void MulOp::backward(const std::vector<Tensor*>& input, Tensor& output) {
    double grad = output.grad();

    double x = input[0]->value();
    double y = input[1]->value();

    input[0]->addGrad(grad * y);
    input[1]->addGrad(grad * x);
}

void SubOp::forward(const std::vector<Tensor*>& input, Tensor& output) {
    output.setValue(input[0]->value() - input[1]->value());
}

void SubOp::backward(const std::vector<Tensor*>& input, Tensor& output) {
    double grad = output.grad();

    input[0]->addGrad(grad);
    input[1]->addGrad(grad * -1);
}