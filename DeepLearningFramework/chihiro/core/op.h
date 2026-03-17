#pragma once

#include "Tensor.h"

#include <vector>
#include <string>

class Op{
public:
    virtual ~Op(){}
    virtual void compute(const std::vector<Tensor*>& input, Tensor& output) = 0;
    virtual void forward(const std::vector<Tensor*>& input, Tensor& output) = 0;
    virtual void backward(const std::vector<Tensor*>& input, Tensor& output) = 0;
    virtual const std::string name() const = 0;
};

class AddOp : public Op {
public:
    AddOp() {}
    ~AddOp() {}

    void compute(const std::vector<Tensor*>& input, Tensor& output) override {
        double result = input[0]->value() + input[1]->value();
        output.setValue(result);
    }
    void forward(const std::vector<Tensor*>& input, Tensor& output) override;
    void backward(const std::vector<Tensor*>& input, Tensor& output) override;

    const std::string name() const override {
        return "Add";
    }
};

class MulOp : public Op {
public:
    MulOp() {}
    ~MulOp() {}

    void compute(const std::vector<Tensor*>& input, Tensor& output) override {
        output.setValue(input[0]->value() * input[1]->value());
    }

    void forward(const std::vector<Tensor*>& input, Tensor& output) override;
    void backward(const std::vector<Tensor*>& input, Tensor& output) override;

    const std::string name() const override {
        return "Mul";
    }
};