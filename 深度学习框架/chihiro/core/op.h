#pragma once

#include "Tensor.h"

#include <vector>
#include <string>

class Op{
public:
    virtual ~Op(){}
    virtual void compute(const std::vector<Tensor*>& input, Tensor& output) = 0;
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

    const std::string name() const override {
        return "Add";
    }
};