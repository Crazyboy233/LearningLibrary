#pragma once

#include "Tensor.h"

#include <vector>
#include <string>

class Op{
public:
    virtual ~Op(){}
    virtual void forward(const std::vector<Tensor*>& input, Tensor& output) = 0;
    virtual void backward(const std::vector<Tensor*>& input, Tensor& output) = 0;
    virtual const std::string name() const = 0;
};

class AddOp : public Op {
public:
    AddOp() {}
    ~AddOp() {}

    void forward(const std::vector<Tensor*>& input, Tensor& output) override;
    void backward(const std::vector<Tensor*>& input, Tensor& output) override;

    const std::string name() const override { return "Add"; }
};

class MulOp : public Op {
public:
    MulOp() {}
    ~MulOp() {}

    void forward(const std::vector<Tensor*>& input, Tensor& output) override;
    void backward(const std::vector<Tensor*>& inputs, Tensor& output) override;

    const std::string name() const override { return "Mul"; }
};

class SubOp : public Op {
public:
    void forward(const std::vector<Tensor*>& input, Tensor& output) override;
    void backward(const std::vector<Tensor*>& input, Tensor& output) override;

    const std::string name() const override { return "Sub"; }
};

class SumOp : public Op {
public:
    void forward(const std::vector<Tensor*>& inputs, Tensor& output) override;
    void backward(const std::vector<Tensor*>& inputs, Tensor& output) override;

    const std::string name() const override { return "Sum"; }
};