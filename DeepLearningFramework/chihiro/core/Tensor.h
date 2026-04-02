#pragma once
#include <vector>
#include <iostream>
#include <assert.h>

class Node;

class Tensor{
public:
    Tensor() = default;

    // 构造时指定 shape，data 按行主序（row-major）展开
    explicit Tensor(const std::vector<size_t>& shape, const std::vector<double>& value){
        shape_ = shape;
        value_ = value;
        grad_.resize(value_.size(), 0.0);
    }
    
    ~Tensor(){}

    const std::vector<double>& value() { return value_; }
    void updateValue(const std::vector<double>& value);
    void setValue(const std::vector<size_t>& shape, const std::vector<double>& value);

    const std::vector<size_t>& shape() const { return shape_; } 
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return value_.size(); }

    // 二维访问（主要给 MatMulOp 使用）
    size_t rows() const { assert(shape_.size() == 2); return shape_[0]; }
    size_t cols() const { assert(shape_.size() == 2); return shape_[1]; } 


    void resize(size_t n) {
        value_.resize(n);
        grad_.assign(n, 0.0);
    }

    Node* producer() { return producer_; }
    void setProducer(Node* node) { producer_ = node; }

    std::vector<double> grad() { return grad_; }
    void addGrad(std::vector<double> grad);    // 累计梯度
    void zeroGrad();

private:
    std::vector<size_t> shape_;
    std::vector<double> value_;
    std::vector<double> grad_;
    Node* producer_ = nullptr;

};