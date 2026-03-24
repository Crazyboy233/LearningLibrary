#pragma once
#include <vector>
#include <iostream>

class Node;

class Tensor{
public:
    Tensor() {}

    explicit Tensor(const std::vector<double> value){
        value_ = value;
        grad_.resize(value_.size(), 0.0);
    }
    
    ~Tensor(){}

    std::vector<double> value() { return value_; }
    void setValue(const std::vector<double> value) {
        // 这里同步整个Tensor的状态, tensor 是一个整体
        value_ = value;

        if (grad_.size() != value_.size()) {
            grad_.assign(value_.size(), 0.0);
        }
    }

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
    Node* producer_ = nullptr;
    std::vector<double> value_;
    std::vector<double> grad_;
};