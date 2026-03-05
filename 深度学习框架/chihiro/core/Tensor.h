#pragma once

class Node;

class Tensor{
public:
    Tensor() :value_(0.0) {}

    explicit Tensor(double value){
        value_ = value;
    }
    
    ~Tensor(){}

    double value() { return value_; }
    void setValue(const double& value) { value_ = value; }

    Node* producer() { return producer_; }
    void setProducer(Node* node) { producer_ = node; }

    double grad() { return grad_; }
    void addGrad(double grad) { grad_ += grad; }    // 累计梯度
    void zeroGrad() { grad_ = 0.0; }

private:
    double grad_;
    Node* producer_ = nullptr;
    double value_;
};