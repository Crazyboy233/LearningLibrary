#include "Tensor.h"

#include <vector>
#include <cassert>

void Tensor::addGrad(std::vector<double> grad) {
    assert(value_.size() == grad_.size());
    assert(grad.size() == grad_.size());
    for (size_t i = 0; i < grad_.size(); ++i) {
        grad_[i] += grad[i];
    }
}

void Tensor::zeroGrad() {
    for (auto& grad : grad_) {
        grad = 0.0;
    }
}

void Tensor::updateValue(const std::vector<double>& value) {
    assert(value.size() == value_.size());
    value_ = value;
    
    grad_.assign(value_.size(), 0.0);   // 梯度清零
}   

void Tensor::setValue(const std::vector<size_t>& shape, const std::vector<double>& value) {
    // 这里同步整个Tensor的状态, tensor 是一个整体
    size_t total = 1;
    for (auto s : shape) {
        total *= s;
    }
    assert(total == value.size());  // shape 和数据量必须匹配。

    shape_ = shape;
    value_ = value;

    grad_.assign(value_.size(), 0.0);   // 梯度清零
}