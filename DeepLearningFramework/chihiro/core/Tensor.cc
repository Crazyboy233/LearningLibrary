#include "Tensor.h"

#include <vector>
#include <cassert>

void Tensor::addGrad(std::vector<double> grad) {
    assert(value_.size() == grad_.size());
    for (int i = 0; i < grad_.size(); ++i) {
        grad_[i] += grad[i];
    }
}

void Tensor::zeroGrad() {
    for (auto& grad : grad_) {
        grad = 0.0;
    }
}