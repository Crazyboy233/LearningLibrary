#include "optimizer.h"
#include <cassert>

void SGD::step() {
    for (auto& p : params_) {
        std::vector<double> result;
        assert(p->value().size() == p->grad().size());
        for (size_t i = 0; i < p->value().size(); ++i) {
            result.push_back(p->value()[i] - lr_ * p->grad()[i]); 
        }
        p->updateValue(result); // 这里优化器只更新参数值，不动 grad.
    }
}

void SGD::zeroGrad() {
    for (auto& p : params_) {
        p->zeroGrad();
    }
}