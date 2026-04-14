#include "optimizer.h"
#include <cassert>

void SGD::step() {
    for (auto& p : params_) {
        const auto& grad = p->grad();
        auto val = p->value();  // 拷贝，updateValue 需要
        auto& vel = velocity_[p.get()];

        if (momentum_ == 0.0) {
            // 普通SGD
            for (size_t i = 0; i < val.size(); ++i) {
                val[i] -= lr_ * grad[i];
            }
        } else {
            // SGD + Momentum
            for (size_t i = 0; i < val.size(); ++i) {
                vel[i] = momentum_ * vel[i] + grad[i];
                val[i] -= lr_ * vel[i];
            }
        }
        
        p->updateValue(val); // 这里优化器只更新参数值，不动 grad.
    }
}

void SGD::zeroGrad() {
    for (auto& p : params_) {
        p->zeroGrad();
    }
}