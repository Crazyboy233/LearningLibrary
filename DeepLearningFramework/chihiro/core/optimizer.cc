#include "optimizer.h"

void SGD::step() {
    for (auto& p : params_) {
        double new_value = p->value() - lr_ * p->grad();
        p->setValue(new_value);
    }
}

void SGD::zeroGrad() {
    for (auto& p : params_) {
        p->zeroGrad();
    }
}