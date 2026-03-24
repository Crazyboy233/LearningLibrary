#include "optimizer.h"

void SGD::step() {
    for (auto& p : params_) {
        std::vector<double> result;
        for (int i = 0; i < p->value().size() && i < p->grad().size(); ++i) {
            result.push_back(p->value()[i] - lr_ * p->grad()[i]); 
        }
        p->setValue(result);
    }
}

void SGD::zeroGrad() {
    for (auto& p : params_) {
        p->zeroGrad();
    }
}