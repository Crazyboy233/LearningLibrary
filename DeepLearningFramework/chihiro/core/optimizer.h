#pragma once
#include "parameter.h"

#include <vector>
class SGD {
public:
    SGD(const std::vector<Parameter*>& params, double lr)
        : params_(params), lr_(lr) {}

    void step();
    void zeroGrad();
    
private:
    std::vector<Parameter*> params_;
    double lr_;
};