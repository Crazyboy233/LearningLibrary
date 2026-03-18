#pragma once

#include "Tensor.h"

class Parameter : public Tensor{
public:
    using Tensor::Tensor;   // 这里是使用了 Tensor 的构造函数

    bool isParameter() const { return true; }
};