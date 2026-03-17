#include "Tensor.h"

class Parameter : public Tensor{
public:
    using Tensor::Tensor;

    bool isParameter() const { return true; }
};