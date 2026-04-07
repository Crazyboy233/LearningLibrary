#pragma once
#include "Tensor.h"
#include "parameter.h"
#include "op.h"
#include "Graph.h"

class Linear {
public:
    Linear(size_t in_features, size_t out_fratures);
    Tensor& forward(Tensor& input, Graph& graph);

    std::vector<Parameter*> parameter() {
        return { w_.get(), b_.get() };
    }
private:
    size_t in_, out_;
    std::unique_ptr<Parameter> w_;
    std::unique_ptr<Parameter> b_;
    MatMulOp mat_mul_op_;
    AddOp add_op_;
};