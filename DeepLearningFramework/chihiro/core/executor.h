#pragma once

#include "Graph.h"

class Executor{
public:
    Executor() {}
    ~Executor() {}

    void forward(Graph& graph);
    void backward(Graph& graph, Tensor& loss);

    void zeroGrad(Graph& graph);
};