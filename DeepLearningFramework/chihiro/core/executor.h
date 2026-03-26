#pragma once

#include "Graph.h"

class Executor{
public:
    explicit Executor(Graph& graph) 
        : graph_(&graph), order_(graph.topoSort()) {}

    void forward();
    void backward(Tensor& loss);

    void zeroGrad();
private:
    Graph* graph_;
    std::vector<Node*> order_;
};