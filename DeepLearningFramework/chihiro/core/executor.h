#pragma once

#include "Graph.h"

class Executor{
public:
    explicit Executor(Graph& graph) 
        : graph_(&graph), order_(graph.topoSort()) {}

    void forward();
    void backward(Tensor& loss);

    void zeroGrad();

    void rebuildOrder();
private:
    Graph* graph_;
    std::vector<Node*> order_;
};