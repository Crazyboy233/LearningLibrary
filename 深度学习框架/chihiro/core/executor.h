#pragma once

#include "Graph.h"

class Executor{
public:
    Executor() {}
    ~Executor() {}

    void run(Graph& graph) {
        auto order = graph.topoSort();
        for (auto& node : order) {
            node->compute();
        }
    }
};