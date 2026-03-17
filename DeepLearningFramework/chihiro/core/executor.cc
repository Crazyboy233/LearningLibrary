#include "executor.h"

void Executor::forward(Graph& graph) {
    auto order = graph.topoSort();
    for(auto& node : order) {
        node->forward();
    }
}

void Executor::backward(Graph& graph, Tensor& loss) {
    auto order = graph.topoSort();
    loss.addGrad(1.0);  // dL/dL = 1

    for(auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* node = *it;
        node->backward();
    }
}