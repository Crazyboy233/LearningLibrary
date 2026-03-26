#include "executor.h"
#include <iostream>
#include <cassert>

void Executor::forward() {
    // auto order = graph.topoSort();
    for(auto& node : order_) {
        node->forward();
    }
}

void Executor::backward(Tensor& loss) {
    if (loss.value().size() != 1) {
        throw std::runtime_error("Loss must be scalar. Use ReduceOp (e.g., sum or mean).");
    }

    // auto order = graph.topoSort();
    loss.addGrad({1.0});

    for(auto it = order_.rbegin(); it != order_.rend(); ++it) {
        Node* node = *it;
        node->backward();
    }
}
void Executor::zeroGrad() {
    for (auto& node : graph_->nodes()) {
        node.get()->output().zeroGrad();
    }
    for (auto& t : graph_->inputs()) {
        t->zeroGrad();
    }
}