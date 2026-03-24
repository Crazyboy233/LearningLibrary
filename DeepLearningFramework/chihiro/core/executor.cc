#include "executor.h"
#include <iostream>
#include <cassert>

void Executor::forward(Graph& graph) {
    auto order = graph.topoSort();
    // std::cout << "topoSort success" << std::endl;
    for(auto& node : order) {
        // std::cout << "run graph" << std::endl;
        node->forward();
    }
}

void Executor::backward(Graph& graph, Tensor& loss) {
    if (loss.value().size() != 1) {
        throw std::runtime_error("Loss must be scalar. Use ReduceOp (e.g., sum or mean).");
    }

    auto order = graph.topoSort();
    // 加这行，打印节点总数和最后一个节点的op
    // std::cout << "拓扑排序节点数：" << order.size() 
    //       << " 最后一个节点op：" << order.back()->op()->name() << std::endl;
    // loss.addGrad(std::vector<double>(loss.value().size(), 1.0));  // dL/dL = 1
    // 优化：
    loss.addGrad({1.0});

    
    for(auto it = order.rbegin(); it != order.rend(); ++it) {
        // std::cout << "executor backward" << std::endl;
        Node* node = *it;
        if (node == nullptr) {
            // std::cout << "node 为 nullptr" << std::endl;
        }
        // std::cout << "该节点的 op：" << node->op()->name() << std::endl;
        // std::cout << "该节点的inputs (类型：vector<Tensor>)：" << std::endl;
        for(int i = 0; i < node->inputs().size(); ++i) {
            // std::cout << "[";
            for(int j = 0; j < node->inputs()[i]->value().size(); ++j) {
                // 每行是一个Tensor
                // std::cout << node->inputs()[i]->value()[j] << ", ";
            }
            // std::cout << "]" << std::endl;
        }
        node->backward();
        // std::cout << "node backward" << std::endl;
    }
}

void Executor::zeroGrad(Graph& graph) {
    for (auto& node : graph.nodes()) {
        node.get()->output().zeroGrad();
    }
}