#include "Graph.h"

#include <unordered_map>
#include <queue>
#include <iostream>

std::vector<Node*> Graph::topoSort() {
    std::unordered_map<Node*, int> indegree;
    std::unordered_map<Node*, std::vector<Node*>> adj;   // 邻接表：key=节点，value=该节点指向的所有节点

    // 初始化所有节点入度为0
    for(auto& nodePtr : nodes_) {
        Node* node = nodePtr.get();
        indegree[node] = 0;
    }

    // 建图
    for(auto& nodePtr : nodes_) {
        Node* node = nodePtr.get();
        for(Tensor* input : node->inputs()) {
            // 获取生成该张量的节点（即当前节点的前置依赖节点）
            Node* producer = input->producer();
            // 如果producer为nullptr，则说明无前置节点
            if(producer == nullptr) {
                continue;
            }
            indegree[node]++;
            adj[producer].push_back(node);  // 邻接表：producer -> node（producer 指向 node）
        }
    }

    std::queue<Node*> q;
    for(auto& pair : indegree) {
        if(pair.second == 0) {
            q.push(pair.first);
        }
    }

    // 拓扑排序
    std::vector<Node*> result;
    while(!q.empty()) {
        Node* curr_node = q.front();
        q.pop();
        result.push_back(curr_node);

        for(Node* next_node : adj[curr_node]) {
            indegree[next_node]--;

            if(indegree[next_node] == 0) {
                q.push(next_node);
            }
        }
    }

    if(result.size() != nodes_.size()) {
        std::cout << "result size = " << result.size() << " nodes_ size = " << nodes_.size() << std::endl;
        throw std::runtime_error("Graph has cycle");
    }

    return result;
}