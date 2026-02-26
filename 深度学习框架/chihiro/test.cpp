#include <functional>
#include <vector>
#include <iostream>

struct Tensor {
    double value;
};

// Node 描述怎么计算
struct Node {
    std::function<void(const std::vector<Tensor*>& inputs, Tensor& output)> op;
    std::vector<Tensor*> inputs;
    Tensor output;
};

struct Graph {
    std::vector<Node*> nodes;
};

void execute(Graph& graph) {
    // 这里假设 graph.nodes 是拓扑排序好的。
    // TODO: excutor 增加拓扑排序功能。Executor 是否应该在 run 时排序？
    for (auto node : graph.nodes) {
        node->op(node->inputs, node->output);
    }
}

Node* make_add(Tensor* a, Tensor* b) {
    Node* node = new Node;
    node->inputs = {a, b};
    node->op = [](const std::vector<Tensor*>& inputs, Tensor& output){
        output.value = inputs[0]->value + inputs[1]->value;
    };
    return node;
}

Node* make_sub(Tensor*a, Tensor* b) {
    Node* node = new Node;
    node->inputs = {a, b};
    node->op = [](const std::vector<Tensor*>& inputs, Tensor& output) {
        output.value = inputs[0]->value - inputs[1]->value;
    };
    return node;
}

int main() {
    Tensor a{3.0};
    Tensor b{4.0};
    Tensor c{5.0};

    Graph g;

    Node* node1 = make_add(&a, &b);
    g.nodes.push_back(node1);
    Node* node2 = make_add(&node1->output, &c);
    g.nodes.push_back(node2);

    execute(g);

    std::cout << node2->output.value << std::endl;
}