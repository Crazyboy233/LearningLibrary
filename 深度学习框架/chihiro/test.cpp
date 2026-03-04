#include <functional>
#include <vector>
#include <string>
#include <iostream>

/*
===写在前面===
现在的结构：
    Graph = 结构
    Node = 依赖 + 输出
    Op = 计算规则
    Executor = 执行策略
*/ 

class Tensor{
public:
    Tensor() :value_(0.0) {}

    explicit Tensor(double value){
        value_ = value;
    }
    
    ~Tensor(){}

    double value() {
        return value_;
    }

    void setValue(const double& value) {
        value_ = value;
    }

private:
    double value_;
};

class Op{
public:
    virtual ~Op(){}
    virtual void compute(const std::vector<Tensor*>& input, Tensor& output) = 0;
    virtual const std::string name() const = 0;
};

class AddOp : public Op {
public:
    AddOp() {}
    ~AddOp() {}

    void compute(const std::vector<Tensor*>& input, Tensor& output) override {
        double result = input[0]->value() + input[1]->value();
        output.setValue(result);
    }

    const std::string name() const override {
        return "Add";
    }
};

class Node{
public:
    Node(Op* op, const std::vector<Tensor*>& inputs)
        :op_(op), inputs_(inputs) {}
    
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    void compute(){
        op_->compute(inputs_, output_);
    }

    Op* op() const {
        return op_;
    }

    Tensor& output() {
        return output_;
    }

    const Tensor& output() const {
        return output_;
    }

    const std::vector<Tensor*> inputs() {
        return inputs_;
    }
private:
    Op* op_;
    std::vector<Tensor*> inputs_;
    Tensor output_;
};

class Graph{
public:
    Graph() {}
    ~Graph() {}

    void addNode(std::unique_ptr<Node> node) {
        nodes_.push_back(std::move(node));
    }

    const std::vector<std::unique_ptr<Node>>& nodes() const  {
        return nodes_;
    }

private:
    std::vector<std::unique_ptr<Node>> nodes_;
};

class Executor{
public:
    Executor() {}
    ~Executor() {}

    void run(Graph& graph) {
        for (auto& node : graph.nodes()) {
            node->compute();
        }
    }
};

int main() {
    Tensor a{3.0};
    Tensor b{4.0};
    Tensor c{5.0};

    AddOp add_op;
    Graph graph;
    auto node = std::make_unique<Node>(&add_op, std::vector<Tensor*>{&a, &b});
    auto node2 = std::make_unique<Node>(&add_op, std::vector<Tensor*>{&a, &b});
    Node* node_ptr = node.get();
    Node* node2_ptr = node2.get();

    graph.addNode(std::move(node));
    graph.addNode(std::move(node2));

    Executor executor;
    executor.run(graph);

    std::cout << node_ptr->output().value() << std::endl;
    std::cout << node2_ptr->output().value() << std::endl;
}