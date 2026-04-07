#pragma once

#include "Node.h"

class Graph{
public:
    Graph() {}
    ~Graph() {}

    void addNode(std::unique_ptr<Node> node) {
        nodes_.push_back(std::move(node));
        graph_dirty_ = true;    // 标记图已被修改
    }

    const std::vector<std::unique_ptr<Node>>& nodes() const  {
        return nodes_;
    }

    // 这里把排序好的数据返回出来，Executor 使用 拓扑序的 Node* 来执行
    // nodes_ 不建议通过排序修改，这样会导致图结构发生变化
    std::vector<Node*> topoSort();

    // 注册外部输入数据，以便对外部数据进行zerograd。
    void addInput(Tensor* t);
    std::vector<Tensor*> inputs();

    // 验证所有输入 Tensor 是否已被赋值
    bool validateInputs() const;
    // 获取未赋值的输入 Tensor 列表
    std::vector<Tensor*> getUninitializedInputs() const;

    bool isDirty() const { return graph_dirty_; }
    void clearDirty() { graph_dirty_ = false; }
private:
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<Tensor*> inputs_;
    bool graph_dirty_ = false;
};