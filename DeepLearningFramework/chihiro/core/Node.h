#pragma once

#include "op.h"

#include <vector>

class Node{
public:
    Node(Op* op, const std::vector<Tensor*>& inputs)
        :op_(op), inputs_(inputs) 
    {
        // 应该在构造时就指定producer，而不是compute时。因为排序发生在compute之前。
        output_.setProducer(this);
    }
    
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    // void compute(){
    //     op_->compute(inputs_, output_)
    // }

    Op* op() const { return op_; }
    Tensor& output() { return output_; }
    const Tensor& output() const { return output_; }
    const std::vector<Tensor*> inputs() { return inputs_; }

    void forward();
    void backward();
private:
    Op* op_;
    std::vector<Tensor*> inputs_;
    Tensor output_;
};