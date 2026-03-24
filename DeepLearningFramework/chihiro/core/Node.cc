#include "Node.h"
#include <iostream>

void Node::forward() {
    // std::cout << "node forward 转发" << std::endl;
    op_->forward(inputs_, output_);
    // std::cout << "node forward 转发完成" << std::endl;
}

void Node::backward() {
    // std::cout << "node backward 转发" << std::endl;
    op_->backward(inputs_, output_);
    // std::cout << "node backward 转发完成" << std::endl;
}