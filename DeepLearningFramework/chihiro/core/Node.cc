#include "Node.h"
#include <iostream>

void Node::forward() {
    op_->forward(inputs_, output_);
}

void Node::backward() {
    op_->backward(inputs_, output_);
}