#pragma once
#include "Tensor.h"

#include <vector>
#include <string>

/*
==============================================================
    Module — 所有网络层的基类
    
    子类需要实现：
        forward(x)    : 前向计算
        parameters() : 返回该层所有可训练参数（叶节点，requires_grad=true）
        name()       : 层名，用于打印

    典型用法：
        Linear fc(2, 4);
        auto y = fc.forward(x);
        loss->backward();
        sgd.step();
        sgd.zero_grad();
==============================================================
*/
class Module {
public:
    virtual ~Module() = default;

    // 前向计算
    virtual TensorPtr forward(const TensorPtr& x) = 0;

    // 返回该层所有可训练参数，供 optimizer 遍历
    virtual std::vector<TensorPtr> parameters() = 0;

    // 层名，子类可以覆盖
    virtual std::string name() const { return "Module"; }

    // 将所有参数梯度清零
    void zeroGrad() {
        for (auto& p : parameters()) {
            p->zeroGrad();
        }
    }
};