#pragma once
#include "module.h"

/*
==============================================================
    Linear - 全连接层
    
    forward : y = x @ w + b
        x : [batch, in_features]
        w : [in_features, out_features] (requires_grad=true)
        b : [1, out_features]           (requires_grad=true)
        y : [batch, out_features]

    权重初始化：
        W — He 初始化（适合 ReLU 激活）
             W ~ N(0, sqrt(2 / in_features))
        b — 全零
 
    参数访问：
        parameters() 返回 {W_, b_}
        W() / b()    直接访问权重/偏置
==============================================================
*/
class Linear : public Module {
public:
    /*
        in_features  : 输入维度
        out_features : 输出维度
        seed         : 随机种子，默认 42，传 0 则用随机设备
    */
    Linear(size_t in_features, size_t out_features, unsigned seed = 42);

    TensorPtr forward(const TensorPtr& x) override;

    std::vector<TensorPtr> parameters() override { return {W_, b_}; }

    std::string name() const override { return "Linear"; }

    // 参数访问
    TensorPtr W() const { return W_; }
    TensorPtr b() const { return b_; }
    size_t inFeatures()  const { return in_features_; }
    size_t outFeatures() const { return out_features_; }

private:
    size_t in_features_;
    size_t out_features_;
 
    TensorPtr W_;   // [in_features, out_features]
    TensorPtr b_;   // [1, out_features]
};