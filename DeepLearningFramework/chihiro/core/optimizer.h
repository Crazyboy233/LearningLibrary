#pragma once
#include "parameter.h"

#include <vector>

class Optimizer {
public:
    explicit Optimizer(std::vector<TensorPtr> params) 
        :params_(std::move(params)) {}

    virtual ~Optimizer() = default;

    virtual void step() = 0;
    
    // 清理所有被管理的参数
    void zeroGrad() {
        for(auto& p : params_) {
            p->zeroGrad();
        }
    }

protected:
    std::vector<TensorPtr> params_;
};

/*
==============================================================
    SGD — 随机梯度下降（含可选 Momentum）
 
    无 Momentum：
        θ ← θ - lr * ∇θ
 
    有 Momentum（Polyak heavy ball）：
        v ← momentum * v + ∇θ
        θ ← θ - lr * v
 
    构造参数：
        params   : 需要优化的参数列表，通常来自 model.parameters()
        lr       : 学习率
        momentum : 动量系数，0.0 表示不使用（默认 0.0）
==============================================================
*/
class SGD : public Optimizer{
public:
    SGD(const std::vector<TensorPtr>& params, double lr, double momentum = 0.0)
        : Optimizer(std::move(params)), lr_(lr), momentum_(momentum) 
    {
        if (lr_ <= 0.0) {
            throw std::invalid_argument("SGD: lr must be > 0");
        }
        if (momentum_ < 0.0 || momentum_ >= 1.0) {
            throw std::invalid_argument("SGD: momentum must be in [0, 1)");
        }
        
        // 初始化 velocity buffer（momentum=0 时不使用，但预分配无妨）
        for (auto& p : params_) {
            velocity_[p.get()].assign(p->size(), 0.0);
        }
 
    }

    void step() override;
    void zeroGrad();
    
private:
    double lr_;
    double momentum_;

    // key: 原始指针（不拥有所有权，params_ 已持有 shared_ptr）
    std::unordered_map<Tensor*, std::vector<double>> velocity_;
};