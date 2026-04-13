#pragma once
#include "grad_fn.h"
#include <memory>
#include <iostream>
#include <assert.h>

/*
==============================================================
    NoGradGuard - 推理时关闭计算图

    用法：
    {
        NoGradGuard guard;
        auto out = model.forward(x);    // 不创建 GradFn
    }
==============================================================
*/

// 注：当前阶段其实用不到该类，该类是用于推理优化。
class NoGradGuard {
public:
    NoGradGuard() { enabled_ = false; }
    ~NoGradGuard() { enabled_ = true; }
    static bool isEnabled() { return enabled_; }
private:
    static inline bool enabled_  = true;
};


/*
==============================================================
    Tensor

    动态图中 Tensor 是一等公民，通过 shared_ptr 传递。

    叶节点（输入数据 / Parameter）：
        grad_fn == nullptr;
        requires_grad_ 由用户指定

    中间节点（op 的输出）：
        grad_fn 指向生成它的 GradFn
        requires_grad_ 自动为 true（只要有一个输入 requires_grad）
==============================================================
*/
class Tensor : public std::enable_shared_from_this<Tensor>{
public:
    // ----------------- 工厂函数 -----------------

    // 创建叶节点：输入数据或者 Parameter
    static TensorPtr create(const std::vector<size_t> shape,
                        const std::vector<double> value,
                        bool requires_grad = false) {
        return std::shared_ptr<Tensor>(new Tensor(shape, value, requires_grad, nullptr));
    }

    // 创建中间节点：由 ops 函数内部调用
    static TensorPtr createFromOp(const std::vector<size_t> shape,
                                const std::vector<double> value,
                                std::shared_ptr<GradFn> grad_fn) {
        return std::shared_ptr<Tensor>(new Tensor(shape, value, /*requires_grad=*/true, grad_fn));
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // --------------- 数据访问 ---------------
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<double>& value() const { return value_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return value_.size(); }
    size_t rows() const { assert(shape_.size() == 2); return shape_[0]; }
    size_t cols() const { assert(shape_.size() == 2); return shape_[1]; }
    
    // --------------- 梯度 ---------------
    const std::vector<double> grad() const { return grad_; }
    bool requireGrad() const { return requires_grad_; }

    void zeroGrad() {
        grad_.assign(value_.size(), 0.0);
    }
    
    // --------------- 累加梯度 ---------------
    void addGrad(const std::vector<double>& grad) {
        if(grad.empty()) {
            grad_.assign(value_.size(), 0.0);
        }
        assert(grad.size() == grad_.size());
        for (size_t i = 0; i < grad_.size(); ++i) {
            grad_[i] += grad[i];
        }
    }

    // --------------- 参数更新（optimizer 使用） ---------------
    void updateValue(const std::vector<double>& new_value) {
        assert(new_value.size() == value_.size());
        value_ = new_value;
    }

    // --------------- 反向传播 ---------------
    // 只能在标量（size() == 1) Tensor 上调用，即 loss.backward()
    // 内部做拓扑排序 + 链式求导
    void backward();

    // --------------- 计算图信息 ---------------
    bool isLeaf() { return grad_fn_ == nullptr; }
    std::shared_ptr<GradFn> gradFn() { return grad_fn_; }

private:
    Tensor(std::vector<size_t> shape, std::vector<double> value, bool requires_grad, std::shared_ptr<GradFn> grad_fn)
        : shape_(shape)
        , value_(value)
        , requires_grad_(requires_grad)
        , grad_fn_(grad_fn)
    {
        grad_.assign(value.size(), 0.0);
    }

    std::vector<size_t> shape_;
    std::vector<double> value_;
    std::vector<double> grad_;
    bool requires_grad_ = false;
    std::shared_ptr<GradFn> grad_fn_ = nullptr;
};