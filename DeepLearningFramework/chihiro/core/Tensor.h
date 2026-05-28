#pragma once
#include "grad_fn.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <functional>

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
    Shape 工具函数（全局，供 ops / grad_fn 共用）
==============================================================
*/

// 返回 Tensor 中元素的总个数
inline size_t shapeNumel(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    
    size_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    return n;
}

// 行主序 strides：strides[i] = product(shape[i+1..end])
// 维度到内存地址的换算
inline std::vector<size_t> shapeStrides(const std::vector<size_t>& shape) {
    size_t ndim = shape.size();
    std::vector<size_t> st(ndim, 1);
    for(int i = (int)ndim - 2; i >= 0; --i) {
        st[i] = st[i + 1] * shape[i + 1];
    }
    return st;
}

/*
    broadcast_shape：末尾对齐，每维取 max
    [2,1,4] 和 [3,4] → [2,3,4]
*/
inline std::vector<size_t> broadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t ndim = std::max(a.size(), b.size());
    std::vector<size_t> out(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        int ia = (int)a.size() - 1 - (int)i;
        int ib = (int)b.size() - 1 - (int)i;
        size_t da = (ia >= 0) ? a[ia] : 1;
        size_t db = (ib >= 0) ? b[ib] : 1;
        if (da != db && da != 1 && db != 1) {
            throw std::runtime_error("broadcastShape: incompatible dims");
        }
        out[ndim - 1 - i] = std::max(da, db);
    }
    return out;
}

/*
    reduceTo：将 grad（大 shape）reduce 回 target_shape（小 shape）
    broadcast 反向时，凡被扩展的维度都要 sum 折叠
    实现参考：../doc/function.md
*/
inline std::vector<double> reduceTo(const std::vector<double>& grad,
                                    const std::vector<size_t>& grad_shape,
                                    const std::vector<size_t>& target_shape) {
    size_t ndim = grad_shape.size();
    // target 左补 1 对齐
    std::vector<size_t> ts(ndim, 1);
    size_t offset = ndim - target_shape.size();
    for (size_t i = 0; i < target_shape.size(); ++i) {
        ts[offset + i] = target_shape[i];
    }

    size_t total = shapeNumel(grad_shape);
    size_t out_total = shapeNumel(ts);
    auto out_st = shapeStrides(ts);

    std::vector<double> result(out_total, 0.0);
    std::vector<size_t> idx(ndim);

    for (size_t flat = 0; flat < total; ++flat) {
        // 把展平的索引 → 还原成多维坐标
        size_t tmp = flat;
        for (int d = (int)ndim - 1; d >= 0; --d) {
            idx[d] = tmp % grad_shape[d];
            tmp /=  grad_shape[d];
        }
        // 广播处理：把多维坐标 → 映射到目标形状的索引
        size_t res_flat = 0;
        for (size_t d = 0; d < ndim; ++d) {
            size_t i = (ts[d] == 1) ? 0 : idx[d];
            res_flat += i * out_st[d];
        }
        result[res_flat] += grad[flat];
    }
    return result;
}
 
// 支持负维度索引
inline int normalizeDim(int dim, int ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::out_of_range("dim out of range");
    return dim;
}


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
    size_t numel() const { return value_.size(); }
    size_t size() const { return numel(); } // 兼容旧接口
    
    // 取最后两维（matmul 等使用）
    size_t rows() const { 
        assert(shape_.size() >= 2); 
        return shape_[shape_.size() - 2]; 
    }

    size_t cols() const { 
        assert(!shape_.empty());
        return shape_.back();
    }
    
    std::vector<size_t> strides() const { return shapeStrides(shape_); }

    // --------------- 梯度 ---------------
    const std::vector<double>& grad() const { return grad_; }
    bool requireGrad() const { return requires_grad_; }

    void zeroGrad() {
        grad_.assign(value_.size(), 0.0);
    }
    
    // --------------- 累加梯度 ---------------
    void addGrad(const std::vector<double>& grad) {
        assert(!grad.empty());
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