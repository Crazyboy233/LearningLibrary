#include "ops.h"
#include "grad_fn.h"
#include <cassert>
#include <cmath>

/*
============================================================
    辅助：判断至少一个输入需要梯度
    只要有一个输入 requires_grad，输出就需要记录计算图
============================================================
*/
static bool anyRequiresGrad(const std::vector<TensorPtr>& inputs) {
    for (auto& t : inputs) {
        if((t)->requireGrad()) {
            return true;
        }
    }
    return false;
}

/*
    broadcast forward：把 a / b 扩展到 out_shape，逐元素运算
    返回两个已 broadcast 的值向量（用于 MulBackward 保存）
*/
static void broadcastExpand(const std::vector<double>& src,
                            const std::vector<size_t>& src_shape,
                            const std::vector<size_t>& out_shape,
                            std::vector<double>& dst) {
    
    size_t ndim = out_shape.size();
    size_t total = shapeNumel(out_shape);
    dst.resize(total);

    // src_shape 左补 1 对齐
    std::vector<size_t> ss(ndim, 1);
    size_t offset = ndim - src_shape.size();
    for (size_t i = 0; i < src_shape.size(); ++i) {
        ss[offset + i] = src_shape[i];
    }

    auto out_st = shapeStrides(out_shape);
    auto ss_st = shapeStrides(ss);

    std::vector<size_t> idx(ndim);
    for (size_t flat = 0; flat < total; ++flat) {
        size_t tmp = flat;
        for (int d = (int)ndim - 1; d >= 0; --d) {
            idx[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        size_t src_flat = 0;
        for (size_t d = 0; d < ndim; ++d) {
            size_t i = (ss[d] == 1) ? 0 : idx[d];   // ss[d] == 1 说明这一维是 broadcast 出来的
            src_flat += i * ss_st[d];
        }
        dst[flat] = src[src_flat];
    }
}

/*
============================================================
    add
============================================================
*/
TensorPtr ops::add(const TensorPtr& a, const TensorPtr& b) {
    auto out_shape = broadcastShape(a->shape(), b->shape());
    size_t n = shapeNumel(out_shape);
    
    std::vector<double> va, vb;
    broadcastExpand(a->value(), a->shape(), out_shape, va);
    broadcastExpand(b->value(), b->shape(), out_shape, vb);

    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = va[i] + vb[i];
    }

    if(!anyRequiresGrad({a, b})) {
        return Tensor::create(out_shape, result);
    }

    auto fn = std::make_shared<AddBackward>();
    fn->shapeA_ = a->shape();
    fn->shapeB_ = b->shape();
    fn->shapeOut_ = out_shape;
    fn->saved_inputs_ = {a, b};

    return Tensor::createFromOp(out_shape, result, fn);
}

/*
============================================================
    sub
============================================================
*/
TensorPtr ops::sub(const TensorPtr& a, const TensorPtr& b) {
    auto out_shape = broadcastShape(a->shape(), b->shape());
    size_t n = shapeNumel(out_shape);

    std::vector<double> va, vb;
    broadcastExpand(a->value(), a->shape(), out_shape, va);
    broadcastExpand(b->value(), b->shape(), out_shape, vb);
    
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = va[i] - vb[i];
    }


    if(!anyRequiresGrad({a, b})) {
        return Tensor::create(out_shape, result);
    }

    auto fn = std::make_shared<SubBackward>();
    fn->shapeA_ = a->shape();
    fn->shapeB_ = b->shape();
    fn->shapeOut_ = out_shape;
    fn->saved_inputs_ = {a, b};

    return Tensor::createFromOp(out_shape, result, fn);
}

/*
============================================================
    mul
============================================================
*/
TensorPtr ops::mul(const TensorPtr& a, const TensorPtr& b) {
    auto out_shape = broadcastShape(a->shape(), b->shape());
    size_t n = shapeNumel(out_shape);

    std::vector<double> va, vb;
    broadcastExpand(a->value(), a->shape(), out_shape, va);
    broadcastExpand(b->value(), b->shape(), out_shape, vb);

    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = va[i] * vb[i];
    }

    if(!anyRequiresGrad({a, b})) {
        return Tensor::create(out_shape, result);
    }

    auto fn = std::make_shared<MulBackward>();
    fn->same_tensor_ = (a.get() == b.get());
    fn->shapeA_ = a->shape();
    fn->shapeB_ = b->shape();
    fn->shapeOut_ = out_shape;
    fn->x_val_ = va;    // 已 broadcast 的值，backward 直接用
    fn->y_val_ = vb;
    if (fn->same_tensor_) {
        fn->saved_inputs_ = {a};
    } else {
        fn->saved_inputs_ = {a, b};
    }
    
    return Tensor::createFromOp(out_shape, result, fn);
}

/*
============================================================
    matmul  (batched)
    A: [..., m, k]   B: [..., k, n]   →   C: [..., m, n]
============================================================
*/
TensorPtr ops::matmul(const TensorPtr& a, const TensorPtr& b) {
    assert(a->ndim() >= 2 && b->ndim() >= 2);

    size_t m = a->rows();
    size_t k = a->cols();
    size_t n = b->cols();
    assert(k == b->rows());

    // batch shape = broadcast(a.shape[:-2], b.shape[:-2])
    std::vector<size_t> bA(a->shape().begin(), a->shape().end() - 2);
    std::vector<size_t> bB(b->shape().begin(), b->shape().end() - 2);
    auto batch_shape = broadcastShape(bA.empty() ? std::vector<size_t>{} : bA,
                                    bB.empty() ? std::vector<size_t>{} : bB);
    
    size_t batch = shapeNumel(batch_shape.empty() ? std::vector<size_t>{1} : batch_shape);
    
    std::vector<size_t> out_shape = batch_shape;
    out_shape.push_back(m);
    out_shape.push_back(n);
    
    size_t out_total = shapeNumel(out_shape);
    std::vector<double> result(out_total, 0.0);

    // Expand A and B to full batch shape for simplicity
    std::vector<size_t> A_full_shape = batch_shape;
    A_full_shape.push_back(m);
    A_full_shape.push_back(k);
    std::vector<size_t> B_full_shape = batch_shape;
    B_full_shape.push_back(k);
    B_full_shape.push_back(n);

    std::vector<double> A_exp, B_exp;
    broadcastExpand(a->value(), a->shape(), A_full_shape, A_exp);
    broadcastExpand(b->value(), b->shape(), B_full_shape, B_exp);

    for(size_t b_idx = 0; b_idx < batch; ++b_idx){
        size_t A_off = b_idx * m * k;
        size_t B_off = b_idx * k * n;
        size_t C_off = b_idx * m * n;

        for(size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                for (size_t j = 0; j < n; ++j) {
                    result[C_off + i * n + j] += A_exp[A_off + i * k + p] * B_exp[B_off + p * n + j];
                }
            }
        }
    }

    if (!anyRequiresGrad({a, b})) {
        return Tensor::create(out_shape, result);
    }

    auto fn = std::make_shared<MatMulBackward>();
    fn->A_val_ = A_exp;     // 保存 broadcast 之后的值
    fn->B_val_ = B_exp;
    fn->shapeA_ = A_full_shape;
    fn->shapeB_ = B_full_shape;
    fn->m_ = m;
    fn->k_ = k;
    fn->n_ = n;
    fn->saved_inputs_ = {a, b};

    return Tensor::createFromOp(out_shape, result, fn);
}

/*
============================================================
    relu : max(0, x)
============================================================
*/
TensorPtr ops::relu(const TensorPtr& a) {
    size_t n = a->numel();
    
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = (a->value()[i] > 0.0) ? a->value()[i] : 0.0;
    }

    if (!anyRequiresGrad({a})) {
        return Tensor::create(a->shape(), result);
    }

    auto fn = std::make_shared<ReLUBackward>();
    fn->x_val_ = a->value();
    fn->saved_inputs_ = {a};

    return Tensor::createFromOp(a->shape(), result, fn);
}

/*
============================================================
    sigmoid : 1 / (1 + exp(-x))
============================================================
*/
TensorPtr ops::sigmoid(const TensorPtr& a) {
    size_t n = a->numel();

    std::vector<double> result(n);

    for (size_t i = 0; i < n; ++i) {
        result[i] = 1.0 / (1.0 + std::exp(-a->value()[i]));
    }

    if(!anyRequiresGrad({a})) {
        return Tensor::create(a->shape(), result);
    }

    auto fn = std::make_shared<SigmoidBackward>();
    fn->y_val_ = result;    // 这里保存输出值，反向要使用
    fn->saved_inputs_ = {a};

    return Tensor::createFromOp(a->shape(), result, fn);
}

/*
============================================================
    sum : 全局
============================================================
*/
TensorPtr ops::sum(const TensorPtr& a) {
    double s = 0.0;
    for (auto v : a->value()) {
        s += v;
    }

    if (!anyRequiresGrad({a})) {
        return Tensor::create({1}, {s});
    }

    auto fn = std::make_shared<SumBackward>();
    fn->input_shape_ = a->shape();
    fn->sum_dim_ = -1;
    fn->saved_inputs_ = {a};

    return Tensor::createFromOp({1}, {s}, fn);
}
/*
============================================================
    sum : (沿 dim)
============================================================
*/
TensorPtr ops::sum(const TensorPtr& a, int dim, bool keepdim) {
    // 因为支持负维度，这里拿到真正的dim
    int d = normalizeDim(dim, (int)a->ndim());

    std::vector<size_t> out_shape;
    for (size_t i = 0; i < a->ndim(); ++i) {
        if ((int)i == d) {
            if (keepdim) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(a->shape()[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    size_t in_total = a->numel();
    size_t out_total = shapeNumel(out_shape);

    // keepdim=true 版本的 out_shape，用于 stride 计算
    std::vector<size_t> out_kd_shape = a->shape();
    out_kd_shape[d] = 1;
    auto out_kd_st = shapeStrides(out_kd_shape);

    std::vector<double> result(out_total, 0.0);

    std::vector<size_t> idx(a->ndim());
    // 相当于在一维打平的数组中操作，通过shapeStrides拿到需要的坐标
    for (size_t flat = 0; flat < in_total; ++flat) {
        size_t tmp = flat;
        for (int dd = (int)a->ndim() - 1; dd >= 0; --dd) {
            idx[dd] = tmp % a->shape()[dd];
            tmp /= a->shape()[dd];
        }

        // 映射到输出（去掉 d 维或置 0）
        size_t out_flat = 0;
        if (keepdim) {
            for (size_t dd = 0; dd < a->ndim(); ++dd) {
                size_t i = ((int)dd == d) ? 0 : idx[dd];
                out_flat += i * out_kd_st[dd];
            }
        } else {
            // out_shape 去掉了第 d 维
            size_t stride = 1;
            for (int dd = (int)a->ndim() - 1; dd >= 0; --dd) {
                if ((int)dd == d) continue;
                out_flat += idx[dd] * stride;
                stride *= a->shape()[dd];
            }
            // 上面的 stride 计算是反向的，重新计算一次
            out_flat = 0;
            // 用 out_shape 的 strides
            auto ost = shapeStrides(out_shape);
            size_t oi = 0;
            for (size_t dd = 0; dd < a->ndim(); ++dd) {
                if ((int)dd == d) continue;
                out_flat += idx[dd] * ost[oi];
                oi++;
            }
        }
        result[out_flat] += a->value()[flat];
    }

    if (!anyRequiresGrad({a})) {
        return Tensor::create(out_shape, result);
    }

    auto fn = std::make_shared<SumBackward>();
    fn->input_shape_ = a->shape();
    fn->sum_dim_ = d;
    fn->keepdim_ = keepdim;
    fn->saved_inputs_ = {a};

    return Tensor::createFromOp(out_shape, result, fn);
}

/*
============================================================
    bce_with_logits_loss : 数值稳定的 BCE，接受 sigmoid 之前的 logits

    forward 用 log-sum-exp trick 避免 log(0)：
        L_i = max(x,0) - x*y + log(1 + e^{-|x|})
    mean 规约到标量 {1}

    使用时不需要提前调用 ops::sigmoid，直接传 fc 的输出
============================================================
*/
TensorPtr ops::bceWithLogitsLoss(const TensorPtr& logits, const TensorPtr& target) {
    assert(logits->shape() == target->shape());

    size_t n = logits->size();

    // forward：numerically stable BCE
    std::vector<double> sig_val(n);
    double loss_val = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double x = logits->value()[i];
        double y = target->value()[i];
        // log(1 + e^{-|x|}) 不会下溢
        loss_val += std::max(x, 0.0) - x * y + std::log(1 + std::exp(-std::abs(x)));
        sig_val[i] = 1.0 / (1.0 + std::exp(-x));    // 反向需要
    }

    loss_val /= static_cast<double>(n);

    if(!anyRequiresGrad({logits})) {
        return Tensor::create({1}, {loss_val});
    }

    auto fn = std::make_shared<BCEWithLogitsBackward>();
    fn->sigmoid_val_ = sig_val;
    fn->target_val_ = target->value();
    fn->n_ = n;
    fn->saved_inputs_ = {logits};   // target 无梯度，不入图

    return Tensor::createFromOp({1}, {loss_val}, fn);
}

/*
============================================================
    crossEntropyLoss
    logits: [N, C]   target: [N] class indices
============================================================
*/
TensorPtr ops::crossEntropyLoss(const TensorPtr& logits, const std::vector<size_t>& target) {
    assert(logits->ndim() == 2);
    // N 指的是 batch size  C 指的是任务类别数
    size_t N = logits->shape()[0];
    size_t C = logits->shape()[1];
    assert(target.size() == N);

    // softmax per row，数值稳定
    std::vector<double> soft(N * C);
    for (size_t n = 0; n < N; ++n) {
        double mx = *std::max_element(logits->value().begin() + n * C, logits->value().begin() + (n + 1) * C);
        double sum_exp = 0.0;
        for (size_t c = 0; c < C; ++c) {
            soft[n * C + c] = std::exp(logits->value()[n * C + c] - mx);
            sum_exp += soft[n * C + c];
        }
        for (size_t c = 0; c < C; ++c) {
            soft[n * C + c] /= sum_exp;
        }
    }

    // NLL loss
    double loss = 0.0;
    for (size_t n = 0; n < N; ++n) {
        loss -= std::log(soft[n * C + target[n]] + 1e-12);
    }
    loss /= (double)N;

    if (!anyRequiresGrad({logits})) {
        return Tensor::create({1}, {loss});
    }

    auto fn = std::make_shared<CrossEntropyBackward>();
    fn->softmax_val_ = soft;
    fn->target_ = target;
    fn->N_ = N;
    fn->C_ = C;
    fn->saved_inputs_ = {logits};

    return Tensor::createFromOp({1}, {loss}, fn);
}


/*
============================================================
    cat : 沿 dim=1（列方向）拼接任意数量的 2D Tensor
          inputs[0]: [m, n0]
          inputs[1]: [m, n1]
          ...
          output   : [m, n0+n1+...]
============================================================
*/
TensorPtr ops::cat(const std::vector<TensorPtr>& inputs, int dim) {
    assert(!inputs.empty());
    size_t ndim = inputs[0]->ndim();
    int d = normalizeDim(dim, (int)ndim);

    // 校验：除 cat_dim 外所有维度必须相同
    std::vector<size_t> out_shape = inputs[0]->shape();
    // 记录所有tensor要拼接的dim的shape值
    std::vector<size_t> split_sizes;
    split_sizes.push_back(inputs[0]->shape()[d]);

    for (size_t i = 1; i < inputs.size(); ++i) {
        assert(inputs[i]->ndim() == ndim);
        // ndim = shape.size(),遍历 shape，除了要拼接的dim外，其余维度shape必须保持一致。
        for (size_t dd = 0; dd < ndim; ++dd) {
            if ((int)dd != d) {
                assert(inputs[i]->shape()[dd] == out_shape[dd]);
            }
        }
        // 计算out_shape的dim维度的shape值
        out_shape[d] += inputs[i]->shape()[d];
        split_sizes.push_back(inputs[i]->shape()[d]);
    }

    size_t out_total = shapeNumel(out_shape);
    std::vector<double> result(out_total, 0.0);
    auto out_st = shapeStrides(out_shape);

    // 当前 tensor 在拼接维度上的起始偏移。
    size_t cat_offset = 0;
    for (size_t inp = 0; inp < inputs.size(); ++inp) {
        const auto& in_shape = inputs[inp]->shape();
        size_t in_total = shapeNumel(in_shape);
        // 保存多维坐标
        std::vector<size_t> idx(ndim);

        for (size_t flat = 0; flat < in_total; ++flat) {
            // 把一维坐标恢复为多维坐标
            size_t tmp = flat;
            for (int dd = (int)ndim - 1; dd >= 0; --dd) {
                idx[dd] = tmp % in_shape[dd];
                tmp /= in_shape[dd];
            }
            // 在 out 中的位置：cat_dim 加上 cat_offset
            size_t out_flat = 0;
            for (size_t dd = 0; dd < ndim; ++dd) {
                size_t i = ((int)dd == d) ? (idx[dd] + cat_offset) : idx[dd];
                // 多维坐标 -> flatten 下标
                out_flat += i * out_st[dd];
            }
            result[out_flat] = inputs[inp]->value()[flat];
        }
        cat_offset += split_sizes[inp];
    }

    if (!anyRequiresGrad(inputs)) {
        return Tensor::create(out_shape, result);
    }

    auto fn = std::make_shared<CatBackward>();
    fn->out_shape_ = out_shape;
    fn->cat_dim_ = d;
    fn->split_sizes_ = split_sizes;
    fn->saved_inputs_ = inputs;

    return Tensor::createFromOp(out_shape, result, fn);
}
/*
============================================================
    transpose 交换两个dim的维度
============================================================
*/
TensorPtr ops::transpose(const TensorPtr& a, int dim0, int dim1) {
    int ndim = (int)a->ndim();
    int d0 = normalizeDim(dim0, ndim);
    int d1 = normalizeDim(dim1, ndim);
 
    std::vector<size_t> out_shape = a->shape();
    std::swap(out_shape[d0], out_shape[d1]);
 
    size_t total = a->numel();
    std::vector<double> result(total);
 
    auto out_st = shapeStrides(out_shape);
    std::vector<size_t> idx(ndim);
 
    for (size_t flat = 0; flat < total; ++flat) {
        size_t tmp = flat;
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = tmp % a->shape()[d];
            tmp /= a->shape()[d];
        }
        // 交换两维
        std::swap(idx[d0], idx[d1]);
        size_t out_flat = 0;
        for (int d = 0; d < ndim; ++d)
            out_flat += idx[d] * out_st[d];
        result[out_flat] = a->value()[flat];
        std::swap(idx[d0], idx[d1]);
    }
 
    if (!anyRequiresGrad({a}))
        return Tensor::create(out_shape, result);
 
    auto fn = std::make_shared<TransposeBackward>();
    fn->in_shape_ = a->shape();
    fn->dim0_ = d0;
    fn->dim1_ = d1;
    fn->saved_inputs_ = {a};
    return Tensor::createFromOp(out_shape, result, fn);
}

/*
============================================================
    reshape
============================================================
*/
TensorPtr ops::reshape(const TensorPtr& a, const std::vector<size_t>& new_shape) {
    assert(shapeNumel(new_shape) == a->numel());
 
    if (!anyRequiresGrad({a}))
        return Tensor::create(new_shape, a->value());
 
    auto fn = std::make_shared<ReshapeBackward>();
    fn->in_shape_ = a->shape();
    fn->saved_inputs_ = {a};
    return Tensor::createFromOp(new_shape, a->value(), fn);
}

/*
============================================================
    softmax（沿 dim）
============================================================
*/
TensorPtr ops::softmax(const TensorPtr& a, int dim) {
    int d = normalizeDim(dim, (int)a->ndim());
    size_t D   = a->shape()[d];
    size_t total = a->numel();
 
    auto st = a->strides();
    size_t stride_d = st[d];
 
    std::vector<double> result(total);
 
    // 枚举所有"条"，每条长度 D
    size_t n_strips = total / D;
 
    // 用 flat index 处理
    // strip_id = (flat / (stride_d * D)) * stride_d + flat % stride_d
    for (size_t flat = 0; flat < total; ++flat) {
        size_t strip_id = (flat / (stride_d * D)) * stride_d + flat % stride_d;
        (void)strip_id;
    }
 
    // 先找每条的 max，再 exp，再归一化
    std::vector<double> strip_max(n_strips, -1e300);
    std::vector<double> strip_sum(n_strips, 0.0);
 
    for (size_t flat = 0; flat < total; ++flat) {
        size_t sid = (flat / (stride_d * D)) * stride_d + flat % stride_d;
        strip_max[sid] = std::max(strip_max[sid], a->value()[flat]);
    }
    for (size_t flat = 0; flat < total; ++flat) {
        size_t sid = (flat / (stride_d * D)) * stride_d + flat % stride_d;
        result[flat] = std::exp(a->value()[flat] - strip_max[sid]);
        strip_sum[sid] += result[flat];
    }
    for (size_t flat = 0; flat < total; ++flat) {
        size_t sid = (flat / (stride_d * D)) * stride_d + flat % stride_d;
        result[flat] /= strip_sum[sid];
    }
 
    if (!anyRequiresGrad({a}))
        return Tensor::create(a->shape(), result);
 
    auto fn = std::make_shared<SoftmaxBackward>();
    fn->y_val_  = result;
    fn->shape_  = a->shape();
    fn->dim_    = d;
    fn->saved_inputs_ = {a};
    return Tensor::createFromOp(a->shape(), result, fn);
}

/*
============================================================
    layerNorm
    x: [..., D]   w: [D]   b: [D]
    归一化最后一维
============================================================
*/
TensorPtr ops::layerNorm(const TensorPtr& x, const TensorPtr& w, const TensorPtr& b, double eps) {
    size_t D = x->shape().back();
    assert(w->numel() == D && b->numel() == D);
 
    size_t total  = x->numel();
    size_t groups = total / D;
 
    std::vector<double> result(total);
    std::vector<double> x_norm(total);
    std::vector<double> rstd(groups);
 
    const auto& xv = x->value();
    const auto& wv = w->value();
    const auto& bv = b->value();
 
    for (size_t g = 0; g < groups; ++g) {
        size_t base = g * D;
        // mean
        double mean = 0.0;
        for (size_t i = 0; i < D; ++i) mean += xv[base + i];
        mean /= D;
        // var
        double var = 0.0;
        for (size_t i = 0; i < D; ++i) {
            double d = xv[base + i] - mean;
            var += d * d;
        }
        var /= D;
        double rs = 1.0 / std::sqrt(var + eps);
        rstd[g] = rs;
        for (size_t i = 0; i < D; ++i) {
            x_norm[base + i] = (xv[base + i] - mean) * rs;
            result[base + i] = x_norm[base + i] * wv[i] + bv[i];
        }
    }
 
    if (!anyRequiresGrad({x, w, b}))
        return Tensor::create(x->shape(), result);
 
    auto fn = std::make_shared<LayerNormBackward>();
    fn->x_val_ = xv;
    fn->x_norm_ = x_norm;
    fn->w_val_ = wv;
    fn->rstd_ = rstd;
    fn->shape_ = x->shape();
    fn->norm_size_ = D;
    fn->saved_inputs_ = {x, w, b};

    return Tensor::createFromOp(x->shape(), result, fn);
}   