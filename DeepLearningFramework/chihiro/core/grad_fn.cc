#include "grad_fn.h"
#include "Tensor.h"
#include <cmath>
#include <numeric>
#include <cassert>

/*
============================================================
    AddBackward
    dA = reduceTo(grad, out_shape, shapeA)
    dB = reduceTo(grad, out_shape, shapeB)
============================================================
*/
std::vector<std::vector<double>> AddBackward::apply(const std::vector<double>& grad) {
    return { reduceTo(grad, shapeOut_, shapeA_),
            reduceTo(grad, shapeOut_, shapeB_) };
}

/*
============================================================
    SubBackward
    dA =  reduceTo(grad, ...)
    dB = -reduceTo(grad, ...)
============================================================
*/
std::vector<std::vector<double>> SubBackward::apply(const std::vector<double>& grad) {
    auto dA = reduceTo(grad, shapeOut_, shapeA_);
    auto dB = reduceTo(grad, shapeOut_, shapeB_);

    for (auto& v : dB) {
        v = -v;
    }

    return {dA, dB};
}

/*
============================================================
    MulBackward
    dA = reduceTo(grad * B, out_shape, shapeA)
    dB = reduceTo(grad * A, out_shape, shapeB)
============================================================
*/
std::vector<std::vector<double>> MulBackward::apply(const std::vector<double>& grad) {
    size_t n = grad.size();

    if (same_tensor_) {
        // C = A * A  →  dA = 2 * grad * A，但 A 可能被 broadcast
        std::vector<double> g(n);
        for (size_t i = 0; i < n; ++i) {
            g[i] = 2.0 * grad[i] * x_val_[i];
        }
        return { reduceTo(g, shapeOut_, shapeA_) };
    }

    // 先把 grad 逐元素乘另一个操作数（两者都已 broadcast 到 shapeOut_）
    std::vector<double> gA(n), gB(n);
    for (size_t i = 0; i < n; ++i) {
        gA[i] = grad[i] * y_val_[i];
        gB[i] = grad[i] * x_val_[i];
    }

    return { reduceTo(grad, shapeOut_, shapeA_), reduceTo(grad, shapeOut_, shapeB_) };
}

/*
============================================================
    MatMulBackward
    MatMulBackward  (batched)
 
    A: [..., m, k]   B: [..., k, n]   C: [..., m, n]
 
    dA[..., i, p] += grad[..., i, j] * B[..., p, j]   (grad @ B^T)
    dB[..., p, j] += A[..., i, p] * grad[..., i, j]   (A^T @ grad)
 
    batch 维度的 broadcast：shapeA_ / shapeB_ 可能不同，
    反向时需要 reduceTo 折叠回各自的原始 shape。
============================================================
*/
std::vector<std::vector<double>> MatMulBackward::apply(const std::vector<double>& grad) {
    // 推导 batch shape（broadcast 后的前缀）

    // broadcast 后的完整 shape
    std::vector<size_t> shapeC_batch;
    {
        std::vector<size_t> bA(shapeA_.begin(), shapeA_.end() - 2);
        std::vector<size_t> bB(shapeB_.begin(), shapeB_.end() - 2);
        size_t bn = std::max(bA.size(), bB.size());
        shapeC_batch.resize(bn);
        for (size_t i = 0; i < bn; ++i) {
            int ia = (int)bA.size() - 1 - (int)i;
            int ib = (int)bB.size() - 1 - (int)i;
            size_t da = (ia >= 0) ? bA[ia] : 1;
            size_t db = (ib >= 0) ? bB[ib] : 1;
            shapeC_batch[bn - 1 - i] = std::max(da, db);
        }
    }

    // batch 个数
    size_t batch = 1;
    for (auto d : shapeC_batch) {
        batch *= d;
    }

    // dA shape（broadcast 后）= [..., m, k]
    std::vector<size_t> shapeA_bc(shapeC_batch);
    shapeA_bc.push_back(m_); 
    shapeA_bc.push_back(k_);

    // dB shape（broadcast 后）= [..., k, n]
    std::vector<size_t> shapeB_bc(shapeC_batch);
    shapeB_bc.push_back(k_); 
    shapeB_bc.push_back(n_);

    std::vector<double> dA_bc(batch * m_ * k_, 0.0);
    std::vector<double> dB_bc(batch * k_ * n_, 0.0);

    // 计算 A / B 在 broadcast 下的 batch strides
    // 如果某个 batch 维度是 1，stride=0（广播维度重复访问同一块）
 
    // 枚举 batch 维度（shapeC_batch 顺序）
    // 用多维索引展开
    size_t bndim = shapeC_batch.size();
    std::vector<size_t> bidx(bndim, 0);

    for (size_t b = 0; b < batch; ++b) {
        // 计算 A_offset / B_offset / C_offset
        // A batch strides
        size_t A_off = 0, B_off = 0;
        {
            size_t bA = shapeA_.size() - 2;
            size_t bB = shapeB_.size() - 2;
            size_t strA = m_ * k_, strB = k_ * n_;
            for (int d = (int)bndim - 1; d >= 0; --d) {
                // A 在该 batch 维度的实际 dim
                int da_idx = (int)bA - ((int)bndim - d);
                size_t da = (da_idx >= 0) ? shapeA_[da_idx] : 1;
                size_t ia = (da == 1) ? 0 : bidx[d];
                A_off += ia * strA;

                int db_idx = (int)bB - ((int)bndim - d);
                size_t db = (db_idx >= 0) ? shapeB_[db_idx] : 1;
                size_t ib = (db == 1) ? 0 : bidx[d];
                B_off += ib * strB;
 
                strA *= (da == 1 ? 1 : da);
                strB *= (db == 1 ? 1 : db);
            }
        }

        size_t C_off = b * m_ * n_;
        size_t dA_off = b * m_ * k_;
        size_t dB_off = b * k_ * n_;
 
        // dA = grad @ B^T
        for (size_t i = 0; i < m_; ++i) {
            for (size_t p = 0; p < k_; ++p) {
                for (size_t j = 0; j < n_; ++j) {
                    dA_bc[dA_off + i * k_ + p] += grad[C_off + i * n_ + j] * B_val_[B_off + p * n_ + j];                            
                }
            }
        }
    
        // dB = A^T @ grad
        for (size_t i = 0; i < m_; ++i) {
            for (size_t p = 0; p < k_; ++p) {
                for (size_t j = 0; j < n_; ++j) {
                    dB_bc[dB_off + p * n_ + j] += A_val_[A_off + i * k_ + p] * grad[C_off + i * n_ + j];
                }
            }
        }

        // 更新 bidx
        for (int d = (int)bndim - 1; d >= 0; --d) {
            if (++bidx[d] < shapeC_batch[d]) break;
            bidx[d] = 0;
        }
    }

    // reduceTo 折叠 broadcast 维度回原始 shape
    auto dA = reduceTo(dA_bc, shapeA_bc, shapeA_);
    auto dB = reduceTo(dB_bc, shapeB_bc, shapeB_);
    return { dA, dB };
}

/*
============================================================
    ReLUBackward
============================================================
*/
std::vector<std::vector<double>> ReLUBackward::apply(const std::vector<double>& grad) {
    size_t n = x_val_.size();
    std::vector<double> dx(n);
    for (size_t i = 0; i < n; ++i) {
        dx[i] = x_val_[i] > 0.0 ? grad[i] : 0.0;
    }

    return { dx };
}

/*
============================================================
    SigmoidBackward
============================================================
*/
std::vector<std::vector<double>> SigmoidBackward::apply(const std::vector<double>& grad) {
    size_t n = y_val_.size();
    std::vector<double> dx(n);
    for (size_t i = 0; i < n; ++i) {
        dx[i] = grad[i] * y_val_[i] * (1.0 - y_val_[i]);
    }

    return { dx };
}

/*
============================================================
    SumBackward
    
    全局 sum（sum_dim_ == -1）：
        dx = grad[0] 广播到所有元素
 
    沿 dim sum（keepdim or not）：
        grad 先 restore 到 keepdim=true 的 shape，
        再 broadcast 到 input_shape_
============================================================
*/
std::vector<std::vector<double>> SumBackward::apply(const std::vector<double>& grad) {
    size_t in_total = shapeNumel(input_shape_);
    
    if (sum_dim_ == -1) {
        // 全局 sum：grad 是标量
        assert(grad.size() == 1);
        return { std::vector<double>(in_total, grad[0]) };
    }
    
    int d = normalizeDim(sum_dim_, (int)input_shape_.size());

    // grad 对应的 shape（keepdim=false 时少一维）
    std::vector<size_t> grad_shape = input_shape_;
    grad_shape[d] = 1;   // keepdim=true 的 shape

    // 如果 keepdim_==false，grad 的实际 shape 是 grad_shape 去掉第 d 维
    // 但 reduceTo 只关心 numel，这里直接把 grad 当 grad_shape 来处理
    // 用 broadcast 展开到 input_shape_
    std::vector<double> dx(in_total);
    auto in_st = shapeStrides(input_shape_);
    auto g_st  = shapeStrides(grad_shape);

    size_t ndim = input_shape_.size();
    std::vector<size_t> idx(ndim, 0);
    for (size_t flat = 0; flat < in_total; ++flat) {
        // 解码 flat
        size_t tmp = flat;
        for (int dd = (int)ndim - 1; dd >= 0; --dd) {
            idx[dd] = tmp % input_shape_[dd];
            tmp /= input_shape_[dd];
        }
        // 对应 grad 的位置（sum_dim_ 上的索引置 0）
        size_t g_flat = 0;
        for (size_t dd = 0; dd < ndim; ++dd) {
            size_t i = (dd == (size_t)d) ? 0 : idx[dd];
            g_flat += i * g_st[dd];
        }
        dx[flat] = grad[g_flat];
    }

    return { dx };
}

/*
============================================================
    BCEWithLogitsBackward
    forward: loss = mean( max(x,0) - x*y + log(1 + e^{-|x|}) )
    
    ∂loss/∂x_i = (1/N) * (sigmoid(x_i) - y_i)
    
    保存的是 sigmoid(x)，反向直接 p - y，无除法，数值稳定
============================================================
*/
std::vector<std::vector<double>> BCEWithLogitsBackward::apply(const std::vector<double>& grad) {
    assert(grad.size() == 1);
    double g = grad[0];
    std::vector<double> dx(n_);
    for (size_t i = 0; i < n_; ++i) {
        dx[i] = g * (sigmoid_val_[i] - target_val_[i]) / static_cast<double>(n_);
    }

    return { dx };
}

/*
============================================================
    EmbeddingBackward
    forward : out[i] = W[ids[i]]           行索引查表
    backward: dW[ids[i]] += grad[i, :]     梯度写回对应行
 
    grad 是输出 out 的梯度，shape [batch * embedding_dim]（展平）
    dW   shape [num_embeddings * embedding_dim]，全零初始化后按 ids 散射累加
 
    散射累加（scatter-add）示意：
        for i in range(batch):
            dW[ids[i], :] += grad[i, :]
 
    注：同一 id 出现多次时 += 保证梯度正确累加，不能用赋值 =
============================================================
*/
std::vector<std::vector<double>> EmbeddingBackward::apply(const std::vector<double>& grad) {
    size_t batch = ids_.size();

    assert(grad.size() == batch * embedding_dim_);

    // dW 全零初始化，shape [num_embeddings * embedding_dim]
    std::vector<double> dW(num_embeddings_ * embedding_dim_, 0.0);

    // 散射累加：把 grad[i, :] 加到 dW[ids[i], :]
    for (size_t i = 0; i < batch; ++i) {
        size_t row = ids_[i];   // 对应 embedding 的行号
        for (size_t j = 0; j < embedding_dim_; ++j) {
            dW[row * embedding_dim_ + j] += grad[i * embedding_dim_ + j]; 
        }
    }

    // saved_inputs_ = {W}，只返回一个梯度
    return { dW };
}

/*
============================================================
    CatBackward（沿任意 dim）
 
    把 grad 按 split_sizes_ 在 cat_dim_ 上切回各输入
============================================================
*/
std::vector<std::vector<double>> CatBackward::apply(const std::vector<double>& grad) {
    size_t ndim = out_shape_.size();
    size_t n_inputs = split_sizes_.size();
 
    size_t total = shapeNumel(out_shape_);
    assert(grad.size() == total);
 
    std::vector<std::vector<double>> d_inputs(n_inputs);
 
    // 每个输入的 shape（只有 cat_dim_ 不同）
    size_t cat_offset = 0;
    for (size_t inp = 0; inp < n_inputs; ++inp) {
        std::vector<size_t> in_shape = out_shape_;
        in_shape[cat_dim_] = split_sizes_[inp];
        size_t in_total = shapeNumel(in_shape);
        d_inputs[inp].assign(in_total, 0.0);
        auto in_st = shapeStrides(in_shape);

        // 遍历 out 的所有元素，属于本 input 的（cat_dim_ 上 idx ∈ [offset, offset+size)）就 copy
        std::vector<size_t> idx(ndim, 0);
        for (size_t flat = 0; flat < total; ++flat) {
            size_t tmp = flat;
            for (int d = (int)ndim - 1; d >= 0; --d) {
                idx[d] = tmp % out_shape_[d];
                tmp /= out_shape_[d];
            }
            size_t cat_idx = idx[cat_dim_];
            if (cat_idx >= cat_offset && cat_idx < cat_offset + split_sizes_[inp]) {
                // 映射到 in_shape 的 flat index
                size_t in_flat = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    size_t i = (d == (size_t)cat_dim_) ? (cat_idx - cat_offset) : idx[d];
                    in_flat += i * in_st[d];
                }
                d_inputs[inp][in_flat] = grad[flat];
            }
        }
        cat_offset += split_sizes_[inp];
    }

    return d_inputs;
}

/*
============================================================
    TransposeBackward
    grad 只需做同样的 transpose（transpose 是自逆的，
    前提是维度对换相同的两维）
============================================================
*/
std::vector<std::vector<double>> TransposeBackward::apply(const std::vector<double>& grad) {
    size_t ndim = in_shape_.size();
    // out_shape 是 in_shape 把 dim0/dim1 互换
    std::vector<size_t> out_shape = in_shape_;
    std::swap(out_shape[dim0_], out_shape[dim1_]);
 
    auto out_st = shapeStrides(out_shape);
    auto in_st  = shapeStrides(in_shape_);
 
    size_t total = shapeNumel(in_shape_);
    std::vector<double> dx(total);
    std::vector<size_t> idx(ndim);
 
    // 遍历 out（grad）的每个元素，映射回 in
    size_t out_total = shapeNumel(out_shape);
    for (size_t flat = 0; flat < out_total; ++flat) {
        size_t tmp = flat;
        for (int d = (int)ndim - 1; d >= 0; --d) {
            idx[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        // 交换 dim0, dim1
        std::swap(idx[dim0_], idx[dim1_]);
        size_t in_flat = 0;
        for (size_t d = 0; d < ndim; ++d)
            in_flat += idx[d] * in_st[d];
        dx[in_flat] = grad[flat];
        std::swap(idx[dim0_], idx[dim1_]); // 还原
    }
    return { dx };
}
 
/*
============================================================
    ReshapeBackward — 把 grad reshape 回 in_shape_
============================================================
*/
std::vector<std::vector<double>> ReshapeBackward::apply(const std::vector<double>& grad) {
    // grad 内存布局不变，只是 shape 解释不同
    return { grad };
}
 
/*
============================================================
    SoftmaxBackward
 
    Jacobian: dL/dx_i = sum_j( dL/dy_j * y_j * (δ_ij - y_i) )
                      = y_i * ( dL/dy_i - dot(dL/dy, y) )
 
    沿 dim_ 计算。
============================================================
*/
std::vector<std::vector<double>> SoftmaxBackward::apply(const std::vector<double>& grad) {
    size_t ndim  = shape_.size();
    size_t total = shapeNumel(shape_);
    int d = normalizeDim(dim_, (int)ndim);
    size_t D = shape_[d];
 
    auto st = shapeStrides(shape_);
 
    std::vector<double> dx(total);
 
    // 枚举除 d 维以外的所有位置
    // 用"跳过 d 维"的方式枚举
 
    // 更简洁的方式：枚举所有 flat 位置，对每个 flat 找其所在"条"的 dot
    // 先计算每个"条"的 dot(grad, y)
    // 条的 id = flat 去掉第 d 维后的 index
    // 用 stride：同一条内的元素 flat 之间相差 st[d]
 
    // 每条的长度是 D，条数是 total/D
    // flat 到"条id + 条内idx"：
    //   条id  = (flat / (st[d] * D)) * st[d] + flat % st[d]
    //   条内idx = (flat / st[d]) % D
 
    std::vector<double> dot_val(total / D, 0.0);
 
    for (size_t flat = 0; flat < total; ++flat) {
        size_t strip_id = (flat / (st[d] * D)) * st[d] + flat % st[d];
        dot_val[strip_id] += grad[flat] * y_val_[flat];
    }
 
    for (size_t flat = 0; flat < total; ++flat) {
        size_t strip_id = (flat / (st[d] * D)) * st[d] + flat % st[d];
        dx[flat] = y_val_[flat] * (grad[flat] - dot_val[strip_id]);
    }
 
    return { dx };
}
 
/*
============================================================
    LayerNormBackward
 
    x_norm = (x - mean) / sqrt(var + eps)
    y = x_norm * w + b
 
    反向（对 x, w, b）：
        dL/db = sum(grad, over norm dims)
        dL/dw = sum(grad * x_norm, over norm dims)
        dL/dx = (dL/dx_norm) * w / sqrt(var+eps)
                其中 dL/dx_norm 需要修正均值和方差的梯度
 
    公式（每个归一化组独立）：
        dx_norm = grad * w
        N = norm_size_
        dx = rstd * (dx_norm - mean(dx_norm) - x_norm * mean(dx_norm * x_norm))
============================================================
*/
std::vector<std::vector<double>> LayerNormBackward::apply(const std::vector<double>& grad) {
    size_t total = shapeNumel(shape_);
    size_t N     = norm_size_;
    size_t groups = total / N;   // 归一化组数（batch * seq_len）
 
    std::vector<double> dx(total), dw(N, 0.0), db(N, 0.0);
 
    for (size_t g = 0; g < groups; ++g) {
        size_t base = g * N;
        double rstd = rstd_[g];
 
        // dw, db（累加跨所有 group）
        for (size_t i = 0; i < N; ++i) {
            dw[i] += grad[base + i] * x_norm_[base + i];
            db[i] += grad[base + i];
        }
 
        // dx_norm = grad * w
        std::vector<double> dx_norm(N);
        for (size_t i = 0; i < N; ++i)
            dx_norm[i] = grad[base + i] * w_val_[i];
 
        // mean(dx_norm) 和 mean(dx_norm * x_norm)
        double mean_dxn = 0.0, mean_dxn_xn = 0.0;
        for (size_t i = 0; i < N; ++i) {
            mean_dxn    += dx_norm[i];
            mean_dxn_xn += dx_norm[i] * x_norm_[base + i];
        }
        mean_dxn    /= N;
        mean_dxn_xn /= N;
 
        for (size_t i = 0; i < N; ++i) {
            dx[base + i] = rstd * (dx_norm[i] - mean_dxn
                                   - x_norm_[base + i] * mean_dxn_xn);
        }
    }
 
    return { dx, dw, db };
}
 
/*
============================================================
    CrossEntropyBackward
 
    forward: softmax + NLLLoss
    dL/d(logits[n, c]) = (softmax[n,c] - 1{c==target[n]}) / N
============================================================
*/
std::vector<std::vector<double>> CrossEntropyBackward::apply(const std::vector<double>& grad) {
    assert(grad.size() == 1);
    double g = grad[0];
    std::vector<double> dx(N_ * C_);
    for (size_t n = 0; n < N_; ++n) {
        for (size_t c = 0; c < C_; ++c) {
            double indicator = (target_[n] == c) ? 1.0 : 0.0;
            dx[n * C_ + c] = g * (softmax_val_[n * C_ + c] - indicator) / (double)N_;
        }
    }
    return { dx };
}
 