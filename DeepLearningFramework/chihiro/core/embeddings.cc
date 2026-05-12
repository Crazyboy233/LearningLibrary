#include "embeddings.h"
#include <random>

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim, unsigned seed = 42) 
    :num_embeddings_(num_embeddings), embedding_dim_(embedding_dim)
{
    // ---- 随机引擎 ----
    std::mt19937 rng;
    if (seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }

    // ---- 初始化：W ~ N(0, 1)，与 PyTorch 默认一致 ----
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> w_data(num_embeddings * embedding_dim);
    for (auto& v : w_data) {
        v = dist(rng);
    }

    W_ = Tensor::create({num_embeddings_, embedding_dim_}, w_data, /*requeires_grad=*/true);
}

TensorPtr Embedding::forward(const std::vector<size_t>& ids) {
    size_t batch = ids.size();

    // 越界检查
    for (size_t i = 0; i < batch; ++i) {
        if(ids[i] > num_embeddings_) {
            throw std::out_of_range(
                "Embedding::forward: id " + std::to_string(ids[i]) +
                " >= num_embeddings " + std::to_string(num_embeddings_)
            );
        }
    }

    // ---- forward：逐行查表，拼成 [batch, embedding_dim] ----
    //
    //   out[i, j] = W[ids[i], j]
    //
    //   W 在内存里是行主序展平的一维数组：
    //       W[row, col] = w_data[row * embedding_dim + col]
    std::vector<double> out_data(batch * embedding_dim_);
    const auto& w = W_->value();

    for (size_t i = 0; i < batch; ++i) {
        size_t row = ids[i];
        for (size_t j = 0; j < embedding_dim_; ++j) {
            out_data[i * embedding_dim_ + j] = w[row * embedding_dim_ + j];
        }
    }

    // ---- 不需要梯度则直接返回 ----
    if (!W_->requireGrad()) {
        return Tensor::create({batch, embedding_dim_}, out_data);
    }

    // ---- 构造 EmbeddingBackward，挂到输出 Tensor 上 ----
    auto fn = std::make_shared<EmbeddingBackward>();
    fn->ids_ = ids;
    fn->num_embeddings_ = this->num_embeddings_;
    fn->embedding_dim_ = this->embedding_dim_;
    fn->saved_inputs_ = {W_};   // 只有 W 需要梯度，ids 不入图

    return Tensor::createFromOp({batch, embedding_dim_}, out_data, fn);
}

TensorPtr Embedding::forward(const TensorPtr& x) {
    // Embedding 的输入是整数 ID，不是浮点 Tensor
    // 请使用 forward(const std::vector<size_t>& ids) 版本
    throw std::logic_error(
        "Embedding::forward(TensorPtr) is not supported. "
        "Use forward(const std::vector<size_t>& ids) instead."
    );
}