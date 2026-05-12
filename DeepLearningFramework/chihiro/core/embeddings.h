#pragma once
#include "module.h"
#include <vector>

/*
==============================================================
    Embedding — 查表层（稀疏 ID → 稠密向量）
 
    本质：一个可学习的权重矩阵 W[num_embeddings, embedding_dim]
          给定一批整数 ID，返回对应行组成的矩阵
 
    forward :
        ids : [batch_size]          （size_t 整数列表，不是 Tensor）
        W   : [num_embeddings, embedding_dim]  (requires_grad=true)
        out : [batch_size, embedding_dim]
 
        out[i] = W[ids[i]]          ← 行索引，不是矩阵乘法
 
    backward :
        dW[ids[i]] += grad[i, :]    ← 只有被查到的行才累加梯度
                                       同一 id 多次出现则多次累加
 
    权重初始化：
        W ~ N(0, 1)                 （与 PyTorch 默认一致）
 
    参数访问：
        parameters() 返回 {W_}
        weight()     直接访问权重矩阵
 
    注意：
        forward 接受 std::vector<size_t>，而非 TensorPtr
        因为 ID 天然是整数，语义上与 Linear 的浮点输入不同
        Module 基类的 forward(TensorPtr) 在此类中不应被调用
==============================================================
*/
class Embedding : public Module {
public:
    /*
        num_embeddings : 词表大小，比如 1000 个用户、500 个商品
        embedding_dim  : 每个 ID 映射到的向量维度，比如 16
        seed           : 随机种子，默认 42，传 0 则用随机设备
    */
    Embedding(size_t num_embeddings, size_t embedding_dim, unsigned seed = 42);

    /*
        主接口：接受整数 ID 列表，返回对应 embedding 矩阵
        ids : 长度为 batch_size 的整数列表，每个值 < num_embeddings
        out : [batch_size, embedding_dim]
    */
    TensorPtr forward(const std::vector<size_t>& ids);

    /*
        Module 基类接口，Embedding 不走浮点 Tensor 输入路径
        调用时会抛出异常，提示使用 forward(ids) 版本
    */
    TensorPtr forward(const TensorPtr& x) override;

    std::vector<TensorPtr> parameters() override { return {W_}; }
 
    std::string name() const override { return "Embedding"; }

    // 参数访问
    TensorPtr weight()         const { return W_; }
    size_t numEmbeddings()     const { return num_embeddings_; }
    size_t embeddingDim()      const { return embedding_dim_; }
private:
    size_t num_embeddings_;
    size_t embedding_dim_;
 
    TensorPtr W_;   // [num_embeddings, embedding_dim]
};
