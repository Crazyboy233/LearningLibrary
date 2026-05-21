#include "../core/Tensor.h"
#include "../core/ops.h"
#include "../core/embeddings.h"
#include "../core/optimizer.h"
#include "../core/linear.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

// 编译命令：
// g++ -std=c++17 ./core/*.cc ./test/08_test_N_dim.cpp && ./a.out

// ============================================================
//  工具：打印带标题的分隔线
// ============================================================
static void section(const std::string& title) {
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout << "  " << title << "\n";
    std::cout << "╚══════════════════════════════════════════════╝\n";
}

static void printVec(const std::string& name,
                     const std::vector<double>& v,
                     size_t max_show = 8) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << name << " [";
    size_t n = std::min(v.size(), max_show);
    for (size_t i = 0; i < n; ++i)
        std::cout << v[i] << (i + 1 < n ? ", " : "");
    if (v.size() > max_show) std::cout << ", ...";
    std::cout << "]\n";
}

// ============================================================
//  TEST 1：基础算子正确性（add / sub / mul / matmul / relu / sigmoid）
// ============================================================
void test_basic_ops() {
    section("TEST 1: 基础算子 (add/sub/mul/matmul/relu/sigmoid)");

    // --- add broadcast: [2,3] + [1,3] ---
    auto a = Tensor::create({2,3}, {1,2,3,4,5,6}, true);
    auto b = Tensor::create({1,3}, {10,20,30}, true);
    auto c = ops::add(a, b);
    std::cout << "[add broadcast 2x3 + 1x3]\n";
    printVec("  result", c->value());
    // expect: [11,22,33, 14,25,36]

    // --- sub ---
    auto s = ops::sub(a, b);
    printVec("  sub result", s->value());

    // --- mul ---
    auto m = ops::mul(a, b);
    printVec("  mul result", m->value());

    // --- matmul [2,3] x [3,2] ---
    auto W = Tensor::create({3,2}, {1,0, 0,1, 1,1}, true);
    auto r = ops::matmul(a, W);
    std::cout << "[matmul 2x3 @ 3x2]\n";
    printVec("  result", r->value());

    // --- relu / sigmoid ---
    auto x = Tensor::create({4}, {-2.0, -0.5, 0.5, 2.0}, true);
    printVec("  relu",    ops::relu(x)->value());
    printVec("  sigmoid", ops::sigmoid(x)->value());

    // --- backward 通过 add ---
    auto loss = ops::sum(c);
    loss->backward();
    printVec("  grad a (should be all 1s)", a->grad());
    printVec("  grad b (should be [2,2,2])", b->grad());
    std::cout << "✅[PASS]\n";
}

// ============================================================
//  TEST 2：Embedding forward + backward
// ============================================================
void test_embedding() {
    section("TEST 2: Embedding forward + backward");

    Embedding emb(5, 3, /*seed=*/42);
    std::vector<size_t> ids = {0, 2, 2};  // id=2 出现两次

    auto out = emb.forward(ids);
    std::cout << "[Embedding(5,3), ids=[0,2,2], out shape=3x3]\n";
    printVec("  out[0]", {out->value().begin(),     out->value().begin() + 3});
    printVec("  out[1]", {out->value().begin() + 3, out->value().begin() + 6});
    printVec("  out[2]", {out->value().begin() + 6, out->value().end()});

    // backward：给 out 一个全 1 的梯度
    auto loss = ops::sum(out);
    loss->backward();
    // dW[2] 应该是 out[1] + out[2] 的梯度，即 2 倍（两次出现）
    auto& dW = emb.weight()->grad();
    printVec("  dW[2] (should be 2.0 each)", {dW.begin() + 6, dW.begin() + 9});
    double ok = std::abs(dW[6] - 2.0) < 1e-9 &&
                std::abs(dW[7] - 2.0) < 1e-9 &&
                std::abs(dW[8] - 2.0) < 1e-9;
    std::cout << "  grad accumulation for repeated id: " << (ok ? "✅[PASS]" : "[FAIL]") << "\n";
}

// ============================================================
//  TEST 3：LayerNorm forward + backward 数值检验
// ============================================================
void test_layernorm() {
    section("TEST 3: LayerNorm forward + backward");

    // x: [2, 4]
    auto x = Tensor::create({2,4}, {1,2,3,4, 2,4,6,8}, true);
    auto w = Tensor::create({4}, {1,1,1,1}, true);
    auto b = Tensor::create({4}, {0,0,0,0}, true);

    auto y = ops::layerNorm(x, w, b);
    std::cout << "[LayerNorm [2,4], w=1, b=0]\n";
    printVec("  y[0] (mean≈0, std≈1)", {y->value().begin(), y->value().begin()+4});
    printVec("  y[1]",                 {y->value().begin()+4, y->value().end()});

    // mean of each row should be ≈0
    double mean0 = 0, mean1 = 0;
    for (int i = 0; i < 4; ++i) { mean0 += y->value()[i]; mean1 += y->value()[4+i]; }
    mean0 /= 4; mean1 /= 4;
    std::cout << "  row0 mean=" << mean0 << " (want ≈0)\n";
    std::cout << "  row1 mean=" << mean1 << " (want ≈0)\n";

    auto loss = ops::sum(y);
    loss->backward();
    printVec("  dw", w->grad());
    printVec("  db", b->grad());
    std::cout << "✅[PASS]\n";
}

// ============================================================
//  TEST 4：CrossEntropyLoss
// ============================================================
void test_cross_entropy() {
    section("TEST 4: CrossEntropyLoss");

    // logits: [3, 4]  targets: [0, 2, 1]
    auto logits = Tensor::create({3,4},
        {2.0, 1.0, 0.1, 0.1,
         0.1, 0.1, 2.0, 0.1,
         0.1, 2.0, 0.1, 0.1}, true);
    std::vector<size_t> targets = {0, 2, 1};

    auto loss = ops::crossEntropyLoss(logits, targets);
    std::cout << "[CrossEntropy [3,4]], targets=[0,2,1]\n";
    std::cout << "  loss = " << loss->value()[0]
              << "  (expect small, max-class matches target)\n";

    loss->backward();
    printVec("  dlogits[0]", {logits->grad().begin(),     logits->grad().begin()+4});
    printVec("  dlogits[1]", {logits->grad().begin()+4,   logits->grad().begin()+8});
    std::cout << "✅[PASS]\n";
}

// ============================================================
//  TEST 5：推荐系统（来自题目示例）
// ============================================================
void test_recommendation() {
    section("TEST 5: 推荐系统 (Embedding + cat + Linear + BCE)");

    const size_t EMB_DIM = 4;
    const size_t HIDDEN  = 8;
    const size_t N_USERS = 10;
    const size_t N_ITEMS = 20;
    const double LR      = 0.05;
    const size_t EPOCHS  = 100;

    Embedding user_emb(N_USERS, EMB_DIM, 1);
    Embedding item_emb(N_ITEMS, EMB_DIM, 2);
    Linear fc(EMB_DIM * 2, HIDDEN, 3);
    Linear out_layer(HIDDEN, 1, 4);

    std::vector<TensorPtr> params;
    for (auto& p : user_emb.parameters()) params.push_back(p);
    for (auto& p : item_emb.parameters()) params.push_back(p);
    for (auto& p : fc.parameters())       params.push_back(p);
    for (auto& p : out_layer.parameters()) params.push_back(p);

    SGD sgd(params, LR, /*momentum=*/0.9);

    std::vector<size_t> user_ids = {0, 2, 5, 7};
    std::vector<size_t> item_ids = {3, 1, 8, 5};
    auto target = Tensor::create({4,1}, {1.0, 1.0, 0.0, 1.0});

    double first_loss = -1.0, last_loss = -1.0;

    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        sgd.zeroGrad();

        auto u_emb  = user_emb.forward(user_ids);
        auto i_emb  = item_emb.forward(item_ids);
        auto concat = ops::cat({u_emb, i_emb});
        auto h      = ops::relu(fc.forward(concat));
        auto logits = out_layer.forward(h);
        auto loss   = ops::bceWithLogitsLoss(logits, target);

        if (epoch == 0) first_loss = loss->value()[0];
        last_loss = loss->value()[0];

        if (epoch % 10 == 0) {
            std::cout << "================================\n";
            std::cout << "epoch = " << epoch << "\n";
            printVec("  loss",           loss->value());
            auto prob = ops::sigmoid(logits);
            printVec("  sigmoid(logits)", prob->value());
            std::cout << "================================\n";
        }

        loss->backward();
        sgd.step();
    }

    std::cout << "\n  first_loss=" << first_loss
              << "  last_loss=" << last_loss << "\n";
    bool converging = last_loss < first_loss;
    std::cout << "  Loss decreased: " << (converging ? "✅[PASS]" : "[FAIL]") << "\n";
}

// ============================================================
//  TEST 6：Mini Transformer Encoder（单头自注意力 + FFN + LayerNorm）
//
//  架构：
//    输入 token ids → Embedding → [seq, d_model]
//    Self-Attention（单头，缩放点积）：
//        Q = x @ Wq,  K = x @ Wk,  V = x @ Wv
//        scores = Q @ K^T / sqrt(d_k)
//        attn   = softmax(scores, dim=-1)
//        ctx    = attn @ V
//    残差 + LayerNorm → Add & Norm 1
//    FFN：Linear(d_model, 4*d_model) → ReLU → Linear(4*d_model, d_model)
//    残差 + LayerNorm → Add & Norm 2
//    输出 logits = Linear(d_model, vocab_size)
//    CrossEntropy loss
// ============================================================
void test_transformer_encoder() {
    section("TEST 6: Mini Transformer Encoder (单头自注意力 + FFN)");

    // 超参
    const size_t VOCAB   = 16;
    const size_t D_MODEL = 8;
    const size_t D_K     = 8;    // 单头，d_k = d_model
    const size_t D_FF    = 16;   // FFN 中间维
    const size_t SEQ     = 4;    // 序列长度
    const double LR      = 0.02;
    const size_t EPOCHS  = 200;

    // ---- 参数层 ----
    Embedding tok_emb(VOCAB, D_MODEL, 10);

    // Attention 投影矩阵
    Linear Wq(D_MODEL, D_K, 11);
    Linear Wk(D_MODEL, D_K, 12);
    Linear Wv(D_MODEL, D_K, 13);
    Linear Wo(D_K, D_MODEL, 14);      // output projection

    // LayerNorm 参数（两组）
    auto ln1_w = Tensor::create({D_MODEL}, std::vector<double>(D_MODEL, 1.0), true);
    auto ln1_b = Tensor::create({D_MODEL}, std::vector<double>(D_MODEL, 0.0), true);
    auto ln2_w = Tensor::create({D_MODEL}, std::vector<double>(D_MODEL, 1.0), true);
    auto ln2_b = Tensor::create({D_MODEL}, std::vector<double>(D_MODEL, 0.0), true);

    // FFN
    Linear ff1(D_MODEL, D_FF, 15);
    Linear ff2(D_FF, D_MODEL, 16);

    // 输出 head
    Linear lm_head(D_MODEL, VOCAB, 17);

    // ---- 收集参数 ----
    std::vector<TensorPtr> params;
    auto add_params = [&](std::vector<TensorPtr> ps) {
        for (auto& p : ps) params.push_back(p);
    };
    add_params(tok_emb.parameters());
    add_params(Wq.parameters());
    add_params(Wk.parameters());
    add_params(Wv.parameters());
    add_params(Wo.parameters());
    params.push_back(ln1_w); params.push_back(ln1_b);
    params.push_back(ln2_w); params.push_back(ln2_b);
    add_params(ff1.parameters());
    add_params(ff2.parameters());
    add_params(lm_head.parameters());

    SGD sgd(params, LR, 0.9);

    // ---- 固定样本：输入序列 [1,3,5,7]，目标序列 [3,5,7,9]（预测下一个 token）----
    std::vector<size_t> src_ids = {1, 3, 5, 7};
    std::vector<size_t> tgt_ids = {3, 5, 7, 9};
    double scale = 1.0 / std::sqrt((double)D_K);

    double first_loss = -1.0, last_loss = -1.0;

    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        sgd.zeroGrad();

        // ---- Embedding: [SEQ, D_MODEL] ----
        auto x = tok_emb.forward(src_ids);   // [4, 8]

        // ---- Self-Attention ----
        auto Q = Wq.forward(x);   // [4, 8]
        auto K = Wk.forward(x);
        auto V = Wv.forward(x);

        // scores = Q @ K^T / sqrt(d_k)  → [4, 4]
        auto Kt     = ops::transpose(K, 0, 1);          // [8, 4]
        auto scores = ops::matmul(Q, Kt);               // [4, 4]
        // scale：手动 mul by scalar tensor
        auto scale_t = Tensor::create({1}, {scale});
        scores = ops::mul(scores, scale_t);
        auto attn   = ops::softmax(scores, 1);          // [4, 4]
        auto ctx    = ops::matmul(attn, V);             // [4, 8]
        auto ctx_proj = Wo.forward(ctx);                // [4, 8]

        // Add & Norm 1
        auto x1 = ops::add(x, ctx_proj);               // 残差
        auto x1n = ops::layerNorm(x1, ln1_w, ln1_b);  // [4, 8]

        // ---- FFN ----
        auto ff_out = ops::relu(ff1.forward(x1n));     // [4, 16]
        auto ff_out2 = ff2.forward(ff_out);            // [4, 8]

        // Add & Norm 2
        auto x2  = ops::add(x1n, ff_out2);
        auto x2n = ops::layerNorm(x2, ln2_w, ln2_b);  // [4, 8]

        // ---- LM Head ----
        auto logits = lm_head.forward(x2n);            // [4, 16]

        // ---- Loss ----
        auto loss = ops::crossEntropyLoss(logits, tgt_ids);

        if (epoch == 0) first_loss = loss->value()[0];
        last_loss = loss->value()[0];

        if (epoch % 40 == 0) {
            std::cout << "--------------------------------\n";
            std::cout << "epoch=" << epoch
                      << "  loss=" << loss->value()[0] << "\n";

            // 打印 argmax 预测
            std::cout << "  predictions: [";
            for (size_t s = 0; s < SEQ; ++s) {
                size_t best = 0;
                double best_v = logits->value()[s * VOCAB];
                for (size_t v = 1; v < VOCAB; ++v) {
                    if (logits->value()[s * VOCAB + v] > best_v) {
                        best_v = logits->value()[s * VOCAB + v];
                        best = v;
                    }
                }
                std::cout << best << (s+1 < SEQ ? "," : "");
            }
            std::cout << "]  targets: [";
            for (size_t s = 0; s < SEQ; ++s)
                std::cout << tgt_ids[s] << (s+1 < SEQ ? "," : "");
            std::cout << "]\n";
        }

        loss->backward();
        sgd.step();
    }

    std::cout << "\n  first_loss=" << first_loss
              << "  last_loss=" << last_loss << "\n";
    bool ok = last_loss < first_loss;
    std::cout << "  Transformer encoder loss decreased: "
              << (ok ? "✅[PASS]" : "[FAIL]") << "\n";
}

// ============================================================
//  main
// ============================================================
int main() {
    std::cout << std::fixed << std::setprecision(6);

    test_basic_ops();
    test_embedding();
    test_layernorm();
    test_cross_entropy();
    test_recommendation();
    test_transformer_encoder();

    std::cout << "\n✅  所有测试完成\n";
    return 0;
}