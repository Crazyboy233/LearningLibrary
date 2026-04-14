#include "../core/Tensor.h"
#include "../core/ops.h"
#include "../core/grad_fn.h"

#include <iostream>
#include <cmath>
#include <cassert>

// 编译命令
// g++ -std=c++17 -O0 ./core/grad_fn.cc ./core/Tensor.cc ./core/ops.cc ./test/01_test_autograd.cpp && ./a.out


// ============================================================
//  工具函数
// ============================================================
 
// 数值梯度：用有限差分验证解析梯度是否正确
// f(x+h) - f(x-h) / (2h)
double numericalGrad(
    std::function<double()> loss_fn,  // 当前 loss 值
    TensorPtr param,
    size_t idx,
    double h = 1e-4)
{
    auto val = param->value();
 
    // f(x + h)
    auto val_plus = val;
    val_plus[idx] += h;
    param->updateValue(val_plus);
    double loss_plus = loss_fn();
 
    // f(x - h)
    auto val_minus = val;
    val_minus[idx] -= h;
    param->updateValue(val_minus);
    double loss_minus = loss_fn();
 
    // 恢复原值
    param->updateValue(val);
 
    return (loss_plus - loss_minus) / (2.0 * h);
}
 
static int pass_count = 0;
static int fail_count = 0;

void check(const std::string& name, bool ok) {
    if (ok) {
        std::cout << "  [PASS] " << name << "\n";
        pass_count++;
    } else {
        std::cout << "  [FAIL] " << name << "\n";
        fail_count++;
    }
}
 
void checkNear(const std::string& name, double a, double b, double tol = 1e-5) {
    check(name, std::abs(a - b) < tol);
}

// ============================================================
//  Test 1: matmul forward
// ============================================================
void test_matmul_forward() {
    std::cout << "\n[Test 1] matmul forward\n";
    // A:[2,3]  B:[3,2]  =>  C:[2,2]
    auto A = Tensor::create({2, 3}, {1, 2, 3, 4, 5, 6}, false);
    auto B = Tensor::create({3,2}, {7,8, 9,10, 11,12}, false);
    auto C = ops::matmul(A, B);

    // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    // C[0,1] = 1*8 + 2*10+ 3*12 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 139
    // C[1,1] = 4*8 + 5*10+ 6*12 = 154
    checkNear("C[0,0]", C->value()[0], 58.0);
    checkNear("C[0,1]", C->value()[1], 64.0);
    checkNear("C[1,0]", C->value()[2], 139.0);
    checkNear("C[1,1]", C->value()[3], 154.0);
    check("no grad_fn when requires_grad=false", C->isLeaf());
}

// ============================================================
//  Test 2: 数值梯度 vs 解析梯度 — matmul + sum
// ============================================================
void test_matmul_grad() {
    std::cout << "\n[Test 2] matmul gradient (numerical check)\n";

    // loss = sum(X @ W)，X 不需要梯度，W 需要
    auto W = Tensor::create({2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, true);
    
    auto compute_loss = [&]() -> double {
        auto X = Tensor::create({2, 2}, {1, 2, 3, 4}, false);
        auto out = ops::matmul(X, W);
        auto loss = ops::sum(out);
        return loss->value()[0];
    };

    // 解析梯度
    {
        auto X = Tensor::create({2, 2}, {1, 2, 3, 4}, false);
        auto out = ops::matmul(X, W);
        auto loss = ops::sum(out);
        loss->backward();
    }
    auto analytic = W->grad();

    // 数值梯度
    for (size_t i = 0; i < W->size(); ++i) {
        double ng = numericalGrad(compute_loss, W, i);
        W->zeroGrad();
        checkNear("W grad[" + std::to_string(i) + "]", analytic[i], ng, 1e-4);
    }
}
// ============================================================
//  Test 3: sigmoid 梯度
// ============================================================
void test_sigmoid_grad() {
    std::cout << "\n[Test 3] sigmoid gradient (numerical check)\n";

    auto x = Tensor::create({1,3}, {0.0, 1.0, -1.0}, true);

    auto compute_loss = [&]() {
        auto y = ops::sigmoid(x);
        auto loss = ops::sum(y);
        return loss->value()[0];
    };

    // 解析梯度
    {
        auto y = ops::sigmoid(x);
        auto loss = ops::sum(y);
        loss->backward();
    }
    auto analytic = x->grad();

    for (size_t i = 0; i < x->size(); ++i) {
        x->zeroGrad();
        double ng = numericalGrad(compute_loss, x, i);
        checkNear("x grad[" + std::to_string(i) + "]", analytic[i], ng, 1e-4);
    }
}

// ============================================================
//  Test 4: add broadcast 梯度
// ============================================================
void test_add_broadcast_grad() {
    std::cout << "\n[Test 4] add broadcast gradient\n";

    // A:[3,2] + b:[1,2]，b 被 broadcast
    auto A = Tensor::create({3,2}, {1,2, 3,4, 5,6}, false);
    auto b = Tensor::create({1,2}, {0.5, -0.5}, true);

    auto y = ops::add(A, b);
    auto loss = ops::sum(y);
    loss->backward();

    // db = sum over rows of grad (grad 全1)，每列各加3次
    checkNear("b grad[0]", b->grad()[0], 3.0);
    checkNear("b grad[1]", b->grad()[1], 3.0);
}

// ============================================================
//  Test 5: relu 梯度
// ============================================================
void test_relu_grad() {
    std::cout << "\n[Test 5] relu gradient\n";

    auto x = Tensor::create({1,4}, {-2.0, -0.5, 0.5, 2.0}, true);
    auto y = ops::relu(x);
    auto loss = ops::sum(y);
    loss->backward();

    // x<0 → grad=0，x>0 → grad=1
    checkNear("relu grad[-2.0]", x->grad()[0], 0.0);
    checkNear("relu grad[-0.5]", x->grad()[1], 0.0);
    checkNear("relu grad[0.5]",  x->grad()[2], 1.0);
    checkNear("relu grad[2.0]",  x->grad()[3], 1.0);
}

// ============================================================
//  Test 6: requires_grad=false 的输入不累积梯度
// ============================================================
void test_no_grad_propagation() {
    std::cout << "\n[Test 6] requires_grad=false stops gradient\n";

    auto X = Tensor::create({1,2}, {1.0, 2.0}, false);  // 不需要梯度
    auto W = Tensor::create({2,2}, {1,0, 0,1}, true);

    auto out = ops::matmul(X, W);
    auto loss = ops::sum(out);
    loss->backward();

    // X.grad 应该为空（保持全零初始化，不被累加）
    bool x_grad_zero = true;
    for (double g : X->grad()) {
        if (g != 0.0) { 
            x_grad_zero = false; 
            break; 
        }
    }
    check("X.grad stays zero", x_grad_zero);
    check("W.grad non-zero",   W->grad()[0] != 0.0);
}

// ============================================================
//  Test 7: 完整训练循环 — 单层线性 + sigmoid + MSE
//  y = sigmoid(X @ W + b)，拟合 XOR
// ============================================================
void test_training_loop() {
    std::cout << "\n[Test 7] training loop — XOR single step\n";

    // 参数
    auto W = Tensor::create({2,1}, {0.5, -0.3}, true);
    auto b = Tensor::create({1,1}, {0.1}, true);

    // 单个样本：X=[0,1]，target=1
    auto X      = Tensor::create({1,2}, {0.0, 1.0}, false);
    auto target = Tensor::create({1,1}, {1.0}, false);

    double lr = 0.5;
    double initial_loss = -1.0;
    double final_loss   = -1.0;

    for (size_t step = 0; step < 20; ++step) {
        // forward
        auto y = ops::matmul(X, W);
        auto z = ops::add(y, b);
        auto pred = ops::sigmoid(z);

        // MSE loss = sum((pred - target)^2)
        auto diff = ops::sub(pred, target);
        auto diff2 = ops::mul(diff, diff);
        auto loss = ops::sum(diff2);

        if (step == 0) {
            initial_loss = loss->value()[0];
        }
        if (step == 19) {
            final_loss = loss->value()[0];
        }

        // backward
        loss->backward();

        // SGD update
        auto wv = W->value();
        auto bv = b->value();
        for (size_t i = 0; i < wv.size(); ++i) {
            wv[i] -= lr * W->grad()[i];
        }
        for (size_t i = 0; i < bv.size(); ++i) {
            bv[i] -= lr * b->grad()[i];
        }
        W->updateValue(wv);
        b->updateValue(bv);

        // 梯度清零
        W->zeroGrad();
        b->zeroGrad();
    }
    std::cout << "  initial loss = " << initial_loss << "\n";
    std::cout << "  final   loss = " << final_loss   << "\n";
    check("loss decreased after 20 steps", final_loss < initial_loss);
}

// ============================================================
//  Test 8: 多次 backward 梯度累积
// ============================================================
void test_grad_accumulation() {
    std::cout << "\n[Test 8] gradient accumulation\n";

    auto W = Tensor::create({1,1}, {2.0}, true);

    // 两次 forward+backward，不清零
    for (int i = 0; i < 2; ++i) {
        auto x    = Tensor::create({1,1}, {3.0}, false);
        auto out  = ops::matmul(x, W);   // out = 2*3 = 6
        auto loss = ops::sum(out);
        loss->backward();
        // dW = x = 3.0，两次累积应为 6.0
    }

    checkNear("accumulated grad = 6.0", W->grad()[0], 6.0);
}


int main() {
    std::cout << "=== Dynamic Autograd Tests ===\n";
 
    test_matmul_forward();
    test_matmul_grad();
    test_sigmoid_grad();
    test_add_broadcast_grad();
    test_relu_grad();
    test_no_grad_propagation();
    test_training_loop();
    test_grad_accumulation();
 
    std::cout << "\n==============================\n";
    std::cout << "PASS: " << pass_count << "  FAIL: " << fail_count << "\n";
    return fail_count == 0 ? 0 : 1;
}