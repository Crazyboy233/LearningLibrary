#include "../core/Tensor.h"
#include "../core/ops.h"
#include "../core/grad_fn.h"

#include <iostream>
#include <cmath>
#include <cassert>

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
    std::cout << "\n[Test] matmul forward\n";
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
    std::cout << "\n[Test] matmul gradient (numerical check)\n";

    // loss = sum(X @ W)，X 不需要梯度，W 需要
    auto W = Tensor::create({2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, true);
    
    auto compute_loss = [&]() -> double {
        auto X = Tensor::create({2, 2}, {1, 2, 3, 4}, false);
        auto out = ops::matmul(X, W);
        auto loss = ops::sum(out);
        return loss->value()[0];
    };


}

int main() {

    return 0;
}