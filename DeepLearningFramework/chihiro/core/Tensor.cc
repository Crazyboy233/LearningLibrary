#include "Tensor.h"
#include <unordered_set>

/*
============================================================
Tensor::backward()

    从当前 Tensor(必须是标量 loss) 出发，反向遍历计算图

    1、从 loss 开始，沿 grad_fn_ 链做拓扑排序(后序)
    2、将逆拓扑排序依次调用每个 GradFn::apply()
    3、将返回的梯度累加到对应输入 Tensor 的 grad_ 上

注意：只处理 requires_grad = true 的 Tensor
============================================================
*/
void Tensor::backward() {
    if (numel() != 1) {
        throw std::runtime_error("backward() 只能从标量调用，请先用 ops::sum() 归约");
    }

    // loss 的初始梯度为 1.0
    grad_ = {1.0};

    /*
    ----------------------------------------------------------
    第一步：DFS 拓扑排序
    从 loss 出发，沿 grad_fn_->saved_inputs_ 做 DFS
    收集所有需要参与反向的 Tensor，得到逆拓扑序列表。
    ----------------------------------------------------------
    */
    std::vector<TensorPtr> topo;        // 逆拓扑序（从 loss 到叶节点）
    std::unordered_set<Tensor*> visited;

    // 递归 DFS, 收集节点
    std::function<void(TensorPtr)> build_topo = [&](TensorPtr t) {
        if (visited.count(t.get())) return; // 已访问 则跳过
        visited.insert(t.get());

        if (t->grad_fn_) {
            for (auto& input : t->grad_fn_->saved_inputs_) {
                if (input && input->requireGrad()) {
                    build_topo(input);
                }
            }
        }
        topo.push_back(t);  // 后序；先处理依赖，再压自己
    };

    build_topo(shared_from_this());

    /*
    topo 现在是从叶节点到 loss 的顺序，反向遍历即逆拓扑序
    ----------------------------------------------------------
    第二步：逆拓扑序调用各 GradFn::apply()
    ----------------------------------------------------------
    */
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        TensorPtr t = *it;
        
        // 叶节点没有 grad_fn_,梯度已经累加完毕，跳过
        if(!t->grad_fn_) {
            continue;
        }

        // 拿到该节点的输出梯度
        const std::vector<double>& grad_out = t->grad_;

        // 调用 GradFn,得到各输入的梯度
        auto grads = t->grad_fn_->apply(grad_out);

        // 将梯度累加到对应输入 Tensor 上
        const auto& inputs = t->grad_fn_->saved_inputs_;
        assert(grads.size() == inputs.size());

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input = inputs[i];
            if (input && input->requireGrad()) {
                input->addGrad(grads[i]);
            }
        }
    }
}