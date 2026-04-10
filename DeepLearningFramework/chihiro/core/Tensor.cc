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
    if (size() != 1) {
        throw std::runtime_error("backward() 只能从标量调用，请先用 ops::sum() 规约");
    }

    // loss 的初始梯度为 1.0
    grad_ = {1.0};

    /*
    ----------------------------------------------------------
    第一步：拓扑排序
    从 loss 出发，沿 grad_fn_->saved_inputs_ 做 DFS
    收集所有需要参与反向的 Tensor，得到逆拓扑序列表。
    ----------------------------------------------------------
    */
    std::vector<TensorPtr> topo;        // 逆拓扑序（从 loss 到叶节点）
    std::unordered_set<Tensor*> visited;

    // 递归 DFS, 收集节点
    std::function<void(TensorPtr)> build_topo = [&](TensorPtr t) {
        if (visited.count(t.get())) return;
        visited.insert(t.get());

        if (t->grad_fn_) {
            for (auto& weak_input : t->grad_fn_->saved_inputs_) {
                auto input = weak_input.lock();
                if (input && input->requireGrad()) {
                    build_topo(input);
                }
            }
        }

        topo.push_back(t);  // 后序；先处理依赖，再压自己
    };


}