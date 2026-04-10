#pragma once
#include "Tensor.h"

/*
============================================================
    namespace ops

    所有算子以自由函数形式暴露
    每个函数：
        1、执行 forward 计算
        2、若 NoGradGuard 未激活，构造对应的 GradFn，保存所需中间值
        3、返回挂载了 GradFn 的输出 TensorPtr
============================================================
*/
namespace ops {

    TensorPtr add(const TensorPtr& a, const TensorPtr& b);
    TensorPtr sub(const TensorPtr& a, const TensorPtr& b);
    TensorPtr mul(const TensorPtr& a, const TensorPtr& b);
    TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
    TensorPtr relu(const TensorPtr& a);
    TensorPtr sigmoid(const TensorPtr& a);
    TensorPtr sum(const TensorPtr& a);  // 全局求和 → 标量，用于构造 loss

}   // namespace ops