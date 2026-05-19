#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/Tensor.h"
#include "../core/ops.h"
#include "../core/linear.h"
#include "../core/optimizer.h"
#include "../core/embeddings.h"

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    m.doc() = "chihiro core";

    // ----------------------------------------------------------------
    // NoGradGuard
    // 注：当前阶段其实用不到该类，该类仅用于推理优化。
    // ----------------------------------------------------------------
    py::class_<NoGradGuard>(m, "NoGradGuard",
        "Context manager that disables gradient computation.\n\n",
        "Usage::\n\n"
        "    with core.NoGradGuard():\n"
        "        out = model.forward(x)  # no GradFn created\n")
    .def(py::init<>())
    .def("__enter__", [](NoGradGuard& g) -> NoGradGuard& { return g; })
    .def("__exit__", [](NoGradGuard&, py::object, py::object, py::object) {});

    // ----------------------------------------------------------------
    // Tensor
    // ----------------------------------------------------------------
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        // 工厂：从 Python list 创建叶节点
        // t = core.Tensor([2, 3], [1,2,3,4,5,6], requires_grad=True)
        .def(py::init([](std::vector<size_t> shape,
                        std::vector<double> data,
                        bool requires_grad) {
            return Tensor::create(shape, data, requires_grad);
            }), py::arg("shape"), py::arg("data"), py::arg("requires_grad") = false)
        
        // 属性
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("data", &Tensor::value)
        .def_property_readonly("grad", &Tensor::grad)
        .def_property_readonly("requires_grad", &Tensor::requireGrad)
        .def_property_readonly("is_leaf", &Tensor::isLeaf)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("size", &Tensor::size)

        // 反向 / 梯度操作
        .def("backward", &Tensor::backward,
            "Start reverse-mode AD from this scalar tensor.")   // 从该标量张量开始反向自动微分
        .def("zero_grad", &Tensor::zeroGrad,
            "Reset gradient buffer to zero.")
        .def("update_value", &Tensor::updateValue,
            "Overwrite parameter values in-place (used by optimizers).")
        
        // 方便调试
        .def("__repr__", [](const Tensor& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            s += "], requires_grad=" + std::string(t.requireGrad() ? "True" : "False") + ")";
            return s;
        });
    
    // ----------------------------------------------------------------
    // ops namespace  →  minigrad.ops 子模块
    // ----------------------------------------------------------------
    py::module_ mops = m.def_submodule("ops", "Differentiable ops");
    
    mops.def("add",     &ops::add,      py::arg("a"), py::arg("b"));
    mops.def("sub",     &ops::sub,      py::arg("a"), py::arg("b"));
    mops.def("mul",     &ops::mul,      py::arg("a"), py::arg("b"));
    mops.def("matmul",  &ops::matmul,   py::arg("a"), py::arg("b"));
    mops.def("relu",    &ops::relu,     py::arg("a"));
    mops.def("sigmoid", &ops::sigmoid,  py::arg("a"));
    mops.def("sum",     &ops::sum,      py::arg("a"));
    mops.def("bce_with_logits_loss", &ops::bceWithLogitsLoss, py::arg("logits"), py::arg("target"));
    mops.def("cat",     &ops::cat,      py::arg("inputs"),
        "Concatenate tensors along dim=1 (column-wise)");
    
    // ----------------------------------------------------------------
    // Linear
    // ----------------------------------------------------------------
    py::class_<Linear>(m, "Linear",
        "Fully-connected layer: y = x @ W + b\n\n"
        "Parameters\n----------\n"
        "in_features  : int\n"
        "out_features : int\n"
        "seed         : int, optional (default 42, 0 = random device)\n")
        .def(py::init<size_t, size_t, unsigned>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("seed") = 42)
        .def("forward",    &Linear::forward, py::arg("x"))
        .def("parameters", &Linear::parameters)
        .def("zero_grad",  &Linear::zeroGrad)
        .def_property_readonly("W",            &Linear::W)
        .def_property_readonly("b",            &Linear::b)
        .def_property_readonly("in_features",  &Linear::inFeatures)
        .def_property_readonly("out_features", &Linear::outFeatures)
        .def("__call__",   &Linear::forward, py::arg("x"))  // 实例方法，让实例可像函数一样调用(实例())
        .def("__repr__",   [](const Linear& l) {    // 定义 repr(实例) 的输出
            return "Linear(in=" + std::to_string(l.inFeatures()) +
                   ", out=" + std::to_string(l.outFeatures()) + ")";
        });
    
    // ----------------------------------------------------------------
    // Embedding
    // ----------------------------------------------------------------
    py::class_<Embedding>(m, "Embedding",
        "Lookup-table layer: out[i] = W[ids[i]]\n\n"
        "Parameters\n----------\n"
        "num_embeddings : int  — vocabulary size\n"
        "embedding_dim  : int  — vector dimension\n"
        "seed           : int, optional (default 42)\n")
        .def(py::init<size_t, size_t, unsigned>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"), py::arg("seed") = 42)
        // forward 接受整数 id 列表
        .def("forward",
             py::overload_cast<const std::vector<size_t>&>(&Embedding::forward),
             py::arg("ids"))
        .def("parameters",      &Embedding::parameters)
        .def("zero_grad",       &Embedding::zeroGrad)
        .def_property_readonly("weight",         &Embedding::weight)
        .def_property_readonly("num_embeddings", &Embedding::numEmbeddings)
        .def_property_readonly("embedding_dim",  &Embedding::embeddingDim)
        .def("__call__",
             py::overload_cast<const std::vector<size_t>&>(&Embedding::forward),
             py::arg("ids"))
        .def("__repr__", [](const Embedding& e) {
            return "Embedding(" + std::to_string(e.numEmbeddings()) +
                   ", " + std::to_string(e.embeddingDim()) + ")";
        });
    
    // ----------------------------------------------------------------
    // SGD
    // ----------------------------------------------------------------
    py::class_<SGD>(m, "SGD",
        "Stochastic Gradient Descent (with optional Polyak momentum).\n\n"
        "Parameters\n----------\n"
        "params   : list[Tensor]\n"
        "lr       : float\n"
        "momentum : float, optional (default 0.0)\n")
        .def(py::init<const std::vector<TensorPtr>&, double, double>(),
             py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0)
        .def("step",     &SGD::step,     "Apply one gradient-descent step.")
        .def("zero_grad",&SGD::zeroGrad, "Zero gradients of all managed parameters.")
        .def("__repr__", [](const SGD& s) {
            // SGD 没有暴露 lr_ 字段，简单显示类名
            return "SGD(...)";
        });
}