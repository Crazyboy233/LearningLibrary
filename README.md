# 个人学习使用

### 目录说明

| 目录 / 文件 | 内容 |
| --- | --- |
| `c++/` | C++ 笔记（CMake 用法、杂项） |
| `DeepLearningFramework/chihiro/` | 手写深度学习框架（C++ + pybind，CMake 构建） |
| `pytorch/` | PyTorch 练习，以及 Transformer 从零实现（`work/02_transformer.py`） |
| `tf/` | TensorFlow 练习，以及同一个 Transformer 的 TF 版实现 |
| `doc/` | Transformer / 名词解释 / 训练流程图等笔记，`doc/site/` 是汇总的静态站点 |
| `env/` | 环境配置记录（Miniconda、WSL2） |
| `linux/` | Linux 常用命令 |
| `lua/` | Lua 基础、进阶笔记与数据描述 |
| `python/` | Python 内置函数、容器及方法示例 |
| `剑指offer/` | 刷题代码（C++） |
| `各种三方库.md` | 第三方库速查 |

### 跑示例

虚拟环境就近放在各自目录下（`pytorch/torch-env`、`tf/tf-env`，已被 gitignore）：

```bash
cd pytorch && source torch-env/bin/activate && python work/02_transformer.py
cd tf      && source tf-env/bin/activate    && python work/02_transformer.py
```

训练产物写到当前目录的 `results_transformer/`。
