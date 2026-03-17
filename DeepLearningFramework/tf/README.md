# 环境安装
## Ubuntu
```bash
cd DeepLearningFramework/tf
python3 -m venv tf-env
source tf-env/bin/activate
# CPU 版（推荐新手）
python3 -m pip install tensorflow

# GPU 版（需英伟达显卡 + CUDA/cuDNN，需先装驱动和依赖）
# 先装 CUDA/cuDNN 依赖（以 Ubuntu 20.04/22.04 为例）
# sudo apt install nvidia-cuda-toolkit libcudnn8
python3 -m pip install tensorflow[and-cuda]  # 自动适配 CUDA/cuDNN
```

## MacOS
```bash
cd DeepLearningFramework/tf
python3 -m venv tf-env
source tf-env/bin/activate
# 通用安装命令（自动适配 Intel/M 芯片）
python3 -m pip install tensorflow

# 可选：安装苹果官方加速插件（M 芯片推荐）
python3 -m pip install tensorflow-metal
```
---
## 通用
**验证安装是否成功**
```python
# 验证 tensorflow
import tensorflow as tf
print(f"TensorFlow 版本: {tf.__version__}")
print(f"TensorFlow 是否可用: {tf.test.is_built_with_cuda() if tf.test.is_built_with_cuda() else 'CPU 模式'}")
```

**虚拟环境**
```bash
# 使用之前需要激活虚拟环境
source torch-env/bin/activate
# 退出虚拟环境
deactivate
```