# 环境安装
## Ubuntu
```bash
cd DeepLearningFramework/pytorch
python3 -m venv torch-env
source torch-env/bin/activate
# CPU 版（通用）
python3 -m pip install torch torchvision torchaudio

# GPU 版（英伟达显卡）
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## MacOS
```bash
cd DeepLearningFramework/pytorch
python3 -m venv torch-env
# 激活虚拟环境
source torch-env/bin/activate
# PyTorch 对 macOS 优化极好，M 芯片支持 GPU 加速（Apple Silicon）
# 通用命令（自动适配 Intel/M 芯片，M 芯片自动启用 Metal 加速）
python3 -m pip install torch torchvision torchaudio
```

---
## 通用
**验证安装是否成功**

```python
# 验证 PyTorch
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")  # Ubuntu GPU 版会返回 True
print(f"MPS 可用 (macOS M 芯片): {torch.backends.mps.is_available()}")  # macOS M 芯片返回 True
```

**虚拟环境**
```bash
# 使用之前需要激活虚拟环境
source torch-env/bin/activate
# 退出虚拟环境
deactivate
```