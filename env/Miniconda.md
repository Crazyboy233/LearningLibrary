## 安装 Miniconda
```bash
# Ubuntu
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 这里一路 Enter+yes
bash Miniconda3-latest-Linux-x86_64.sh 

source ~/.bashrc

# 验证
conda --version
```

## 使用
```bash
# 创建新环境
conda create -n pytorch python=3.12
# 这里 -n pytorch 环境名字
# python=3.12 python 版本

# 进入base环境
conda activate base

# 查看所有环境
conda env list

# 激活环境
conda activate pytorch

# 退出环境
conda deactivate

# 删除环境
conda remove -n pytorch --all

# 查看当前python
which python
```