# git

##  **基础配置（首次使用必做）**

```bash
# 设置全局身份标识（和GitHub账号一致）
git config --global user.name "BoBolilla"
git config --global user.email "1326511780@qq.com"

# 启用颜色标记（查看diff更直观）
git config --global color.ui auto

# 设置默认编辑器为nano（避免不熟悉的vim）
git config --global core.editor nano
```

## **克隆你的GitHub项目**

```bash
cd ~/work/paper  # 进入你的工作目录
git clone https://github.com/你的用户名/项目名.git
cd 项目名
```

## 环境迁移的最佳实践

####  本地操作（在你的 Windows/Mac 电脑上）：

1. 打开 Anaconda Prompt 或终端：

   ```bash
   conda activate 你的本地环境名
   conda env export --no-builds > environment.yml
   ```

   > 📌 `--no-builds` 是关键！可避免依赖特定平台编译版本导致失败

####  服务器操作（在 VSCode 远程终端）：

```bash
# 上传生成的 environment.yml 到项目根目录
cd ~/你的项目名
conda env create -f environment.yml --name 你的环境名
```

### 🚀 环境优化技巧

```bash
conda install cudatoolkit=11.8 cudnn=8.9  # 需匹配服务器驱动版本
nvidia-smi  # 查看支持的 CUDA 版本(右上角显示)
```

### 🧪 项目启动流程

```bash
conda activate 你的环境名
pip install -e .  # 安装项目自身依赖
python train.py  # 启动主程序
```

### ⚠️ 必看避坑指南

1. **路径差异处理：**

   ```python
   # 建议在代码开头添加
   import os
   DATA_PATH = os.path.expanduser('~/dataset/')  # 自动适配不同系统
   ```

2. **大文件传输：**

   bash

   bash

   复制

   ```bash
   # 使用 rsync 传输大文件
   rsync -avzP ./big_data/ hyzheng@172.23.148.118:~/project/data/
   ```

3. **环境诊断命令：**

   bash

   bash

   复制

   ```bash
   conda list  # 验证包安装
   nvidia-smi -l 5  # 实时监控 GPU 使用
   ```

### 💡 高级技巧

创建 `launch.sh` 避免重复操作：

```bash
#!/bin/bash
conda activate your_env
python train.py --batch-size=256 --epochs=100
```

运行方式：`chmod +x launch.sh && ./launch.sh`

### 📊 环境重建流程总结

| 步骤        | 本地操作                     | 服务器操作                        |
| ----------- | ---------------------------- | --------------------------------- |
| 1. 导出环境 | `conda env export > env.yml` | 上传 env.yml                      |
| 2. 创建环境 | -                            | `conda env create -f env.yml`     |
| 3. 激活环境 | -                            | `conda activate your_env`         |
| 4. 附加安装 | -                            | `pip install -r requirements.txt` |

**重要提示：** 首次运行时使用 `nohup python -u train.py > log.txt &` 可防止网络中断导致训练终止

> 🔭 建议在环境稳定后创建镜像：`conda env export --no-builds --from-history > clean_env.yml` 可精简环境描述文件

你现在可以流畅地在服务器上启动实验了！当需要更新代码时：

```bash
git pull origin main  # 同步最新代码
conda env update -f environment.yml  # 更新环境
```

开始你的科研旅程吧！遇到具体报错时可随时提供日志，我会帮你诊断问题根源 🚀

## requirments.txt导出

```bash
pip install pipreqs
pipreqs /path/to/project --force  # --force 覆盖已有文件
```

## conda

删除环境

```bash
conda env remove --n tf_gpu_env

pip install -r requirements.txt
```

### 安装tensorflow

```bash
conda create -n tf python=3.9
conda activate tf
pip install tensorflow==2.12
# 下载 CUDA Toolkit 11.8
cd ~
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run

./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-11.8
nvcc --version

# https://developer.nvidia.com/rdp/cudnn-download
# 选择：cuDNN v8.6.x → for CUDA 11.x → Linux → Local Installer (tar)
# 将其上传到服务器(拖到服务器)
cd ~
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
cp cudnn-*-archive/include/* $HOME/cuda-11.8/include
cp cudnn-*-archive/lib/* $HOME/cuda-11.8/lib64

# 配置环境变量（添加到虚拟环境）
export PATH=$HOME/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-11.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

# 验证
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"


```

