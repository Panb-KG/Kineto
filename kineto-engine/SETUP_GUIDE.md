# Kineto Engine - 环境配置指南
# 适用于 Intel AI Box / MoFang M01（**Intel Arc Pro B60，24GB 显存，Battlemage**）或本地开发机

> ⚠️ **重要更正**：这台 Intel AI Box **不是 NVIDIA 机器，没有 CUDA，没有 `nvidia-smi`**。
> 它用 **PyTorch XPU** 后端（`torch>=2.5` 原生支持）。本文早期版本曾假设它为
> “NVIDIA CUDA 24GB”，现已修正。**生产环境请走下面的「方案 1：Intel XPU」。**
>
> 完整的设备侧部署流程（只读探测 → 可逆停用 MoFang 上层 → Intel GPU 驱动 → venv/XPU 安装
> → 模型 rsync → systemd → Cloudflare Tunnel → 验收/回滚）见仓库根目录
> **[`DEPLOY_MOFANG.md`](../DEPLOY_MOFANG.md)** 与 `deploy/` 工具包。本文只讲“环境怎么装”。

## 📋 环境前置要求

### 硬件配置
- **显卡支持**（按本项目优先级排序）：
  - ★ **Intel Arc GPU + PyTorch XPU** —— **Intel AI Box / MoFang M01 的生产路径**
    （Arc Pro B60 24GB；需 Intel GPU 驱动：内核 `i915`/`xe` + compute-runtime + Level Zero）
  - NVIDIA GPU + CUDA 12.1+（RTX 4080 / A100 等）—— 仅适用于**另外的** N 卡机器，不是这台 AI Box
  - Apple Silicon Mac (M1/M2/M3/M4) - 使用 MPS（开发调试）
  - ❌ CPU 模式——**在 Intel AI Box 上生产不可行**，仅用于本机验证代码逻辑

- **内存要求**：
  - 显存：Intel AI Box 为 **24GB（Arc Pro B60）**，HMR2 ViT-H 峰值占用约 8-16GB，余量充足
  - 系统内存：**至少 16GB**（门禁硬要求，见 `deploy/discover_device.sh` 的 G1）
  - 可用磁盘：**至少 25GB**（venv ≈ 8G + 模型 2.6G + job 产物）

### 软件先决条件
- **OS**：Ubuntu **24.04 LTS**（或 26.04）。PyTorch XPU wheel 只在 24.04/26.04 验证过；
  **22.04 需走 IPEX 路径或先升级系统**。macOS 仅用于开发。
- **Intel GPU 驱动**（设备上，需 sudo）：`intel-opencl-icd`、`intel-level-zero-gpu`、`level-zero`、
  `libze1`、`libze-intel-gpu1`；`/dev/dri/renderD128` 必须存在，运行用户必须在 **`render`** 与 **`video`** 组
- **Python** 3.10 或 3.11（3.12 可能有兼容性问题）。设备上推荐 **3.11 + venv**；
  Ubuntu 24.04 受 PEP 668 保护，**不要**往系统 Python 里 `pip install`
- **Conda 或 Mamba**（仅本地开发机推荐 Mamba，速度快 10 倍）；设备上用 `python3.11 -m venv` 即可
- **Git**、**ffmpeg**、**libosmesa6**（无显示器环境渲染 demo 视频靠 OSMesa）

---

## 🔧 快速配置步骤

### ★ 方案 1: Intel XPU 环境（**推荐用于 Intel AI Box / MoFang M01**）

```bash
# ------------------------------------------------------------------
# 第 0 步：只读探测（不改任何状态，末尾自带 PASS/FAIL 门禁汇总）
# ------------------------------------------------------------------
bash deploy/discover_device.sh

# ------------------------------------------------------------------
# 第 1 步：Intel GPU 驱动（需 sudo；已装则 apt 提示已是最新，幂等）
# 完整步骤含 Intel APT 源与 GPG key 配置，见 DEPLOY_MOFANG.md §5.3
# ------------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero \
                        libze1 libze-intel-gpu1 \
                        libgl1 libglib2.0-0 libsm1 libxext6 libxrender1 \
                        libosmesa6 ffmpeg
sudo usermod -aG render,video $USER      # 之后必须重新登录才生效
ls -l /dev/dri                           # 应看到 card0 与 renderD128

# ------------------------------------------------------------------
# 第 2 步：隔离环境（Ubuntu 24.04 上用 venv，不用 conda）
# ------------------------------------------------------------------
sudo apt-get install -y python3.11 python3.11-venv
sudo mkdir -p /opt/kineto && sudo chown $USER:$USER /opt/kineto
python3.11 -m venv /opt/kineto/venv
source /opt/kineto/venv/bin/activate
pip install --upgrade pip wheel setuptools

# ------------------------------------------------------------------
# 第 3 步：★ 先装 PyTorch XPU（顺序不可颠倒！）
# requirements.txt 里有裸 torch>=2.0.0，先装它会把 XPU 轮子覆盖成 CPU 轮子
# ------------------------------------------------------------------
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 可选增强：Intel Extension for PyTorch（版本必须与 torch 主次版本对齐：2.7.10 ↔ torch 2.7.x）
# kineto_core.py 会机会性 import 它，装了能拿到算子融合等加速；不装也不影响 XPU 可用性
pip install intel-extension-for-pytorch==2.7.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# ------------------------------------------------------------------
# 第 4 步：★ 验证 XPU（唯一可信判据）
# ------------------------------------------------------------------
python -c "import torch;print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
#   期望输出形如: True Intel(R) Arc(TM) Pro B60 Graphics

# ------------------------------------------------------------------
# 第 5 步：Kineto 依赖（装完立即复查 torch 仍是 +xpu）
# ------------------------------------------------------------------
pip install -r requirements.txt
python -c "import torch;print(torch.__version__, torch.xpu.is_available())"
#   若变成了 CPU 轮子，修回：
#   pip install --force-reinstall --no-deps torch torchvision torchaudio \
#       --index-url https://download.pytorch.org/whl/xpu

# ------------------------------------------------------------------
# 第 6 步：4DHumans
# ★ 不要装 '[all]' extra：它会拖入 detectron2 源码编译，在边缘设备上极易失败且本引擎不需要
# ------------------------------------------------------------------
# 选项 A（本地开发机）: 从源码安装
#   ⚠️ 上游原版在 torch>=2.6 下加载 .ckpt 会 UnpicklingError（weights_only 默认变 True）。
#   git clone 拿到的是**未打补丁**的上游版本，clone 后必须立即打上本仓库的补丁（含 2 个 hunk：
#   模块级 torch.load→weights_only=False monkey-patch + load 前 add_safe_globals）：
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
# ★ 不可跳过：上游在 torch>=2.6 下加载 .ckpt 会 UnpicklingError（weights_only 默认变 True），补丁含 2 个 hunk。
#   路径是 ../../deploy —— 4D-Humans 在 kineto-engine/ 下、deploy 在仓库根，故要上**两**级；
#   写成 ../deploy 会解析成 kineto-engine/deploy（不存在）→ 补丁打不上 → 之后被静默吞错出合成数据。
PATCH=../../deploy/patches/hmr2_weights_only.patch
#   幂等守卫：先 --check；能正向应用就应用，已应用过（可反向）就跳过，都不行则大声失败退出。
if git apply --check "$PATCH" 2>/dev/null; then
    git apply -p1 "$PATCH" && echo '[patch] 已应用 hmr2_weights_only.patch'
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo '[patch] 已应用过，跳过（幂等）'
else
    echo '[patch][FATAL] 补丁既不能正向也不能反向应用 —— 上游版本可能已变，请人工核对' >&2; exit 1
fi
#   阻断式断言：两处改动必须都在，否则**大声失败**（绝不让它静默吞掉 → 合成关键点 exit 0）
grep -q weights_only hmr2/models/__init__.py && grep -q add_safe_globals hmr2/models/__init__.py \
    && echo '[patch] 两处补丁均 OK（weights_only + add_safe_globals）' \
    || { echo '[patch][FATAL] 断言失败：hmr2/models/__init__.py 缺 weights_only/add_safe_globals，补丁未生效' >&2; exit 1; }
pip install -e . && cd ..
# 选项 B（设备）: ⚠️ .gitignore 排除了 kineto-engine/4D-Humans/ 与 yolov8n.pt，设备端 git clone
#                拿不到本地这份（含 torch>=2.6 的两处补丁），必须 rsync 过去：
#                · 代码树（含 4D-Humans/，已带补丁）→ DEPLOY_MOFANG.md §5.5 的 rsync 命令
#                · 若设备端改用 git clone，则同样要在 cd 4D-Humans 后 `git apply -p1 ../../deploy/patches/hmr2_weights_only.patch`
#                · 模型权重与 SMPL      → 下面这个脚本（含 sha256 校验，只增不删）
bash deploy/transfer_models.sh --device <user>@192.168.1.107 --wire

# ------------------------------------------------------------------
# 第 7 步：预加载模型权重（首次运行会自动下载 2.5GB，但设备上应预先 rsync 过去）
# 缓存路径由 $HOME 决定：hmr2/configs 里 CACHE_DIR = $HOME/.cache → $HOME/.cache/4DHumans
# ------------------------------------------------------------------
python -c "import torch;from kineto_core import PoseExtractor; \
pe=PoseExtractor(torch.device('xpu')); print('Models loaded, mode =', pe.mode)"
#   期望：mode 为 '4dhumans'（不是 'fallback'）——这是 api.py 的上线硬门禁

# ------------------------------------------------------------------
# 第 8 步：跑完整流程（detect_device() 会自动选到 xpu；CLI 没有 --device 参数）
# --output 是**输出目录**，pose_data.json 与 demo_output.mp4 都写到该目录下
# ------------------------------------------------------------------
OMP_NUM_THREADS=4 PYOPENGL_PLATFORM=osmesa \
    python kineto_core.py --input input_video.mp4 --output ./output
#   日志应出现：[Device] Intel XPU: Intel(R) Arc(TM) Pro B60 Graphics
```

> 生产部署不要手工跑第 8 步，而是用 `deploy/kineto-engine.service`（systemd）把
> `uvicorn api:app --host 127.0.0.1 --port 8000` 托管起来，再用 `deploy/validate.sh` 跑 G1..G10 验收。

---

### 方案 2: Apple Silicon Mac (M1/M2/M3) - MPS 后端

```bash
# 1. 创建环境
conda create -n kineto-engine python=3.11 -y
conda activate kineto-engine

# 2. 安装 PyTorch + MPS 支持
pip install torch torchvision torchaudio

# 3. 验证 MPS 支持
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}'); print(f'Built with MPS: {torch.backends.mps.is_built()}')"

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装 4DHumans（同上）
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
pip install -e .
```

**注意**：MPS 在某些操作上可能不支持（如某些 SMPL-X 计算），此时代码会自动降级到 CPU。

---

### ⚠️ 方案 3: CPU 专用（测试/调试环境）— **Intel AI Box 上生产不可行**

> ❌ **不要**在 Intel AI Box 上把 CPU 当生产后端。这台机器有 24GB 的 Arc Pro B60，跑 CPU
> 等于把它全程闲置；10 秒视频需 3-8 小时，而引擎是**串行队列（concurrency=1）**，
> 根本不可能满足 SLA。CPU 仅适用于：本机验证代码逻辑、无 GPU 的 CI、
> 以及 XPU 排障时的**数值对照实验**（见 `DEPLOY_MOFANG.md` §4.4 的 XPU-vs-CPU 一致性检查）。

```bash
# 1. 创建环境
conda create -n kineto-engine python=3.11 -y
conda activate kineto-engine

# 2. 安装 PyTorch CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 4DHumans（同样不装 '[all]' extra）
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
pip install -e .

# ⚠️ 警告：CPU 推理非常缓慢（单帧 2-5 秒，1 秒视频可能需要 2-5 分钟），仅用于测试
# 用 --no-refine 跳过迭代修正，把耗时从 N 轮压到 1 轮
python kineto_core.py --input input_video.mp4 --output ./output --no-refine
```

> 💡 在设备上临时降到 CPU 做对照，**不要**去找不存在的 `--device cpu` 参数；
> 用 `sudo chmod 000 /dev/dri/renderD128` 隐藏设备节点（实验后 `chmod 0666` 恢复），
> `detect_device()` 会自然降回 CPU。

---

### 方案 4: NVIDIA CUDA 环境（**仅适用于另外的 N 卡机器，不是 Intel AI Box**）

> 本项目的边缘设备是 Intel Arc（方案 1）。下面这套只在你手上另有一台
> RTX 40xx / A100 / H100 服务器时才用得上。容器路径也只有 NVIDIA 才用 `--gpus all`；
> Intel Arc 必须用 `--device /dev/dri`，两者不可互换。

```bash
# 1. 创建独立的 Conda 环境
conda create -n kineto-engine python=3.11 -y
conda activate kineto-engine

# 2. 安装 CUDA 工具链（Conda 方式，避免系统依赖冲突）
conda install -c conda-forge cuda-nvcc cuda-runtime -y

# 3. 安装 PyTorch + CUDA 12.1（官方方式；同样必须先于 requirements.txt）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 验证 CUDA 支持
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 5. 安装 Kineto 依赖
pip install -r requirements.txt

# 6. 安装 4DHumans（从源代码；**不要** pip install '4D-Humans[all]'，那会拖入 detectron2）
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
pip install -e .

# 7. 预加载模型权重
python -c "import torch;from kineto_core import PoseExtractor; \
pe=PoseExtractor(torch.device('cuda')); print('Models loaded, mode =', pe.mode)"
```

---

## 🎯 验证安装完整性

```bash
# 激活环境（设备上是 source /opt/kineto/venv/bin/activate）
conda activate kineto-engine

# 运行完整性检查脚本（三后端通用，会自己挑出当前可用设备）
python -c "
import torch
import cv2
import numpy as np
from smpl_x import SMPLX
import requests

has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

print('✓ PyTorch:', torch.__version__)
print('✓ XPU  Available:', has_xpu, ('-> ' + torch.xpu.get_device_name(0)) if has_xpu else '')
print('✓ CUDA Available:', torch.cuda.is_available())
print('✓ MPS  Available:', has_mps)
print('✓ Device:', 'cuda' if torch.cuda.is_available() else 'xpu' if has_xpu else 'mps' if has_mps else 'cpu')
print('✓ OpenCV:', cv2.__version__)
print('✓ SMPL-X loaded successfully')
print('✓ Requests library ready')
print('\n✅ All dependencies verified!')
"
```

**在 Intel AI Box 上，上面必须看到 `XPU Available: True -> Intel(R) Arc(TM) Pro B60 Graphics`
且 `Device: xpu`。** 若 `Device` 落到 `cpu`，回到方案 1 的第 1/3/4 步查驱动、组权限与 torch 版本。

完整的设备侧验收请用：

```bash
bash deploy/validate.sh          # G1..G10：/health、XPU、POST /jobs、轮询计时、
                                 # extraction_mode==4dhumans、final_quality_score>=0.6、
                                 # demo mp4、OOM、重启持久化、MoFang 回归
```

---

## 🚀 开发工作流

### 在 VS Code 中配置 Python 环境

1. 打开 VS Code 命令面板：`Cmd + Shift + P`
2. 选择 "Python: Select Interpreter"
3. 本地开发机：`/path/to/conda/envs/kineto-engine/bin/python`
   Intel AI Box：`/opt/kineto/venv/bin/python`

### 运行第一个测试

```bash
# 确保有 input_video.mp4 在引擎目录下
cd <repo>/kineto-engine        # Mac: /Users/panbo/Projects/Kineto/kineto-engine
                              # 设备: /opt/kineto/kineto-engine

# 运行核心解算脚本（设备已自动选择 cuda > xpu > mps > cpu）
# ⚠️ --output 是**输出目录**，不是文件名；产物为 <dir>/pose_data.json 与 <dir>/demo_output.mp4
python kineto_core.py --input input_video.mp4 --output ./output

# 可用参数只有这几个（**没有** --device / --demo / --skip-demo / --batch-size）：
#   --input/-i  输入视频        --output/-o  输出目录
#   --max-iter  迭代修正轮数(3)   --quality-thresh 质量阈值(0.6)
#   --no-refine 跳过迭代修正（CPU 排障时推荐）
```

### 在设备上以 HTTP 服务方式运行（生产形态）

```bash
# 手动调试
cd /opt/kineto/kineto-engine
HOME=/srv/kineto KINETO_JOBS_DIR=/srv/kineto/jobs \
KINETO_API_KEY='<强随机密钥>' KINETO_CORS_ORIGINS='https://<your-zeabur-app>' \
    /opt/kineto/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000

# 生产：交给 systemd（unit 文件已写好，拷过去改密钥即可）
sudo cp deploy/kineto-engine.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now kineto-engine
journalctl -u kineto-engine -f
```

---

## ⚙️ 常见问题排查

### ★ 问题 0（Intel AI Box 最常见）: XPU 检测不到

**症状**：`torch.xpu.is_available()` 返回 `False`，日志里出现 `[Device] CPU (推理较慢)`

**按命中率排查**：
```bash
# 1) torch 被 requirements.txt 里的裸 torch>=2.0.0 覆盖成 CPU 轮子了？
python -c "import torch;print(torch.__version__)"      # 必须包含 '+xpu'
pip install --force-reinstall --no-deps torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu

# 2) 设备节点与组权限
ls -l /dev/dri                                          # 应有 renderD128
id -nG | tr ' ' '\n' | grep -E 'render|video'            # 两个都要在
sudo usermod -aG render,video kineto && sudo systemctl restart kineto-engine

# 3) Level Zero / 内核驱动
dmesg | grep -iE 'i915|xe|drm|arc' | tail -20
sudo lshw -C display

# 4) OS 版本（22.04 上 XPU wheel 未验证）
cat /etc/os-release | grep VERSION_ID
```
> 一键采集：`bash deploy/discover_device.sh`（只读）。逐条对策：`DEPLOY_MOFANG.md` §9 的 XPU-1..XPU-4。

> ℹ️ **已修正**：`api.py` 的 `_detect_device()` **已支持 xpu**（按 CUDA > XPU > MPS > CPU 自动选择），
> 因此在 Arc 上 `GET /health` 的 `device` 字段应如实报 **`xpu`**（`/health` 现需带 `X-API-Key`；公开存活用 `/healthz`）。
> 若报 `cpu`，**是真故障**（非“已知无害缺口”）：按序排查 ① `kineto` 用户是否在 `render`/`video` 组、
> ② `/dev/dri/renderD128` 是否存在、③ torch 是否被换成 CPU 轮子（应带 `+xpu`）；`deploy/validate.sh` 的 **G1a 对此硬判 FAIL**。

### 问题 1: 显存/内存溢出（OOM）

**症状**：`RuntimeError: XPU out of memory` / `CUDA out of memory`，
或系统 OOM-killer（`dmesg | grep -i oom` 有记录、`systemctl show kineto-engine -p NRestarts` 计数上升）

**解决方案**：
```python
# kineto_core.py 已实现 clear_device_memory(device)，每帧推理后立即清理，三后端均覆盖：
import torch
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
elif device.type == "xpu":                     # ← Intel Arc Pro B60
    torch.xpu.empty_cache()
elif device.type == "mps":
    torch.mps.empty_cache()
```
```bash
# 运行时层面的缓解手段（CLI 里**没有** --batch-size / --aggressive-cleanup）：
python kineto_core.py --input video.mp4 --output ./output --max-iter 1   # 降迭代轮数
python kineto_core.py --input video.mp4 --output ./output --no-refine    # 单次输出
ffmpeg -i long.mp4 -t 10 -vf scale=-2:720 short_720p.mp4                 # 缩短/降分辨率
sudo systemctl edit kineto-engine     # 抬高 MemoryMax=（需 cgroup v2），然后 daemon-reload + restart
```

### 问题 2: 4DHumans 模型加载失败

**症状**：`ModuleNotFoundError: No module named '4dhumans'`

**解决方案**：
```bash
# 确保从源代码正确安装
cd 4D-Humans
pip install -e . --no-cache-dir --force-reinstall
```

### 问题 3: MPS 不支持的操作

**症状**：`NotImplementedError: The operator ... is not implemented for MPS`

**解决方案**：代码中已实现自动降级到 CPU 的逻辑，无需手动处理。

### 问题 4: ComfyUI API 连接失败（❗历史遗留，当前引擎不依赖）

> 当前 `kineto_core.py` / `api.py` **没有任何 ComfyUI 依赖**（全仓库只有本文提到它）。
> 下面的内容仅作历史记录保留，**部署时不需要启 ComfyUI，也不需要开 8188 端口**。

**症状**：`ConnectionError: Failed to connect to http://127.0.0.1:8188`

**解决方案**：
```bash
# 确保 ComfyUI 在另一个终端已启动
# ComfyUI 通常在本地 8188 端口运行
# 确认 ComfyUI 服务是否启动：
curl -s http://127.0.0.1:8188/api/
```

---

## 📦 环境导出与共享

如果需要与团队共享同一环境配置：

```bash
# 导出当前环境为 YAML
conda env export -n kineto-engine > kineto_engine_env.yml

# 他人恢复环境
conda env create -f kineto_engine_env.yml
```

---

## 💾 显存管理最佳实践（Arc Pro B60 24GB 优化）

在 `kineto_core.py` 中应遵循以下原则（**注意 XPU 分支，这是 Intel AI Box 实际走的路径**）：

```python
# ✓ 正确做法：每帧推理后清理（引擎已封装为 clear_device_memory(device)）
for frame in video_frames:
    pose = model.inference(frame)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'xpu':                 # ← Intel Arc Pro B60
        torch.xpu.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

# ✗ 错误做法：积累所有推理结果后再清理（会导致 OOM）
all_poses = []
for frame in video_frames:
    pose = model.inference(frame)
    all_poses.append(pose)  # 把张量留在显存里，24GB 也会快速耗尽
```

**额外约束（设备侧）**：
- `deploy/kineto-engine.service` 里设了 `MemoryMax=12G` 与 `OMP_NUM_THREADS=4`，给 nginx/MoFang/系统留余量
- api.py 是**串行队列（concurrency=1）**，绝不并行推理；不要把 worker 数改大
- 监控：`sudo intel_gpu_top`（Intel 版 `nvidia-smi`）+ `torch.xpu.memory_allocated()`
- job 产物目录 `KINETO_JOBS_DIR=/srv/kineto/jobs` 会随时间膨胀，需定期清理（见 `DEPLOY_MOFANG.md` §9 OPS 条目）

---

## 📝 下一步

完成上述配置后，请：
1. ✅ 确认环境创建成功，且 `torch.xpu.is_available() == True`（Intel AI Box 上必须）
2. ✅ 运行验证脚本确认所有依赖可用，且 `PoseExtractor.mode == '4dhumans'`（非 `fallback`）
3. ✅ 准备 `input_video.mp4` 测试文件
4. ✅ 转至仓库根目录 **[`DEPLOY_MOFANG.md`](../DEPLOY_MOFANG.md)**，按 Phase 0 → Phase 1（共存验证）
   → Phase 2（生产部署）→ Phase 3（`deploy/validate.sh` 验收 + reboot 复验）推进

---

## 📞 支持

- 🏭 **设备侧部署手册**：[`../DEPLOY_MOFANG.md`](../DEPLOY_MOFANG.md)
- 🛠️ **部署工具包**：`../deploy/`（`discover_device.sh` / `transfer_models.sh` / `validate.sh` /
  `kineto-engine.service` / `Dockerfile.engine` / `docker-compose.yml` / `cloudflared/`）
- 🖥️ **设备选择与后端对比**：`DEVICE_SELECTION.md`
- **Intel XPU 后端文档**：https://pytorch.org/docs/stable/notes/get_start_xpu.html
- **Intel Extension for PyTorch**：https://intel.github.io/intel-extension-for-pytorch/
- **4DHumans 文档**：https://github.com/shubham-goel/4D-Humans
- **SMPL-X 文档**：https://github.com/vchoutas/smplx
- **PyTorch 文档**：https://pytorch.org/docs/stable/index.html
- **FastAPI 文档**：https://fastapi.tiangolo.com/
