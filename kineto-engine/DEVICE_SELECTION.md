# Kineto Engine - 设备选择指南

> ⚠️ **重要更正**：本项目的目标边缘设备 **MoFang M01 / Intel AI Box 搭载的是
> Intel Arc Pro B60（24GB 显存，Battlemage 架构）**，**不是 NVIDIA GPU，没有 CUDA，没有 `nvidia-smi`，
> Docker 也不用 `--gpus`**。它的 PyTorch 后端是 **XPU**（`torch>=2.5` 原生支持，`torch.xpu.*` API，`device='xpu'`）。
> 本文早期版本曾把这台机器误标为「NVIDIA CUDA 24GB」、并把 Intel Arc 一行写成 8GB，现已修正。
>
> 设备侧完整部署手册（探测 / 可逆剥离 MoFang / 驱动 / systemd / Cloudflare Tunnel / 验收）见
> 仓库根目录 **[`DEPLOY_MOFANG.md`](../DEPLOY_MOFANG.md)** 与 `deploy/` 工具包。

## 🎯 快速决策树

```
您使用什么硬件？
│
├─ NVIDIA GPU (RTX 40xx, A100, H100 等)
│  ├─ Windows/Linux ✅
│  │  └─> 使用 CUDA 12.1
│  │     pip install torch --index-url https://download.pytorch.org/whl/cu121
│  │     性能：⭐⭐⭐⭐⭐ (最快)
│  │
│  └─ Mac (不推荐，驱动问题多)
│     └─> 降级到 CPU
│        pip install torch --index-url https://download.pytorch.org/whl/cpu
│        性能：⭐ (最慢)
│
├─ Apple Silicon (M1/M2/M3/M4)
│  └─ macOS 12.3+
│     └─> 使用 MPS (Metal Performance Shaders)
│        pip install torch torchvision torchaudio
│        性能：⭐⭐⭐ (中等)
│
├─ Intel Arc GPU (Arc Pro B60 / A770 / A380 / Xe 等)   ← ★ Intel AI Box 生产路径
│  └─ Ubuntu 24.04/26.04 + Intel GPU 驱动 (内核 i915/xe + compute-runtime + Level Zero)
│     └─> 使用 PyTorch XPU（torch>=2.5 原生，不需要 oneAPI 全家桶）
│        pip3 install torch torchvision torchaudio \
│            --index-url https://download.pytorch.org/whl/xpu
│        可选加速：pip install intel-extension-for-pytorch==2.7.10+xpu \
│            --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
│        性能：⭐⭐⭐⭐ (Arc Pro B60 = 24GB 显存，生产可用)
│
└─ CPU 专用 (Intel Xeon, AMD Ryzen 等)
   └─> 使用 CPU 模式
      pip install torch --index-url https://download.pytorch.org/whl/cpu
      性能：⭐⭐ (较慢，仅测试用)
      ❌ 在 Intel AI Box 上属「生产不可行」：10 秒视频要 3-8 小时，24GB 显存全程闲置，
         而引擎是串行队列 (concurrency=1)，会直接压垮 SLA。CPU 只允许用于本机脚本逻辑自测。
```

---

## 📊 性能对比表

| 设备 | 推荐场景 | 单帧推理 | 10秒视频 (60fps) | 显存 | 安装难度 |
|------|--------|--------|-----------------|------|--------|
| **Intel Arc XPU**（Arc Pro B60） | **生产环境 = Intel AI Box / MoFang M01** | ~100-150ms | ~20-30 min | **24GB** | ⭐⭐ 中等（需装 Intel GPU 驱动） |
| **NVIDIA CUDA** | 有 N 卡的服务器/工作站（**不是**本项目的 AI Box） | ~80ms | ~15 min | 16-24GB | ⭐ 简单 |
| **Apple Silicon MPS** | MacBook 开发调试 | ~150ms | ~30 min | 8-16GB（统一内存） | ⭐ 简单 |
| **CPU (多核)** | 仅脚本逻辑自测 | ~2-5s | 3-8 小时 | RAM | ⭐⭐ 中等 |

> ❌ **CPU-only 在 Intel AI Box 上不具备生产可行性**（单帧 2-5s、10 秒视频 3-8 小时、24GB 显存闲置）。
> Arc XPU 一行的耗时为工程估算，真实值以 `deploy/validate.sh` 的 **G4 计时门禁**为准。

---

## 🔧 详细配置指南

### ✅ 情景 1: Intel AI Box / MoFang M01 — **Intel Arc Pro B60 24GB (Battlemage)** ← 本项目生产环境

**最佳选择**: **PyTorch XPU**（不是 CUDA，不是 oneAPI 全家桶）

**前置条件（缺一不可）**

| 项 | 要求 |
|---|---|
| OS | Ubuntu **24.04 LTS**（或 26.04）。XPU wheel 只在 24.04/26.04 验证过；**22.04 需走 IPEX 路径或先升级系统** |
| 内核 | ≥ 6.8（Battlemage 支持较完整），`i915` 或 `xe` 驱动已加载 |
| 用户态 | **compute-runtime（NEO OpenCL ICD）+ Level Zero** |
| 设备节点 | `/dev/dri/card*`、`/dev/dri/renderD128` 存在 |
| 权限 | 运行用户必须在 **`render`** 与 **`video`** 组里，否则 `torch.xpu.is_available()` 为 False |
| Python | 3.11（venv；24.04 上受 PEP 668 保护，**不要**往系统 Python 里 pip install） |

```bash
# 0. 只读探测（在设备上跑，不改任何状态）：确认 GPU/RAM/磁盘/驱动/端口/MoFang 清单
bash deploy/discover_device.sh

# 1. 安装 Intel GPU 用户态驱动（需要 sudo；已装则 apt 会提示已是最新，幂等）
#    详细步骤 + Intel APT 源配置见 DEPLOY_MOFANG.md §5.3
sudo apt-get update
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero \
                        libze1 libze-intel-gpu1
sudo usermod -aG render,video $USER      # 之后需重新登录生效
ls -l /dev/dri                            # 应看到 renderD128

# 2. 创建隔离 venv（Python 3.11）
sudo apt-get install -y python3.11 python3.11-venv
python3.11 -m venv /opt/kineto/venv
source /opt/kineto/venv/bin/activate
pip install --upgrade pip wheel setuptools

# 3. ★ 先装 PyTorch XPU（必须第一个装！requirements.txt 里有裸 torch>=2.0.0，
#    后装会被 PyPI 的 CPU 轮子覆盖掉 XPU 支持）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 4. （可选）Intel Extension for PyTorch —— 版本必须与 torch 主次版本对齐（2.7.10 ↔ torch 2.7.x）
#    kineto_core.py 会机会性 import 它，装了能拿到算子融合等加速，不装也不影响 XPU 可用性
pip install intel-extension-for-pytorch==2.7.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# 5. ★ 验证 XPU（唯一可信判据）
python -c "import torch;print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
#   期望输出形如: True Intel(R) Arc(TM) Pro B60 Graphics

# 6. 安装项目依赖（装完再复查一次 torch 仍是 +xpu 版本，被覆盖就 --force-reinstall --no-deps 修回）
pip install -r requirements.txt
python -c "import torch;print(torch.__version__, torch.xpu.is_available())"

# 7. 安装 4DHumans —— ★ 不要装 '[all]' extra（会拖入 detectron2 源码编译，边缘设备上极易失败）
cd 4D-Humans && pip install -e . && cd ..

# 8. 运行完整流程（kineto_core.py 的 detect_device() 会自动选到 xpu，无需也不能手指定）
#    注：--output 是**输出目录**，pose_data.json 与 demo_output.mp4 都会写到该目录下
python kineto_core.py --input input_video.mp4 --output ./output
```

**预期输出**：
- ✅ 启动日志出现 `[Device] Intel XPU: Intel(R) Arc(TM) Pro B60 Graphics`
- ✅ 单帧推理 ~100-150ms，10 秒视频 ~20-30min
- ✅ 峰值显存占用 ~8-16GB（24GB 上限内，`MemoryMax=12G` 的 systemd 限制也留了余量）
- ✅ 无 OOM、`dmesg | grep -i oom` 无新增记录
- ✅ `pose_data.json` 的 `metadata.extraction_mode == "4dhumans"`（引擎的上线硬门禁）

**容器路径注意**：Docker 直通 Intel GPU 用 `--device /dev/dri`（compose 里 `devices: ["/dev/dri"]`），
**不是** `--gpus` / nvidia-container-runtime。见 `deploy/Dockerfile.engine` + `deploy/docker-compose.yml`。

---

### ✅ 情景 2: MacBook Pro M3 (Apple Silicon)

**最佳选择**: MPS (Metal Performance Shaders)

```bash
# 系统环境假设：macOS 13.x + M3 芯片

# 1. 创建隔离环境
conda create -n kineto-engine python=3.11 -y
conda activate kineto-engine

# 2. 安装 PyTorch (MPS 支持)
pip install torch torchvision torchaudio

# 3. 验证 MPS
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装 4DHumans
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans && pip install -e . && cd ..

# 6. 运行，自动使用 MPS（--output 是输出目录）
python kineto_core.py --input input_video.mp4 --output ./output
```

**注意**：
- 某些操作不支持 MPS，会自动降级到 CPU（代码已处理）
- 内存占用会更高（共享系统内存）
- M3 的 8GB 统一内存足够处理中等长度视频

---

### ⚠️ 情景 3: 开发测试环境 (CPU Only) — **Intel AI Box 上生产不可行**

> ❌ **不要**在 Intel AI Box 上把 CPU 当作生产后端：这台机器有 24GB 的 Arc Pro B60，
> 跑 CPU 等于把它闲置，同时 10 秒视频要 3-8 小时，串行队列下无法交付。
> CPU 路径**仅**适用于：本机快速验证代码逻辑、无 GPU 的 CI、以及 XPU 排障时的对照实验
> （见 `DEPLOY_MOFANG.md` §4.4 的 XPU-vs-CPU 数值一致性检查）。

**何时选择**: 
- 快速验证代码逻辑
- 没有 GPU 的环境
- 低成本 CI/CD 环境

```bash
# 1. 创建环境
conda create -n kineto-engine python=3.11 -y
conda activate kineto-engine

# 2. 安装 CPU 版 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 4DHumans
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans && pip install -e . && cd ..

# 5. 运行（会很慢！）—— 用 --no-refine 跳过迭代修正，把单视频耗时从 N 轮压到 1 轮
python kineto_core.py --input input_video.mp4 --output ./output --no-refine
```

**期望与限制**：
- ⚠️ 单帧推理 ~2-5 秒（取决于 CPU 核心数）
- ⚠️ 10 秒视频耗时 3-8 小时（仅测试脚本逻辑）
- ⚠️ 建议加 `--no-refine`（跳过迭代修正）与 `--max-iter 1` 以大幅缩短耗时
- ⚠️ 仅用于验证数据管道，**不适合实际生产；在 Intel AI Box 上更不允许作为生产后端**

---

### ✅ 情景 4: NVIDIA CUDA 服务器 / 工作站（**不是**本项目的 Intel AI Box）

**适用**：你手上另有一台带 N 卡的机器（RTX 40xx / A100 / H100）。本项目的边缘设备是 Intel Arc，
**这一节与 Intel AI Box 无关**，仅作为云端/离线大批量推理的备选。

```bash
# 系统环境假设：Ubuntu 22.04/24.04 + NVIDIA GPU + 已装对应驱动（nvidia-smi 可用）
conda create -n kineto-engine python=3.11 -y && conda activate kineto-engine

# 1. 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. 验证 CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# 3. 项目依赖 + 4DHumans（同样不装 '[all]' extra）
pip install -r requirements.txt
cd 4D-Humans && pip install -e . && cd ..

# 4. 运行
python kineto_core.py --input input_video.mp4 --output ./output
```

**容器路径注意**：只有 NVIDIA 才用 `--gpus all`；Intel Arc 必须用 `--device /dev/dri`，两者不可互换。

---

## 🔍 运行时设备检测与覆盖

### 自动检测
```python
# kineto_core.py 的 detect_device() 会自动选择最佳设备
# 优先级: CUDA > XPU (Intel Arc) > MPS > CPU
# 并在检测到 intel_extension_for_pytorch 时打印其版本（缺失则静默跳过，不影响 XPU 可用性）

python kineto_core.py --input input_video.mp4 --output ./output
# Intel AI Box 上的日志形如：
#   [IPEX] intel-extension-for-pytorch 2.7.10+xpu 已加载     ← 装了 IPEX 才有（前缀是 [IPEX]，不是 [Device]）
#   [Device] Intel XPU: Intel(R) Arc(TM) Pro B60 Graphics
# 其它环境：[Device] NVIDIA CUDA: ... / [Device] Apple Silicon MPS / [Device] CPU (推理较慢)
```

### 关于“手动指定设备”
⚠️ **`kineto_core.py` 的 CLI 目前没有 `--device` 参数**，设备完全由 `detect_device()` 自动判定。
实际可用的参数只有：

| 参数 | 说明 |
|---|---|
| `--input` / `-i` | 输入视频路径（默认 `input_video.mp4`） |
| `--output` / `-o` | **输出目录**（默认 `output`），产物为 `<dir>/pose_data.json` 与 `<dir>/demo_output.mp4` |
| `--max-iter` | 最大迭代修正次数（默认 3） |
| `--quality-thresh` | 质量阈值（默认 0.6，对应 `metadata.pipeline.final_quality_score`；与 env `KINETO_QUALITY_THRESHOLD` 同源） |
| `--no-refine` | 跳过迭代修正，仍跑**真实审计**得诚实质量分（不再强制 0）；CPU 排障时推荐 |

**退出码**：

| 码 | 含义 |
|---|---|
| 0 | 正常完成 |
| 1 | 真实崩溃（异常未捕获） |
| 2 | argparse 用法错误（参数不合法） |
| 3 | StrictModeRefused（`KINETO_STRICT=1` 触发：假数据/缺权重/detector降级） |

**运行时环境变量**（引擎内置默认，可由 systemd EnvironmentFile / docker compose / shell env 覆盖）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `KINETO_STRICT` | `1` | 只守假数据/缺权重/detector降级的 fail-fast（退出码 3）；❗ **不再对质量做门禁** |
| `KINETO_QUALITY_GATE` | `warn` | 质量门禁：`warn`=不达标仍 done 附 degraded；`fail`=不达标判 failed；`off`=不门禁 |
| `KINETO_QUALITY_THRESHOLD` | `0.6` | 质量分阈值（与 `--quality-thresh` CLI 同源） |
| `KINETO_AUDIT_VISUALIZE` | `1`(on) | 审计时是否生成可视化叠帧 |
| `KINETO_AUDIT_CACHE_FRAMES` | `48` | 审计帧缓存容量（≤0 禁用复用） |
| `KINETO_CLEAR_MEM_EVERY` | `30` | 每 N 帧强制释放显存 |
| `KINETO_SUPPORTED_FRAME_CEILING` | `9000` | 架构级帧数上限（≥此值拒绝处理） |
| `KINETO_QUALITY_LOG_MAX_BYTES` | `5242880` | quality_log.jsonl 轮转大小（0=不轮转） |
| `KINETO_TIMEOUT_FLOOR_SEC` | `7200` | 子进程超时下限 |
| `KINETO_TIMEOUT_BASE_SEC` | `1800` | 超时基线 |
| `KINETO_TIMEOUT_PER_FRAME_SEC` | `0.8` | 每帧超时增量 |
| `KINETO_TIMEOUT_MAX_SEC` | `21600` | 超时上限 |

**要强制降回 CPU 做对照实验**，不要找不存在的 `--device cpu`，用下面两种方法之一：

```bash
# 方法 A（推荐）：隐藏 GPU 设备节点，detect_device() 自然降回 CPU
sudo chmod 000 /dev/dri/renderD128        # 实验后记得恢复：sudo chmod 0666 /dev/dri/renderD128

# 方法 B：在 Python 层直接调（XPU-vs-CPU 数值一致性检查，见 DEPLOY_MOFANG.md §4.4）
python -c "import torch;from kineto_core import PoseExtractor;pe=PoseExtractor(device=torch.device('cpu'))"
```

---

## 💾 显存管理策略

### 引擎实际实现：`clear_device_memory(device)`（已覆盖 XPU）

`kineto_core.py` 每帧推理后调用它释放显存，三条后端分支均已实现：

```python
def clear_device_memory(device):
    """每帧推理后释放显存/内存"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":                       # ← Intel Arc Pro B60 走这里
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
```

### XPU 显存监控（Intel AI Box 上排查 OOM 的第一手信息）
```python
import torch
assert torch.xpu.is_available()
print("已分配 :", torch.xpu.memory_allocated() / 1e9, "GB")
print("已预留 :", torch.xpu.memory_reserved()  / 1e9, "GB")
torch.xpu.empty_cache()
# 峰值统计（部分 torch 版本未实现，先 hasattr 再调）
if hasattr(torch.xpu, "reset_peak_memory_stats"):
    torch.xpu.reset_peak_memory_stats()
    print("峰值   :", torch.xpu.max_memory_allocated() / 1e9, "GB")
```

> 系统级监控：`sudo intel_gpu_top`（来自 `intel-gpu-tools`）可看 Arc 的实际占用率与显存带宽，
> 它相当于 Intel 平台上的 `nvidia-smi`。**这台机器上没有也不需要 `nvidia-smi`。**

### CUDA 显存优化（仅情景 4 的 N 卡机器适用）
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()                 # 立即释放
    torch.cuda.reset_peak_memory_stats()     # 重置峰值记录
    print(f"CUDA Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### MPS 显存优化（仅 MacBook 开发机适用）
```python
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

---

## 🐛 常见设备问题

### 问题 0（Intel AI Box 最常见）: XPU 检测不到

**症状**：`torch.xpu.is_available()` 返回 `False`，或日志里出现 `[Device] CPU (推理较慢)`

**排查步骤（按命中率从高到低）**：
```bash
# 1. torch 是不是被 requirements.txt 里的裸 torch>=2.0.0 覆盖成了 CPU 轮子？
python -c "import torch;print(torch.__version__)"      # 应包含 '+xpu'
#    被覆盖就修回（--no-deps 避免连带回退其它包）：
pip install --force-reinstall --no-deps torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu

# 2. 设备节点存在吗？
ls -l /dev/dri                                          # 应有 card0 / renderD128

# 3. 当前用户在 render / video 组里吗？（systemd 下靠 unit 的 SupplementaryGroups）
id -nG | tr ' ' '\n' | grep -E 'render|video'
sudo usermod -aG render,video $USER                     # 加完必须重新登录/重启服务

# 4. Level Zero 能看到 GPU 吗？（装了 intel-level-zero-gpu 后有 ze_info 等工具）
dmesg | grep -iE 'i915|xe|drm|arc' | tail -20
sudo lshw -C display

# 5. Ubuntu 22.04？XPU wheel 未在该版本验证——走 IPEX 路径或先升级到 24.04
cat /etc/os-release | grep VERSION_ID
```
> 一键采集以上全部信息：`bash deploy/discover_device.sh`（只读，不改状态）。
> 逐条对策见 `DEPLOY_MOFANG.md` §9 排障表的 XPU-1..XPU-4。

### 问题 0b: `/health` 报告 `device=cpu`（但 XPU 实际可用）—— 这是**真故障**

**说明**：`kineto-engine/api.py` 的 `_detect_device()` **已支持 xpu**（按 CUDA > XPU > MPS > CPU 自动选择），
Arc 上 `GET /health` 应如实报 **`device=xpu`**（`/health` 需带 `X-API-Key`；公开存活探针用 `/healthz`）。
因此 `device=cpu` **不再是“已知无害缺口”**，而是真故障，按序排查：
① `kineto` 服务用户是否在 `render`/`video` 组（`id kineto`）；② `/dev/dri/renderD128` 是否存在且属组 `render`；
③ torch 是否被 `requirements.txt` 换成了 CPU 轮子（`torch.__version__` 应带 `+xpu`）。
可对照引擎日志的 `[Device] Intel XPU: ...` 与 `torch.xpu.is_available()`；`deploy/validate.sh` 的 **G1a 对此硬判 FAIL**（非 WARN）。

### 问题 1: CUDA 检测不到（仅情景 4 的 N 卡机器）

**症状**：`torch.cuda.is_available() = False`

**排查步骤**：
```bash
# 1. 检查驱动
nvidia-smi  # 应输出 GPU 信息和驱动版本

# 2. 检查 CUDA 版本
nvcc --version  # 确认 CUDA 12.1 已安装

# 3. 重新安装 PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# 4. 测试
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 2: MPS 报告 "operator not implemented"

**症状**：某些操作在 MPS 上不支持

**解决**：代码中已实现自动回退逻辑
```python
try:
    result = tensor.to('mps').compute()
except NotImplementedError:
    result = tensor.to('cpu').compute()  # 自动降级
```

### 问题 3: 显存溢出 (OOM)

**症状**：`RuntimeError: CUDA out of memory` / `RuntimeError: XPU out of memory`，
或系统级 OOM-killer（`dmesg | grep -i oom` 有记录，systemd 显示服务被杀后重启）

**快速修复**：
```bash
# 方法 1（最有效）：降低迭代修正轮数——每轮都会重复推理与缓存中间结果
python kineto_core.py --input video.mp4 --output ./output --max-iter 1

# 方法 2：完全跳过修正，单次输出
python kineto_core.py --input video.mp4 --output ./output --no-refine

# 方法 3：缩短输入视频 / 降低分辨率（先用 ffmpeg 预处理）
ffmpeg -i long.mp4 -t 10 -vf scale=-2:720 short_720p.mp4

# 方法 4：抬高 systemd 内存上限（仅当确认是系统 RAM 而非显存不足；需 cgroup v2）
sudo systemctl edit kineto-engine        # 覆盖 MemoryMax=，然后 daemon-reload + restart
```
> ⚠️ CLI 里**没有** `--batch-size`、`--aggressive-cleanup`、`--device cpu` 这些参数（旧文档误写）。
> 引擎内部本来就是逐帧处理，并在每帧后调 `clear_device_memory(device)`。
> `deploy/validate.sh` 的 **G8** 会自动检查 dmesg/journal 中的 OOM 与服务重启次数。
> 另有 **G12（G8-SSOT）** 检查骨架 SSOT 一致性（`skeleton_spec.py` ↔ `skeleton.ts`）。

---

## 📋 部署检查清单

使用前请确认：

### ★ XPU 环境（Intel AI Box / MoFang M01 — 本项目生产环境）
- [ ] `lspci -nn | grep -iE 'vga|display|3d'` 能看到 Intel 显卡（记下 `[8086:xxxx]` PCI ID）
- [ ] `ls -l /dev/dri` 有 `card*` 与 `renderD128`
- [ ] Ubuntu **24.04 / 26.04**（`cat /etc/os-release`）；22.04 需走 IPEX 或升级
- [ ] 内核 ≥ 6.8，`dmesg | grep -iE 'i915|xe'` 无报错
- [ ] 已装 `intel-opencl-icd` + `intel-level-zero-gpu` + `level-zero` + `libze-intel-gpu1`
- [ ] 运行用户属于 `render` 与 `video` 组（`id -nG`）
- [ ] `python -c "import torch;print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"` → `True Intel(R) Arc(TM) Pro B60 Graphics`
- [ ] `torch.__version__` 包含 `+xpu`（没被 requirements.txt 的裸 torch 覆盖）
- [ ] 系统内存 ≥ 16GB，可用磁盘 ≥ 25GB，8000 端口空闲
- [ ] 容器路径用 `devices: ["/dev/dri"]`，**不是** `--gpus`
- [ ] `bash deploy/validate.sh` 的 G1..G12 全绿（含 `extraction_mode==4dhumans`、`final_quality_score>=0.6`、骨架 SSOT 一致性）
- [ ] `sudo reboot` 后重跑一次 validate 仍然全绿（重启持久化）

> 一键采集：`bash deploy/discover_device.sh`（只读，末尾自带 PASS/FAIL 门禁汇总）。

### CUDA 环境（仅情景 4 的 N 卡机器）
- [ ] `nvidia-smi` 显示 GPU 信息
- [ ] `python -c "import torch; torch.cuda.is_available()"` 返回 True
- [ ] CUDA 版本 ≥ 12.0
- [ ] 显存 ≥ 16GB (推荐 24GB)

### MPS 环境（MacBook 开发机）
- [ ] macOS 版本 ≥ 12.3
- [ ] Apple Silicon 芯片 (M1+)
- [ ] `python -c "import torch; torch.backends.mps.is_available()"` 返回 True
- [ ] 统一内存 ≥ 8GB

### CPU 环境（❌ Intel AI Box 上生产不可行，仅自测）
- [ ] CPU ≥ 8 核心
- [ ] RAM ≥ 16GB
- [ ] Python 3.10+
- [ ] 已确认这台机器**真的没有可用 GPU**（否则不要选这条路）

---

## 🚀 性能优化建议

### XPU（Intel AI Box）
```bash
# 1. 装上 IPEX（与 torch 主次版本对齐），kineto_core.py 会机会性 import 并打印版本
pip install intel-extension-for-pytorch==2.7.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# 2. 线程数：边缘设备留给推理的 CPU 不要太多（systemd unit 里已固定为 4）
export OMP_NUM_THREADS=4

# 3. 无显示器环境下渲染 demo 视频靠 OSMesa（unit 里已设）
export PYOPENGL_PLATFORM=osmesa

# 4. 降低迭代轮数是性价比最高的提速手段（线性减少推理次数）
python kineto_core.py --input video.mp4 --output ./output --max-iter 1

# 5. 监控真实占用（相当于 Intel 版 nvidia-smi）
sudo intel_gpu_top
```

### CUDA
```bash
# 启用 cuDNN 自动优化
export CUDNN_BENCHMARK=1

# 异步执行
export CUDA_LAUNCH_BLOCKING=0

python kineto_core.py --input video.mp4 --output ./output
```

### MPS
```bash
# 启用 MPS 回退
export PYTORCH_ENABLE_MPS_FALLBACK=1

python kineto_core.py --input video.mp4 --output ./output
```

---

## 📞 获取更多帮助

- 🏭 **设备侧部署手册**：[`../DEPLOY_MOFANG.md`](../DEPLOY_MOFANG.md)（探测 / 可逆剥离 MoFang / 驱动 / systemd / Cloudflare Tunnel / 验收 / 回滚）
- 🛠️ **部署工具包**：`../deploy/`（`discover_device.sh` / `transfer_models.sh` / `validate.sh` / `strip_mofang.sh` / `patches/hmr2_weights_only.patch` / `kineto-engine.service` / `Dockerfile.engine` / `docker-compose.yml` / `cloudflared/`）
- 📖 查看 SETUP_GUIDE.md 详细步骤
- 🚀 查看 QUICKSTART.md 快速开始
  > ⚠️ QUICKSTART.md 早期版本曾出现 **NVIDIA CUDA / `--device cuda` / `--batch-size` / `--verbose`** 等表述，
  > 这些**均已过时**（本项目为 Intel Arc **XPU**，且 CLI **没有** `--device`）；现版已对齐。若仍见到任何
  > CUDA / `--device` 残留段落，**一律以本文（DEVICE_SELECTION.md）为准**：真实 CLI 只有
  > `--input/-i`、`--output/-o`（目录）、`--max-iter`、`--quality-thresh`、`--no-refine`。
- 🔧 运行 `python check_environment.py` 自动诊断
