# Kineto Engine - 快速开始指南

> **目标设备：Intel Arc（Pro B60 / 24GB / Battlemage）+ PyTorch XPU**。本引擎面向云边协同的
> 边缘算力节点，**不是 NVIDIA，没有 CUDA，没有 `nvidia-smi`**。`kineto_core.py` 的 `detect_device()`
> 与 `api.py` 的 `_detect_device()` 都会按 **CUDA > XPU > MPS > CPU** 自动选择设备，**无 `--device` 参数**。
> 真实 CLI 只有：`--input/-i`、`--output/-o`（**目录**）、`--max-iter`、`--quality-thresh`、`--no-refine`。
> 设备侧完整部署（驱动 / XPU 验证 / systemd / 隧道）见根目录 [`../DEPLOY_MOFANG.md`](../DEPLOY_MOFANG.md)；
> 设备选型细节见 [`DEVICE_SELECTION.md`](./DEVICE_SELECTION.md)。

## ⚡ 30 秒快速上手

### 步骤 1: 自动环境检测

```bash
cd kineto-engine
python check_environment.py
```

该脚本将自动检测：
- ✅ Python 版本
- ✅ Conda 环境
- ✅ 计算设备（Intel **XPU** / Apple MPS / CPU）
- ✅ PyTorch 状态（XPU 版本应带 `+xpu`）
- ✅ 依赖库完整性

### 步骤 2: 创建环境（仅首次）

#### Intel Arc XPU 环境（本项目目标平台，推荐）
```bash
conda create -n kineto-engine python=3.11 -y
conda activate kineto-engine
# torch 必须【第一个】装：requirements.txt 里有裸 torch>=2.0.0，后装会被 PyPI 的 CPU 轮子覆盖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
# 可选增强：Intel Extension for PyTorch（版本须与 torch 主次版本对齐，2.7.10 ↔ torch 2.7.x）
pip install intel-extension-for-pytorch==2.7.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install -r requirements.txt
# 验证（唯一可信判据）：期望形如 True Intel(R) Arc(TM) Pro B60 Graphics
python -c "import torch;print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```
> 前置：需先装 Intel GPU 用户态运行时（compute-runtime / Level Zero），`kineto` 用户须在 `render`/`video` 组。
> 详见 [`../DEPLOY_MOFANG.md`](../DEPLOY_MOFANG.md) §5.3/§5.4 与 [`DEVICE_SELECTION.md`](./DEVICE_SELECTION.md)。

#### Apple Silicon Mac（本地开发/联调）
```bash
conda create -n kineto-engine python=3.11 -y
conda activate kineto-engine
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

#### CPU 模式（兜底，慢 10-100 倍）
```bash
conda create -n kineto-engine python=3.11 -y
conda activate kineto-engine
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 步骤 3: 安装 4DHumans 模型库（含 torch>=2.6 必需补丁）

```bash
# 方式 A: 从源代码安装（完整功能）
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
# ⚠️ torch>=2.6 起 torch.load 默认 weights_only=True，会在加载 .ckpt 时崩溃；
#    必须打仓库自带的补丁（把 torch.load 包成 weights_only=False）：
git apply -p1 ../deploy/patches/hmr2_weights_only.patch
grep -q weights_only hmr2/models/__init__.py && grep -q add_safe_globals hmr2/models/__init__.py && echo "[patch] 两处补丁均 OK"   # 校验补丁（2 hunk）已生效
pip install -e .
cd ..

# 方式 B: 如果已有预编译包（仍需确认已含 weights_only 补丁，否则加载 .ckpt 会崩）
pip install 4dhumans
```
> 该补丁受版本控制于 `deploy/patches/hmr2_weights_only.patch`；`deploy/Dockerfile.engine` 构建期会
> `git apply` 并断言 `weights_only` 存在，缺失则禁止出图。详见 [`SETUP_GUIDE.md`](./SETUP_GUIDE.md)。

### 步骤 4: 准备测试视频

将您的测试视频放在项目根目录：
```bash
cp /path/to/your/video.mp4 ./input_video.mp4
```

### 步骤 5: 验证环境

```bash
python check_environment.py
# 所有检查都应为 ✅
```

---

## 🎯 核心工作流（稍后）

环境就绪后，运行核心脚本（设备由 `detect_device()` 自动选择，**无需也不能**传 `--device`）：

```bash
# 激活环境
conda activate kineto-engine

# 运行完整流程：视频 → 3D 姿态 JSON（+ 可视化）写入【输出目录】
python kineto_core.py \
  --input input_video.mp4 \
  --output output
```

**输出**（`--output` 是**目录**，默认 `output`，产物写入其中）：
- ✅ `output/pose_data.json` - 符合 README 规范的科研级 3D 姿态数据
- ✅ `output/demo_output.mp4` - 骨架叠加可视化视频
- ✅ `output/audit_results.json` - 质量审计报告（机器可读 verdict）
- ✅ `output/audit_iter*/` - 每轮迭代的审计中间产物

#### 产物 schema（canonical 整改后）

| 字段 | 形状/类型 | 说明 |
|---|---|---|
| `keyframes[].joints_3d` | `[24, 3]` float | **canonical SMPL 24 关节序**（由 `skeleton_spec.py` SSOT 定义） |
| `keyframes[].smpl_thetas` | `[72]` float | 轴角表示（与 joints_3d 逐帧一致；refine 改动帧会重算） |
| `keyframes[].betas` | `[10]` float | **additive**（仅 4dhumans 模式产出；旧消费方不受影响） |
| `keyframes[].cam_t` | `[3]` float | 相机平移 |
| `keyframes[].confidence_score` | float | 帧级置信度 |
| `metadata.pipeline.final_quality_score` | float | **诚实质量分**（不再被 `--no-refine` 强制为 0） |
| `metadata.pipeline.refine_applied` | bool | 是否应用了迭代修正 |
| `audit_results.json` → `verdict` | `"pass"\|"warn"\|"fail"` | 机器可读质量判定 |
| `audit_results.json` → `total_issues` | int | 审计发现的问题总数 |
| `audit_results.json` → `failure_reason` | string\|null | verdict=fail 时的原因说明 |

> **关节序 SSOT**：24 关节的顺序、父子关系、骨架边均由 `kineto-engine/skeleton_spec.py` 权威定义，
> 前端镜像在 `kineto-web/lib/skeleton.ts`。`deploy/validate.sh` 的 **G12（G8-SSOT）** 会检查二者一致性。

---

## 🔧 常见命令

### 查看帮助
```bash
python kineto_core.py --help
```

### 调节迭代修正（真实参数）
```bash
# 最大迭代修正次数（默认 3）与质量阈值（默认 0.6）
python kineto_core.py --input input_video.mp4 --output output --max-iter 5 --quality-thresh 0.7

# 跳过修正，仍跑真实审计得诚实质量分（不再强制 0）
python kineto_core.py --input input_video.mp4 --output output --no-refine
```

### 退出码

| 码 | 含义 |
|---|---|
| 0 | 正常完成 |
| 1 | 真实崩溃（未捕获异常） |
| 2 | argparse 用法错误 |
| 3 | StrictModeRefused（`KINETO_STRICT=1` 下检测到假数据/缺权重/detector降级） |

### 长视频天花板

≥ **9000 帧**（`KINETO_SUPPORTED_FRAME_CEILING`）的视频被引擎拒绝处理（架构级上限：JSON/mp4 线性膨胀、前端全量加载、超时按帧数缩放）。

### 关于设备选择
```bash
# 设备由 detect_device() 自动按 CUDA > XPU > MPS > CPU 选择，【没有】--device 参数。
# 在 Intel Arc 上应自动命中 XPU；若日志显示用了 CPU，先查 render/video 组与 torch 是否为 +xpu 轮子。
python -c "import torch;print('xpu', torch.xpu.is_available())"
```

---

## 🐛 快速故障排除

| 问题 | 解决方案 |
|------|--------|
| `ModuleNotFoundError: torch` | `pip install torch...` (见上面的环境创建步骤) |
| 加载 `.ckpt` 报 `weights_only` / UnpicklingError | torch>=2.6 需打补丁：`cd 4D-Humans && git apply -p1 ../deploy/patches/hmr2_weights_only.patch` |
| Intel Arc 上却用了 CPU / `xpu.is_available()` 为 False | `kineto` 用户不在 `render`/`video` 组，或 torch 被装成了 CPU 轮子（应带 `+xpu`）——见 DEPLOY_MOFANG §9 XPU-1/XPU-2 |
| XPU 显存不足 / OOM | 降低 `--max-iter`、缩短输入视频；确认 `dmesg` 无 OOM（`deploy/validate.sh` 会查） |
| `cannot import 4DHumans` | 确保从源代码安装了 4DHumans: `cd 4D-Humans && pip install -e .`（并打上补丁） |
| `no video input found` | 检查 `input_video.mp4` 是否存在于当前目录 |
| 推理超级慢 | 确认是否回退到了 CPU；在 Intel Arc 上命中 XPU 会快 10-100 倍 |

---

## 📊 性能基准

在 Intel Arc Pro B60（24GB 显存，XPU）下的典型性能（仅参考，以实机为准）：

| 操作 | 时间 | 显存占用 |
|------|------|--------|
| 单帧推理 (4DHumans) | ~50-100ms | ~8GB |
| 显存清理 | ~10ms | 释放 ~8GB |
| 整个工作流 (60fps, 10s 视频) | ~15-20 min | 峰值接近 24GB |

---

## 📚 详细文档

更多信息请查看：
- [SETUP_GUIDE.md](./SETUP_GUIDE.md) - 详细的环境配置指南
- [../README.md](../README.md) - 项目整体架构

---

## ✅ 确认清单

在开始实际工作前，请确认：

- [ ] `check_environment.py` 运行结果全部为 ✅
- [ ] `input_video.mp4` 已放在当前目录
- [ ] 环境已激活 (`conda activate kineto-engine`)
- [ ] 所有依赖已安装 (`pip list | grep torch`)；在 Intel Arc 上 `torch.__version__` 应带 `+xpu`
- [ ] `python -c "import torch;print(torch.xpu.is_available())"` 在 Intel Arc 上为 `True`
- [ ] 4DHumans 已正确安装且已打 `weights_only` 补丁 (`grep -q weights_only 4D-Humans/hmr2/models/__init__.py`)

✨ 一切就绪后，我将提供 `kineto_core.py` 源码。
