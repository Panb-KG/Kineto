# DEPLOY_MOFANG.md — Kineto Engine 设备侧部署与底层复用手册
### 目标设备：MoFang M01 / Intel AI Box（Intel Arc Pro B60 24GB · Ubuntu · nginx · SSH:22）

> **执行边界（务必先读）**
> 云端 Agent **没有** 设备 SSH 凭据，因此本手册中所有"设备端"命令都由**你（用户）在设备上亲自执行**。
> `deploy/` 下的脚本全部满足三条铁律：
> 1. **安全** —— 探测脚本纯只读；部署脚本在 `systemctl disable` / 驱动安装 / `rsync` 之前一律 `read -p` 显式确认；
> 2. **幂等** —— 可重复执行，已完成的步骤自动跳过；模型同步不带 `--delete`，**只增不删**；
> 3. **可逆** —— 停用 MoFang 用 `disable --now`（不是 `mask`、不是卸载、不是重刷），随时 `enable --now` 复原。
>
> 云端 Agent 不会 SSH、不会跑破坏性命令，也不会修改 `kineto-engine/kineto_core.py`、
> `kineto-engine/requirements.txt`（另有 Owner）与 `kineto-web/` 下任何文件。

---

## 0. TL;DR — 推荐路径

```
Phase 0  探测（只读）           →  bash deploy/discover_device.sh            [10 min]
            │  决策门禁 GO？
            ▼
Phase 1  OPTION (ii) 共存验证   →  建独立 venv + 装 torch-XPU + 跑 XPU/CPU 数值一致性
         （零风险，不动 MoFang）    验证通过后 MoFang 全绿                    [30-60 min]
            │
            ▼
Phase 2  OPTION (i) 生产部署    →  可逆停用 MoFang 上层 → 装 Intel GPU 驱动 →
         （推荐的生产默认）         /opt/kineto venv(py3.11) → torch-XPU → requirements →
                                  4D-Humans -e . → rsync 模型 → systemd → Cloudflare Tunnel  [2-4 h]
            │
            ▼
Phase 3  验收                   →  bash deploy/validate.sh   (G1..G12 全绿才算上线)  [30 min]
            │
            ▼
         reboot 复验（硬性要求） →  sudo reboot && bash deploy/validate.sh
```

**OPTION (iii) 全量重装 Ubuntu：仅文档，最后手段。** 见 §6，含砖机/Secure Boot/厂商恢复镜像警告。

**一句话结论**：保留 Ubuntu + SSH + nginx + GPU 驱动，只"关掉" MoFang 的上层业务服务，
把整机当成一台**纯 Intel XPU 推理盒**复用；Kineto Engine 以 systemd 服务跑在 `127.0.0.1:8000`，
由 Cloudflare Tunnel 出公网给 Zeabur 前端。

---

## 1. 权威事实基线（不可协商，其它文档与此冲突时以此为准）

### 1.1 硬件 / 计算后端

| 项 | 事实 |
|---|---|
| GPU | **Intel Arc Pro B60，24GB 显存，Battlemage 架构** |
| ❌ 不是 | **不是 NVIDIA，没有 CUDA，没有 `nvidia-smi`，没有 `--gpus`** |
| 计算后端 | **PyTorch XPU**（`torch>=2.5` 原生支持，`torch.xpu.*` API，`device='xpu'`） |
| 必需驱动 | Intel GPU 内核驱动（`i915` 或 `xe`）+ **compute-runtime (NEO OpenCL ICD)** + **Level Zero** |
| 设备节点 | `/dev/dri/card*`、`/dev/dri/renderD128`；进程用户必须属于 **`render` 与 `video`** 组 |
| wheel 验证平台 | Ubuntu **24.04 / 26.04**（**22.04 未验证** → 走 IPEX 路径或先升级系统） |
| 容器 GPU 直通 | `--device /dev/dri`（compose 里 `devices: ["/dev/dri"]`），**不是** `--gpus` / nvidia-container-runtime |

**PyTorch XPU 安装（确切命令，Phase 1/2 都用这一套）**

```bash
# ① torch 必须**第一个**装（requirements.txt 里有裸 torch>=2.0.0，后装会被 PyPI 的 CPU 轮子覆盖）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# ② 可选增强：Intel Extension for PyTorch（版本必须与 torch 主次版本对齐，2.7.10 ↔ torch 2.7.x）
pip install intel-extension-for-pytorch==2.7.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# ③ 验证（唯一可信的判据）
python -c "import torch;print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
#   期望输出形如: True Intel(R) Arc(TM) Pro B60 Graphics
```

> `kineto-engine/kineto_core.py` 的 `detect_device()` 与 `kineto-engine/api.py` 的 `_detect_device()`
> **都已实现 CUDA > XPU > MPS > CPU 的自动选择**（api.py 已原生识别 `xpu`），并在检测到
> `intel_extension_for_pytorch` 时打印其版本（缺失则静默跳过，不影响 XPU 可用性）。
> ✅ 因此在 Intel Arc 上 `GET /health` 的 `device` 字段应为 **`xpu`**。若报 `cpu`，**是真故障**（非“已知无害缺口”）：
> 按顺序排查 ① `kineto` 服务用户是否在 `render`/`video` 组、② `/dev/dri/renderD128` 是否存在且属组 `render`、
> ③ torch 是否被 `requirements.txt` 换成了 CPU 轮子（`torch.__version__` 应带 `+xpu`）。`deploy/validate.sh` 的 **G1a 对此硬判 FAIL**。
> ⚠️ `/health` 现**需鉴权**（调用时带 `-H "X-API-Key: <key>"`）；另有公开的 `GET /healthz` 作极简存活探针。
> （`kineto-engine/api.py` 另有 Owner，本手册不改其代码，只按其现有契约描述。）

**官方资料来源**（本手册结论所依据的上游文档，集中列出以便核对）

| 结论 / 命令 | 官方来源 |
|---|---|
| Intel GPU 用户态运行时 APT 源（compute-runtime / Level Zero） | `https://repositories.intel.com/gpu/`（Intel GPU 软件栈安装指引） |
| PyTorch XPU 轮子索引（`--index-url …/whl/xpu`） | `https://download.pytorch.org/whl/xpu` |
| Intel Extension for PyTorch（IPEX）XPU 轮子 | `https://pytorch-extension.intel.com/release-whl/stable/xpu/us/` |
| IPEX 文档与版本对齐（2.7.10 ↔ torch 2.7.x） | `https://intel.github.io/intel-extension-for-pytorch/` |
| PyTorch XPU 入门 / `torch.xpu.*` API | `https://pytorch.org/`（Get Started · Intel XPU 平台页） |
| 硬件 SKU（Arc Pro B60 24GB / Battlemage） | Intel ARK 检索（`https://ark.intel.com/` 搜 “Arc Pro B60”）——**确切 SKU / PCI ID 需上机 Q1 实测确认**（`lspci -nn`），本手册不杜撰精确 URL |

> 硬件型号以设备上 `lspci -nn | grep -iE 'vga|display|3d'` 的**实测输出**为准（见 §12 Q1）；
> 上表链接为“去哪里查”的指引，不代替上机核实。

### 1.2 引擎 HTTP API（`kineto-engine/api.py`，已存在，不需重写）

| 项 | 事实 |
|---|---|
| 框架 | FastAPI + uvicorn；**串行 job 队列，并发恒为 1**（边缘设备绝不允许并行推理） |
| 启动 | `uvicorn api:app --host 127.0.0.1 --port 8000` |
| `GET /healthz` | **公开存活探针**（无需鉴权）→ 极简 200；供隧道/容器/监控判活 |
| `GET /health` | **需鉴权**（`X-API-Key`）→ `{status, extraction_mode, device, model_loaded, detector_loaded, queue_depth}`；`device` 在 Arc 上为 `xpu` |
| `POST /jobs` | multipart 字段 `video`，**或** JSON `{"video_path": "…"}`（限制在 inbox 目录内）→ `202 {job_id}` |
| `GET /jobs/{id}` | → `{state: queued|running|done|failed, progress, quality_score, extraction_mode, error?}` |
| `GET /jobs/{id}/pose_data.json` | 姿态数据（`state != done` 时 409） |
| `GET /jobs/{id}/demo_output.mp4` | 演示视频（同上） |
| 环境变量 | `KINETO_JOBS_DIR`（产物目录）、`KINETO_API_KEY`（`X-API-Key` 鉴权）、`KINETO_ALLOW_NO_AUTH=1`（**仅本机调试**）、`KINETO_STRICT=1`（**生产必开**：只守假数据/缺权重/detector降级的 fail-fast，退出码 3；❗**不再对质量做门禁**）、`KINETO_QUALITY_GATE=warn\|fail\|off`（质量门禁：warn=不达标仍 done 附 degraded；fail=不达标判 failed；off=不门禁）、`KINETO_QUALITY_THRESHOLD`（默认 0.6）、`KINETO_CORS_ORIGINS`（逗号分隔） |
| `model_loaded` / `detector_loaded` 语义 | `/health.model_loaded` **仅**反映 4DHumans 权重是否齐全（`.ckpt`+`model_config.yaml`+SMPL `.pkl` 三者都在才 `True`）；`detector_loaded` 单独反映 YOLOv8n（`yolov8n.pt`）。二者已拆分，`model_loaded=False` 不再被 yolo 架空——它如实代表 4DHumans 未布线（→ job 回退合成关键点并被判 FAILED） |
| **OpenPose fallback** | OpenPose fallback 权重（`pose_iter_440000.caffemodel` 等）**不随部署提供，fallback 属有意禁用**；缺权重时引擎会退化为合成关键点，并被 api.py 按 `extraction_mode != '4dhumans'` **硬判 FAILED** |
| **上线硬门禁** | `pose_data.json` 的 `metadata.extraction_mode != '4dhumans'` → job 判 **FAILED**，拒绝交付合成/假姿态数据 |
| **质量分位置** | `metadata.pipeline.final_quality_score`（**不在** `metadata` 顶层，写断言时别看错层级） |
| 鉴权缺失行为（fail-closed） | `KINETO_API_KEY` 未设**且**未设 `KINETO_ALLOW_NO_AUTH=1` → 受保护端点（`/health`、`/jobs` 等）直接返回 **503 `auth not configured`**（不再 WARN+放行）；公开存活探针 `/healthz` 不受影响。本机调试可临时 `KINETO_ALLOW_NO_AUTH=1`，生产绝不允许 |

> **关于 `KINETO_CORS_ORIGINS`**：前端已改为经 **Next.js 服务端代理**同源调用 `/api/*`（见根目录 `DEPLOY_ZEABUR.md`）：
> 浏览器→Zeabur 是**同源**请求（不触发跨源 CORS），Zeabur 服务端→引擎是**服务器间**调用（不受浏览器 CORS 约束）。
> 因此浏览器不再直连引擎，引擎的 CORS 对浏览器**功能上已非必需**；**但**设备侧两处启动守卫
> （`docker-compose.yml` 的 `${KINETO_CORS_ORIGINS:?}`、`kineto-engine.service` 的 `ExecStartPre grep`）
> 仍**要求它为非空的精确 origin**（纵深防御）——留空会导致 `docker compose up` 报错或 systemd 启动失败。
> 照填前端 Zeabur 精确 origin 即可（如 `https://kineto.<你的域名>`）；详见 `DEPLOY_ZEABUR.md` §5。

### 1.3 设备现状与复用意图

| 项 | 处置 |
|---|---|
| Ubuntu OS / SSH(22) / nginx / GPU 驱动 | **保留**（nginx 与 Kineto 无端口冲突：Kineto 只绑回环 8000） |
| MoFang 配对 UI `https://<device>/next/MoFang.html` | 上层应用 → 共存阶段保留，生产阶段可**可逆停用** |
| MoFang assistant / `/bridge/v2/*` 业务 API | 同上 |
| MoFang 的固件/看门狗/风扇/OTA 类服务 | **不要动**（见 §5.2 的"禁止停用清单"） |
| 前端 | 部署在 Zeabur（公网），必须经 **Cloudflare Tunnel** 才能调到局域网内的 API |

---

## 2. Phase 0 — 设备探测（READ-ONLY，不改任何状态）

### 2.1 执行

```bash
# Mac 上：把探测脚本拷到设备并执行（也可以直接 ssh 粘贴执行）
scp -P 22 deploy/discover_device.sh <user>@192.168.1.107:/tmp/
ssh <user>@192.168.1.107 'bash /tmp/discover_device.sh 2>&1 | tee ~/kineto_discover_$(date +%F).log'

# 若设备上没有免密 sudo，用 root 再跑一次以补齐 lshw / dmesg / nginx -T 三段
ssh <user>@192.168.1.107 'sudo bash /tmp/discover_device.sh'
```

脚本会依次采集：OS 版本（`lsb_release -a`、`/etc/os-release`）、内核（`uname -r`）、架构
（`dpkg --print-architecture`）、GPU 身份（`lspci -nn`、`/dev/dri`、`lshw -C display`、
`dmesg | grep -iE 'i915|xe|drm|arc'`、Intel GPU 用户态包清单）、CPU（`lscpu`、`nproc`、AVX/AMX flags）、
RAM（`free -h`、`/proc/meminfo`）、磁盘（`df -hT`、`lsblk`、inode）、docker 版本与 cgroup 版本、
python3 版本、systemd 状态、监听端口（`ss -tlnp`，含 8000 是否空闲）、
**MoFang 服务清单**（`systemctl list-units --type=service --state=running | grep -iE 'mofang|bridge|ai'` +
`list-unit-files` + `docker ps -a` + MoFang 站点根目录线索）、
**nginx 摘要**（`sudo nginx -T | grep -iE 'server_name|location|proxy_pass|root|listen'`）、
既有 Kineto 部署痕迹（幂等性检查），最后打印 PASS/FAIL 门禁汇总。退出码 0 = GO，1 = NO-GO。

### 2.2 决策门禁表（DECISION GATE）

| # | 门禁 | 阈值 | 取自 | 未通过怎么办 |
|---|---|---|---|---|
| **G1** | 系统内存 | **≥ 16 GB** | `/proc/meminfo` MemTotal | HMR2 ViT-H 加载 + 视频缓冲需要；< 16G 只能上更小模型，**不具备生产条件** |
| **G2** | 可用磁盘 | **≥ 25 GB** | `df -k /`（若 `/srv`、`/opt` 独立分区，看脚本 §5 的逐挂载点报告） | venv ≈ 8G + 模型 2.6G + job 产物；清 MoFang 缓存/旧视频/`docker system prune` |
| **G3** | Intel GPU 存在 | `lspci` 有 Intel 显卡 **且** `/dev/dri/renderD*` 存在 | 脚本 §2 | 只有 `lspci` 命中但无 renderD → **驱动未就绪**，先做 §5.3 |
| **G4** | docker 存在 | `docker --version` 可用 | 脚本 §6 | **不阻断**：systemd 是主路径；只有想走容器路径才需要 |
| A1 | Ubuntu 版本 | 24.04 / 26.04 | `/etc/os-release` | 22.04 → 见 §12 开放问题 Q2；先走 IPEX 路径或升级 |
| A2 | 内核版本 | ≥ 6.8（Battlemage 支持较完整） | `uname -r` | 偏旧 → 装 HWE 内核（`linux-generic-hwe-24.04`），**但要先有恢复镜像**（§6） |
| A3 | 架构 | `amd64` | `dpkg --print-architecture` | 非 amd64 → 无 XPU wheel，方案不成立 |
| A4 | 8000 端口 | 空闲 | `ss -tlnp` | 被占 → 改 unit 端口并同步改 tunnel config |
| A5 | 出网能力 | 可达 pypi.org / download.pytorch.org | 脚本 §9 | 不通 → 走离线 wheel 方案（在能上网的机器 `pip download` 后 rsync 过去） |

### 2.3 必须回传给云端的信息

1. 完整探测日志（`~/kineto_discover_*.log`）；
2. `lspci -nn | grep -iE 'vga|display|3d'` 的**原始行**（含 `[8086:xxxx]` PCI ID，用于确认 B60 具体 SKU）；
3. `systemctl list-unit-files --type=service | grep -iE 'mofang|bridge|claw|rag|assistant'` 的**完整输出**
   （§5.2 的停用清单必须以此为准，不能凭猜测）；
4. `sudo nginx -T` 的 server 段（判断是否存在 443/80 反代到 MoFang，以及是否会与 tunnel 冲突）；
5. `mokutil --sb-state`（Secure Boot 状态，影响驱动安装方式）。

---

## 3. 三条路线对比

| | **OPTION (ii) 共存 / 先验证** | **OPTION (i) 生产默认（可逆剥离）** | **OPTION (iii) 全量重装** |
|---|---|---|---|
| 做什么 | 建一个隔离 venv，只装 torch-XPU 并验证算力，**完全不碰 MoFang** | `systemctl disable --now` 停掉 MoFang 上层业务服务，然后正常部署 Kineto | 抹掉整机，装纯净 Ubuntu Server 24.04 |
| 风险 | **≈0**（只多占 ~8GB 磁盘） | 低（全部可逆；最坏情况 `enable --now` 复原） | **高**：可能砖机、丢厂商驱动/恢复分区、丢 MoFang 授权、Secure Boot 拒启 |
| 可逆性 | 删掉 venv 目录即可 | 每条命令都有对应逆操作（§10） | **不可逆**（除非有厂商恢复镜像） |
| 何时选 | **永远先做这一步**（Phase 1） | 共存验证通过 + 确认不再需要 MoFang 上层业务 | 只有当 MoFang 的私有栈污染到无法隔离（例如其 Python/驱动被钉死且与 Intel GPU 用户态冲突），且**已拿到厂商恢复镜像** |
| MoFang 状态 | 完全可用（回归必须全绿） | 上层 UI/业务 API 停用；系统层（GPU/nginx/SSH）保留 | 全部消失 |
| 对应章节 | §4 | §5 | §6（仅文档） |

> **推荐顺序**：Phase 1 (OPTION ii) → 验收 MoFang 回归 → Phase 2 (OPTION i)。
> OPTION (iii) 除非万不得已不要碰；本手册只记录，不提供脚本。

---

## 4. Phase 1 — OPTION (ii)：共存验证（零风险，不动 MoFang）

**目标**：在不影响 MoFang 任何服务的前提下，证明"这台机器的 Intel Arc Pro B60 能被 PyTorch XPU 用起来，
且数值正确"。这一步一旦通过，Phase 2 的所有不确定性就只剩工程性问题。

> 所有命令在**设备上**执行。整个过程只新增一个目录 `~/kineto-probe`，不写系统路径、不动服务。

### 4.1 准备：系统级只读依赖检查

```bash
# 编译/运行期需要的系统库（opencv-python、pyrender、ffmpeg）。
# 若 §2 探测显示已存在则跳过；安装系统包属于"状态变更"，请自行确认后再执行：
sudo apt update
sudo apt install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    build-essential git ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libosmesa6

# Ubuntu 24.04 若源里没有 python3.11（默认 3.12），用 deadsnakes：
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 -V          # 期望 Python 3.11.x
```

> ⚠️ **不要** 顺手执行 `sudo apt upgrade` / `do-release-upgrade`：升级内核可能让 MoFang 的
> 私有驱动/DKMS 模块失效，且无法轻易回滚。只装明确需要的包。

### 4.2 建隔离 venv + 安装 PyTorch XPU

```bash
mkdir -p ~/kineto-probe && cd ~/kineto-probe
python3.11 -m venv probe-venv                      # 与系统 python、与 MoFang 完全隔离
source probe-venv/bin/activate
pip install --upgrade pip wheel setuptools

# ★ 确切的 XPU 安装命令（与 §1.1 一致）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 可选：IPEX（kineto_core.detect_device() 会尝试 import 它；torch>=2.5 已原生支持 XPU，不装也能跑）
pip install intel-extension-for-pytorch==2.7.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

python -c "import torch;print(torch.__version__)"  # 期望形如 2.7.x+xpu  ← 必须带 +xpu 后缀
```

> **权限坑（Phase 1 就会暴露）**：若下一步 `torch.xpu.is_available()` 返回 `False`，
> 九成是当前用户不在 `render` 组：
> ```bash
> ls -l /dev/dri                      # renderD128 的属组通常是 render
> id                                  # 看自己有没有 render / video
> sudo usermod -aG render,video $USER && newgrp render    # 加组后必须重新登录/换组才生效
> ```

### 4.3 验证 ①：XPU 设备可见

```bash
python -c "import torch;print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
#   期望: True Intel(R) Arc(TM) Pro B60 Graphics
#   False → 看 §9 排障表 "XPU-1"

python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("xpu available:", torch.xpu.is_available())
print("device count :", torch.xpu.device_count())
print("device name  :", torch.xpu.get_device_name(0))
p = torch.xpu.get_device_properties(0)
print("properties   :", p)
try:
    print("VRAM (GB)    :", round(p.total_memory / 1e9, 1))   # 期望 ≈ 24
except Exception as e:
    print("VRAM 查询失败 :", e)
try:
    import intel_extension_for_pytorch as ipex
    print("IPEX         :", ipex.__version__)
except Exception:
    print("IPEX         : 未安装（可选）")
PY
```

**通过判据**：`xpu available: True`、`device count >= 1`、`device name` 含 `Arc`/`B60`、`VRAM ≈ 24 GB`。

### 4.4 验证 ②：XPU vs CPU 数值一致性（sanity check）

> 这一步比"能跑起来"更重要：它证明 XPU 后端算子**结果正确**，而不是返回全 0/NaN。

```bash
python - <<'PY'
import torch, time, sys

torch.manual_seed(0)
FAIL = []

def check(name, a, b, atol=1e-3, rtol=1e-3):
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    nan = bool(torch.isnan(b).any())
    print(f"{'PASS' if ok and not nan else 'FAIL'}  {name:28s} max|Δ|={ (a-b).abs().max().item():.3e}  nan={nan}")
    if not (ok and not nan):
        FAIL.append(name)

if not torch.xpu.is_available():
    print("FAIL  torch.xpu.is_available() == False —— 先解决驱动/组权限"); sys.exit(1)

dev = torch.device("xpu")

# 1) 矩阵乘（SYCL/oneMKL 主链路）
A = torch.randn(1024, 1024); B = torch.randn(1024, 1024)
check("matmul fp32", (A @ B), (A.to(dev) @ B.to(dev)).cpu())

# 2) 卷积（HMR2 ViT-H 的 patch embed 就是 conv/linear 组合）
x = torch.randn(2, 3, 256, 256); w = torch.randn(16, 3, 3, 3)
check("conv2d fp32",
      torch.nn.functional.conv2d(x, w, padding=1),
      torch.nn.functional.conv2d(x.to(dev), w.to(dev), padding=1).cpu())

# 3) LayerNorm + GELU + Softmax（Transformer 关键算子）
ln = torch.nn.LayerNorm(512); y = torch.randn(4, 64, 512)
check("layernorm", ln(y), ln.to(dev)(y.to(dev)).cpu())
check("gelu", torch.nn.functional.gelu(y), torch.nn.functional.gelu(y.to(dev)).cpu())
s = torch.randn(8, 24)
check("softmax", torch.softmax(s, -1), torch.softmax(s.to(dev), -1).cpu())

# 4) bf16（Arc 支持 bf16；若失败说明该 dtype 路径未就绪，需强制 fp32）
try:
    Ab, Bb = A.bfloat16(), B.bfloat16()
    check("matmul bf16", (Ab @ Bb).float(), (Ab.to(dev) @ Bb.to(dev)).cpu().float(), atol=2e-2, rtol=2e-2)
except Exception as e:
    print("WARN  bf16 路径异常:", e)

# 5) 显存分配 + 真实吞吐（24GB 卡应能轻松吃下 4GB 张量）
try:
    big = torch.empty(int(4e9 // 4), device=dev)   # 4 GB fp32
    big.fill_(1.0); torch.xpu.synchronize()
    print(f"PASS  显存分配 4GB             allocated={torch.xpu.memory_allocated()/1e9:.2f} GB")
    del big; torch.xpu.empty_cache()
except Exception as e:
    FAIL.append("vram-4GB"); print("FAIL  显存分配 4GB:", e)

N = 2048
Ac, Bc = torch.randn(N, N, device=dev), torch.randn(N, N, device=dev)
for _ in range(3): Ac @ Bc                      # warmup
torch.xpu.synchronize(); t0 = time.time()
for _ in range(10): Ac @ Bc
torch.xpu.synchronize(); t_xpu = (time.time() - t0) / 10
Ah, Bh = A.clone(), B.clone()
t0 = time.time()
for _ in range(2): Ah @ Bh
t_cpu = (time.time() - t0) / 2
gf = 2 * N**3 / 1e9
print(f"INFO  matmul {N}x{N}: XPU {t_xpu*1e3:.1f} ms ({gf/t_xpu:.1f} GFLOPS) | CPU {t_cpu*1e3:.1f} ms ({gf/t_cpu:.1f} GFLOPS) | speedup x{t_cpu/t_xpu:.1f}")
if t_xpu >= t_cpu:
    print("WARN  XPU 没跑赢 CPU —— 可能在用 CPU 回退或驱动异常，需人工看一眼")

print("\n== 结论:", "ALL PASS ✅ 可以进入 Phase 2" if not FAIL else f"FAIL ❌ {FAIL}")
sys.exit(1 if FAIL else 0)
PY
```

**通过判据**：全部 `PASS`，`结论: ALL PASS`，且 XPU matmul 明显快于 CPU（Arc Pro B60 上通常 5×~50×）。

### 4.5 验证 ③（可选）：HMR2 权重能否在 XPU 上加载并前向

```bash
cd ~/kineto-probe
pip install -r /opt/kineto/kineto-engine/requirements.txt   # 若代码已同步（Phase 2 才做，可跳过）
python - <<'PY'
# 只验证"权重能 load 到 xpu 且不炸"，不做完整视频推理
import torch
from hmr2.models import load_hmr2
m, cfg = load_hmr2()
m = m.to("xpu").eval()
x = torch.randn(1, 3, 256, 256, device="xpu")
with torch.no_grad():
    out = m(x)
print("forward OK, keys:", list(out.keys())[:6] if isinstance(out, dict) else type(out))
print("xpu mem allocated: %.2f GB" % (torch.xpu.memory_allocated()/1e9))
PY
```

> 若报 `UnpicklingError / weights_only` → 说明用的 4D-Humans 是**上游原版**而非本仓库
> 已打补丁的版本（本仓库 `kineto-engine/4D-Humans/hmr2/models/__init__.py` 顶部有
> `torch.load` 的 `weights_only=False` 补丁）。解决办法：用 §5.5 同步过来的那份，别 `git clone` 上游。

### 4.6 MoFang 回归检查（共存路径下**必须全绿**）

```bash
# ① 配对 UI 仍可用
curl -sk -o /dev/null -w 'MoFang.html      HTTP %{http_code}\n' https://192.168.1.107/next/MoFang.html
#   期望 200

# ② 业务 API 的 agent 健康（openclaw / rag）
curl -sk https://192.168.1.107/bridge/v2/bootstrap | head -c 1200; echo
#   期望 HTTP 200，且响应里 agentHealth 中 openclaw 与 rag 均为 ok/healthy
#   （不同固件字段名可能是 agentHealth / agents / health，人工核对语义即可）

# ③ 服务面没被我们影响
systemctl list-units --type=service --state=running | grep -iE 'mofang|bridge|ai'
sudo systemctl status nginx --no-pager | head -n 5

# ④ GPU 没有被我们独占（MoFang 若也用 GPU，需确认共享可行）
ls -l /dev/dri; sudo dmesg | grep -iE 'i915|xe' | tail -n 10
```

一键版：`bash deploy/validate.sh --mofang-mode coexist`（它会把 G10 回归做成门禁项）。

### 4.7 Phase 1 判定

| 结果 | 动作 |
|---|---|
| §4.3 `True` + §4.4 ALL PASS + §4.6 MoFang 全绿 | ✅ **进 Phase 2**（§5）。此时 MoFang 依然完好，Kineto 只是多了一个 venv |
| §4.3 `False` | 查 §9 的 XPU-1 / XPU-2；90% 是 `render` 组或驱动缺失 |
| §4.4 出现 NaN / 不 close | 记录 torch 与 IPEX 版本、GPU 型号，回传云端；先不要进 Phase 2 |
| §4.6 MoFang 挂了 | **立刻回滚**：`rm -rf ~/kineto-probe`，然后排查是谁动了服务（本阶段不应该有任何服务变化） |
| 清理（任何时候都可） | `deactivate; rm -rf ~/kineto-probe` —— 完全无痕 |

---

## 5. Phase 2 — OPTION (i)：生产部署（可逆剥离 MoFang 上层）

> 约定符号：下文 `DEV=<user>@192.168.1.107` 是设备 SSH 目标；`/opt/kineto` 是代码与 venv 根，
> `/srv/kineto` 是模型/产物/服务账号 HOME 根。两者分开是为了让「代码可整目录替换」而「2.6GB 模型不动」。

### 5.0 部署前置检查清单

- [ ] Phase 0 门禁 G1/G2/G3 全 PASS（`bash deploy/discover_device.sh` 退出码 0）
- [ ] Phase 1 §4.3/§4.4 全 PASS，且 §4.6 MoFang 回归全绿
- [ ] 已拿到 §2.3 的 MoFang 单元清单（§5.2 要用）
- [ ] 已确认 §12 的开放问题 Q3（保留 vs 剥离 MoFang）与 Q9/Q10（域名 + Zeabur origin）
- [ ] 手边有设备的物理访问途径（万一需要断电重启）

### 5.1 存证：把 MoFang 当前状态完整备份（回滚的前提）

```bash
mkdir -p ~/kineto-rollback-evidence && cd ~/kineto-rollback-evidence
systemctl list-unit-files --type=service  > all_services_before.txt
systemctl list-units --type=service --state=running > running_services_before.txt
systemctl list-timers --all               > timers_before.txt      # 看有没有厂商看门狗会重新拉起
sudo nginx -T                             > nginx_full_before.conf 2>/dev/null
sudo lshw -C display                      > gpu_before.txt 2>/dev/null
dpkg -l                                   > dpkg_before.txt
apt-mark showhold                         > apt_holds_before.txt
ip -4 -o addr                             > network_before.txt
ss -tlnp                                  > ports_before.txt 2>/dev/null
df -hT                                    > disk_before.txt
tar czf ~/mofang_evidence_$(date +%F).tgz -C ~ kineto-rollback-evidence
ls -lh ~/mofang_evidence_*.tgz
```

> 这份存证是 §10 回滚手册的唯一依据。**不要跳过**。

### 5.2 停用 MoFang 上层业务服务（可逆；`disable --now`，不 mask、不 purge）

**禁止停用清单（动了就可能变砖或失联）**

> 记法：下面是 `strip_mofang.sh` 里 `FORBIDDEN_RE` 的**前导词边界 token**（不是 glob 子串）。
> `fan`/`drm`/`thermal`/`power` 等按 **token 词首**命中（`fancontrol`、`drm`、`thermald`…），
> **不会**误伤 `mofang` 里的 "fan"（其前是词字符 `o`，无词边界）；`xe` 用**精确 token**（`\bxe\b`）
> 拦 Intel Xe GPU 驱动单元，**不含 `xeon`** —— 本机是 Xeon 平台 Intel AI Box，`mofang-xeon-agent.service`
> 会被正确 **ALLOW**（旧写法 `xe*` 会误拦 xeon，已修正）。命中即 `exit 1`；确属误拦才用 `--allow-forbidden --yes`（会记 `[OVERRIDE]` 审计行）。

```
ssh   sshd   openssh   network   systemd   systemd-timesyncd   getty   dbus
nginx docker  containerd  udev  kmod  intel  i915  xe(精确token,不含xeon)  gpu  drm
fwupd thermald thermal  fan  power  battery  watchdog  cron  crond  rsyslog
chrony ntp ntpsec  unattended-upgrades  snapd  polkit  apparmor  audit
ota   firmware  license  （厂商 OTA / 固件升级 / 授权 / 时钟同步类服务）
```

**操作步骤（一律走 `deploy/strip_mofang.sh` —— 它内置禁停硬拦截、确认、undo 文件、`--rollback`）**

> ⚠️ **不要再手工 `for u in ...; do systemctl disable --now "$u"; done`。**
> `deploy/strip_mofang.sh` 会在停用前对上面的「禁止停用清单」做**正则硬拦截**：命中即 `exit 1`
> 且**绝不停用任何单元**（避免手滑把 `sshd`/`network`/`intel-*` 停掉导致设备失联或变砖）。
> 停用一律用 `disable --now`（可逆），**不 mask、不 purge**；并生成 undo 文件供一键回滚。

```bash
# ① 只读发现候选（rag 必须用词边界 \brag\b，否则 'rag' 会误命中 storage 的 "sto-rag-e"）：
systemctl list-unit-files --type=service | grep -iE 'mofang|bridge|claw|\brag\b|assistant'

# ② 干跑：只打印「将停用」与「将拒绝（命中禁停清单）」，不实际改动任何东西
sudo bash deploy/strip_mofang.sh --dry-run

# ③ 先看清每个候选单元是什么、谁依赖它（只读；strip 脚本不做这步，人工核对）
for u in mofang-assistant.service mofang-bridge.service; do   # 换成你真机上的名字
  echo "===== $u ====="; systemctl cat "$u" 2>/dev/null | head -n 25
  systemctl list-dependencies --reverse "$u" 2>/dev/null | head -n 15
done
systemctl list-timers --all | grep -iE 'mofang|bridge' || echo '(无相关 timer)'   # 有无看门狗会拉回来

# ④ 正式停用（脚本会打印将停用清单并要求显式确认；命中禁停清单则直接 exit 1）
sudo bash deploy/strip_mofang.sh --units "mofang-assistant.service mofang-bridge.service"
#   或让脚本自己现场发现候选： sudo bash deploy/strip_mofang.sh
#   脚本会在 $HOME/kineto-rollback-evidence/strip_undo_<时间戳>.txt 记下 undo 清单（回滚用），务必连同 §5.1 存证一起保存。

# ⑤ 复核：上层 UI 应当已不可达，而 nginx/SSH/GPU 必须照常
curl -sk -o /dev/null -w 'MoFang.html → HTTP %{http_code}\n' https://192.168.1.107/next/MoFang.html
systemctl is-active nginx ssh docker 2>/dev/null
ls -l /dev/dri
```

**如何恢复（可逆性证明 —— 请现在就记下来）**

```bash
# 首选：用 strip 脚本生成的 undo 文件一键回滚（内部对每个单元 systemctl enable --now）
sudo bash deploy/strip_mofang.sh --rollback "$HOME/kineto-rollback-evidence/strip_undo_<时间戳>.txt"

# 或直接手工恢复：
sudo systemctl enable --now mofang-assistant.service mofang-bridge.service
# 与存证比对是否还有遗漏：
#   diff <(systemctl list-unit-files --type=service) ~/kineto-rollback-evidence/all_services_before.txt
# 若曾被 mask（本手册与 strip 脚本都不用 mask）：sudo systemctl unmask <unit> && sudo systemctl enable --now <unit>
curl -sk -o /dev/null -w '%{http_code}\n' https://192.168.1.107/next/MoFang.html   # 应回到 200
```

> **nginx 说明**：MoFang 的静态页由 nginx 提供。停用后端服务后，`/next/MoFang.html` 可能仍返回 200
> （静态文件还在），但页面里的 `/bridge/v2/*` 调用会 502。这是**预期**的，不要为此去改 nginx 配置——
> 保留 nginx 原样才能让 MoFang 随时恢复，也避免与 Cloudflare Tunnel 产生不必要的耦合。

### 5.3 安装 Intel GPU 驱动（compute-runtime + Level Zero）

> **先看 §2 探测报告的 §2b 段**：MoFang 出厂机器很可能**已经装了**驱动（`intel-opencl-icd` /
> `level-zero` 已在 dpkg 列表里，`/dev/dri/renderD128` 已存在）。若已存在 → **跳过本节**，
> 直接做 §5.4。重复安装/升级驱动是本手册里少数可能影响 MoFang 的操作。

```bash
# ① 加 Intel 官方 GPU 软件栈源（noble = 24.04；22.04 用 jammy，但见 §12 Q2）
sudo mkdir -p --mode=0755 /etc/apt/keyrings
wget -qO- https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --dearmor --yes --output /etc/apt/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/etc/apt/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list
sudo apt update

# ② 【关键】安装前先干跑 + 快照 dpkg，评估「连带升级面」（驱动升级是本手册少数可能影响 MoFang 的操作）
apt-cache policy intel-opencl-icd intel-level-zero-gpu level-zero libze1     # 先看会装什么版本
# 干跑：-s 只模拟不落地，重点看 "The following packages will be REMOVED / upgraded"——
#   若它要连带升级 libc/mesa/内核相关或移除 MoFang 依赖的包，先停下来核对，不要盲目 -y
sudo apt-get -s install intel-opencl-icd intel-level-zero-gpu level-zero libze1 libze-intel-gpu1 | grep -iE 'Remv|Inst|upgraded|newly installed' 
dpkg -l > ~/kineto-rollback-evidence/dpkg_before_driverinstall.txt      # 安装前包快照

# ③ 确认干跑无异常后再真正安装（只装用户态运行时，内核态 i915/xe 由 Ubuntu 内核提供）
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero libze1 libze-intel-gpu1
sudo apt install -y clinfo intel-media-va-driver-non-free      # 可选：诊断/硬解
dpkg -l > ~/kineto-rollback-evidence/dpkg_after_driverinstall.txt       # 安装后包快照
diff ~/kineto-rollback-evidence/dpkg_before_driverinstall.txt \
     ~/kineto-rollback-evidence/dpkg_after_driverinstall.txt | grep -iE '^[<>]'   # 连带升级面：逐条核对

# ④ 验证（三条都要过）
ls -l /dev/dri                       # 应有 card0/card1 + renderD128
sudo clinfo | grep -iE 'Device Name|Driver Version|Global memory size' | head -n 10
ls -l /etc/OpenCL/vendors/           # 应有 intel.icd
sudo dmesg | grep -iE 'i915|xe' | tail -n 10
```

> 若 ② 的干跑或 ③ 的 `diff` 显示驱动包**连带升级了 MoFang 依赖的库**（如 mesa/libc/内核），
> 先回 §5.1 存证确认 MoFang 可回滚，再决定是否继续；必要时用 `apt-mark hold` 钉住敏感包。

> Secure Boot 开启时（`mokutil --sb-state` → `SecureBoot enabled`），第三方内核模块会被拒载。
> 本节的包都是用户态，**不涉及**模块签名；但若你还需要装 DKMS 内核模块，必须先准备 MOK 签名密钥，
> 否则重启后 GPU 直接消失。这也是 §6 重装路线被标为高危的原因之一。

### 5.4 创建服务账号与目录骨架

```bash
# ① 专用非特权账号（HOME=/srv/kineto —— 这决定了 4DHumans 缓存路径，别改）
sudo useradd --system --create-home --home-dir /srv/kineto --shell /usr/sbin/nologin kineto || true
sudo usermod -aG render,video kineto        # ★ XPU 访问的硬性要求
id kineto                                    # 必须看到 render、video

# ② 目录
sudo mkdir -p /opt/kineto /srv/kineto/{models,jobs,inbox,.cache}
sudo chown -R kineto:kineto /srv/kineto
sudo chmod 0755 /srv/kineto/jobs

# ③ 自检：kineto 用户能否打开 GPU 设备节点
sudo -u kineto bash -c 'test -r /dev/dri/renderD128 && test -w /dev/dri/renderD128 && echo "renderD128 OK" || echo "renderD128 权限不足"'
```

### 5.5 同步代码（Mac → 设备）

```bash
# 在 Mac 上执行。注意 4D-Humans/ 与 yolov8n.pt 在仓库 .gitignore 里，
# 所以设备上 git clone 拿不到 —— 必须 rsync 过去（本仓库这份还含 torch>=2.6 的补丁）。
DEV=<user>@192.168.1.107
ssh $DEV 'sudo mkdir -p /opt/kineto && sudo chown $(id -un) /opt/kineto'

rsync -az --partial --info=progress2 \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'output/' --exclude 'output_test/' --exclude 'output_closed/' \
  --exclude 'checkpoints/' --exclude '*.mp4' --exclude '.DS_Store' \
  --exclude '.git/' --exclude 'hmr2.egg-info/' \
  ~/Projects/Kineto/kineto-engine/  $DEV:/opt/kineto/kineto-engine/

rsync -az --partial ~/Projects/Kineto/deploy/            $DEV:/opt/kineto/deploy/
rsync -az         ~/Projects/Kineto/DEPLOY_MOFANG.md     $DEV:/opt/kineto/

ssh $DEV 'ls -l /opt/kineto/kineto-engine/api.py /opt/kineto/kineto-engine/4D-Humans/setup.py'
```

### 5.6 建 venv（Python 3.11）并按**严格顺序**安装依赖

```bash
DEV=<user>@192.168.1.107
ssh $DEV
sudo -u kineto bash <<'EOF'
set -e
python3.11 -m venv /opt/kineto/venv
/opt/kineto/venv/bin/pip install --upgrade pip wheel setuptools

# ① ★ torch XPU 必须第一个装
cd /opt/kineto/kineto-engine
/opt/kineto/venv/bin/pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu
/opt/kineto/venv/bin/python -c "import torch;print('torch', torch.__version__, 'xpu', torch.xpu.is_available())"

# ② 再装项目依赖（裸 torch>=2.0.0 已被 ① 满足，不会被覆盖成 CPU 轮子）
/opt/kineto/venv/bin/pip install -r requirements.txt

# ③ 防呆校验：确认 torch 仍是 +xpu 版本，被覆盖就强制修回来
/opt/kineto/venv/bin/python -c "import torch,sys;v=torch.__version__;sys.exit(0 if '+xpu' in v or getattr(torch.version,'xpu',None) else 1)" \
  || /opt/kineto/venv/bin/pip install --force-reinstall --no-deps torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/xpu

# ④ 最后装 4D-Humans（editable）。★ 不要装 '[all]' extra —— detectron2 无 CUDA 工具链会编译失败，
#    而且 Kineto 用 YOLOv8n 做人体检测，根本不需要 ViTDet/detectron2
cd /opt/kineto/kineto-engine/4D-Humans
/opt/kineto/venv/bin/pip install -e .
/opt/kineto/venv/bin/python -c "import hmr2; print('hmr2 import OK')"

# ⑤ 可选：IPEX（版本要与 torch 主次版本对齐）
# /opt/kineto/venv/bin/pip install intel-extension-for-pytorch==2.7.10+xpu \
#     --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
EOF

# ⑥ 以 kineto 身份复验 XPU（这一步才是服务真实运行环境下的结论）
sudo -u kineto env HOME=/srv/kineto /opt/kineto/venv/bin/python -c \
  "import torch;print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

> Ubuntu 24.04 的 PEP 668（externally-managed-environment）会阻止往系统 python 里 pip install ——
> 全程用 `/opt/kineto/venv` 就没这个问题，**不要**用 `--break-system-packages`。

### 5.7 同步模型权重（Mac → 设备）+ 缓存布线

```bash
# 在 Mac 上执行；脚本会先打印完整计划并 read -p 确认，再开始 rsync（不带 --delete）
bash deploy/transfer_models.sh --device <user>@192.168.1.107 --port 22 --wire
#   先演练一遍（不传数据）：加 --dry-run
#   只要校验大小、跳过 2.5GB 的 sha256：加 --quick
```

脚本产出布局与解析逻辑：

```
/srv/kineto/models/4DHumans/…            ← 规范副本（2.5GB ckpt + model_config.yaml + data/）
/srv/kineto/models/engine/yolov8n.pt
/srv/kineto/models/engine/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

systemd 路径: HOME=/srv/kineto → $HOME/.cache/4DHumans
              --wire 会建软链 /srv/kineto/.cache/4DHumans → /srv/kineto/models/4DHumans
              并把 yolov8n.pt / basicModel*.pkl 软链进 /opt/kineto/kineto-engine（引擎用相对路径）
Docker  路径: 卷挂载 /srv/kineto/models:/root/.cache，HOME=/root
              → /root/.cache/4DHumans == /srv/kineto/models/4DHumans（无需软链）
```

> 为什么排除 `hmr2_data.tar.gz`：它只是 `logs/` + `data/` 的打包来源（2.5GB），已解压过，传它是纯浪费。
> `hmr2.models.download_models()` 会在它缺失时尝试重新下载，但引擎走的是 `load_hmr2()`，不会触发下载。

布线后自检（**上线前必跑**）：

```bash
sudo -u kineto env HOME=/srv/kineto /opt/kineto/venv/bin/python -c \
 "from hmr2.configs import CACHE_DIR_4DHUMANS as C; import os; \
  print(C); print('dir exists :', os.path.isdir(C)); \
  print('ckpt exists:', os.path.isfile(C+'/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt')); \
  print('cfg  exists:', os.path.isfile(C+'/logs/train/multiruns/hmr2/0/model_config.yaml')); \
  print('smpl exists:', os.path.isfile(C+'/data/smpl/SMPL_NEUTRAL.pkl'))"
# 四个都必须是 True；任一 False → 引擎会静默回退到合成关键点，api.py 直接把 job 判 FAILED
ls -l /opt/kineto/kineto-engine/yolov8n.pt      # 应是软链或实体文件
```

### 5.8 密钥 / CORS / systemd 部署

```bash
# ① 生成密钥并写入受保护的 env 文件（unit 用 EnvironmentFile=- 读取，密钥不入库）
sudo mkdir -p /etc/kineto
printf 'KINETO_API_KEY=%s\nKINETO_CORS_ORIGINS=%s\n' \
  "$(openssl rand -hex 32)" \
  "https://kineto-web.zeabur.app" | sudo tee /etc/kineto/kineto-engine.env
sudo chown root:kineto /etc/kineto/kineto-engine.env
sudo chmod 0640 /etc/kineto/kineto-engine.env
sudo cat /etc/kineto/kineto-engine.env      # 把 KINETO_API_KEY 抄给前端（Zeabur 环境变量）

# ② 安装 unit（模板里的 __SET_ME__ / __SET_ZEABUR_ORIGIN__ 会被 ① 的 EnvironmentFile 覆盖）
#    ⚠️ unit 内置了 ExecStartPre 守卫：若 ① 未做（env 文件缺失/密钥仍是占位符/
#       CORS 不以 http(s):// 开头），服务会**故意启动失败**而不是带占位密钥裸奔。
#       失败后：sudo systemctl reset-failed kineto-engine 再 restart（详见 §9 API-4）。
sudo install -m 0644 /opt/kineto/deploy/kineto-engine.service /etc/systemd/system/kineto-engine.service
sudo systemctl daemon-reload

# ③ 先前台冒烟一次（比直接 enable 更好排障）
#    ⚠️ api.py 现为 fail-closed：不传 KINETO_API_KEY 时受保护端点（/health、/jobs）直接返回 503，
#       只有公开存活探针 /healthz 仍 200。冒烟只为验「能起 + XPU 可见」，故临时加 KINETO_ALLOW_NO_AUTH=1
#       放行 /health（**仅本机调试**，生产绝不允许）；正式跑一定走 ④ 的 systemd（带 EnvironmentFile 的真密钥）。
sudo -u kineto env HOME=/srv/kineto KINETO_JOBS_DIR=/srv/kineto/jobs PYOPENGL_PLATFORM=osmesa \
  KINETO_ALLOW_NO_AUTH=1 \
  /opt/kineto/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --app-dir /opt/kineto/kineto-engine
#   另开一个终端:
#     curl -fsS http://127.0.0.1:8000/healthz                        → {"status":"ok"}（公开存活）
#     curl -fsS http://127.0.0.1:8000/health | python3 -m json.tool  → 看 "device":"xpu"（因加了 ALLOW_NO_AUTH 才可达）
#   然后 Ctrl-C 停掉。
#   注意：kineto_core.py 的 `[Device] Intel XPU: …` 是**跑推理 job 时**（子进程内 detect_device() 惰性调用）才打印，
#         且被 api.py 以 capture_output 捕获——裸起 uvicorn（不提交 job）在终端/journal 里都看不到它。
#         冒烟阶段以 /health 的 "device":"xpu" 字段为准（权威判据另见 validate.sh G1a/G2）。

# ④ 正式启动 + 开机自启
sudo systemctl enable --now kineto-engine
systemctl status kineto-engine --no-pager
# XPU 权威判据：curl -fsS http://127.0.0.1:8000/health -H "X-API-Key: <key>" 看 "device":"xpu"
#   （kineto_core 的 [Device] Intel XPU 行在子进程内被 capture_output 捕获，成功时不进 journalctl；
#    别以「journal 里没有 [Device] 行」误判 XPU 失败——以 /health device 字段与 validate.sh G1a 为准）
journalctl -u kineto-engine -f          # 看启动日志/报错；提交首个 job 后看 queue/worker 活动
```

### 5.9 Cloudflare Tunnel（公网入口，必需）

完整步骤、DNS CNAME 说明、100MB 上传上限对策、排障表见 **`deploy/cloudflared/README.md`**。摘要：

```bash
sudo apt install -y cloudflared                                  # 官方 apt 源，见 README §2
sudo cloudflared tunnel login                                     # 浏览器授权
sudo cloudflared tunnel create kineto-engine                      # 记下 UUID
sudo cloudflared tunnel route dns kineto-engine kineto-api.<YOUR_DOMAIN>   # 自动建 CNAME
sudo cp /opt/kineto/deploy/cloudflared/config.yml /etc/cloudflared/config.yml
sudo sed -i -e 's/__TUNNEL_UUID__/<UUID>/g' -e 's/__YOUR_DOMAIN__/<域名>/g' /etc/cloudflared/config.yml
sudo cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate
sudo install -m 0644 /opt/kineto/deploy/cloudflared/cloudflared.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now cloudflared
curl -fsS https://kineto-api.<YOUR_DOMAIN>/healthz                # 从 Mac 上验证存活（公开）；/health 需带 X-API-Key
```

### 5.10 Phase 2 完成判据

- [ ] `systemctl is-active kineto-engine` = `active`，`is-enabled` = `enabled`
- [ ] `curl -fsS http://127.0.0.1:8000/health -H "X-API-Key: <key>"` 的 `"device"` = `xpu`（**不能是** `cpu`），或 `validate.sh` G1a=PASS（`[Device] Intel XPU` 行由子进程 capture_output 捕获、成功时不进 journalctl，勿以其缺失误判 XPU 失败）
- [ ] `curl -fsS http://127.0.0.1:8000/healthz` 返回 200（公开存活；`/health` 需带 `-H "X-API-Key: <key>"` 才返回 200）
- [ ] `curl -fsS https://kineto-api.<YOUR_DOMAIN>/healthz` 返回 200（公网链路存活）
- [ ] `bash deploy/validate.sh` 的 G1..G12 全 PASS → 进 §8

---

## 6. Phase 3 — OPTION (iii)：全量重装纯 Ubuntu（**仅文档 · 最后手段**）

> ⚠️⚠️⚠️ **这一节没有脚本，是故意的。** 重装是**不可逆**操作，必须你亲自、逐字确认后才能做。

### 6.1 为什么默认不推荐

| 风险 | 后果 |
|---|---|
| **砖机 (brick)** | MoFang M01 这类一体机常把引导/恢复分区、风扇控制、看门狗、BIOS 定制项放在厂商镜像里。抹盘后可能**无法开机、风扇不转、无法进 BIOS**，只能返厂 |
| **没有厂商恢复镜像** | 一旦抹掉，无法回到出厂状态；MoFang 的授权/设备绑定/云端注册可能永久失效 |
| **Secure Boot / 引导锁定** | 若 BIOS 开了 Secure Boot 且只允许厂商签名引导项，自装 Ubuntu 可能根本**无法引导**（需先关 Secure Boot，而部分机型该选项被锁或需厂商密码） |
| **GPU 驱动丢失** | 厂商镜像里可能预装了针对 Arc Pro B60 的定制 compute-runtime/内核参数（如 `i915.force_probe`、`xe` 模块参数）。重装后需自己重建，Battlemage 在旧内核上可能不被识别 |
| **硬件适配** | 存储控制器/网卡/Wi-Fi/串口可能需厂商驱动，纯 Ubuntu 下可能没网——而没网就装不了驱动，死循环 |
| **收益为零** | Kineto 只需一个 venv + 一个 systemd 服务 + 一个出站隧道。§5 已经能在**不抹盘**的前提下拿到全部收益 |

### 6.2 只有在以下**全部**成立时才考虑重装

1. §5.3 的 Intel GPU 用户态与 MoFang 预装栈**硬冲突**（例如 MoFang 钉死了旧版 compute-runtime 且 `apt-mark hold`，无法升级）；且
2. 已从厂商**书面拿到**：恢复镜像（U 盘/下载链）+ BIOS 密码（如有）+ 重装后 GPU 驱动清单；且
3. 已物理接触设备（接了显示器/键盘，能进 BIOS，能插 U 盘），且
4. 已接受“MoFang 上层应用永久丢失”并由业务方书面确认。

### 6.3 重装步骤纲要（仅记录，不提供自动化）

```
0. 先做全盘镜像备份（不是文件备份！）：
     用另一台机器 + USB 存储，dd/Clonezilla 把整盘包括 ESP/恢复分区完整克隆一份，
     并验证镜像可回写。没这一步就不要开始。
1. BIOS：关 Secure Boot（若可）、设 UEFI 优先从 USB 引导、记下原始引导项顺序
2. 介质：Ubuntu Server 24.04.x LTS amd64（建议选带 HWE 内核的 24.04.2+，对 Battlemage 更友好）
3. 分区：至少给 / 预留 60GB（venv 8G + 模型 2.6G + jobs）；保留/重建 ESP；
         若厂商恢复分区存在，**不要删**（给自己留后路）
4. 安装后第一事：确认 GPU
     lspci -nn | grep -iE 'vga|display|3d'
     ls -l /dev/dri
     若无 renderD → 按 §5.3 装 compute-runtime + Level Zero（此时可能需要 HWE 内核）
5. 再按 §5.4 → §5.9 正常部署（跳过 §5.1/§5.2 的 MoFang 部分，因为已经没了）
6. 验收：§8 + reboot 复验
```

### 6.4 回滚（重装的“回滚”只有一种）

用步骤 0 的全盘镜像回写。**若没有步骤 0，就没有回滚。**

---

## 7. 备选路径 — Docker 容器（`deploy/Dockerfile.engine` + `docker-compose.yml`）

### 7.1 什么时候选容器

| 选容器 | 选 systemd（默认） |
|---|---|
| 想把环境完全封装、便于换机重建 | 设备就是这一台，不打算迁移 |
| 不想在宿主装 Python 3.11/编译链 | 宿主环境干净，或已按 §5.6 装好 |
| 希望镜像可归档/可版本化 | 希望日志直接进 journald、排障路径最短 |

> 两条路径**互斥**：都要占 8000。切换前先 `sudo systemctl disable --now kineto-engine`。
> 两条路径**共用同一份模型**（`/srv/kineto/models`），不会重复占 2.6GB。
> ⚠️ 监听地址不同：**systemd 路径绑 `127.0.0.1:8000`**（宿主回环，天然不对 LAN 暴露）；
> **容器路径内部必须绑 `0.0.0.0:8000`**（容器的 127.0.0.1 对外不可达），安全性靠
> compose **不发布端口**（只有 `expose`）来保证 —— 未映射的端口在宿主/LAN 上都访问不到。

### 7.2 基础镜像选型（已确定：`ubuntu:24.04`）

| 候选 | 结论 | 理由 |
|---|---|---|
| **`ubuntu:24.04` + Intel GPU APT 源用户态** | ✅ **选定** | 与宿主同发行版（同 glibc / 同 Level Zero ABI）；XPU wheel 官方验证平台就是 24.04；Intel 官方 GPU 源只给 Ubuntu 发包；torch XPU 版本由我们完全掌控；镜像 ~4-5GB |
| `python:3.11-slim` (Debian bookworm) + 手装 compute-runtime .deb | ❌ | Intel 官方 APT 源**不支持 Debian**，只能从 GitHub Release 手抓 .deb `dpkg -i`，版本漂移不可复现 |
| `intel/ai-stacks/*`（oneAPI / pytorch-xpu 一体化） | ⚠️ 作为 Plan B | 体积 10GB+，且**已经钉死**自己的 torch/oneAPI 版本；再叠 `pip install -r requirements.txt` 容易触发解析器把 torch 换成 CPU 轮子或与镜像内 IPEX 冲突。需要它时直接换基底：`--build-arg BASE_IMAGE=…`（compose 里也可用 `.env` 的 `BASE_IMAGE=`） |

容器只需**用户态**（`intel-opencl-icd` / `intel-level-zero-gpu` / `level-zero` / `libze1`）；
内核态驱动（`i915`/`xe`）永远在宿主机，通过 `devices: ["/dev/dri"]` 透传。

### 7.3 构建与运行

```bash
# 在设备上（代码已按 §5.5 同步到 /opt/kineto）
cd /opt/kineto/deploy

# 密钥：优先用模板（compose 用 ${KINETO_API_KEY:?} 强校验，不设不让起）
cp .env.example .env && nano .env
#   KINETO_API_KEY=<与 §5.8 同一把>          # openssl rand -hex 32
#   KINETO_CORS_ORIGINS=https://kineto-web.zeabur.app
chmod 0600 .env                              # ★ 里面有密钥，别留 644

sudo mkdir -p /srv/kineto/{models,jobs}
docker compose config                        # 干跑一次，确认变量已注入
docker compose build                         # 首次 15-30 min（torch XPU 轮子 ~1.6GB）
docker compose up -d engine
docker compose logs -f engine

# 容器内验 XPU（对应 §4.3）
docker exec kineto-engine python -c "import torch;print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"

# 隧道：要么用 compose 里的 cloudflared（profiles: tunnel，ingress 改 http://engine:8000），
#       要么用宿主 systemd 版 —— 后者需给 engine 加 `ports: ["127.0.0.1:8000:8000"]`
#       或 `network_mode: host`（compose 文件里已写好注释）。两者都不做 → 宿主隧道 502。
docker compose --profile tunnel up -d
```

### 7.4 构建注意事项

- **构建上下文 = `../kineto-engine`**（不是仓库根），避免把 `kineto-web/node_modules`、`.venv`（1.5GB）拖进去。
- 忽略规则在 `deploy/Dockerfile.engine.dockerignore`（BuildKit 的 per-Dockerfile ignore）；
  老版 Docker 请 `cp deploy/Dockerfile.engine.dockerignore kineto-engine/.dockerignore`。
- **4D-Humans 优先用随代码同步过去的本地副本**（含 `torch.load(weights_only=False)` 补丁）；
  上下文里没它时 Dockerfile 会自动 `git clone` 上游 —— 但上游在 torch≥2.6 上加载 .ckpt 会报
  `UnpicklingError`，所以生产构建请确保副本存在。
- **不装 `.[all]` extra**（detectron2）：无 CUDA 工具链会编译失败，且引擎用 YOLOv8n 做检测，不需 ViTDet。
- Dockerfile 末尾有一道**自愈步骤**：若 `requirements.txt` 把 torch 换成了非 XPU 版本，自动
  `--force-reinstall --no-deps` 回装 XPU 轮子。
- **密钥不烘进镜像**：Dockerfile 里 `KINETO_API_KEY` / `KINETO_CORS_ORIGINS` 默认为空，
  实际值由 `deploy/.env` 经 compose 注入（`docker inspect` 能看到镜像 ENV，所以绝不写死）。
  `.env` 不要提交进版本库。

> **已接受残留风险（本轮有意延后，仅记录不改代码）**
>
> - **MJ9 — 4D-Humans 全局 monkey-patch `torch.load(weights_only=False)`**：`hmr2_weights_only.patch`
>   把 `torch.load` 全局改成 `weights_only=False`（并 `add_safe_globals([DictConfig,ListConfig])`），以绕过
>   torch≥2.6 加载 `.ckpt` 的 `UnpicklingError`。这**降低了反序列化安全边界**（理论上 pickle 可内嵌可执行代码）。
>   **接受理由**：① 这是当前**唯一验证跑通**的 4D-Humans 推理路径，改动它风险高于收益；② 权重来源已被
>   `deploy/transfer_models.sh` 的 **sha256 校验**锁定（只从可信 Mac 侧 rsync，不从公网拉）；③ 检测器权重
>   `yolov8n.pt` 为**预置**而非运行时下载。缓解后剩余风险可接受；若日后 4D-Humans 支持 `weights_only=True`
>   安全加载，应第一时间移除该 monkey-patch。
> - **MN5 — 补丁“双写”**：`hmr2_weights_only.patch` 同时存在于**权威副本** `deploy/patches/hmr2_weights_only.patch`
>   与 `deploy/Dockerfile.engine` 的**内联副本**（`RUN <<'PATCH' cat > /tmp/hmr2_weights_only.patch`，因构建上下文是
>   `kineto-engine/` 而非仓库根、拿不到 `deploy/patches/`）。两份**必须逐字节同步**，否则容器路径与裸机路径行为漂移。
>   **接受理由**：Dockerfile 末尾已有**构建期断言**（§6b：`hmr2.models` 源码须同时含 `weights_only` 与 `add_safe_globals`，
>   缺一即**拒绝出图**），能在构建时兜住漂移；故仅记录、不引入额外单源机制。改了 `deploy/patches/` 务必同步改 Dockerfile 内联副本（2 个 hunk）。

---

## 8. 验收门禁（`deploy/validate.sh`）

```bash
# 设备上（本地回环，用 JSON {video_path} 提交，最省带宽）
sudo KINETO_API_KEY=<key> bash /opt/kineto/deploy/validate.sh \
     --video /opt/kineto/kineto-engine/input_video.mp4

# Mac 上穿隧道验（multipart 上传，顺带验证公网链路）
bash deploy/validate.sh --base https://kineto-api.<YOUR_DOMAIN> --api-key <key> \
     --upload --video ~/Projects/Kineto/videos/input_video.mp4
```

| 门禁 | 断言 | 期望 | 失败时看 |
|---|---|---|---|
| **G1** | `GET /healthz` + `GET /health` | `/healthz` 公开存活；`/health` 需 `X-API-Key`，200 且返回体含 `extraction_mode` 字段 | §9 API-1 |
| **G1a** | `/health.device` | **Arc 上必须为 `xpu`；`device=cpu` 判 FAIL**（api.py 的 `_detect_device()` 已支持 xpu） | §1.1 / §9 XPU-1/3 |
| **G2** | `torch.xpu.is_available()` | `True` + `get_device_name(0)` 含 Arc/B60 | §9 XPU-1/2 |
| **G3** | `POST /jobs` | `202` + `job_id`（401=密钥错，413=隧道上传超限） | §9 API-2 |
| **G4** | 轮询 `/jobs/{id}` | 最终 `state == done`，**并计时打印全程耗时**（超 3600s 判 FAIL） | §9 JOB-1 |
| **G5** | `metadata.extraction_mode` | **必须等于 `4dhumans`**（合成回退 = 不得上线） | §9 MODEL-1 |
| **G6** | `metadata.pipeline.final_quality_score` | **≥ KINETO_QUALITY_THRESHOLD**（默认 0.6；注意在 `pipeline` 子层） | §9 MODEL-2 |
| G7 | `GET …/demo_output.mp4` | 200 且 > 10KB（引擎 `--skip-demo` 时为 WARN） | — |
| **G8** | `dmesg` / `journalctl -k` | 无 `oom-kill`；额外报 `NRestarts`（>0 说明曾崩溃） | §9 OOM-1 |
| **G9** | 重启持久化 | `is-enabled=enabled`（容器：`restart=unless-stopped`）+ 打印 reboot 复验清单 | §9 OPS-1 |
| **G10** | MoFang 回归 | `--mofang-mode coexist`：两项必须绿；`stripped`：确认已停 | §11 |
| G11 | Zeabur 前端可达性 | `--web-base`：站点 2xx/3xx + 同源代理 `/api/health`=200；仅当另传 `--web-origin`（仍直连）时附带 CORS 预检 | `DEPLOY_ZEABUR.md` |
| **G12** | 骨架 SSOT 一致性（G8-SSOT） | `skeleton_spec.py` ↔ `skeleton.ts` 关节名/父表/边/part_map 逐项一致（collar 13/14 父=9） | `deploy/check_ssot.py` |

**reboot 复验是硬性要求**（不是可选）：`sudo reboot` 后重跑一次 `validate.sh`，
重点看 `/dev/dri` 权限与两个服务是否自启。退出码 0 才算验收通过。

---

## 9. 排障表

| ID | 症状 | 根因 | 处置 |
|---|---|---|---|
| **XPU-1** | `torch.xpu.is_available() == False` | 用户不在 `render`/`video` 组；或驱动未装 | `ls -l /dev/dri`；`sudo usermod -aG render,video <user>` 后**重新登录**；再不行走 §5.3 |
| **XPU-2** | 同上，且 `ls /dev/dri` 无 `renderD*` | 内核驱动未加载（`i915`/`xe`） | `sudo dmesg \| grep -iE 'i915\|xe'`；内核太旧→装 HWE；Secure Boot 开着→查模块拒载 |
| **XPU-3** | `torch.__version__` 不带 `+xpu` | 装成了 PyPI 的 CPU 轮子（先装了 requirements） | `pip install --force-reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu` |
| **XPU-4** | `ModuleNotFoundError: triton` / SYCL 运行库缺失 | XPU 轮子依赖 `triton-xpu`/`intel-sycl-rt` 未拉到 | 安装时**只用** `--index-url`（不要加 `--no-deps`），该索引托管了全部运行时依赖 |
| **MODEL-1** | job FAILED，error 含 `fell back to non-4dhumans mode` | 权重/缓存没就位 → 引擎用了内置简化模型 | 跑 §5.7 的自检（四个 True）；确认 `HOME=/srv/kineto` 与软链 |
| **MODEL-2** | `final_quality_score` < 0.6 | 视频模糊/人体太小/倒立姿态；或 YOLO 检测框偏 | 看 `journalctl -u kineto-engine` 里的 `[RotHack]` 警告；换清晰、人体占比大的短视频复测 |
| **MODEL-3** | `UnpicklingError` / `weights_only` | 用了上游 4D-Humans（无补丁） | 用 §5.5 rsync 过去的本仓库副本；若设备端是 `git clone` 上游，则 `cd 4D-Humans && git apply -p1 ../../deploy/patches/hmr2_weights_only.patch`（**注意 `../../`**：deploy 在仓库根、不在 4D-Humans 内；幂等守卫与阻断式断言见 `SETUP_GUIDE.md`） |
| **API-1** | `/health` 连不上或 401 | 服务未启/端口不对/隧道未通；或 `/health` 现需鉴权而未带 `X-API-Key`（401） | 先 `curl …/healthz`（公开存活）区分“没起”与“鉴权”；`systemctl status kineto-engine`；`ss -tlnp \| grep 8000`；`journalctl -u kineto-engine -n 100` |
| **API-2** | `POST /jobs` 401 | `KINETO_API_KEY` 不一致 | `sudo cat /etc/kineto/kineto-engine.env` 与 **Zeabur 控制台的服务端变量 `KINETO_API_KEY`**（非 `.env.local`、非 `NEXT_PUBLIC_*`）逐字比对；见 `DEPLOY_ZEABUR.md` |
| **API-3** | `POST /jobs` 413 | Cloudflare 100MB 上传上限 | 改用 `{"video_path": …}`：先 rsync 到 `/srv/kineto/inbox/` 再提交 |
| **API-4** | 服务起不来，`systemctl status` 显示 `status=1/FAILURE`且无 uvicorn 日志 | unit 里的 **ExecStartPre 守卫**拦下了：`/etc/kineto/kineto-engine.env` 不存在，或 `KINETO_API_KEY` 缺失/短于 16 位/仍是 `__SET_ME__`，或 `KINETO_CORS_ORIGINS` 不以 `http(s)://` 开头 | 按 §5.8 写真实密钥与 CORS；`sudo systemctl reset-failed kineto-engine && sudo systemctl restart kineto-engine`。（守卫故意不让服务带着占位密钥裸奔；本机调试可临时注释掉那两行 ExecStartPre） |
| **API-5** | `/health`、`/jobs` 等返回 **503 `auth not configured`**（但 `/healthz` 仍 200） | api.py **fail-closed**：`KINETO_API_KEY` 未设**且**未设 `KINETO_ALLOW_NO_AUTH=1` | 生产：按 §5.8① 写真实 `KINETO_API_KEY` 到 `/etc/kineto/kineto-engine.env` 后 `systemctl restart kineto-engine`；本机调试：临时 `KINETO_ALLOW_NO_AUTH=1`（**绝不可用于生产**）。公开存活探针 `/healthz` 不受影响 |
| **JOB-1** | 长时间卡在 `running` | 串行队列（并发=1）前面还有 job；或首次加载 2.5GB 权重 | `curl /health` 看 `queue_depth`；首次推理包含模型加载，耐心等；单 job 超 2h 会被 api.py 强制失败 |
| **OOM-1** | `dmesg` 有 `oom-kill`，或 `NRestarts > 0` | `MemoryMax=12G` 被撞；或 MoFang 残留服务占内存 | `systemctl show kineto-engine -p MemoryMax -p MemoryCurrent`；调高 unit 的 `MemoryMax`（需 RAM 充裕）后 `daemon-reload && restart` |
| **OPS-1** | 重启后服务没起 | 忘了 `systemctl enable` | `sudo systemctl enable kineto-engine cloudflared` |
| **OPS-2** | `Permission denied` 写 `/srv/kineto/jobs` | 属主不对 | `sudo chown -R kineto:kineto /srv/kineto` |
| **OPS-3** | pyrender 报 OpenGL/EGL 错 | 未走 OSMesa | 确认 `PYOPENGL_PLATFORM=osmesa` 与 `libosmesa6` 已装 |
| **OPS-4** | 前端报 CORS 错 | 新代理架构下浏览器**不直连引擎**，通常不应出现 CORS 错；若出现说明前端走了旧的直连路径 | 优先改回同源代理 `/api/*`（见 `DEPLOY_ZEABUR.md`）；若确实保留直连，才需把 Zeabur 精确 origin 回填 `KINETO_CORS_ORIGINS` 后 `sudo systemctl restart kineto-engine` |
| **TUN-1** | 公网 502 / 404 | engine 未监听回环（容器路径未发布端口/未用 host 网络），或 hostname 拼错命中兜底规则 | 见 `deploy/cloudflared/README.md` §8 |
| **DOCKER-1** | 容器恒 `unhealthy`、cloudflared 不启动（`depends_on: service_healthy` 永不满足）→ 隧道/公网入口 100% 失效 | HEALTHCHECK 误打**需鉴权**的 `/health`：compose 强制注入 `KINETO_API_KEY`，无 `X-API-Key` 的探测恒 401 → 容器永久 unhealthy | HEALTHCHECK 必须打**公开** `/healthz`（本仓库 `Dockerfile.engine`/`docker-compose.yml` 已修正）；排查 `docker inspect -f '{{json .State.Health}}' kineto-engine` 看探测输出/退出码；`validate.sh` G9c 对此硬断言 |

---

## 10. 回滚手册（per-step rollback）

| 步骤 | 做了什么 | 如何撑销 | 可逆 |
|---|---|---|---|
| §4 Phase 1 | 建了 `~/kineto-probe` | `rm -rf ~/kineto-probe` | ✅ 完全 |
| §5.1 | 写了存证文件 | 不需撑销（只是快照） | — |
| §5.2 停用 MoFang | `systemctl disable --now <units>` | `sudo systemctl enable --now <units>`；对照存证 `diff` 逐个核 | ✅ |
| §5.2（若曾 mask） | `systemctl mask` | `sudo systemctl unmask <unit>` 后 `enable --now` | ✅ |
| §5.3 驱动 | 新增 Intel APT 源 + 用户态包 | 一般**不需回滚**（向后兼容）。确实要撑：`sudo apt remove intel-opencl-icd intel-level-zero-gpu level-zero libze1 libze-intel-gpu1 && sudo rm /etc/apt/sources.list.d/intel-gpu-noble.list && sudo apt update`。⚠️ 若 MoFang 原本依赖这些包，移除前先 `apt-cache rdepends --installed intel-opencl-icd` 查反向依赖 | ⚠️ 谨慎 |
| §5.4 账号/目录 | 新增 `kineto` 用户、`/opt/kineto`、`/srv/kineto` | `sudo systemctl disable --now kineto-engine; sudo rm /etc/systemd/system/kineto-engine.service; sudo systemctl daemon-reload; sudo userdel kineto; sudo rm -rf /opt/kineto /srv/kineto /etc/kineto` | ✅ |
| §5.5–§5.7 代码/模型 | 写入上述两个目录 | 同上（删目录）；模型可重传（transfer_models.sh 幂等） | ✅ |
| §5.8 systemd | 新增 unit + `/etc/kineto` | `sudo systemctl disable --now kineto-engine && sudo rm -f /etc/systemd/system/kineto-engine.service && sudo systemctl daemon-reload` | ✅ |
| §5.9 Tunnel | 新增 cloudflared + DNS CNAME | `sudo systemctl disable --now cloudflared; sudo cloudflared tunnel delete kineto-engine`（同时删 DNS 记录）；详见 `deploy/cloudflared/README.md` §8 | ✅ |
| §7 容器 | 镜像 + 容器 | `docker compose down --rmi local --volumes=false`（**不加 `--volumes`**，否则可能误删挂载数据） | ✅ |
| §6 重装 | 抹盘 | **不可逆**，只能用 §6.3-0 的全盘镜像回写 | ❌ |

**一键全量回滚到“只有 MoFang”的初始状态**（按顺序执行，每步都先看后做）：

```bash
sudo systemctl disable --now kineto-engine cloudflared 2>/dev/null
sudo rm -f /etc/systemd/system/kineto-engine.service /etc/systemd/system/cloudflared.service
sudo systemctl daemon-reload
sudo cloudflared tunnel delete kineto-engine 2>/dev/null || true
sudo rm -rf /opt/kineto /srv/kineto /etc/kineto ~/kineto-probe
sudo userdel kineto 2>/dev/null || true
sudo systemctl enable --now <§5.2 里停用过的每个 MoFang 单元>
curl -sk -o /dev/null -w 'MoFang.html → %{http_code}\n' https://192.168.1.107/next/MoFang.html   # 应回 200
```

---

## 11. MoFang 回归检查清单（共存路径 OPTION ii 专用）

> 目标：证明“Kineto 跑起来了，MoFang 一点没坏”。在 Phase 1 结束后、以及 Phase 2 的每一步之后都可以跑。

| # | 检查项 | 命令 | 期望 |
|---|---|---|---|
| R1 | 配对 UI 仍可访问 | `curl -sk -o /dev/null -w '%{http_code}\n' https://192.168.1.107/next/MoFang.html` | **200** |
| R2 | 业务 API bootstrap | `curl -sk https://192.168.1.107/bridge/v2/bootstrap` | **200**，响应体中 `agentHealth` 的 **`openclaw`** 与 **`rag`** 仍为 ok/healthy |
| R3 | MoFang 服务仍在跑 | `systemctl list-units --type=service --state=running \| grep -iE 'mofang\|bridge\|ai'` | 与 §5.1 存证**逐项一致** |
| R4 | nginx 未被改动 | `sudo nginx -T \| md5sum` 对比存证 | 哈希一致（共存阶段不应改 nginx） |
| R5 | GPU 共享无冲突 | `sudo dmesg \| grep -iE 'i915\|xe\|reset\|hang' \| tail` | 无 reset/hang/gpu hang 记录 |
| R6 | 端口无争用 | `ss -tlnp \| grep -E ':(80\|443\|8000)\b'` | 80/443 仍属 nginx；8000 属 uvicorn |
| R7 | 内存未被吃干 | `free -h`；`systemctl show kineto-engine -p MemoryCurrent` | Kineto 峰值不超 `MemoryMax=12G`，且 MoFang 未被 OOM |

一键版：`bash deploy/validate.sh --mofang-mode coexist`（G10 会把 R1/R2 做成硬门禁）。

> R2 的字段结构因固件版本而异（可能是 `agentHealth.openclaw.status`、也可能是 `agents[]` 数组）。
> `validate.sh` 只做**存在性**匹配（`grep -qiE 'openclaw'` / `'rag'`）并把完整响应体打印出来供你人工核对语义，
> 不会因为字段名差异而误判 FAIL。**把实际响应体回传云端，我们可以把断言改成精确版。**

---

## 12. OPEN QUESTIONS — 必须由你在设备上确认（云端无法自行得出）

| # | 问题 | 为什么重要 | 怎么查 |
|---|---|---|---|
| **Q1** | **确切 GPU 型号与 PCI ID**：是单芯 Arc Pro B60 24GB，还是双芯版本？`lspci -nn` 原始行是什么？ | 影响显存预算、是否需要 `ZE_AFFINITY_MASK`/`ONEAPI_DEVICE_SELECTOR` 锁定设备，以及性能预期基线 | `lspci -nn \| grep -iE 'vga\|display\|3d'`；`sudo lshw -C display` |
| **Q2** | **Ubuntu 到底是 22.04 还是 24.04？内核版本？** | XPU wheel 官方只验证了 24.04/26.04；22.04 要么走 IPEX 路径，要么先升 HWE/升级系统（有风险） | `lsb_release -a`；`uname -r` |
| **Q3** | **保留还是剥离 MoFang？** 具体哪些单元可以停？哪些（OTA/看门狗/风扇/授权）**绝对不能停**？ | 直接决定走 OPTION (ii) 还是 (i)；停错一个可能让机器失联或过热 | `systemctl list-unit-files --type=service \| grep -iE 'mofang\|bridge\|claw\|rag'`；`systemctl cat <unit>` |
| **Q4** | **是否需要离设备 CUDA 兜底？**（另一台 NVIDIA 机器 / 云端 GPU 作为 fallback） | 影响前端是否要做双后端路由、以及是否要在 api.py 外层加调度（两者都超出本任务边界，需单独开任务） | 业务决策 |
| **Q5** | Secure Boot 是否开启？BIOS 是否有密码？ | 影响驱动安装与 §6 重装可行性 | `mokutil --sb-state` |
| **Q6** | 磁盘布局：`/` 与 `/srv` 是否同一分区？各自可用空间？ | 模型 2.6G + venv 8G + jobs 增长；分区不当会把系统盘写满 | `df -hT`；`lsblk` |
| **Q7** | 出网能力：能否直连 `download.pytorch.org` / `pypi.org` / `repositories.intel.com` / `*.cloudflare.com`？是否需代理或国内镜像？ | torch XPU 轮子 ~1.6GB，不能下载就部署不了；需提前准备离线 wheel | `discover_device.sh` §9 的连通性输出 |
| **Q8** | MoFang 是否也在用这块 GPU？它的推理服务会不会与 Kineto 争显存？ | 24GB 看似充裕，但 HMR2 ViT-H 峰值可到 20GB+；共存时可能 OOM | 共存验证时跑一次 MoFang 推理，同时 `watch -n1 'sudo dmesg \| tail'` |
| **Q9** | Cloudflare 域名是什么？你是否有该 Zone 的编辑权（能建 CNAME）？ | 没有域名/权限就只能用临时隧道，URL 每次重启都变，不能上生产 | Cloudflare 控制台 |
| **Q10** | Zeabur 前端的**确切 origin** 是什么？ | 写进 `KINETO_CORS_ORIGINS`，错一个字符浏览器就拦 | Zeabur 服务详情页 |
| **Q11** | 8000 端口是否空闲？nginx 是否已占用或反代了该端口？ | 被占就得改 unit 与 tunnel config（两处必须同步） | `ss -tlnp \| grep 8000` |
| **Q12** | 设备上是否已有旧版 Kineto 部署残留？ | 幂等部署要求先识别旧物（旧 venv/旧 unit/旧模型） | `discover_device.sh` §12 |
| ~~**Q13**~~ | **已关闭**：`kineto-engine/api.py` 的 `_detect_device()` **已支持 xpu**（CUDA>XPU>MPS>CPU） | Arc 上 `/health` 会如实报 `device=xpu`；若报 `cpu` 是真故障（组权限/驱动/CPU 轮子），`validate.sh` G1a 硬判 FAIL | 无需再查；以 §1.1 为准 |
| **Q14** | （跨 Agent 依赖）`requirements.txt` 末尾的 IPEX 占位段何时锁版本？ | 现在靠本手册人工指定 `2.7.10+xpu`；锁版后才能保证可复现构建 | 引擎侧 Owner |

---

## 13. `deploy/` 文件清单与命令速查

```
deploy/
├── discover_device.sh              Phase 0 只读体检 + 决策门禁 PASS/FAIL（不改状态）
├── strip_mofang.sh                 §5.2 剥离 MoFang 上层服务（禁停硬拦截 + --dry-run + undo + --rollback）
├── kineto-engine.service           systemd unit（主路径；含 render/video 组、HOME、MemoryMax 注释）
├── Dockerfile.engine               容器备选路径（ubuntu:24.04 + Intel GPU 用户态 + torch XPU + 构建期断言）
├── Dockerfile.engine.dockerignore  BuildKit per-Dockerfile 忽略规则（排除 .venv/产物/权重）
├── docker-compose.yml              engine(devices:/dev/dri, 不发布端口) + 可选 cloudflared(profile)
├── .env.example                    compose 变量模板（API_KEY / CORS / STRICT / QUALITY_GATE / 性能 env）
├── transfer_models.sh              Mac → 设备 模型 rsync + sha256 校验 + 可选 --wire 布线
├── validate.sh                     G1..G12 端到端验收 + SSOT一致性 + 计时 + OOM + 重启持久化 + MoFang回归
├── check_ssot.py                   G12/G8-SSOT 骨架 SSOT↔前端镜像一致性闸门（validate.sh 调用）
├── patches/
│   └── hmr2_weights_only.patch     4D-Humans torch>=2.6 weights_only=False 补丁（git clone 后必须 git apply）
└── cloudflared/
    ├── config.yml                  kineto-api.<YOUR_DOMAIN> → http://127.0.0.1:8000 + 404 兜底
    ├── cloudflared.service         隧道 systemd unit（--no-autoupdate，专用非特权账号）
    └── README.md                   安装/login/create/route dns/CNAME/验证/100MB 上限/排障
DEPLOY_MOFANG.md                    本文件（设备侧）
DEPLOY_ZEABUR.md                    前端上云指南（仓库根目录；Zeabur 服务端代理架构）
```

所有 shell 脚本均已通过 `bash -n` 语法校验并加了可执行位；均支持 `-h/--help`，
未知参数退出码 2（避免误用）。

### 一页速查（按顺序跑完就是完整部署）

```bash
# ---- Mac ----
scp -P 22 deploy/discover_device.sh <user>@192.168.1.107:/tmp/
ssh <user>@192.168.1.107 'bash /tmp/discover_device.sh | tee ~/discover.log'      # Phase 0

# ---- 设备 ----
#   Phase 1: §4.1 → §4.6（共存验证，零风险）
#   Phase 2: §5.1 存证 → §5.2 停 MoFang → §5.3 驱动 → §5.4 账号目录

# ---- Mac ----
rsync -az --exclude '.venv/' --exclude '__pycache__/' --exclude 'output*/' \
      --exclude '*.mp4' --exclude '.git/' \
      ~/Projects/Kineto/kineto-engine/ <user>@192.168.1.107:/opt/kineto/kineto-engine/   # §5.5
rsync -az ~/Projects/Kineto/deploy/ <user>@192.168.1.107:/opt/kineto/deploy/
bash deploy/transfer_models.sh --device <user>@192.168.1.107 --wire                       # §5.7

# ---- 设备 ----
#   §5.6 venv + torch-XPU → requirements → 4D-Humans -e .
#   §5.8 /etc/kineto/kineto-engine.env（KINETO_API_KEY / KINETO_CORS_ORIGINS；unit 已内置 STRICT=1 + QUALITY_GATE=warn）+ systemd enable --now
#   §5.9 cloudflared（公网入口）
#   ★ 验收务必带 --web-base，否则 G11（Zeabur 前端可达性）恒 SKIP：
sudo bash /opt/kineto/deploy/validate.sh --video /opt/kineto/kineto-engine/input_video.mp4 \
     --web-base https://kineto.<你的前端域名>                                   # §8（G1..G12）

# ---- Zeabur 前端（与设备侧并行；完整步骤见 DEPLOY_ZEABUR.md）----
#   §1 Root Directory = Projects/Kineto/kineto-web → §2 部署 → §3 配**服务端**变量：
#      ENGINE_API_BASE=https://kineto-api.<YOUR_DOMAIN>、KINETO_API_KEY（与设备侧同一把，**无** NEXT_PUBLIC_ 前缀）
#   §4 绑定域名 → §6 验证：浏览器 ?job=<id> 来源徽标须为 LIVE API（非 FIXTURE）

# 重启持久化复验：reboot 会断掉当前 SSH，**不能**用 `sudo reboot && ...`
sudo reboot
# —— 等 60-120s 后重新 ssh 上去，再跑一次（同样带 --web-base）：
sudo bash /opt/kineto/deploy/validate.sh --video /opt/kineto/kineto-engine/input_video.mp4 \
     --web-base https://kineto.<你的前端域名>
systemctl is-enabled kineto-engine && systemctl is-active kineto-engine   # 两个都要 enabled/active
```

### 不可触碰的文件（另有 Owner）

- `kineto-engine/kineto_core.py` —— 引擎核心（XPU 适配由引擎 Owner 推进）
- `kineto-engine/requirements.txt` —— 依赖清单（IPEX 锁版由引擎 Owner 推进）
- `kineto-web/**` —— 前端（部署在 Zeabur）

本手册修正了与之矛盾的两份旧文档（把 Intel AI Box 误写成 NVIDIA/CUDA 24GB、把 Intel Arc 写成 8GB）：
`kineto-engine/DEVICE_SELECTION.md`、`kineto-engine/SETUP_GUIDE.md`。与本文冲突时，**以本文为准**。


