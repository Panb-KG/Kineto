# Kineto 项目规则

> 适用于本仓库内所有 AI 辅助开发与人工编码。违反「硬约束」条目的改动一律拒绝。
> 进展沉淀与踩坑记录维护在 `DEVELOPMENT_MEMORY.md`，与本规则互补。

---

## 1. 项目定位与架构（不可违背）

**Kineto = 云边协同的运动康复 AI 平台**：视频 → 3D 姿态（SMPL-X）→ 标准化动作处方。

| 模块 | 位置 | 技术栈 | 职责 |
|---|---|---|---|
| `kineto-engine/` | 设备侧 AI BOX（Intel Arc Pro B60, Ubuntu 24.04） | Python 3.12 + FastAPI + 4DHumans | Video-to-Pose 推理，算力本地闭环 |
| `kineto-web/` | Zeabur 云端 | Next.js 14 + React Three Fiber + TS | 处方交付、3D/2D 互动展示 |
| `deploy/` | 部署工具包 | bash + systemd + Docker | 设备体检、剥离、同步、验收门禁 |

**请求链路（唯一正确架构）**：浏览器 → 同源 `/api/*` → Next.js 服务端代理（`kineto-web/app/api/[...path]/route.ts`，注入 `X-API-Key`，视频 ReadableStream 流式中转不落盘）→ 引擎。
**禁止**浏览器直连引擎（引擎需 `X-API-Key`，浏览器不持有）。

**ADR-001 已定案**：前端留 Zeabur、引擎留 AI BOX。重启评估仅限三种触发：私有化一体机交付 / 纯内网环境 / Zeabur 不可用。

---

## 2. 引擎硬约束（kineto-engine）

1. **单 worker 串行推理**：`uvicorn api:app --workers 1`。边缘设备绝不允许并行推理（OOM）。`kineto_core.py` 经 subprocess 调用，`cwd` 必须为 ENGINE_DIR（内部使用相对路径）。
2. **Job 状态字段是 `state`**（queued/running/done/failed），不是 `status`；另有 progress / quality_score / extraction_mode。
3. **JSON 模式 `video_path` 必须位于 `KINETO_INBOX` 内**（设备为 `/srv/kineto/inbox`），其他路径一律拒绝。
4. **上线门禁**：`extraction_mode != '4dhumans'` 判 failed，禁止合成/假姿态数据出引擎。
5. **骨架常量 SSOT**：所有关节名、kintree、骨长界、rest 模板只从 `skeleton_spec.py` import，禁止任何文件硬编码（含 `kineto_core.py` / `pose_audit.py` / `skeleton_validator.py`）。索引一律 SMPL canonical 24 关节序；HMR2 的 OpenPose Body-25 序不可混用。前端 `kineto-web/lib/skeleton.ts` 必须镜像 SSOT（G12 闸门校验）。
6. **产物 metadata 判别位**：`joint_order=smpl-canonical` + `schema_version=2`（schema 只做 additive 演进，向后兼容）。
7. **4D-Humans 旋转三步流程（不得省略）**：检测横卧（bbox 宽≥高）→ 旋转帧 90° 推理 → **joints_3d/cam_t/global_orient 逆旋转回原始帧坐标系**。输出必须与原视频姿态一致。
8. **env 数值读取统一走 `_int_env` / `_float_env`**（api.py），禁止 `int(os.environ.get(K, D) or 0)` 式空串吞噬写法。

---

## 3. 前端硬约束（kineto-web）

1. **禁止前端 spine 旋转校正**：引擎已将 3D 坐标复原到原始帧坐标系。原视频里人是躺着的，3D 就显示躺着。任何「自动直立」逻辑都会造成多层旋转叠加错位（此坑已反复出现 ≥3 次）。
2. **相机系→世界系转换必须保留**：引擎输出 Y 下、Z 向前；Three.js 是 Y 上、Z 向后。`computeFraming`/`computeMeshFraming` 中 `y → -y`、`z → -z`，不得删除。
3. **密钥永不进浏览器包**：`ENGINE_API_BASE` / `KINETO_API_KEY` 是服务端变量（无前缀）；任何密钥禁止 `NEXT_PUBLIC_` 前缀。`NEXT_PUBLIC_*` 构建期内联，改动需重新部署。
4. **`NEXT_PUBLIC_API_BASE` 留空**，必须走同源代理。
5. **`ENGINE_API_BASE` 运行时读取**（`force-dynamic`），按部署侧选可达地址：
   - Mac 开发（tailnet）：`http://aibox:8000`
   - 客户局域网：`http://juxin.lan:8000`
   - Zeabur 生产：`https://aibox.tail6791a3.ts.net`（Tailscale Funnel；**Zeabur 不在 tailnet/客户内网**，aibox:8000 与 juxin.lan:8000 从 Zeabur 均不可达）
   - 前端生产域名（2026-09-09 定案）：`https://kineto.标智云.中国`（punycode `kineto.xn--9kqt69c97a.xn--fiqs8s`；引擎 CORS 白名单须用 punycode，浏览器 Origin 头为 ASCII 序列化）
6. **引擎不可达时降级不误报**：回退内置 fixture `public/fixtures/pose_data.sample.json`；轮询超时保留 job_id 提示继续等待，而非报「未连接」。
7. **`validatePoseData` 全帧校验**（非只首帧）；`joint_order` 非 canonical、`degraded` 时必须有横幅告警。

---

## 4. 数据结构规范（科研级，不可降级）

- 时间戳锚点：每个数据点必须含 `timestamp_ms`（相对毫秒数），不可只依赖 `frame_index`（为测力台/EMG 1000Hz 对齐预留）。
- `metadata` 必含 `video_fps` / `total_frames` / `resolution` / `model_version`。
- 全帧原始数据保留原则：完整保留未平滑滤波的逐帧参数与置信度，禁止为省存储降维。
- 算法解耦：推理器封装为可替换 class（工厂模式），替换模型不改 API 与时间戳逻辑。

---

## 5. 安全铁律

1. **密钥不入库**：`deploy/.env`、`/etc/kineto/kineto-engine.env` 的 `KINETO_API_KEY` 绝不进 git、不进文档。`.gitignore` 已覆盖 `*.env`（`!*.env.example` 例外）。
2. **SMPL 授权模型文件不入库**：`kineto-engine/4D-Humans/`、`checkpoints/`、`yolov8n.pt` 已 gitignore。设备端**不能** git clone 上游获取（本仓库版含 `weights_only` 补丁，上游原版会 UnpicklingError）——同步用 `DEPLOY_MOFANG.md §5.5` 的 rsync 命令。
3. 引擎仅暴露 8000 端口；受保护端点全部要求 `X-API-Key`（`/healthz` 例外，公开存活探针）。
4. 大文件（>100MB 或公网链路）先 `scp` 到 `/srv/kineto/inbox/`，再 POST `video_path`。

---

## 6. 设备与部署硬约束

| 约束 | 内容 |
|---|---|
| 系统 | Ubuntu 24.04 (Noble)，**系统 Python 3.12**，禁用 deadsnakes PPA（曾拖垮 network-manager 断网） |
| torch-XPU | 走 SJTU 镜像 `https://mirror.sjtu.edu.cn/pytorch-wheels/xpu/`（官方源 KB/s） |
| chumpy 0.70 | 必须源码补丁 + `--no-deps --no-build-isolation`（补丁在 `deploy/patches/`） |
| PyOpenGL | 必须 3.1.7（pyrender 钉的 3.1.0 缺 `OSMesaCreateContextAttribs`） |
| omegaconf | 必须显式安装（HMR2 .ckpt 反序列化需要） |
| WiFi | 省电必须关（`wifi.powersave 2`），否则高吞吐传输整机掉线 |
| 设备网络 | GitHub TCP 常被封锁；cloudflared 用 `pkg.cloudflare.com`，Tailscale 用官方 install.sh |
| 部署顺序 | MoFang 剥离必须先 `strip_mofang.sh --dry-run` 核对 5 unit + openclaw 容器，再正式执行；回滚 `--rollback` |
| 设备准入 | `discover_device.sh` 四门禁：RAM≥16G / disk≥25G / Intel GPU / Docker |
| 验收 | `KINETO_API_KEY=$(grep KINETO_API_KEY /etc/kineto/kineto-engine.env \| cut -d= -f2) bash deploy/validate.sh --mofang-mode stripped`，G1-G13 全过才算交付 |
| 设备布局 | 代码 `/opt/kineto/kineto-engine`，venv `/opt/kineto/venv`，模型 `/srv/kineto/models`，任务 `/srv/kineto/jobs`，密钥 `/etc/kineto/kineto-engine.env` |
| SSH | `ssh juxin@aibox`（tailnet）或 `ssh juxin@juxin.lan`（局域网） |

---

## 7. 代码规范

1. **语言**：注释、提交信息、文档一律中文（与现有代码一致）；标识符英文。
2. **Python**：`from __future__ import annotations`；类型标注；错误处理完善但不过度——只校验系统边界（HTTP 入参、外部文件），信任内部调用。
3. **TypeScript**：严格模式（`tsc --noEmit` 必须过）；共享类型集中在 `kineto-web/lib/types.ts`。
4. **简洁优先**：不为单一用途建抽象；不加用不到的配置项；bug 修复不顺手重构。
5. **提交信息**：`feat:` / `fix:` 前缀 + 中文描述，重大修复标注 issue 号（如 `fix(#66,#68)`）。

---

## 8. 工作流规范

1. **改动后验证**：
   - 前端：`cd kineto-web && npm run typecheck && npm run lint`
   - 骨架一致性（改了 skeleton_spec.py 或 skeleton.ts）：`python3 deploy/check_ssot.py`（静态、只读、不需引擎在线）
   - 引擎/部署改动：设备侧跑 `deploy/validate.sh` 门禁。
2. **DEVELOPMENT_MEMORY.md 持续维护**（用户明确要求）：每个里程碑后，进展追加到 §2 时间线、新坑追加到 §4/§5、核心信息变更同步 §3。核心信息速查（§3）是访问地址/路径/端点的唯一权威，改部署布局必须同步。
3. **坐标/旋转类改动**（引擎坐标输出或前端渲染）：必须先读本规则 §2.7、§3.1、§3.2，并在真机视频上 E2E 验证姿态方向，不允许只跑单测。
4. **模糊需求主动澄清**，拒绝猜测；特别是涉及 API 字段名、路径、部署模式时。

---

## 9. 禁止事项速查（红线）

- ❌ 浏览器直连引擎 / 密钥进 `NEXT_PUBLIC_*`
- ❌ 前端对 3D 姿态做任何旋转校正
- ❌ 并行推理 / uvicorn 多 worker
- ❌ 骨架常量硬编码（绕过 `skeleton_spec.py`）
- ❌ 合成姿态数据当真数据出引擎（extraction_mode 必须 4dhumans）
- ❌ 把 `.env` / API Key / SMPL 模型文件提交进 git
- ❌ 设备上用 deadsnakes / 从 GitHub 下载安装包
- ❌ 引擎 job 状态字段写成 `status`
- ❌ 删除相机系→世界系转换（`y→-y, z→-z`）
