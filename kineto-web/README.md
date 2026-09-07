# Kineto Web · 运动康复 3D 姿态可视化前端

Kineto 云边协同架构中的「业务交互大脑」，部署于 Zeabur（Serverless 2C 4G）。
基于 **Next.js (App Router) + React Three Fiber** 构建，读取 kineto-engine 输出的
`pose_data.json`，以医疗极简风格渲染可交互的 SMPL 24 关节三维骨架。

> **架构（修复后）**：浏览器只与【同源】的 `/api/*`（Next.js Route Handler）通信；
> 服务端代理把请求转发到引擎基址（服务端 env `ENGINE_API_BASE`），并在【服务端】
> 注入 `X-API-Key`（服务端 env `KINETO_API_KEY`）。浏览器**不直连引擎、不持有任何密钥**。

> 本目录为 Phase 2（task #3）从零搭建，完全独立于 `kineto-engine/`。

---

## 技术栈与依赖版本

| 依赖 | 版本 | 说明 |
| --- | --- | --- |
| `next` | 14.2.35 | App Router（已升级到已修补安全版本）|
| `react` / `react-dom` | 18.3.1 | 与 R3F v8 兼容 |
| `three` | 0.169.0 | WebGL 渲染内核 |
| `@react-three/fiber` | 8.17.10 | React 的 three 渲染器 |
| `@react-three/drei` | 9.114.3 | `OrbitControls` / `Grid` 等辅助组件 |
| `@tweenjs/tween.js` | 21.0.0 | **选定动画库**（见下方理由） |
| `typescript` | 5.6.3 | 严格模式 |

**为何选 tween.js（而非 gsap / react-spring）**：`PROJECT_PLAN.md` 任务 2.1 与项目既有
技术栈均明确要求「利用 Tween.js 在关键姿态间平滑循环播放」。为与规划保持一致、减少
依赖分叉，本项目的播放头（playhead）由 tween.js 的线性 Tween 驱动；关节位置则在相邻
关键帧之间做插值（见 `lib/timeline.ts`）。

---

## 目录结构

```
kineto-web/
├── app/
│   ├── layout.tsx            # 根布局 + 字体（Instrument Serif / IBM Plex Sans / Plex Mono）
│   ├── page.tsx              # 首页外壳：上传面板 + 页头 + 关键帧区 + 3D 视图区 + 页脚
│   ├── api/[...path]/route.ts # 服务端代理：同源 /api/* → 引擎，注入 X-API-Key（密钥不出服务端）
│   └── globals.css           # "Clinical Editorial" 设计系统
├── components/
│   ├── UploadPanel.tsx       # 极简上传闭环：选 .mp4 → uploadJob → 轮询 → 切到 ?job=<id>
│   ├── ViewerStage.tsx       # 客户端主舞台：数据加载 + 降级横幅 + 时间轴装配 + 组合
│   ├── SkeletonViewer.tsx    # R3F <Canvas>：关节球 + 骨骼圆柱 + 网格/坐标轴/光照
│   ├── KeyframeCards.tsx     # 上半部分 4 张 2D 关键帧指导图（占位）
│   ├── TimelineControls.tsx  # 播放/暂停 + 进度条 scrubber + 时间/帧读数
│   └── MetadataPanel.tsx     # 元数据面板 + 假数据告警徽章
├── lib/
│   ├── types.ts              # 与 pose_data.json / jobs 契约严格对应的 TS 接口
│   ├── skeleton.ts           # SMPL 关节名/父表/骨骼对/部位配色（镜像 SSOT skeleton_spec.py）
│   ├── timeline.ts           # PoseTimeline（tween.js）+ 关键帧插值 + 取景归一化
│   ├── useTimeline.ts        # React hook：播放/暂停/拖拽 + 时间订阅
│   ├── api.ts                # 浏览器侧 API 客户端（同源 /api，无密钥；上传字段=video，轮询读 state）
│   └── poseData.ts           # 数据加载：API 优先，失败优雅回退 fixture
├── public/
│   └── fixtures/
│       └── pose_data.sample.json   # 复制自 output_test/pose_data.json（canonical，离线开发用）
├── .env.example
├── .nvmrc                    # Node 版本（20）
├── next.config.js
├── tsconfig.json
└── package.json
```

---

## 运行

> 要求 **Node ≥ 18.17**（仓库附 `.nvmrc`，可用 `nvm use` 切到 Node 20）。

```bash
cd kineto-web
npm install

# 开发（完全离线，使用内置 fixture，无需后端）
npm run dev          # http://localhost:3000

# 生产构建 / 启动
npm run build
npm start

# 仅类型检查
npm run typecheck
```

> 开发默认离线可用：未配置后端时自动加载 `public/fixtures/pose_data.sample.json`。

---

## 数据模式：Fixture vs API

加载逻辑见 `lib/poseData.ts` → `loadPoseData(jobId?)`：

1. **API 模式**：当提供了 `jobId` 时，浏览器请求【同源】的
   `GET /api/jobs/{id}/pose_data.json`（经 `app/api/[...path]/route.ts` 代理转发到引擎，
   代理在服务端注入 `X-API-Key`；浏览器不携带任何密钥）。
   - 通过 URL 查询参数传入任务 ID：`http://localhost:3000/?job=<job_id>`；
   - 上传成功后 `UploadPanel` 会自动切到该地址。
2. **优雅降级**：代理返回 503（引擎未配置）/ 502（不可达）或**真正的网络失败**（fetch
   TypeError）时，自动回退到内置 fixture，舞台顶部显示「当前为样例数据」横幅，元数据
   面板徽章变为醒目告警色并给出回退原因。**降级绝不破坏 3D 查看器。**
   - ❗ **超时不属于离线**：轮询超时（408）不会显示「未连接引擎/样例数据」，而是保留
     `job_id`（写入 state 与 URL `?job=<id>`）并提示「任务仍在处理中，可继续等待或稍后重连」（见下）。
3. **Fixture 模式（默认）**：未提供 `jobId` 时直接加载 fixture，`source` 标记为 `FIXTURE`。

### 上传轮询与超时恢复（MJ1）

引擎处理一段视频最长可达 **2h**，因此 `UploadPanel` 的轮询超时默认为 **1h**，可配置：

- prop：`<UploadPanel timeoutMs={7200000} />`；
- env：`NEXT_PUBLIC_POLL_TIMEOUT_MS`（毫秒，构建期内联）。

超时时抛【可区分】的 `ApiTimeoutError`（status=408），UI **不**将其误判为离线，而是：
保留 `job_id` → 展示「任务仍在处理中」+ `job_id` + 「继续等待」按钮（点击恢复轮询）。
用户也可刷新页面，因 `job_id` 已写入 URL `?job=<id>` 而可重新驱动查看。

### 代理安全：白名单 / 限流 / 最小头转发（MJ3）

`app/api/[...path]/route.ts` 不仅是转发器，还是公网安全边界：

- **方法+路径白名单**：仅放行 `POST /jobs`、`GET /jobs/{id}`、`GET /jobs/{id}/pose_data.json`、
  `GET /jobs/{id}/demo_output.mp4`、`GET /health`、`GET /healthz`；其余一律 **404**，防止代理
  沦为任意 URL 中继。
- **per-IP 令牌桶限流**：`POST /jobs` 每 IP 每 10min 最多 **3 次**，超限返回 **429 + `Retry-After`**。
  客户端 IP 从 `x-forwarded-for` 首跳读取并做兜底。
  > ⚠️ **已接受的残留风险**：限流为**内存态**，仅单实例有效；多副本/多实例部署下
  > 阈值会被放大，若需严格全局限流应下沉到网关/CDN 层（如 Zeabur/Cloudflare 限流规则）。
- **最小头转发**：绝不把客户端 `Cookie`/`Authorization` 透传给引擎，仅转发 multipart 必需的
  `content-type` 与 `accept`；`X-API-Key` 由服务端 `set` 注入，客户端即使伪造也无法覆盖。

### 环境变量（`.env.local`）

```bash
cp .env.example .env.local
# 【服务端】变量（无 NEXT_PUBLIC_ 前缀，仅服务端可见，必需）：
# ENGINE_API_BASE 按部署侧位置选择（多模式，详见 .env.example；运行时读取，改值即生效）：
#   Mac 本地开发   → http://aibox:8000        （Mac 在 Tailscale tailnet 内）
#   客户内网部署   → http://juxin.lan:8000    （路由器本地 DNS）
#   Zeabur 生产    → https://aibox.tail6791a3.ts.net（Tailscale Funnel 公网入口，已启用）
ENGINE_API_BASE=https://kineto-api.<DOMAIN>
KINETO_API_KEY=<your-api-key>
# 【浏览器侧】可选变量（NEXT_PUBLIC_ 构建期内联）：
# NEXT_PUBLIC_API_BASE=          # 留空即默认同源 /api；切勿填旧式绝对 URL
# NEXT_PUBLIC_POLL_TIMEOUT_MS=   # 上传轮询超时（毫秒），默认 1h(3600000)
```

> ⚠️ `NEXT_PUBLIC_*` 是**构建期内联**，改动后必须**重新部署**；
> **任何密钥都不得用 `NEXT_PUBLIC_` 前缀**，否则会泄露进浏览器包。
>
> ⚠️ **`NEXT_PUBLIC_API_BASE` 不要填旧式绝对 URL**（如 `https://kineto-api.<DOMAIN>`）：
> 默认已改为同源 `/api` 代理，绝对 URL 会让浏览器**直连引擎**并因缺少 `X-API-Key` 而 **401**；
> dev 下检测到 `://` 会在控制台打印警告。

### API 端点约定（浏览器 → 同源 `/api` → 代理 → 引擎）

| 方法 | 端点（代理后） | 用途 |
| --- | --- | --- |
| POST | `/jobs`（multipart, 字段 **`video`**） | 上传视频创建任务 → 202 `{ job_id }` |
| GET | `/jobs/{id}` | 轮询任务：`{ state, progress, quality_score?, extraction_mode?, error? }` |
| GET | `/jobs/{id}/pose_data.json` | 拉取姿态数据 |
| GET | `/jobs/{id}/demo_output.mp4` | 下载合成视频（流式透传） |
| GET | `/health` | 引擎健康检查（需鉴权，代理注入密钥） |
| GET | `/healthz` | 公开存活探针 |

> 任务状态契约：规范字段是 **`state`**（非 `status`），完成值 **`done`**、失败值 **`failed`**；
> 后端不返回 `status`/`completed`。上传 multipart 字段名为 **`video`**（非 `file`）。

---

## 数据契约（已与整改后 canonical 样本核对）

```jsonc
{
  "metadata": {
    "video_fps": 20.03,
    "total_frames": 466,
    "resolution": "1280x720",
    "model_version": "4dhumans-v1.0",
    "device": "mps",
    "extraction_mode": "4dhumans",
    "pipeline": { "final_quality_score": 0.9936, "...": "..." }
  },
  "keyframes": [
    {
      "frame_index": 0,
      "timestamp_ms": 0.0,
      "state_label": "initial",
      "joints_3d": [[x, y, z], ...],   // 24 个关节，SMPL canonical 序
      "smpl_thetas": [ ...72 floats ], // global_orient 3 + body_pose 69
      "cam_t": [x, y, z],              // additive（旧样本可能缺失）
      "confidence_score": 1.0,
      "betas": [ ...10 floats ]        // additive（旧样本可能缺失）
    }
  ]
}
```

要点：
- 质量分路径为 `metadata.pipeline.final_quality_score`（**非** metadata 顶层，根对象也没有）。
- `joints_3d` 的 24 个关节顺序为 **SMPL canonical 序**（0=pelvis、9=spine3、12=neck、
  15=head），由引擎在交付前从 HMR2 的 OpenPose Body-25 序归一而来（关节序根因已修）。
- 关节名、父表、骨骼父子对（23 条，**collar 13/14 的父为 spine3 9**）与部位归属严格镜像
  引擎骨架 SSOT `kineto-engine/skeleton_spec.py`（`SMPL_JOINT_NAMES` / `SMPL_PARENTS` /
  `SMPL_SKELETON` / `BONE_PART_MAP`）；一致性由 `deploy/check_ssot.py`（`validate.sh` 的
  **G12**）把关，改前端拓扑前必须先改 SSOT。
- 部位配色镜像 `kineto_core.py` 的 `BONE_COLORS`（配色不属于骨架 SSOT）。
- 后端 `BONE_COLORS` 为 BGR（OpenCV 约定），前端在 `lib/skeleton.ts` 中转换为
  three.js 使用的 RGB，使前后端可视颜色一致。
- `cam_t` / `betas` 为 additive 可选字段，缺失不报错（`validatePoseData` 只硬断言
  `joints_3d` 为 24×[x,y,z]）。
- 任务状态附加标志：门禁 `warn` 模式下质量不达标仍交付 `done`，但会**仅当为真时**
  附加 `degraded` / `quality_warning`（纯 additive，见 `lib/types.ts` 的 `JobStatus`）。
- 假数据守卫：`extraction_mode != "4dhumans"` 时元数据面板显示醒目告警徽章。

---

## 交互说明

- **3D 视图**：鼠标左键拖拽旋转、滚轮缩放、右键平移（drei `OrbitControls`，可从任意
  解剖平面观察）。骨架按全局包围盒居中并归一化，整段动画共用同一变换，不会漂移。
- **时间轴**：底部控制条播放/暂停、拖拽 scrubber 跳转；空格键切换播放。466 帧真实样本
  以 60fps 插值渲染（每帧 O(log n) 定位 + 24×3 次线性插值，命令式更新，无 React 重渲染）。

---

## 本期范围与后续（Follow-ups）

**本期已完成（MVP）**：基于 `joints_3d` 的 24 关节骨架可视化、时间轴插值循环播放、
scrubber、OrbitControls、元数据面板与假数据守卫、离线 fixture + API 双模式、
**同源 `/api` 服务端代理（密钥只在服务端注入）**、**上传/轮询 UI 闭环（`UploadPanel`）**、
**fixture 降级横幅与醒目告警徽章**、生产构建通过。

**后续待办**：
- [ ] **SMPL 网格渲染**：由 `smpl_thetas`（+ `cam_t`）驱动完整 SMPL 人体网格，而非仅关节骨架。
- [ ] **真实 2D 关键帧图**：将 `KeyframeCards` 的占位 SVG 替换为后端 ComfyUI/FLUX.1
      生成的标准四宫格图（`PROJECT_PLAN` 任务 1.2）。
- [ ] **坐标轴朝向校准**：真实样本关节坐标的竖直轴与 three.js 默认 Y-up 约定可能存在
      差异，后续可加入轴重映射开关以严格对齐解剖平面。
- [ ] **合成视频回放**：在任务完成后内嵌播放 `demo_output.mp4`（已经代理可流式下载）。
- [ ] **关键帧标记**：在时间轴上标出 initial/active/final 状态段与同步锚点（科研预留）。
