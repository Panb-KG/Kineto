# Kineto

**智能运动康复动作解析与标准化图谱生成平台**

Kineto 是一个专为运动康复工作室、临床指导与解剖学教学设计的云边协同 AI 平台。系统通过计算机视觉技术将真人教练的运动视频逆向解析为高精度的 3D 骨架数据（SMPL-X），并利用受限的扩散模型（FLUX.1）生成解剖完全可控、角色视觉绝对一致的标准化动作处方图谱，同时在 Web 端提供可全视角互动的 3D 康复动作指导。

核心原则：**解剖严谨优先**。杜绝生成式 AI 原生的"动作幻觉"，确保骨盆-脊柱节律、多关节协同等复杂生物力学特征真实可靠。

> **🚀 部署入口** → 设备侧（Intel Arc / MoFang）见 **[`DEPLOY_MOFANG.md`](./DEPLOY_MOFANG.md)**；前端上云（Zeabur）见 **[`DEPLOY_ZEABUR.md`](./DEPLOY_ZEABUR.md)**。

---

## 🏗 系统架构 (Cloud-Edge Hybrid)

采用云边协同架构，实现重度 GPU 算力本地化与轻量级业务云端化。

| 模块 | 运行环境 | 核心技术栈 | 核心职责 |
| :--- | :--- | :--- | :--- |
| **Kineto Engine (算法引擎)** | 本地算力节点 (Intel AI BOX / 24GB 显存) | Python, 4DHumans, ComfyUI, FastAPI | 执行 Video-to-Pose (3D 姿态逆向解算)；执行 Pose-to-Render (FLUX 2D 四宫格标准图生成)；算力本地闭环保障数据隐私。 |
| **Kineto Web (业务大脑)** | 云端 Serverless (Zeabur 2C 4G) | Next.js, React Three Fiber (R3F) | 面向患者与学生的动作处方交付；提供高保真 2D 图与 3D 骨架的可拖拽互动展示。 |

---

## ✨ 核心特性

1. **精准动作逆向工程 (Video-to-Pose)**：基于单目视频提取高保真人体运动学参数，精准捕获肩关节外展、髋膝屈伸等真实发力轨迹与代偿特征。

2. **完全可控的标准图重绘 (Grid Latent Render)**：基于提取的姿态参数，结合 ControlNet (DensePose + Depth) 约束 FLUX 模型，一次性生成同一角色、同一着装、同一视角的 4 宫格标准化教学图。

3. **Web 端 3D 全视角互动 (Interactive 3D)**：动作处方下发极小体积的 JSON 驱动文件，用户在小程序/Web 端可 360° 拖拽观察动作，直观理解肩胛运动节律与骨盆中立位。

4. **科研级数据拓展层 (Research Ready)**：全量保留带有绝对时间戳的 3D 运动学数据，预留未来与测力台 (Force Plate)、肌电图 (EMG) 等 1000Hz 高频动力学硬件的时间序列同步能力。

---

## 📂 目录结构

```text
Kineto/
├── kineto-engine/           # 设备侧算法引擎 (Python + FastAPI)，跑在 Intel Arc
│   ├── api.py               # FastAPI 服务端点（/jobs、/health 需鉴权、/healthz 公开）
│   ├── kineto_core.py       # Video-to-Pose 核心管线（CLI: --input/--output/--max-iter/--quality-thresh/--no-refine）
│   ├── keyframe_selector.py # 关键帧筛选
│   ├── pose_audit.py        # 姿态审计
│   ├── skeleton_validator.py# 骨架校验
│   ├── check_environment.py # 环境自检
│   ├── requirements.txt     # 依赖（含 Intel XPU / IPEX 安装指引）
│   ├── DEVICE_SELECTION.md / SETUP_GUIDE.md / QUICKSTART.md
│   └── 4D-Humans/           # 上游 4DHumans（.gitignore 排除，需 rsync + weights_only 补丁，见 SETUP_GUIDE）
├── kineto-web/              # 云端前端 (Next.js 14 + React Three Fiber)
│   ├── app/                 # App Router（page.tsx + api/[...path]/route.ts 服务端代理）
│   ├── components/          # UI 组件（R3F 3D 渲染、上传面板、元数据面板）
│   ├── lib/                 # 数据层（api.ts / poseData.ts / types.ts）
│   └── public/              # 静态资源与内置 fixture 样例 JSON
├── deploy/                  # 设备侧部署工具包（脚本 + Dockerfile + systemd + cloudflared）
│   ├── discover_device.sh   # Phase 0 设备探测
│   ├── validate.sh          # 部署门禁 G1..G11 验证
│   ├── strip_mofang.sh      # MoFang 上层可逆剥离（禁停硬拦截 + undo + --rollback）
│   ├── transfer_models.sh   # 模型/代码同步到设备
│   ├── Dockerfile.engine / docker-compose.yml / kineto-engine.service
│   ├── patches/hmr2_weights_only.patch   # 4D-Humans torch>=2.6 补丁（受版本控制）
│   └── cloudflared/         # Cloudflare Tunnel 配置与 README
├── docs/                    # 开发文档、Vibe Coding 提示词资产库
├── DEPLOY_MOFANG.md         # 设备侧（Intel Arc / MoFang）部署与底层改造手册
├── DEPLOY_ZEABUR.md         # 前端上云（Zeabur）指南
└── README.md
```

---

## 🧬 数据结构规范 (科研扩展要求)

为了确保 3D 运动学数据与未来外部传感器数据的对齐，Kineto 导出的姿态数据严格遵循时间戳锚点规范：

```json
{
  "metadata": {
    "video_fps": 60,
    "total_frames": 180,
    "resolution": "1080p",
    "model_version": "4dhumans-v1.0"
  },
  "keyframes": [
    {
      "frame_index": 1,
      "timestamp_ms": 16.67,
      "state_label": "initial", 
      "smpl_thetas": [],
      "confidence_score": 0.95
    }
  ]
}
```

---

## 🚀 敏捷开发路线图 (Vibe Coding)

本项目基于独立开发者 AI 辅助编程（Vibe Coding）范式推进：

### Phase 1: 算法引擎黑盒验证 (Local Engine)

- [ ] 跑通 input.mp4 到 3D 姿态 JSON 的自动化提取，并完成显存防溢出优化
- [ ] 跑通 JSON 参数动态注入 ComfyUI API 并成功切片生成 4 张关键帧图

### Phase 2: 业务大脑可视化 (Cloud Web) —— ✅ 已交付

- [x] 构建 Next.js 前端，实现 React Three Fiber 的 3D 模型平滑渲染与 Tween.js 动画循环
- [x] 使用测试 JSON（内置 fixture）验证前端 3D/2D 双轨交互体验；上传→轮询→可视化闭环打通

### Phase 3: 云边通讯闭环 (Integration) —— ✅ 已交付

- [x] 将 Engine 封装为 FastAPI（`kineto-engine/api.py`：`/jobs`、`/health`、`/healthz`）
- [x] 配置 Cloudflare Tunnel 内网穿透（`deploy/cloudflared/`），实现 Zeabur 云端经**服务端代理**对本地 Intel AI BOX 的异步生图请求与状态轮询

> **剩余（需在真实设备上执行）**：设备侧实机部署（Intel Arc 驱动 / XPU 验证 / MoFang 剥离 / systemd 上线）
> 属物理操作，需用户在设备上按 [`DEPLOY_MOFANG.md`](./DEPLOY_MOFANG.md) 执行并用 `deploy/validate.sh` 过门禁；
> 前端上云按 [`DEPLOY_ZEABUR.md`](./DEPLOY_ZEABUR.md) 在 Zeabur 控制台完成。

---

## ⚠️ 免责声明

本平台生成的动作指导图与 3D 骨架仅供运动训练示意与解剖学教学参考。系统输出不构成临床医学诊断，不能替代专业医师或持证康复师开具的个体化医疗处方。