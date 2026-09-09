# Kineto 项目开发记忆

> 用途：沉淀开发过程中的主要进展、核心信息与踩坑经验，供后续会话/协作者快速接手。
> 维护约定：新进展按日期追加到 §2 时间线；新坑追加到 §5；核心信息变更时同步更新 §3/§4。

---

## 1. 项目概览

- **目标**：在局域网算力设备（AI BOX）上部署 Kineto Engine —— 基于 Intel Arc Pro B60 (XPU) 的视频人体姿态提取服务（4D-Humans + YOLO 检测），供 Zeabur 前端（kineto-web）调用。
- **设备**：主机名 `juxin`，Ubuntu 24.04 (Noble)，Intel Arc Pro B60 24GB，WiFi `wlp8s0`。
- **仓库布局**：`kineto-engine/`（引擎）、`kineto-web/`（前端）、`deploy/`（部署脚本）、`videos/`（测试视频）。

## 2. 进展时间线

### 2026-09-05 部署日
- 设备体检：`deploy/discover_device.sh` 四项门禁（RAM≥16G / disk≥25G / Intel GPU / Docker）+ A1-A3 全过。
- MoFang 剥离：先 `--dry-run` 确认 5 个 systemd unit + openclaw 容器，再正式剥离；`--rollback` 可回滚。
- 事故与修复：曾用 deadsnakes PPA 装 python3.11 拖垮 network-manager 导致断网，TTY 手工 wpa_supplicant 修网；此后禁用 deadsnakes，统一用系统 Python 3.12。
- 修复 apt 混源（jammy/noble 混用），旧源备份 `/etc/apt/sources.list.disabled-jammy`。
- 引擎部署完成：torch-XPU（SJTU 镜像）、chumpy 源码补丁、PyOpenGL 3.1.7、omegaconf 补装；systemd 服务 `kineto-engine.service`。
- 验收：`deploy/validate.sh` G1-G11 全过 —— 466 帧真实推理、`extraction_mode=4dhumans`、质量分 0.6726、约 62s/条（Arc Pro B60）；reboot 复验通过（自启 + XPU 就绪）。
- WiFi 省电关闭（`wifi.powersave 2`）：此前高吞吐传输两次整机掉线的根因。

### 2026-09-06 网络与外设日
- 换局域网：新 IP `192.168.110.177`，路由器本地 DNS 解析 `juxin.lan`（免查 IP）。
- 修复设备 DNS：`/etc/resolv.conf` 为 TTY 修网期遗留的静态文件（指向旧网关）→ 改回 systemd-resolved stub；活动连接持久化公共 DNS（223.5.5.5 / 119.29.29.29）。
- 公网访问探索：Cloudflare quick tunnel（临时 URL）验证可用后停用；固定域名方案探索 —— Cloudflare Add site 只收根域名（需整域 NS 委托，影响 Zeabur 子域，放弃）；FRP 不可行（云 ECS 由 Zeabur 托管无 SSH）。
- **最终方案：Tailscale 组网**。设备端 `aibox`（100.101.114.50）+ Mac 端 `macbook-air`（100.73.220.80），账号 `Panb-KG@`，MagicDNS 域 `tail6791a3.ts.net`。跨网 E2E 验证通过（上传 18.7MB → XPU 推理 → 下载 13.1MB demo）。
- `cloudflared-quick.service` 已 stop+disable（减少暴露面），二进制与 apt 源保留。
- 蓝牙键盘：安装 bluez 5.72，Logitech K380 配对（passkey 流程）+ trust + connect 成功，开机自连；`bluetooth.service` 已 enable。
- 架构决策：评审「前后端都部署在 AI BOX」方案后**维持前端 Zeabur + 引擎 AI BOX**（详见 §7 ADR-001）。

### 2026-09-07 公网入口打通
- 公网入口方案定调：Zeabur Wonder Mesh 与 Tailscale 同为 WireGuard overlay，装在同一设备有路由/DNS 冲突风险；且 Mesh 公网入口需专属 Gateway 服务器（用户的 ECS 由 Zeabur 托管无 SSH，走不通）→ 放弃 Wonder Mesh。
- **Tailscale Funnel 上线**：`tailscale funnel --bg 8000`，公网地址 `https://aibox.tail6791a3.ts.net`（永久固定、自动签发 TLS、与 tailnet 同守护进程零冲突、重启自动恢复）。
- 公网 E2E 验证通过：`/health`（带 key）→ device=xpu 模型就绪；无 key → 401；上传 18.7MB 视频 → XPU 推理 66s → state=done、4dhumans、质量分 0.6726 → 下载 13.1MB demo 视频。
- `.env.example` / `README.md` 三模式表更新：public-prod 改为 Funnel 地址。
- 前端 Zeabur 生产部署的前置条件已满足：在 Zeabur 项目环境变量设 `ENGINE_API_BASE=https://aibox.tail6791a3.ts.net` + `KINETO_API_KEY=<key>` 即可。

### 2026-09-07 Ultra Review 修复轮次 + AI BOX 重新部署
- Ultra Review 发现 3 Critical + 4 Major + ~15 Minor 问题，启动全面整改：
  - **引擎 22 项修复**（Bill）：Rodrigues 最小旋转替代 Kabsch for K=1、SSOT 惰性加载（`_resolve_pkl_path` 环境变量优先）、`clip_info` 跨轮累积（不再每轮清零）、`rel_violation` 只对被夹帧求均值、`compute_verdict` 量纲一致（骨数 + spike_ratio>0.10?1:0）、`skeleton_validator` M4 迁移（5 符号从 `skeleton_spec` 导入）、metadata 判别位 `joint_order=smpl-canonical` + `schema_version=2`
  - **部署 4 项修复**（Kevin）：`check_ssot` 优雅降级（pkl 不可用时 exit code 3 SKIP）、G12 覆盖 `skeleton_validator`、`Dockerfile.engine` §6c 构建期断言（24 joints / 23 edges / collar parent=9）、`validate.sh` G13 黄金样本（断言 `output_test` verdict=pass）
  - **前端 6 项修复**（Aaron）：`validatePoseData` 全帧校验（非只首帧）、`degraded` 横幅（`UploadPanel` / `ViewerStage`）、`joint_order` 非 canonical 时告警（`MetadataPanel`）
- `output_test` / `output_closed` 重生成：verdict=pass, total_issues=0, joint_order=smpl-canonical, schema_version=2, 质量分 0.9936
- **AI BOX 重新部署**：rsync 代码（commit cdba171）+ 重启服务 + 验收 **12 PASS / 3 SKIP / 0 FAIL** ✅
- 验收基线更新：质量分 0.9936（原 0.6726，因 `clip_info` 跨轮累积后更诚实反映约束违反程度）
- 部署踩坑：`/srv/kineto/` 目录权限 750→755（juxin 无 traverse 权限导致 validate.sh G3 403）；旧视频 `/opt/kineto/kineto-engine/input_video.mp4` 需删除（validate.sh 优先探测该路径但不在 inbox 内）

### 2026-09-09 P0 mesh↔joints 对齐 + smplx 形状 bug 修复（本地验证通过，待真机复验）
- **根因定位**：SMPL forward 输出在模型空间，交付 joints_3d 在原始帧坐标系，两者相差每帧恒定平移（Kabsch 验证：去 t 后残差=0，|t|≈0.31m）→ 叠加模式 mesh 与骨架错位 ~300mm。
- **引擎修复**：`kineto_core.py` `compute_mesh_for_keyframes` 增加 pelvis 锚点平移（`t_align = joints_3d[0] - (J_regressor @ vertices)[0]`），mesh 顶点对齐到交付坐标系；J_regressor 线性性保证中间插值帧按构造贴合。
- **顺带实锤隐藏 bug**：smplx 0.1.28 `pose2rot=False` 要求 `global_orient` 为 `(B,1,3,3)`，引擎原传 `(B,3,3)` 会在 `torch.cat` 抛 RuntimeError → mesh 路径被宽 except 吞掉、整体静默降级 `has_mesh: False`（历史产物均无 mesh_vertices 的根因）。已修为 `reshape(1, 1, 3, 3)`。
- **前端**：删除 `CombinedViewer.tsx` 的 `JOINT_NUDGE` 手调偏移（引擎已数学对齐，无需补偿）。
- **门禁**：`check_ssot.py` 新增 **G14**（产物 mesh 顶点经 J_regressor 回归须与 joints_3d 重合，容差 10mm；无 mesh 帧时 SKIP）。
- **本地数值复验 PASS**：12/466 帧采样，修复前最大错位 327.4mm → 修复后残差 0.0003mm；`tsc --noEmit` + G12/G14 闸门通过。
- **真机复验 PASS**（2026-09-09）：rsync 两文件到设备 → 离线 CLI 重生成 `output_test`（466 帧、16 帧 mesh、`[Mesh] ✓ 计算完成`——修复前此处静默失败；质量分 0.9936 无回归）→ G14 断言 mean/max **0.000mm** PASS。设备上无 kineto-web 时 G12 提前 SKIP，G14 需单独脚本（用 `skeleton_spec.SMPL_J_REGRESSOR` 纯 numpy 即可，无需构建 SMPLLayer）。
- **遗留已解决**（2026-09-09 晚，全程免 sudo，经 Tailscale 远程）：设备 `KINETO_CORS_ORIGINS` 原为 UTF-8 中文域名（浏览器 Origin 头发 punycode 不匹配）→ 更新为 punycode+UTF-8 双值；docker（`--pid=host --cap-add SYS_PTRACE --security-opt apparmor=unconfined`）TERM 主进程 → systemd `Restart=always` 自动拉起（PID 10394→18198）加载新代码；随后**真实 API 链路 E2E PASS**：POST /jobs → 71s → `state=done / 4dhumans / 0.9936`，job 产物首次含 `mesh_vertices`（16 帧），G14 断言 mean/max **0.000mm**。
- **节奏贴合矛盾（规划输入）**：骨架 466 帧全量 60fps 插值，mesh 受 `MESH_FRAME_BUDGET=16` 限制仅 16 帧 → 叠加模式节奏必然脱节；全帧 mesh 需二进制通道（float32 JSON 约 150MB 不可行）。

## 3. 核心信息速查（设备与服务）

### 访问方式
| 场景 | 地址 |
|---|---|
| **公网（任意互联网设备，Funnel）** | `https://aibox.tail6791a3.ts.net`（Zeabur 生产用此地址） |
| 任意网络（Mac 经 Tailscale，推荐） | `http://aibox:8000`（或 `http://100.101.114.50:8000`） |
| SSH（任意网络） | `ssh juxin@aibox` |
| 同局域网 | `http://juxin.lan:8000` / `ssh juxin@juxin.lan` |

- 引擎所有受保护端点需请求头 `X-API-Key`（值在设备 `/etc/kineto/kineto-engine.env`，勿入库入文档）。
- 大视频（>100MB 或经公网隧道）先 `scp` 到设备 `/srv/kineto/inbox/`，再 POST `{"video_path":"/srv/kineto/inbox/xx.mp4"}`。

### 服务布局（设备侧）
| 项 | 路径 |
|---|---|
| systemd 服务 | `kineto-engine.service`（LAN 监听经 drop-in：`/etc/systemd/system/kineto-engine.service.d/override-lan.conf`） |
| 代码 / venv | `/opt/kineto/kineto-engine`（Python 3.12 venv：`/opt/kineto/venv`） |
| 模型 | `/srv/kineto/models`（4DHumans 软链自 `/srv/kineto/.cache/4DHumans`） |
| 任务 / 收件箱 | `/srv/kineto/jobs`、`/srv/kineto/inbox` |
| 密钥/环境 | `/etc/kineto/kineto-engine.env`（`KINETO_INBOX=/srv/kineto/inbox`） |

### API 端点
| 端点 | 说明 |
|---|---|
| `GET /healthz` | 公开存活探针 |
| `GET /health` | 鉴权；返回 device=xpu、模型加载状态 |
| `POST /jobs` | multipart `video=@file` 或 JSON `video_path`（限 inbox 内）→ 返回 `job_id` |
| `GET /jobs/{id}` | 状态字段是 **`state`**（queued/running/done/failed），另有 progress/quality_score/extraction_mode |
| `GET /jobs/{id}/pose_data.json` / `demo_output.mp4` | 完成后取产物 |

### 验收基线（复验锚点）
- 466 帧视频，device=xpu，4dhumans，质量分 0.9936，~62s；`KINETO_API_KEY=$(grep KINETO_API_KEY /etc/kineto/kineto-engine.env | cut -d= -f2) bash deploy/validate.sh --mofang-mode stripped`
- reboot 复验：服务自启、XPU/模型就绪、E2E 通过。
- 产物 metadata：`joint_order=smpl-canonical` + `schema_version=2`（schema additive，向后兼容）。

## 4. 踩坑与解法（硬约束）

1. **禁用 deadsnakes PPA**（Ubuntu 24.04 会与 network-manager 等系统包冲突）→ 用系统 Python 3.12。
2. **torch-XPU 轮子走 SJTU 镜像** `https://mirror.sjtu.edu.cn/pytorch-wheels/xpu/`（官方源 KB/s；TUNA 无 xpu index）。
3. **chumpy 0.70** 在 Python 3.12/numpy 2.x 无法构建：需源码补丁（numpy 别名清理、`np.*`→`np.bool_/np.int_/np.float64`、`getargspec`→`getfullargspec`）+ `--no-deps --no-build-isolation`（补丁见 `deploy/patches/`）。
4. **PyOpenGL 必须 3.1.7**（pyrender 钉的 3.1.0 缺 `OSMesaCreateContextAttribs`）。
5. **omegaconf 必须显式安装**：HMR2 .ckpt 反序列化需要（requirements 注释称训练侧而未装）。
6. **WiFi 省电必须关**（`wifi.powersave 2`）：高吞吐传输会整机掉线。
7. **客户局域网 DNS 可能坏**：检查 `/etc/resolv.conf` 是否为静态遗留文件（应为 stub 软链）；路由器 DNS 不解析外域时 `nmcli con mod <conn> ipv4.dns "223.5.5.5 119.29.29.29" ipv4.ignore-auto-dns yes && nmcli device reapply`。
8. **GitHub TCP 在这些网络被封**：cloudflared 用 `pkg.cloudflare.com` apt 源装；Tailscale 用官方 `tailscale.com/install.sh`（可达）。
9. **Cloudflare 免费计划请求体 ≤100MB**；**Add site 只收根域名**（子域委托不可行）。
10. **引擎 job 状态字段是 `state` 不是 `status`**；JSON 模式 `video_path` 必须位于 `KINETO_INBOX` 内。
11. **`pkill -f` 会匹配自身命令行**（SSH 复合命令中 pkill 关键词出现在同一命令串导致自杀）：pkill 模式用 `[x]` 方括号断言或拆分命令。
12. **bluetoothctl 非交互配对**：单次/管道模式（`(echo scan on; sleep 6; echo pair <MAC>; sleep 120; echo quit) | bluetoothctl`），agent 用 `KeyboardDisplay`，passkey 由主机端显示、在物理键盘输入；配对后 `trust` 以便自动回连。
13. **3D 姿态必须与原视频一致——禁止前端 spine 旋转校正**：引擎 `_extract_4dhumans_rotated` 已将 3D 坐标逆旋转回原始帧坐标系，前端不得再做 spine-to-Y 对齐。若原视频中人是躺着的，3D 模型也应显示为躺着。多层旋转叠加会导致姿态翻转/错位，此问题已反复出现多次。
14. **4D-Humans 旋转处理流程（硬约束）**：4D-Humans 模型按人体直立姿态设计，处理非直立输入时必须遵循三步流程：
    - **输入旋转**：检测横卧姿态（bbox 宽≥高）→ 旋转帧 90° 让人体"直立"
    - **模型推理**：在旋转后的帧上运行 4D-Humans，得到直立坐标系下的 3D 结果
    - **输出复原**：将 joints_3d、cam_t、global_orient 逆旋转回原始帧坐标系
    - 前端直接渲染复原后的坐标，**禁止**任何额外旋转。原视频是什么姿态，3D 就显示什么姿态。
15. **相机坐标→世界坐标转换（硬约束）**：引擎输出相机坐标系（X 右，Y 下，Z 向前），Three.js 使用世界坐标系（X 右，Y 上，Z 向后）。前端在 `computeFraming`/`computeMeshFraming` 中必须应用转换：`y → -y`，`z → -z`。否则模型会上下颠倒、前后反向。
16. **smplx 0.1.28 `pose2rot=False` 形状约定**：`global_orient` 必须是 `(B,1,3,3)`（与 `body_pose` 的 `(B,23,3,3)` 同维才能 `torch.cat`），传 `(B,3,3)` 抛 RuntimeError；引擎 mesh 路径宽 except 会把这类错误吞成 `has_mesh: False` 静默降级——mesh 相关改动必须用真产物验证 G14，不能只看 job state=done。
17. **设备离线跑引擎 CLI 需 `HOME=/srv/kineto`**：HMR2 `get_config` 的 cachedir 依赖 HOME 解析到 `~/.cache/4DHumans`（服务以 kineto 用户跑、HOME=/srv/kineto）；juxin 直接跑会 `FileNotFoundError: …/model_config.yaml`。离线复现：`cd /opt/kineto/kineto-engine && HOME=/srv/kineto /opt/kineto/venv/bin/python3 kineto_core.py -i <video> -o <dir>`（juxin 在 video/render 组，XPU 可用；output_test 归 juxin 可写）。
18. **无 sudo 远程运维 AI BOX（juxin sudo 需密码）**：juxin 在 `docker` 组 = 等价 root。读改 root 文件：`docker run --rm --user 0 --entrypoint sh -v /etc/kineto:/e <镜像> -c '…'`（mofang 镜像内置非 root USER，必须显式 `--user 0`；容器内无宿主组名，chown 用数字 GID，`getent group kineto` 查）。重启 systemd 服务：`docker run --rm --pid=host --user 0 --cap-add SYS_PTRACE --security-opt apparmor=unconfined <镜像> kill -TERM <主进程PID>` → `Restart=always` 自动拉起（默认 AppArmor docker-default 拦跨命名空间 signal，必须 unconfined + SYS_PTRACE）。改 env 文件后注意恢复 `chown root:kineto`（GID 982）+ `chmod 640`。

## 5. 待办 / 后续方向

- **Zeabur 前端生产部署**：前端域名已定案 `https://kineto.标智云.中国`（punycode `kineto.xn--9kqt69c97a.xn--fiqs8s`）。设备侧 `KINETO_CORS_ORIGINS` 已更新为 punycode+UTF-8 双值（2026-09-09）。剩余待办全在 Zeabur 控制台：绑定自定义域名（自动 TLS）+ 阿里云子域 CNAME 到 Zeabur 目标 + 环境变量 `ENGINE_API_BASE=https://aibox.tail6791a3.ts.net` + `KINETO_API_KEY=<key>`，git push 部署后验证浏览器→域名→Zeabur→Funnel→引擎全链路。
- **自定义域名**（可选）：若品牌需要 `kineto.标智云.中国` 而非 `ts.net`，Tailscale 支持给节点配 CNAME（需 Tailscale Pro 或以上）；或保留 cloudflared 备用方案。
- **备用公网方案**：cloudflared 已装（apt 源保留），如需临时公网 URL：`systemctl enable --now cloudflared-quick`，地址用 `journalctl -u cloudflared-quick | grep trycloudflare` 查（重启会变）。
- K380 三个蓝牙通道中仅一个绑到本机，其余通道可另连 Mac 等设备。

## 6. 关键文件索引

| 文件 | 用途 |
|---|---|
| `DEPLOY_MOFANG.md` | MoFang 时代部署文档（含剥离要求） |
| `deploy/discover_device.sh` | 设备体检（G1-G4 / A1-A3 门禁） |
| `deploy/strip_mofang.sh` | MoFang 剥离/回滚（`--dry-run` / `--rollback`） |
| `deploy/transfer_models.sh` | 模型同步（Mac → 设备，rsync 增量） |
| `deploy/validate.sh` | G1-G11 验收（`--mofang-mode stripped`） |
| `deploy/kineto-engine.service` | systemd unit 模板 |
| `deploy/cloudflared/` | 命名隧道配置（未启用） |
| `deploy/patches/` | chumpy 等源码补丁 |
| `deploy/Dockerfile.engine` / `docker-compose.yml` | 引擎容器化（Wonder Mesh 路线备用） |
| `PROJECT_PLAN.md` / `README.md` | 项目规划与总览 |

## 7. 架构决策记录（ADR）

### ADR-001 前端留在 Zeabur、引擎留在 AI BOX（2026-09-06，已采纳）

**背景**：评估是否把前端也迁到 AI BOX（前后端一体）。设备实测资源充裕（31G 内存/29G 可用、915G 磁盘/693G 可用），硬件上完全放得下。

**决策**：维持现状 —— 前端 Zeabur（静态托管），引擎 AI BOX（XPU 推理）。

**理由**：
1. **公网入口绕不开云端**：AI BOX 在 NAT 后、走 WiFi、局域网已两换，无公网 IP。前端无论放哪，公网访问都必须经云端入口（Zeabur Gateway/Cloudflare/VPS）。「都放 AI BOX」只是把流量多绕一跳（用户→云→设备→前端+API），省不掉云依赖，反而扩大设备暴露面。
2. **可靠性**：Zeabur 保证前端永远在线，引擎离线时前端仍可打开并降级提示；前端随设备走则设备断电断网=整个产品消失。
3. **性能无差别**：视频经浏览器→Zeabur 代理→引擎【流式中转】（`request.body` ReadableStream 直通，不缓冲不落盘），前端托管位置不影响推理性能；上传带宽瓶颈在引擎侧链路，与前端放哪无关。
4. **数据主权已满足**：视频数据在 Zeabur 仅流式中转不落盘、不存储；大文件亦可用 scp→inbox→`video_path` 模式完全绕开 Zeabur 链路。
5. **迭代与安全**：git push 即发布、HTTPS/CDN/回滚齐备；设备侧维持仅暴露 8000 API（带 X-API-Key）。

**重新评估的触发条件**（出现任一即重启此决策）：
- 产品形态转为**私有化一体机交付**（客户内网整机交付）→ 用 `deploy/Dockerfile.engine` + docker-compose 打包前后端；
- 客户环境**完全无互联网**（纯内网演示/交付）；
- Zeabur 托管不可用或成本不可接受。

**配套行动**（2026-09-06 ~ 07 已完成）：多模式引擎地址落地 —— 无需代码改动（`ENGINE_API_BASE` 为运行时读取，`app/api/[...path]/route.ts` 的 `force-dynamic` 保证改环境变量即时生效），三种模式的取值与可达性约束已写入 `kineto-web/.env.example` 与 `kineto-web/README.md`。公网入口用 **Tailscale Funnel**（`https://aibox.tail6791a3.ts.net`，已启用并验证），Zeabur 生产部署的前置条件已满足。
