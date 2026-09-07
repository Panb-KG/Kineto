# Kineto 前端上云指南 —— Zeabur 部署（`kineto-web`）

> **适用范围**：把 `kineto-web`（Next.js 14 + React Three Fiber 的 3D 姿态可视化前端）部署到
> **Zeabur**，并绑定你自己的域名。设备侧引擎（Intel Arc / FastAPI）的部署见
> **[`DEPLOY_MOFANG.md`](./DEPLOY_MOFANG.md)**；本文只讲前端上云。
>
> **部署入口速查**：设备侧 → `DEPLOY_MOFANG.md`；前端 → 本文 `DEPLOY_ZEABUR.md`。

---

## 0. 架构前提：浏览器不再直连引擎（务必先读）

修复后的云边架构采用 **Next.js 服务端代理**，安全边界在服务端而非浏览器：

```
浏览器  ──同源──▶  https://<你的Zeabur域名>/api/*      (Next.js Route Handler，服务端)
                        │  服务端读取 ENGINE_API_BASE + KINETO_API_KEY
                        │  注入 X-API-Key 请求头
                        ▼
                 https://kineto-api.<你的域名>/…         (设备侧 FastAPI，经 Cloudflare Tunnel 暴露)
```

关键结论（与 `DEPLOY_MOFANG.md`、引擎 `api.py` 的契约一致）：

| 事实 | 说明 |
|---|---|
| 浏览器只与 **同源** `/api/*` 通信 | 见 `kineto-web/app/api/[...path]/route.ts`；浏览器**从不**直连引擎、**从不**持有任何密钥 |
| `X-API-Key` 由**服务端**注入 | 代理在 `route.ts` 里 `headers.set("X-API-Key", KINETO_API_KEY)` |
| 引擎 `GET /health` **需鉴权** | 代理转发时带上密钥即可；另有**公开**的 `GET /healthz` 作极简存活探针 |
| **引擎 CORS 对浏览器已非必需** | 浏览器→Zeabur 是同源；Zeabur 服务端→引擎是**服务器间调用**，不触发浏览器 CORS。设备侧 `KINETO_CORS_ORIGINS` 功能上已非必需，但启动守卫仍要求填值（纵深防御，见 §5） |
| 上传契约 | `POST /jobs` 为 `multipart/form-data`，**字段名 = `video`**；代理流式透传 body 以保留 boundary |

> 因此：**代理架构下浏览器不再直连引擎，CORS 功能上已非必需；但设备侧启动守卫仍要求 `KINETO_CORS_ORIGINS` 非空（纵深防御），
> 必须填入前端精确 origin，详见 §5。**

---

## 1. ⚠️ 头号坑：Zeabur 的 Root Directory 必须填对

本仓库的 **git 根目录在用户 home 级**（`git rev-parse --show-toplevel` → `/Users/<you>`），
远程为 `github.com/Panb-KG/Kineto`。前端项目**嵌套**在仓库的深层子目录里：

```
<git root>/
└── Projects/Kineto/
    ├── kineto-web/        ← 前端（本文要部署的）
    ├── kineto-engine/     ← 设备侧引擎（不在 Zeabur 上跑）
    └── deploy/
```

**所以在 Zeabur 新建服务时，Root Directory（根目录 / 项目路径）必须填：**

```
Projects/Kineto/kineto-web
```

> ❌ 不填、或填成 `Projects/Kineto`、`kineto-web`、`/kineto-web` 都会**构建必失败**：
> Zeabur 会在错误的目录里找 `package.json`，找不到 `npm ci` 的入口。
> ✅ 路径**相对于 git 仓库根**、**不要**以 `/` 开头、**不要**带仓库名前缀。

---

## 2. 部署步骤（Zeabur 控制台）

1. **新建项目 / 服务**：Zeabur 控制台 → 新建 Project → 新建 Service → **从 GitHub 仓库部署**。
2. **连接仓库**：授权并选择 `Panb-KG/Kineto`，分支选你的默认分支（如 `main`）。
3. **设置 Root Directory**：在服务设置里把根目录填为 **`Projects/Kineto/kineto-web`**（见 §1）。
4. **构建 / 启动命令**（Zeabur 通常能从 `package.json` 自动识别；若需手动则填）：
   - 构建（Build Command）：`npm ci && npm run build`
   - 启动（Start Command）：`npm start`
   - 输出目录：`.next`（Next.js 自管，一般无需手填）
5. **Node 版本**：**≥ 18.17**（`kineto-web/package.json` 的 `engines.node`）。
   仓库自带 `kineto-web/.nvmrc = 20`，Zeabur 一般会据此选 Node 20；若构建报 Node 版本过低，
   在服务的运行时/环境变量里显式指定 Node 20。
6. **配置环境变量**（见 §3）→ **绑定域名**（见 §4）→ 触发部署。

> `npm ci` 依赖 `package-lock.json`（仓库已含），可复现构建；**不要**改成 `npm install`。
> Next.js 是标准构建，Zeabur 原生支持，无需自定义 Dockerfile。

---

## 3. 环境变量：在 **Zeabur 控制台**设置（不是 `.env.local`）

生产环境的变量**一律在 Zeabur 控制台的 Environment Variables 里填**。
`.env.local` 只用于本机开发，**不会**（也不该）进生产；仓库里的 `kineto-web/.env.example`
只是模板示例。

| 变量 | 必填 | 前缀 | 示例 / 说明 |
|---|---|---|---|
| `ENGINE_API_BASE` | ✅ | **无 `NEXT_PUBLIC_`**（服务端） | `https://kineto-api.<YOUR_DOMAIN>` —— 设备侧引擎经 Cloudflare Tunnel 暴露的基址 |
| `KINETO_API_KEY` | ✅ | **无 `NEXT_PUBLIC_`**（服务端） | 与**设备侧同一把**密钥（`DEPLOY_MOFANG.md` §5.8 生成、`/etc/kineto/kineto-engine.env` 里那把） |
| `NEXT_PUBLIC_API_BASE` | 可选 | `NEXT_PUBLIC_`（浏览器） | 留空即默认走**同源代理** `/api`（推荐）；仅当需要非默认同源前缀时才填 |

### 🔒 两条安全铁律（务必遵守）

1. **`NEXT_PUBLIC_*` 是构建期内联**：带 `NEXT_PUBLIC_` 前缀的变量会在 **`npm run build` 时被写死进
   浏览器 JS 包**，对所有人可见，且**改动后必须重新部署（重新构建）才会生效**——运行时改控制台不生效。
2. **密钥绝不能用 `NEXT_PUBLIC_` 前缀**：`KINETO_API_KEY` 必须是**服务端变量**（无前缀），
   只在 `app/api/[...path]/route.ts` 里 `process.env.KINETO_API_KEY` 读取。
   ❌ 绝不能写成 `NEXT_PUBLIC_API_KEY`——那会把密钥泄露进浏览器包，任何人都能抓包拿到。

> 代理的运行时行为（用于排障，见 §6）：
> - `ENGINE_API_BASE` 或 `KINETO_API_KEY` **任一未配置** → 代理对 `/api/*` 返回 **503 `engine not configured`**；
> - 已配置但**引擎不可达** → 返回 **502 `engine unreachable`**；
> - 两种情况前端都**不崩溃**，会优雅回退到内置 fixture 样例数据（§6）。

---

## 4. 绑定你自己的域名

1. Zeabur 服务 → Networking / 域名 → **绑定自定义域名**（如 `kineto.<你的域名>`），
   按提示在你的 DNS 服务商加 CNAME 指向 Zeabur 给出的目标；等待证书签发（Let's Encrypt 自动）。
2. 该域名即前端对外的 `https://<你的Zeabur域名>`，浏览器同源访问 `/api/*`。
3. **`ENGINE_API_BASE` 用的是引擎域名**（如 `kineto-api.<YOUR_DOMAIN>`，由设备侧 Cloudflare Tunnel
   提供，见 `DEPLOY_MOFANG.md` §5.9 / `deploy/cloudflared/README.md`），**与前端域名是两个不同的域名**，
   不要混淆。

---

## 5. 设备侧 CORS：功能上非必需，但启动守卫仍要求填值（纵深防御）

在新代理架构下，浏览器只与前端**同源** `/api/*` 通信、**不再直连引擎**（Zeabur 服务端→引擎是服务器间调用，不触发浏览器 CORS），因此 `KINETO_CORS_ORIGINS` 在**功能上已不依赖**。

**但是**，设备侧两处启动守卫仍**硬性要求 `KINETO_CORS_ORIGINS` 为非空的精确 origin**（纵深防御）：
- `deploy/docker-compose.yml` → `${KINETO_CORS_ORIGINS:?请设置...}`（空值时 `docker compose up` 直接报错退出）
- `deploy/kineto-engine.service` → `ExecStartPre ... grep -Eq "^KINETO_CORS_ORIGINS=https?://"`（空值时 systemd 服务启动失败）

因此你**必须**把前端 Zeabur 的精确 origin 填入设备侧的 `KINETO_CORS_ORIGINS`：
```
KINETO_CORS_ORIGINS=https://kineto.<你的前端域名>
```
多个 origin 用逗号分隔；必须带 scheme（`https://`）、不带尾斜杠/路径。

> ⚠️ **留空会导致 `docker compose up` 报错或 systemd 服务启动失败。** 不要留空。

> 结论：代理架构下浏览器不再直连引擎，CORS **功能上非必需**；但启动守卫仍要求填入前端精确 origin（纵深防御），照填即可。

---

## 6. 部署后验证（必做）

### 6.1 手工验证（浏览器）

1. **无 jobId（离线/样例态）**：打开 `https://<你的Zeabur域名>/`
   - 应看到**醒目的“当前为样例数据（内置 fixture）”横幅**（`ViewerStage` 的 `stage-banner`），
     元数据面板的来源徽标显示 **`FIXTURE`**。
   - 这说明前端本身已跑起来、3D 渲染正常（此时未连真实任务）。
2. **带真实 jobId（连通态）**：先在设备侧建好一个任务拿到 `<id>`，再打开
   ```
   https://<你的Zeabur域名>/?job=<id>
   ```
   - 元数据面板来源徽标必须是 **`LIVE API`（即 `source === "api"`）**，**不能**是 `FIXTURE`。
   - 若仍是 `FIXTURE`，看横幅/控制台里的 `fallbackReason`，按 §7 排障。
3. **上传闭环**：用页面上传面板选一个 `.mp4` → 应经同源 `/api/jobs`（`multipart`，字段名 `video`）
   创建任务并轮询到 `state=done`，随后可用 `?job=<新id>` 查看结果。

### 6.2 脚本化验证（`deploy/validate.sh` 的 **G11**）

`deploy/validate.sh` 已内置 **G11：Zeabur 前端可达性 + 同源代理链路**断言。在能访问前端的机器上：

```bash
# 前端可达性 + 同源代理 /api/health 链路（推荐用法）
bash deploy/validate.sh --web-base https://kineto.<你的域名>

# 仅当你仍保留“浏览器→引擎直连”时，额外做 CORS 预检断言：
bash deploy/validate.sh --web-base https://kineto.<你的域名> \
                        --web-origin https://kineto.<你的域名>
```

G11 会检查：
- **G11a** 前端站点可达（HTTP 2xx/3xx）；
- **G11b** 同源代理 `${WEB_BASE}/api/health` 返回 **200**（经服务端注入密钥打到引擎 `/health`）；
  - 若返回 **503** → 前端**未配置** `ENGINE_API_BASE` / `KINETO_API_KEY`（回 §3）；
  - 若返回 **502** → 前端已配置但**引擎不可达**（回 §7、查隧道）；
- **G11c**（仅当给了 `--web-origin`）→ CORS 预检断言（新代理架构下通常**不需要**）。

> 也可用环境变量替代命令行参数：`KINETO_WEB_BASE` / `KINETO_WEB_ORIGIN`。
> 未提供 `--web-base` 时 G11 记为 **SKIP**（不影响其它门禁）。

---

## 7. 排障速查

| 现象 | 大概率原因 | 处理 |
|---|---|---|
| Zeabur 构建失败：找不到 `package.json` / `npm ci` 报错 | **Root Directory 没填或填错** | 改为 `Projects/Kineto/kineto-web`（§1），重新部署 |
| 构建失败：Node 版本过低 / 语法不支持 | Node < 18.17 | 指定 Node 20（`.nvmrc`），重构建 |
| 页面能开，但 `?job=<id>` 仍是 `FIXTURE`，代理返回 **503** | 控制台**未配** `ENGINE_API_BASE` / `KINETO_API_KEY` | 在 Zeabur 控制台补齐两个**服务端**变量（§3），重新部署 |
| 代理返回 **502 `engine unreachable`** | 前端配置对，但引擎不可达 | 查设备侧 Cloudflare Tunnel 是否在线、`ENGINE_API_BASE` 域名是否正确、引擎 `/healthz` 是否 200 |
| 代理返回 **401 / 403** | `KINETO_API_KEY` 与设备侧**不是同一把** | 用 `/etc/kineto/kineto-engine.env` 里那把，重填控制台变量并重部署 |
| 改了 `NEXT_PUBLIC_*` 但不生效 | `NEXT_PUBLIC_` 是**构建期内联** | 必须**重新部署（重新构建）**，运行时改无效 |
| 浏览器抓包看到了密钥 | 误用了 `NEXT_PUBLIC_API_KEY` | **立即轮换密钥**；改为无前缀的 `KINETO_API_KEY`（§3 铁律） |
| 上传 415 / 400 | multipart 字段名不是 `video` | 契约字段名固定为 `video`（代理透传，勿改） |

---

## 8. 与设备侧的一致性检查清单

- [ ] `ENGINE_API_BASE` 的域名 = 设备侧 Cloudflare Tunnel 暴露的引擎域名（`kineto-api.<YOUR_DOMAIN>`）。
- [ ] `KINETO_API_KEY` = 设备侧 `/etc/kineto/kineto-engine.env` 里的**同一把**（`DEPLOY_MOFANG.md` §5.8）。
- [ ] `KINETO_API_KEY` / `ENGINE_API_BASE` **无** `NEXT_PUBLIC_` 前缀。
- [ ] 设备侧引擎已启用 **`KINETO_STRICT=1`**（`kineto-engine.service` / `docker-compose.yml` / `Dockerfile.engine` / `deploy/.env.example` 均已内置；见 `DEPLOY_MOFANG.md` §1.2 / §5.8）：
  - STRICT 只守“假数据/缺权重/detector 降级”的 fail-fast（退出码 3 = StrictModeRefused）；
  - ❗ STRICT **不再对质量做门禁**，质量由 **`KINETO_QUALITY_GATE`**（默认 warn）独立控制。
  - `warn`（默认）= 质量不达标时 job 仍 `done` 但响应体附 `degraded=true` + `quality_warning` 字段，前端可据此展示警告横幅；
  - `fail` = 不达标时 job 判 `failed`；`off` = 从不门禁。
  - 配合 api.py 的 `extraction_mode != '4dhumans'` 硬门禁，前端因此**绝不会**把「合成假姿态」当成 `LIVE API` 真实结果展示。
- [ ] Root Directory = `Projects/Kineto/kineto-web`。
- [ ] Node ≥ 18.17（推荐 20）。
- [ ] `https://<前端域名>/?job=<id>` 的来源徽标 = `LIVE API`（`api`），非 `FIXTURE`。
- [ ] `bash deploy/validate.sh --web-base https://<前端域名>` 的 **G11 = PASS**。
- [ ] （仅老直连路径）设备侧 `KINETO_CORS_ORIGINS` 回填了前端精确 origin；代理架构下此项**可跳过**。
- [ ] 前端已接线 `degraded` 横幅（`UploadPanel` / `ViewerStage`）：质量降级时展示告警。
- [ ] `PoseMetadata` 已声明 `joint_order?: string` / `schema_version?: number`（additive 可选，向后兼容）。
- [ ] `validatePoseData` 已改为全帧校验（非只校验首帧）。

---

*本文只覆盖前端上云；引擎/驱动/MoFang 剥离/隧道等见 [`DEPLOY_MOFANG.md`](./DEPLOY_MOFANG.md) 与 [`deploy/`](./deploy/)。*
