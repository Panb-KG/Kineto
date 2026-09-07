# Cloudflare Tunnel — Kineto Engine 的公网出口

## 为什么这一步是**必需**的

| 事实 | 后果 |
|---|---|
| `deploy/kineto-engine.service` 里 uvicorn `--host 127.0.0.1` | API 只在设备回环上监听，局域网内其它机器都访问不到 |
| `deploy/docker-compose.yml` 故意**不发布端口** | 容器路径同样没有入站入口 |
| MoFang M01 / Intel AI Box 在客户局域网内，无公网 IP | 无法直接 `https://<ip>:8000` 访问 |
| 前端部署在 Zeabur（公网 Serverless） | 必须由公网侧主动调用设备 API |

Cloudflare Tunnel 由设备**出站**建立到 Cloudflare 边缘的加密长连接，因此：

- 不需要公网 IP、不需要路由器端口转发、不需要动设备上的 nginx；
- 设备不新增任何监听端口（`ss -tlnp` 里 8000 仍然只在 `127.0.0.1`）；
- 对外只有一个 HTTPS hostname：`https://kineto-api.<YOUR_DOMAIN>`；
- 兜底 ingress 规则是 `http_status:404`，不会把 nginx / MoFang / SSH 漏到公网。

> **为什么选 Cloudflare Tunnel 而不是 FRP？** 两者都能把设备回环端口暴露到公网，但 Cloudflare Tunnel
> **无需公网 IP、无需入站端口转发、无需改设备 nginx**（纯出站长连接）且自带 HTTPS/证书；
> **FRP 则需要一台有公网 IP 的中转机**（自建 frps）并自行维护端口与 TLS。本项目边缘设备在客户局域网内、
> 无公网 IP，故选 Cloudflare Tunnel；仅当你已有公网中转机且不想依赖 Cloudflare 时，FRP 才是备选。

最终链路（**新代理架构**：浏览器不再直连引擎）：

```
浏览器  ──同源──►  Zeabur 前端（Next.js 服务端代理 /api/*）
                        │  服务端读 ENGINE_API_BASE + KINETO_API_KEY，注入 X-API-Key
                        ▼  （ENGINE_API_BASE=https://kineto-api.<YOUR_DOMAIN>）
                 Cloudflare Edge ──(出站隧道)──► 设备上的 cloudflared
                                                      │  http://127.0.0.1:8000
                                                      ▼
                                              kineto-engine (uvicorn / FastAPI)
```

---

## 1. 前置条件

1. 一个托管在 **Cloudflare DNS** 上的域名（`<YOUR_DOMAIN>`，例如 `example.com`）。
   没有域名 → 可先用临时隧道（见 §6），但 URL 每次重启都变，**不适合生产**。
2. Cloudflare 账号对该域名有编辑权限（Zero Trust 免费额度即可，最多 50 用户）。
3. 设备上 `kineto-engine.service` 已能 `curl -fsS http://127.0.0.1:8000/healthz`（公开存活探针；`/health` 现需带 `X-API-Key`）。
4. 设备能出站访问 `*.cloudflare.com:443`（用 `deploy/discover_device.sh` §9 的连通性检查确认）。

---

## 2. 安装 cloudflared（设备端，官方 apt 源）

```bash
# 以下命令会安装软件包，执行前请自行确认；均需 root
sudo mkdir -p --mode=0755 /etc/apt/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /etc/apt/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/etc/apt/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
cloudflared --version
```

> 备选（无 apt 源访问权限）：直接下载 deb
> `curl -L -o cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared.deb`

---

## 3. 登录 + 创建隧道 + 绑定 DNS（三条关键命令）

```bash
# 3.1 浏览器授权（会打开 Cloudflare 登录页，选中你的域名）
#     成功后生成 /etc/cloudflared/cert.pem
sudo cloudflared tunnel login

# 3.2 创建隧道，记下输出的 UUID（形如 1a2b3c4d-...）
#     同时生成 /etc/cloudflared/<UUID>.json（隧道凭据，等同于密码，切勿外传/入库）
sudo cloudflared tunnel create kineto-engine

# 3.3 在 Cloudflare DNS 里自动创建 CNAME:
#     kineto-api.<YOUR_DOMAIN>  ->  <UUID>.cfargotunnel.com
sudo cloudflared tunnel route dns kineto-engine kineto-api.<YOUR_DOMAIN>
```

**关于 DNS CNAME 这一步**：`tunnel route dns` 会自动在托管区里写入一条
`kineto-api` 的 CNAME 记录指向 `<UUID>.cfargotunnel.com`。如果你想手工加：

| 字段 | 值 |
|---|---|
| Type | `CNAME` |
| Name | `kineto-api` |
| Target | `<TUNNEL_UUID>.cfargotunnel.com` |
| Proxy status | **Proxied（橙色云，必须开启）** |
| TTL | Auto |

> 橙云必须是开启状态。灰云（DNS only）会把解析结果指向一个不可路由的隧道地址，
> 表现为 `502 Bad Gateway` 或连接超时。

---

## 4. 落地配置 + systemd 服务

```bash
# 4.1 用仓库模板生成实际配置（替换两个占位符）
sudo mkdir -p /etc/cloudflared
sudo cp deploy/cloudflared/config.yml /etc/cloudflared/config.yml
sudo sed -i \
  -e "s/__TUNNEL_UUID__/<你的UUID>/g" \
  -e "s/__YOUR_DOMAIN__/<你的域名>/g" \
  /etc/cloudflared/config.yml

# 4.2 校验 ingress 规则（不启动隧道，只做静态校验）
sudo cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate

# 4.3 收紧权限（凭据文件只允许服务账号读）
sudo useradd --system --home /etc/cloudflared --shell /usr/sbin/nologin cloudflared 2>/dev/null || true
sudo chown -R root:cloudflared /etc/cloudflared
sudo chmod 0750 /etc/cloudflared
sudo chmod 0640 /etc/cloudflared/config.yml /etc/cloudflared/*.json

# 4.4 安装并启动服务
sudo install -m 0644 deploy/cloudflared/cloudflared.service /etc/systemd/system/cloudflared.service
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared.service
sudo systemctl status cloudflared --no-pager
journalctl -u cloudflared -f      # 应看到 "Registered tunnel connection" ×4
```

---

## 5. 验证

```bash
# 5.1 设备本地（回环，绕过隧道）—— /healthz 公开存活；/health 需带 X-API-Key
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/health -H "X-API-Key: $KINETO_API_KEY"

# 5.2 从 Mac / 任意公网机器（穿隧道）—— 公开存活用 /healthz
curl -fsS https://kineto-api.<YOUR_DOMAIN>/healthz
#   详细拓扑 /health 现需鉴权：
curl -fsS https://kineto-api.<YOUR_DOMAIN>/health -H "X-API-Key: $KINETO_API_KEY"
#   期望: {"status":"ok","extraction_mode":null,"device":"xpu","model_loaded":...,"queue_depth":0}

# 5.3 鉴权生效（应返回 401）
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://kineto-api.<YOUR_DOMAIN>/jobs
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://kineto-api.<YOUR_DOMAIN>/jobs \
     -H "X-API-Key: $KINETO_API_KEY" -H 'Content-Type: application/json' \
     -d '{"video_path":"/opt/kineto/kineto-engine/input_video.mp4"}'
#   第一条期望 401，第二条期望 202

# 5.4 兜底规则（未配置的主机名应 404，而不是被转发）
curl -s -o /dev/null -w '%{http_code}\n' https://kineto-api.<YOUR_DOMAIN>/../../etc/passwd
```

前端侧（**新代理架构 —— 浏览器不再直连引擎**）：在 **Zeabur 控制台**为前端设置**服务端**环境变量
`ENGINE_API_BASE=https://kineto-api.<YOUR_DOMAIN>` 与 `KINETO_API_KEY=<与设备同一把 key>`
（**均无 `NEXT_PUBLIC_` 前缀**，由 `app/api/[...path]/route.ts` 在服务端注入 X-API-Key）。
详见根目录 [`DEPLOY_ZEABUR.md`](../../DEPLOY_ZEABUR.md)。

> ⚠️ **不要**再把密钥放进 `kineto-web/.env.local` 的 `NEXT_PUBLIC_API_KEY` —— 那会把密钥内联进浏览器包泄露。
> 由于浏览器只与前端**同源** `/api/*` 通信、不再直连引擎，CORS **功能上已非必需**；但设备侧启动守卫（compose/systemd）仍**要求 `KINETO_CORS_ORIGINS` 非空**（纵深防御），
> 必须填入前端精确 origin（如 `https://kineto.<你的域名>`）。留空会导致 `docker compose up` 报错或 systemd 服务启动失败。

---

## 6. 已知限制与对策（务必读完）

| 限制 | 影响 | 对策 |
|---|---|---|
| **上传体积上限**：Cloudflare 免费/Pro 计划请求体上限 **100 MB** | 大视频 `POST /jobs`（multipart）会被 413 拒绝 | ① 前端只上传 ≤100MB 的短片；② 或把视频先 `scp/rsync` 到设备 `/srv/kineto/inbox/`，再用 JSON 形式 `{"video_path": "/srv/kineto/inbox/x.mp4"}` 提交任务（api.py 原生支持，且不占隧道带宽） |
| **连接超时**：Cloudflare 边缘对单个请求约 100s 上限 | 推理本身是异步的，`POST /jobs` 秒回，无影响 | 保持 job 轮询模型；**不要**把推理改成同步接口 |
| **临时隧道**（`cloudflared tunnel --url http://127.0.0.1:8000`）URL 每次变化 | 只适合临时联调 | 生产必须用命名隧道 + 自有域名（本文档方案） |
| **隧道凭据 = 密钥** | `<UUID>.json` 泄露等于把设备 API 暴露到公网 | `chmod 0640` + 不入 git + 泄露后立即 `cloudflared tunnel delete` 重建 |
| **API Key 是唯一鉴权** | 隧道本身不做鉴权，公网可达 | 必须设置 `KINETO_API_KEY`（api.py 现为 fail-closed：未设 key 且未设 `KINETO_ALLOW_NO_AUTH=1` → 受保护端点直接 503，不再 WARN+放行；公开存活用 `/healthz`） |
| Cloudflare 会缓存/压缩响应 | `pose_data.json` 可能被边缘缓存 | 如需强一致，在该 hostname 上建 Cache Rule「Bypass cache」；job 产物路径含随机 job_id，实际冲突概率低 |

---

## 7. 容器化变体（可选）

`deploy/docker-compose.yml` 里带了一个 `cloudflared` 服务（`profiles: ["tunnel"]`，默认不启动）。
若要走容器：

```bash
sudo mkdir -p /srv/kineto/cloudflared
sudo cp /etc/cloudflared/config.yml /etc/cloudflared/<UUID>.json /srv/kineto/cloudflared/
# 容器网络里 engine 用服务名寻址，把 ingress 改成:
sudo sed -i 's#http://127.0.0.1:8000#http://engine:8000#' /srv/kineto/cloudflared/config.yml
cd /opt/kineto/deploy && docker compose --profile tunnel up -d
```

注意此时 **不要** 再启用宿主机上的 `cloudflared.service`（同一条隧道两个副本会互相踢连接）。

---

## 8. 运维速查

```bash
sudo systemctl restart cloudflared              # 改完 config.yml 后
sudo cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate   # 静态校验
sudo cloudflared tunnel info kineto-engine      # 隧道与连接状态
sudo cloudflared tunnel route dns kineto-engine kineto-api2.<YOUR_DOMAIN>       # 追加主机名
journalctl -u cloudflared --since "10 min ago"  # 排障
# 彻底拆除（可逆操作的终点）:
sudo systemctl disable --now cloudflared
sudo cloudflared tunnel delete kineto-engine    # 会同时删除 DNS 记录
sudo rm -f /etc/systemd/system/cloudflared.service && sudo systemctl daemon-reload
```

**故障排查**：

| 症状 | 原因 | 处理 |
|---|---|---|
| 公网 502 / `ERR_FAILED` | engine 没起来或没监听 127.0.0.1:8000 | `curl http://127.0.0.1:8000/healthz`（公开存活探针；`/health` 现需 `X-API-Key`，无密钥会 401/503），`systemctl status kineto-engine` |
| 公网 404 且 body 是 `http_status:404` | hostname 拼错，命中兜底规则 | 核对 `config.yml` 的 hostname 与 DNS 记录完全一致 |
| 1033 / `Argo Tunnel error` | 隧道未运行或凭据文件不可读 | `journalctl -u cloudflared`；检查 `/etc/cloudflared/<UUID>.json` 权限 |
| 前端经代理 502 `engine unreachable` | 隧道/engine 不可达 | `curl https://kineto-api.<YOUR_DOMAIN>/healthz`；`systemctl status cloudflared kineto-engine` |
| （仅老直连路径）前端 CORS 报错 | `KINETO_CORS_ORIGINS` 未含前端 origin | 新代理架构浏览器不直连引擎、无需 CORS；若仍直连，改 `/etc/kineto/kineto-engine.env` 后 `systemctl restart kineto-engine` |
| POST 大视频 413 | 触发 Cloudflare 100MB 上限 | 改用 `video_path` 方式（见 §6） |
