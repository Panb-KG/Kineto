/**
 * lib/api.ts
 * ─────────────────────────────────────────────────────────────────────────
 * 后端任务 API 客户端（浏览器侧）。
 *
 * 云边架构（修复后）：
 *   浏览器 → 同源 Next.js Route Handler `/api/*`（app/api/[...path]/route.ts）
 *          → 服务端代理转发到引擎基址，并在【服务端】注入 X-API-Key。
 *
 * 安全要点：
 *   - 浏览器【不再】直连引擎、【不再】持有任何密钥；
 *   - X-API-Key 由服务端代理注入，此文件绝不附加任何鉴权头；
 *   - API base 默认同源 `/api`（经代理）；如需自定义可设 NEXT_PUBLIC_API_BASE。
 *
 * 端点（经代理，字段名以引擎契约为准）：
 *   POST {BASE}/jobs   (multipart, 字段名 `video`)  → 202 { job_id }
 *   GET  {BASE}/jobs/{id}                            → JobStatus（state: queued|running|done|failed）
 *   GET  {BASE}/jobs/{id}/pose_data.json             → PoseData
 */

import type { JobStatus, PoseData } from "./types";

/**
 * 浏览器侧 API 基址。
 * 默认走同源代理 `/api`；仅在需要指向其他同源前缀时才设 NEXT_PUBLIC_API_BASE。
 * 注意：这里【不是】引擎地址，也【不含】任何密钥——引擎地址与密钥都在服务端。
 */
const RAW_API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";
export const API_BASE = (RAW_API_BASE || "/api").replace(/\/+$/, "");

// MJ2(d)：NEXT_PUBLIC_API_BASE 若为旧式绝对 URL（含 "://"），浏览器会绕过同源
// 代理直连引擎——而引擎需要 X-API-Key（浏览器不持有），大概率 401。默认应留空走 /api。
if (RAW_API_BASE.includes("://") && process.env.NODE_ENV !== "production") {
  // eslint-disable-next-line no-console
  console.warn(
    `[kineto-web] NEXT_PUBLIC_API_BASE 检测到绝对 URL（${RAW_API_BASE}）。` +
      "默认已改为同源 /api 代理；绝对 URL 会让浏览器直连引擎并因缺少 X-API-Key 触发 401。" +
      "如无特殊需要，请留空该变量以走同源代理。",
  );
}

/**
 * 浏览器侧是否应尝试调用后端 API。
 * 采用同源代理后基址默认恒为 `/api`，故通常返回 true；保留此函数供
 * lib/poseData.ts 作为「是否尝试 API、否则直接 fixture」的门控。
 * 引擎是否真正可用由代理在运行时判定（未配置 ENGINE_API_BASE / KINETO_API_KEY
 * 时代理返回 503，上层优雅回退 fixture）。
 */
export function isApiConfigured(): boolean {
  return API_BASE.length > 0;
}

/** API 层错误，携带状态码便于上层区分「未配置 / 不可达 / 4xx / 5xx」。 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * MJ2(a)：轮询超时错误（可区分于网络离线），status=408。
 * 调用方据此保留 jobId 并提示「任务仍在处理中」，而非误判为「未连接引擎」。
 */
export class ApiTimeoutError extends ApiError {
  constructor(message = "轮询超时", readonly timeoutMs?: number) {
    super(message, 408);
    this.name = "ApiTimeoutError";
  }
}

/**
 * MJ2(b)：是否为「真正的网络失败」（fetch 抛 TypeError，如 failed to fetch / 断网 / DNS）。
 * 仅此类才算离线；超时(408)、503、其他 4xx 等均不等同于离线。
 */
export function isNetworkError(err: unknown): boolean {
  if (err instanceof ApiError) {
    const c = err.cause;
    return c instanceof TypeError || (c instanceof Error && c.name === "TypeError");
  }
  return err instanceof TypeError;
}

/**
 * MJ2(c)：从非 ok 响应中解析 { detail } 或 { error }，拼进错误消息，
 * 使 429/507/413/503 等原因能透传给用户。解析失败则回退到 fallback。
 */
async function reasonFromResponse(res: Response, fallback: string): Promise<string> {
  try {
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      const data = (await res.json()) as { detail?: unknown; error?: unknown };
      const reason = data?.detail ?? data?.error;
      if (typeof reason === "string" && reason.trim()) return `${fallback}：${reason.trim()}`;
      if (reason) return `${fallback}：${JSON.stringify(reason)}`;
    } else {
      const text = (await res.text()).trim();
      if (text) return `${fallback}：${text.slice(0, 200)}`;
    }
  } catch {
    /* body 非 JSON / 已消费 / 为空——忽略，使用 fallback */
  }
  return fallback;
}

/**
 * 拉取指定任务的姿态数据 JSON。
 * @param jobId 任务 ID
 * @param signal 可选的中止信号
 */
export async function fetchPoseData(
  jobId: string,
  signal?: AbortSignal,
): Promise<PoseData> {
  const url = `${API_BASE}/jobs/${encodeURIComponent(jobId)}/pose_data.json`;
  let res: Response;
  try {
    // 浏览器侧不附加任何鉴权头；X-API-Key 由服务端代理注入。
    res = await fetch(url, { signal, cache: 'no-store' });
  } catch (err) {
    throw new ApiError(`无法连接后端: ${url}`, undefined, err);
  }
  if (!res.ok) {
    throw new ApiError(
      await reasonFromResponse(res, `获取 pose_data 失败 (${res.status})`),
      res.status,
    );
  }
  return (await res.json()) as PoseData;
}

/**
 * [P1 mesh 节奏贴合] 拉取 SMPL 顶点二进制（mesh_vertices.f32）。
 *
 * 布局：帧数×vertexCount×3 float32 LE（相机系坐标），与 keyframes 1:1。
 * 字节数不符时抛错（契约被破坏不应静默渲染错数据）。
 */
export async function fetchMeshVerticesRaw(
  jobId: string,
  frameCount: number,
  vertexCount: number,
  signal?: AbortSignal,
): Promise<Float32Array> {
  const url = `${API_BASE}/jobs/${encodeURIComponent(jobId)}/mesh_vertices.f32`;
  let res: Response;
  try {
    res = await fetch(url, { signal, cache: "no-store" });
  } catch (err) {
    throw new ApiError(`无法连接后端: ${url}`, undefined, err);
  }
  if (!res.ok) {
    throw new ApiError(
      await reasonFromResponse(res, `获取 mesh_vertices.f32 失败 (${res.status})`),
      res.status,
    );
  }
  const buf = await res.arrayBuffer();
  const expected = frameCount * vertexCount * 3 * 4;
  if (buf.byteLength !== expected) {
    throw new ApiError(
      `mesh_vertices.f32 大小不符：${buf.byteLength}B ≠ 预期 ${expected}B` +
        `（${frameCount}×${vertexCount}×3×float32）`,
    );
  }
  return new Float32Array(buf);
}

/** 轮询单个任务状态。 */
export async function getJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<JobStatus> {
  const url = `${API_BASE}/jobs/${encodeURIComponent(jobId)}`;
  let res: Response;
  try {
    res = await fetch(url, { signal, cache: 'no-store' });
  } catch (err) {
    throw new ApiError(`无法连接后端: ${url}`, undefined, err);
  }
  if (!res.ok) {
    throw new ApiError(
      await reasonFromResponse(res, `查询任务失败 (${res.status})`),
      res.status,
    );
  }
  return (await res.json()) as JobStatus;
}

/**
 * 上传视频创建任务（POST /jobs，multipart/form-data）。
 *
 * 契约：multipart 字段名为 `video`（不是 file）；引擎返回 202 `{ job_id }`。
 * 通过同源代理透传原始 multipart body，保留 boundary；不手动设置 Content-Type。
 *
 * @param file 视频文件（如 .mp4）
 * @returns 新建任务的 job_id
 */
export async function uploadJob(file: File): Promise<string> {
  const form = new FormData();
  // 契约字段名 = video
  form.append("video", file);

  const url = `${API_BASE}/jobs`;
  let res: Response;
  try {
    // 不手动设置 Content-Type，交由浏览器附带 multipart boundary；
    // 不附加 X-API-Key，由服务端代理注入。
    res = await fetch(url, { method: "POST", body: form });
  } catch (err) {
    throw new ApiError(`上传失败: ${url}`, undefined, err);
  }
  if (!res.ok) {
    throw new ApiError(
      await reasonFromResponse(res, `上传失败 (${res.status})`),
      res.status,
    );
  }
  const data = (await res.json()) as { job_id?: string; id?: string };
  const jobId = data.job_id ?? data.id;
  if (!jobId) throw new ApiError("上传响应缺少 job_id");
  return jobId;
}

/**
 * 轮询任务直到完成或超时。
 *
 * 契约：读 `job.state`，成功判定 `"done"`、失败判定 `"failed"`
 * （后端不返回 status/completed）。
 *
 * @param jobId     任务 ID
 * @param options   intervalMs 轮询间隔；timeoutMs 超时；signal 中止；onProgress 进度回调
 */
export async function pollJobUntilDone(
  jobId: string,
  options: {
    intervalMs?: number;
    timeoutMs?: number;
    signal?: AbortSignal;
    onProgress?: (job: JobStatus) => void;
  } = {},
): Promise<JobStatus> {
  const { intervalMs = 2000, timeoutMs = 120_000, signal, onProgress } = options;
  const startedAt = Date.now();

  // eslint-disable-next-line no-constant-condition
  while (true) {
    if (signal?.aborted) throw new ApiError("轮询已中止");
    const job = await getJob(jobId, signal);
    onProgress?.(job);
    if (job.state === "done") return job;
    if (job.state === "failed") {
      throw new ApiError(job.error ?? "任务处理失败");
    }
    if (Date.now() - startedAt > timeoutMs) {
      throw new ApiTimeoutError(
        `任务处理超过 ${Math.round(timeoutMs / 60000)} 分钟仍未完成`,
        timeoutMs,
      );
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
