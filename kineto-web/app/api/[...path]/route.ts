/**
 * app/api/[...path]/route.ts
 * ─────────────────────────────────────────────────────────────────────────
 * 服务端代理（Next.js Route Handler）——云边架构的安全边界。
 *
 * 浏览器只与【同源】的 `/api/*` 通信；本代理在服务端把请求转发到引擎基址，
 * 并在【服务端】注入 `X-API-Key`。浏览器从不直连引擎、从不持有任何密钥。
 *
 * 服务端环境变量（均无 NEXT_PUBLIC_ 前缀，构建期不会内联进浏览器包）：
 *   - ENGINE_API_BASE : 引擎基址，如 https://kineto-api.<DOMAIN>
 *   - KINETO_API_KEY  : 访问引擎所需的密钥（作为 X-API-Key 发送）
 *
 * 未配置任一时返回 503 + 清晰 JSON（不抛异常、不崩溃页面），
 * 由前端上层优雅回退到 fixture 样例数据。
 *
 * 转发要点：
 *   - MJ3(b) 方法+路径白名单：仅放行 POST /jobs、GET /jobs/{id}、
 *     GET /jobs/{id}/pose_data.json、GET /jobs/{id}/mesh_vertices.f32、
 *     GET /jobs/{id}/demo_output.mp4、
 *     GET /jobs/{id}/input.mp4、GET /jobs/{id}/annotated_output.mp4、GET /health、GET /healthz；其余一律 404（防止代理沦为任意 URL 中继）；
 *   - MJ3(a) per-IP 令牌桶限流：POST /jobs 每 IP 每 10min 最多 3 次，超限 429 +
 *     Retry-After（内存态，单实例级；多实例部署下为已接受残留风险）；
 *   - MJ3(c) 最小头转发：绝不把客户端 Cookie/Authorization 透传给引擎，仅转发
 *     multipart 必需的 content-type 与 accept；X-API-Key 由服务端 set 覆盖（客户端无法伪造）；
 *   - POST multipart 直接透传 request.body（ReadableStream）以保持 boundary，不用
 *     formData() 重编码；上游 status / body / content-type 原样返回（GET 亦为流式透传）。
 */

import { NextRequest, NextResponse } from "next/server";

// 该代理依赖 Node 运行时（ReadableStream 透传、服务端 env），不可用 Edge。
export const runtime = "nodejs";
// 每次请求都实时读取 env 与转发，禁止静态化/缓存。
export const dynamic = "force-dynamic";

/** 逐跳（hop-by-hop）头，不应在代理间转发。 */
const HOP_BY_HOP = new Set([
  "host",
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  // 流式透传时长度由传输编码决定，转发原始 content-length 可能造成不匹配。
  "content-length",
]);

interface ProxyResult {
  response: NextResponse | Response;
}

/**
 * MJ3(b)：方法 + 路径白名单。仅放行契约内的端点，其余 404。
 * path 为 catch-all 拆分后的段（如 /api/jobs/{id} → ["jobs", "{id}"]）。
 */
function isAllowedRoute(method: string, path: string[]): boolean {
  const p = path.filter((s) => s.length > 0);
  if (method === "POST") {
    return p.length === 1 && p[0] === "jobs";
  }
  if (method === "GET") {
    if (p.length === 1 && (p[0] === "health" || p[0] === "healthz")) return true;
    if (p.length === 2 && p[0] === "jobs") return true; // /jobs/{id}
    if (
      p.length === 3 &&
      p[0] === "jobs" &&
      (p[2] === "pose_data.json" || p[2] === "demo_output.mp4" || p[2] === "input.mp4" || p[2] === "annotated_output.mp4" || p[2] === "mesh_vertices.f32" || /^grid_\d{2}\.jpg$/.test(p[2]))
    ) {
      return true; // /jobs/{id}/{pose_data.json|demo_output.mp4|input.mp4|annotated_output.mp4|mesh_vertices.f32|grid_XX.jpg}
    }
  }
  return false;
}

/** MJ3(a)：从 x-forwarded-for 首跳读取客户端 IP，做多级兜底。 */
function clientIp(request: NextRequest): string {
  const xff = request.headers.get("x-forwarded-for");
  if (xff) {
    const first = xff.split(",")[0]?.trim();
    if (first) return first;
  }
  const realIp = request.headers.get("x-real-ip");
  if (realIp && realIp.trim()) return realIp.trim();
  return "unknown";
}

/**
 * MJ3(a)：per-IP 令牌桶限流（内存态，单实例级）。
 * POST /jobs：每 IP 每 10min 最多 3 次。
 */
const JOB_RATE_LIMIT = 3;
const JOB_RATE_WINDOW_MS = 10 * 60 * 1000;
interface Bucket {
  tokens: number;
  updatedAt: number;
}
const jobBuckets = new Map<string, Bucket>();

function consumeJobToken(ip: string): { allowed: boolean; retryAfterSec: number } {
  const now = Date.now();
  // 轻量清理：防止 Map 无界增长。
  if (jobBuckets.size > 1000) {
    for (const [key, b] of jobBuckets) {
      if (now - b.updatedAt > JOB_RATE_WINDOW_MS) jobBuckets.delete(key);
    }
  }
  let bucket = jobBuckets.get(ip);
  if (!bucket) {
    bucket = { tokens: JOB_RATE_LIMIT, updatedAt: now };
    jobBuckets.set(ip, bucket);
  }
  // 按经过时间线性补充令牌（上限 = JOB_RATE_LIMIT）。
  const elapsed = now - bucket.updatedAt;
  bucket.tokens = Math.min(
    JOB_RATE_LIMIT,
    bucket.tokens + (elapsed / JOB_RATE_WINDOW_MS) * JOB_RATE_LIMIT,
  );
  bucket.updatedAt = now;

  if (bucket.tokens >= 1) {
    bucket.tokens -= 1;
    return { allowed: true, retryAfterSec: 0 };
  }
  const missing = 1 - bucket.tokens;
  const retryAfterSec = Math.max(
    1,
    Math.ceil((missing / JOB_RATE_LIMIT) * JOB_RATE_WINDOW_MS * 0.001),
  );
  return { allowed: false, retryAfterSec };
}

/**
 * 核心转发逻辑，GET/POST 共用。
 * @returns 上游响应，或在白名单外/超限/未配置/不可达时的清晰 JSON 错误响应。
 */
async function proxy(request: NextRequest, path: string[]): Promise<ProxyResult> {
  const method = request.method;

  // MJ3(b)：方法 + 路径白名单，其余一律 404。
  if (!isAllowedRoute(method, path)) {
    return { response: NextResponse.json({ error: "not found" }, { status: 404 }) };
  }

  // MJ3(a)：POST /jobs 做 per-IP 令牌桶限流。
  if (method === "POST") {
    const { allowed, retryAfterSec } = consumeJobToken(clientIp(request));
    if (!allowed) {
      return {
        response: NextResponse.json(
          { error: "rate limited", detail: "上传过于频繁，请稍后再试", retry_after: retryAfterSec },
          { status: 429, headers: { "Retry-After": String(retryAfterSec) } },
        ),
      };
    }
  }

  const engineBase = process.env.ENGINE_API_BASE;
  const apiKey = process.env.KINETO_API_KEY;

  // 未配置引擎 → 503 + JSON（不抛异常）。前端据此优雅回退 fixture。
  if (!engineBase || !apiKey) {
    return {
      response: NextResponse.json(
        { error: "engine not configured" },
        { status: 503 },
      ),
    };
  }

  const base = engineBase.replace(/\/+$/, "");
  const search = request.nextUrl.search ?? "";
  const url = `${base}/${path.join("/")}${search}`;

  // MJ3(c)：最小头转发——绝不把客户端 Cookie/Authorization 透传给引擎；
  // 仅转发 multipart 必需的 content-type 与 accept。X-API-Key 由服务端 set 注入，
  // 客户端即使伪造 X-API-Key 也不会被拷贝，无法覆盖服务端密钥。
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const accept = request.headers.get("accept");
  if (accept) headers.set("accept", accept);
  // 服务端注入密钥——浏览器永远看不到它。
  headers.set("X-API-Key", apiKey);

  const init: RequestInit & { duplex?: "half" } = { method, headers };
  // 带 body 的方法直接透传 ReadableStream，保持 multipart boundary 不被重编码。
  if (method !== "GET" && method !== "HEAD") {
    init.body = request.body;
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (err) {
    // 引擎不可达 → 502 + JSON，同样不崩溃。
    return {
      response: NextResponse.json(
        { error: "engine unreachable", detail: err instanceof Error ? err.message : String(err) },
        { status: 502 },
      ),
    };
  }

  // 复制上游响应头，剔除逐跳头，其余（content-type 等）原样回传。
  const respHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) respHeaders.set(key, value);
  });

  // 流式透传上游 body（适用于 JSON 与 mp4 下载），保持 status/statusText。
  return {
    response: new NextResponse(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: respHeaders,
    }),
  };
}

type Ctx = { params: { path: string[] } };

export async function GET(request: NextRequest, { params }: Ctx) {
  const { response } = await proxy(request, params.path ?? []);
  return response;
}

export async function POST(request: NextRequest, { params }: Ctx) {
  const { response } = await proxy(request, params.path ?? []);
  return response;
}
