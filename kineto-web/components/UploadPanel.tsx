/**
 * components/UploadPanel.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 极简上传闭环（客户端组件）：选择 .mp4 → uploadJob → pollJobUntilDone
 * → 成功后把地址切到 `?job=<job_id>`，由现有 <ViewerStage/> 加载真实数据。
 *
 * 优雅降级：当同源代理返回 503（未配置引擎）/ 502（不可达）或真正网络异常时，
 * 只在面板内显示清晰的内联提示，绝不破坏下方 fixture 3D 查看器。
 *
 * MJ1（超时与离线区分）：引擎处理最长可达 2h，轮询超时默认 1h（可经 prop/env
 * 配置）。超时时【不】误判为离线，而是保留 jobId 到组件 state 与 URL(?job=<id>)，
 * 渲染「任务仍在处理中，可继续等待或稍后重连」并提供「继续等待」按钮恢复轮询。
 * 视觉与「Clinical Editorial」医疗极简风格保持一致。
 */

"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiTimeoutError,
  isNetworkError,
  pollJobUntilDone,
  uploadJob,
} from "../lib/api";
import type { JobStatus } from "../lib/types";

type UploadPhase =
  | { kind: "idle" }
  | { kind: "uploading" }
  | { kind: "processing"; job: JobStatus | null }
  | { kind: "done"; jobId: string }
  | { kind: "timeout"; jobId: string }
  | { kind: "offline"; message: string }
  | { kind: "error"; message: string };

/**
 * 是否为「离线/引擎不可用」（可优雅降级到样例数据）。
 * MJ2(b)：仅 503（引擎未配置）、502（不可达）或真正的网络失败(fetch TypeError)
 * 才算离线；超时(408) 与其他 4xx（如 429/413）绝不判为离线。
 */
function isOfflineError(err: unknown): boolean {
  if (!(err instanceof ApiError)) return false;
  if (err instanceof ApiTimeoutError) return false;
  if (err.status === 503 || err.status === 502) return true;
  return isNetworkError(err);
}

/** MJ1：轮询间隔与超时（可经 prop 或 env 覆盖）。引擎处理最长可达 2h。 */
const POLL_INTERVAL_MS = 2000;
const DEFAULT_POLL_TIMEOUT_MS = 3_600_000; // 1h

function resolveTimeoutMs(propMs?: number): number {
  if (typeof propMs === "number" && propMs > 0) return propMs;
  const env = Number(process.env.NEXT_PUBLIC_POLL_TIMEOUT_MS);
  if (Number.isFinite(env) && env > 0) return env;
  return DEFAULT_POLL_TIMEOUT_MS;
}

interface UploadPanelProps {
  /** 轮询超时（ms），默认 1h；亦可经 NEXT_PUBLIC_POLL_TIMEOUT_MS 覆盖。 */
  timeoutMs?: number;
}

export default function UploadPanel({ timeoutMs: timeoutProp }: UploadPanelProps = {}) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<UploadPhase>({ kind: "idle" });
  const [fileName, setFileName] = useState<string>("");

  const busy = phase.kind === "uploading" || phase.kind === "processing";
  const timeoutMs = resolveTimeoutMs(timeoutProp);

  const gotoJob = useCallback(
    (jobId: string) => router.replace(`/?job=${encodeURIComponent(jobId)}`),
    [router],
  );

  // 轮询任务；被上传流程与「继续等待」复用。自身消化异常，不外抛。
  const startPolling = useCallback(
    async (jobId: string) => {
      setPhase({ kind: "processing", job: null });
      // MJ1：立即把 jobId 写入 URL，超时/刷新后仍可恢复。
      gotoJob(jobId);
      try {
        const done = await pollJobUntilDone(jobId, {
          intervalMs: POLL_INTERVAL_MS,
          timeoutMs,
          onProgress: (job) => setPhase({ kind: "processing", job }),
        });
        setPhase({ kind: "done", jobId });
        gotoJob(jobId);
        void done;
      } catch (err) {
        // MJ1：单纯超时不得判为离线——保留 jobId，提示可继续等待/稍后重连。
        if (err instanceof ApiTimeoutError) {
          setPhase({ kind: "timeout", jobId });
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        if (isOfflineError(err)) setPhase({ kind: "offline", message });
        else setPhase({ kind: "error", message });
      }
    },
    [gotoJob, timeoutMs],
  );

  const handleFile = useCallback(
    async (file: File) => {
      setFileName(file.name);
      setPhase({ kind: "uploading" });
      try {
        const jobId = await uploadJob(file);
        await startPolling(jobId);
      } catch (err) {
        // 仅上传阶段的错误（startPolling 自行消化轮询异常）。
        const message = err instanceof Error ? err.message : String(err);
        if (isOfflineError(err)) setPhase({ kind: "offline", message });
        else setPhase({ kind: "error", message });
      } finally {
        // 允许重复选择同一文件。
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [startPolling],
  );

  const resume = useCallback(() => {
    if (phase.kind === "timeout") void startPolling(phase.jobId);
  }, [phase, startPolling]);

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void handleFile(file);
  };

  const progressPct = (() => {
    if (phase.kind !== "processing" || !phase.job) return null;
    const p = phase.job.progress;
    if (typeof p !== "number") return null;
    // 兼容 0..1 与 0..100 两种刻度。
    const pct = p <= 1 ? p * 100 : p;
    return Math.max(0, Math.min(100, Math.round(pct)));
  })();

  return (
    <section className="upload-panel" aria-label="上传教学视频">
      <div className="upload-panel__head">
        <h2 className="upload-panel__title">上传教学视频 · Ingest</h2>
        <p className="upload-panel__desc">
          选择真人教学视频（.mp4），经 4DHumans 逆向解析为三维姿态后自动加载到下方视图。
        </p>
      </div>

      <div className="upload-panel__body">
        <label className={`upload-drop${busy ? " upload-drop--busy" : ""}`}>
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,.mp4"
            onChange={onChange}
            disabled={busy}
            className="upload-drop__input"
          />
          <span className="upload-drop__icon" aria-hidden>
            ⭱
          </span>
          <span className="upload-drop__text">
            {busy ? "处理中…" : fileName ? fileName : "点击选择 .mp4 视频"}
          </span>
        </label>

        <div className="upload-status" role="status" aria-live="polite">
          {phase.kind === "uploading" && (
            <span className="upload-note">
              <span className="spinner spinner--sm" aria-hidden /> 正在上传视频…
            </span>
          )}

          {phase.kind === "processing" && (
            <span className="upload-note">
              <span className="spinner spinner--sm" aria-hidden />
              引擎解析中
              {phase.job?.state ? ` · ${phase.job.state}` : ""}
              {progressPct !== null ? ` · ${progressPct}%` : ""}
            </span>
          )}

          {progressPct !== null && (
            <span className="upload-bar" aria-hidden>
              <span style={{ width: `${progressPct}%` }} />
            </span>
          )}

          {phase.kind === "done" && (
            <span className="upload-note upload-note--ok">
              ✓ 任务完成，正在加载姿态数据（job={phase.jobId.slice(0, 8)}…）
            </span>
          )}

          {phase.kind === "timeout" && (
            <div className="upload-note upload-note--warn upload-note--col">
              <span>任务仍在处理中，可继续等待或稍后重连。</span>
              <span className="upload-note__detail">job={phase.jobId}</span>
              <button type="button" className="upload-resume" onClick={resume}>
                继续等待
              </button>
            </div>
          )}

          {phase.kind === "offline" && (
            <span className="upload-note upload-note--warn">
              未连接引擎，正在展示样例数据。
              <span className="upload-note__detail">{phase.message}</span>
            </span>
          )}

          {phase.kind === "error" && (
            <span className="upload-note upload-note--err">
              处理失败：{phase.message}
            </span>
          )}
        </div>
      </div>
    </section>
  );
}
