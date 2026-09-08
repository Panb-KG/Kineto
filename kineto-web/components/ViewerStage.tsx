/**
 * components/ViewerStage.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 客户端主舞台：负责数据加载（API → fixture 优雅回退）、时间轴装配，
 * 并组合上半部分关键帧卡片、下半部分 3D 视图与控制条、以及元数据面板。
 *
 * 3D 视图（依赖 WebGL/window）通过 next/dynamic 以 ssr:false 动态导入，
 * 保证服务端渲染阶段不触碰 three 的浏览器 API。
 */

"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import MetadataPanel from "./MetadataPanel";
import TimelineControls from "./TimelineControls";
import KeyframeCards from "./KeyframeCards";
import VideoCompare, { AnalysisVideo } from "./VideoCompare";
import { loadPoseData, type LoadedPoseData } from "../lib/poseData";
import { getJob } from "../lib/api";
import { useTimeline } from "../lib/useTimeline";
import { useVideoSync } from "../lib/useVideoSync";

const SkeletonViewer = dynamic(() => import("./SkeletonViewer"), {
  ssr: false,
  loading: () => <div className="viewer-skeleton-loading">初始化 3D 引擎…</div>,
});

const MeshViewer = dynamic(() => import("./MeshViewer"), {
  ssr: false,
  loading: () => <div className="viewer-skeleton-loading">初始化 Mesh 引擎…</div>,
});



const CombinedViewer = dynamic(() => import("./CombinedViewer"), {
  ssr: false,
  loading: () => <div className="viewer-skeleton-loading">初始化叠加视图…</div>,
});

/** 3D 视图渲染模式 */
type ViewMode = "skeleton" | "mesh" | "both";

interface ViewerStageProps {
  /** 可选任务 ID；缺省时离线加载内置 fixture。 */
  jobId?: string;
}

type Phase =
  | { kind: "loading" }
  | { kind: "waiting"; message: string }
  | { kind: "ready"; loaded: LoadedPoseData }
  | { kind: "error"; message: string };

export default function ViewerStage({ jobId }: ViewerStageProps) {
  const [phase, setPhase] = useState<Phase>({ kind: "loading" });
  // m13: 跟踪 job 的 degraded 状态，用于显示质量降级横幅
  const [degraded, setDegraded] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    /** 轮询等待 job 完成，间隔 2s，超时 15min */
    async function waitForJob(targetJobId: string): Promise<void> {
      const POLL_INTERVAL = 2000;
      const POLL_TIMEOUT = 900_000;
      const startedAt = Date.now();

      while (true) {
        if (controller.signal.aborted) return;
        const job = await getJob(targetJobId, controller.signal);
        if (!active) return;

        if (job.state === "done") return;
        if (job.state === "failed") {
          throw new Error(job.error ?? "任务处理失败");
        }

        // still running / queued
        const elapsed = Math.round((Date.now() - startedAt) / 1000);
        if (Date.now() - startedAt > POLL_TIMEOUT) {
          // 超时：回退到 loadPoseData（可能拿到数据也可能降级 fixture）
          return;
        }
        setPhase({ kind: "waiting", message: `任务处理中，请稍候…（${elapsed}s）` });
        await new Promise((r) => setTimeout(r, POLL_INTERVAL));
      }
    }

    (async () => {
      try {
        // 有 jobId 时先检查 job 状态
        if (jobId) {
          try {
            const job = await getJob(jobId, controller.signal);
            if (!active) return;

            if (job.state === "running" || job.state === "queued") {
              // job 还在跑，轮询等待
              setPhase({ kind: "waiting", message: "任务处理中，请稍候…" });
              await waitForJob(jobId);
              if (!active) return;
            } else if (job.state === "failed") {
              setPhase({ kind: "error", message: job.error ?? "任务处理失败" });
              return;
            }
            // state === "done" → 继续往下 loadPoseData
          } catch {
            // getJob 失败（网络错误/404 等）→ 回退到 loadPoseData（会降级 fixture）
            if (!active) return;
          }
        }

        const loaded = await loadPoseData(jobId, controller.signal);
        if (active) setPhase({ kind: "ready", loaded });
      } catch (err) {
        if (active)
          setPhase({
            kind: "error",
            message: err instanceof Error ? err.message : String(err),
          });
      }
    })();
    return () => {
      active = false;
      controller.abort();
    };
  }, [jobId]);

  // m13: 当 jobId 存在时，查询 job 状态以获取 degraded 标志
  useEffect(() => {
    if (!jobId) { setDegraded(false); return; }
    const controller = new AbortController();
    getJob(jobId, controller.signal)
      .then((job) => { if (job.degraded) setDegraded(true); })
      .catch(() => { /* 查询失败不影响主流程 */ });
    return () => controller.abort();
  }, [jobId]);

  if (phase.kind === "loading") {
    return (
      <div className="stage-status">
        <div className="spinner" aria-hidden />
        <p>正在加载姿态数据…</p>
      </div>
    );
  }

  if (phase.kind === "waiting") {
    return (
      <div className="stage-status">
        <div className="spinner" aria-hidden />
        <p>{phase.message}</p>
      </div>
    );
  }

  if (phase.kind === "error") {
    return (
      <div className="stage-status stage-status--error" role="alert">
        <strong>数据加载失败</strong>
        <p>{phase.message}</p>
      </div>
    );
  }

  return (
    <ReadyStage loaded={phase.loaded} degraded={degraded} jobId={jobId} />
  );
}

/** 数据就绪后的实际渲染（独立组件以便安全调用 useTimeline）。 */
function ReadyStage({
  loaded,
  degraded,
  jobId,
}: {
  loaded: LoadedPoseData;
  degraded?: boolean;
  jobId?: string;
}) {
  const { data, source, fallbackReason } = loaded;
  const keyframes = useMemo(() => data.keyframes, [data]);
  const { timeline, playing, durationMs, toggle } = useTimeline(keyframes, true);

  // 视频同步 Hook：模型动图与解析动图联动
  const videoSync = useVideoSync();

  // 检查是否有 mesh 数据
  const hasMesh = data.metadata.has_mesh === true && 
    data.mesh_faces !== undefined && 
    data.mesh_faces.length > 0 &&
    keyframes.some((kf) => kf.mesh_vertices && kf.mesh_vertices.length > 0);

  // 视图模式切换（默认骨架，有 mesh 数据时可选）
  const [viewMode, setViewMode] = useState<ViewMode>("skeleton");
  const [showWireframe, setShowWireframe] = useState(false);

  // 切换视图模式的回调
  const cycleViewMode = useCallback(() => {
    if (!hasMesh) return;
    setViewMode((prev) => {
      if (prev === "skeleton") return "mesh";
      if (prev === "mesh") return "both";
      return "skeleton";
    });
  }, [hasMesh]);

  // 空格键播放/暂停
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        toggle();
      }
      // M 键切换视图模式
      if (e.code === "KeyM" && hasMesh) {
        cycleViewMode();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, hasMesh, cycleViewMode]);

  return (
    <>
      {/* 降级横幅：未连接引擎 / 未提供任务时，醒目提示当前为样例数据，
          避免生产未配引擎时把 fixture 误当真实结果。 */}
      {source === "fixture" && (
        <div className="stage-banner" role="status">
          <span className="stage-banner__tag" aria-hidden>
            SAMPLE
          </span>
          <span className="stage-banner__text">
            当前为样例数据（内置 fixture）
            {fallbackReason ? ` · ${fallbackReason}` : ""}
          </span>
        </div>
      )}

      {/* m13: 质量降级横幅 —— 引擎判定产物质量未达标但仍交付 */}
      {degraded && source === "api" && (
        <div className="stage-banner stage-banner--degraded" role="alert">
          <span className="stage-banner__tag" aria-hidden>
            DEGRADED
          </span>
          <span className="stage-banner__text">
            质量降级交付：引擎判定该任务产物质量未达标，结果仅供参考。
          </span>
        </div>
      )}

      {/* 四宫格教学图：仅当 job 完成且有 grid_images 时显示 */}
      {source === "api" && jobId && data.metadata.grid_images && data.metadata.grid_images.length > 0 && (
        <KeyframeCards
          jobId={jobId}
          gridImages={data.metadata.grid_images}
          gridLabels={data.metadata.grid_labels}
        />
      )}

      {/* 模型动图：全宽横幅展示 */}
      {source === "api" && jobId && (
        <VideoCompare
          jobId={jobId}
          modelRef={videoSync.modelRef}
          playing={videoSync.playing}
          togglePlay={videoSync.togglePlay}
          onModelPlay={videoSync.onModelPlay}
          onModelPause={videoSync.onModelPause}
          onModelSeek={videoSync.onModelSeek}
          onModelTimeUpdate={videoSync.onModelTimeUpdate}
          onModelEnded={videoSync.onModelEnded}
        />
      )}

      {/* 下半部分：3D 视图 + 控制条 + 元数据 */}
      <section className="viewer-section" aria-label="3D 交互视图">
        <div className="viewer-frame">
          {/* 根据视图模式渲染 */}
          {viewMode === "skeleton" && (
            <SkeletonViewer keyframes={keyframes} timeline={timeline} />
          )}
          {viewMode === "mesh" && hasMesh && (
            <MeshViewer
              keyframes={keyframes}
              faces={data.mesh_faces!}
              timeline={timeline}
              showWireframe={showWireframe}
            />
          )}
          {viewMode === "both" && hasMesh && (
            <CombinedViewer
              keyframes={keyframes}
              faces={data.mesh_faces!}
              timeline={timeline}
              showWireframe={showWireframe}
            />
          )}

          {/* 视图模式切换按钮 */}
          {hasMesh && (
            <div className="viewer-mode-toggle">
              <button
                className="viewer-mode-btn"
                onClick={cycleViewMode}
                title="切换视图模式 (M)"
              >
                {viewMode === "skeleton" && "🦴 骨架"}
                {viewMode === "mesh" && "👤 Mesh"}
                {viewMode === "both" && "🦴+👤 叠加"}
              </button>
              {(viewMode === "mesh" || viewMode === "both") && (
                <button
                  className="viewer-mode-btn viewer-mode-btn--sm"
                  onClick={() => setShowWireframe((v) => !v)}
                  title="切换线框显示"
                >
                  {showWireframe ? "◈ 线框" : "◇ 线框"}
                </button>
              )}
            </div>
          )}

          <div className="viewer-hint">拖拽旋转 · 滚轮缩放 · 右键平移{hasMesh ? " · M 切换模式" : ""}</div>
        </div>

        <div className="viewer-side">
          <TimelineControls
            timeline={timeline}
            playing={playing}
            durationMs={durationMs}
            onToggle={toggle}
            fps={data.metadata.video_fps}
          />
          <MetadataPanel
            metadata={data.metadata}
            source={source}
            fallbackReason={fallbackReason}
          />
          {/* 解析动图：缩小版，置于元数据面板下方 */}
          {source === "api" && jobId && (
            <AnalysisVideo
              jobId={jobId}
              analysisRef={videoSync.analysisRef}
              onAnalysisPlay={videoSync.onAnalysisPlay}
              onAnalysisPause={videoSync.onAnalysisPause}
              onAnalysisSeek={videoSync.onAnalysisSeek}
            />
          )}
        </div>
      </section>
    </>
  );
}
