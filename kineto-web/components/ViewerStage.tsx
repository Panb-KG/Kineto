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

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import MetadataPanel from "./MetadataPanel";
import TimelineControls from "./TimelineControls";
import KeyframeCards from "./KeyframeCards";
import { loadPoseData, type LoadedPoseData } from "../lib/poseData";
import { getJob } from "../lib/api";
import { useTimeline } from "../lib/useTimeline";

const SkeletonViewer = dynamic(() => import("./SkeletonViewer"), {
  ssr: false,
  loading: () => <div className="viewer-skeleton-loading">初始化 3D 引擎…</div>,
});

const VideoCompare = dynamic(() => import("./VideoCompare"), {
  ssr: false,
});

interface ViewerStageProps {
  /** 可选任务 ID；缺省时离线加载内置 fixture。 */
  jobId?: string;
}

type Phase =
  | { kind: "loading" }
  | { kind: "ready"; loaded: LoadedPoseData }
  | { kind: "error"; message: string };

export default function ViewerStage({ jobId }: ViewerStageProps) {
  const [phase, setPhase] = useState<Phase>({ kind: "loading" });
  // m13: 跟踪 job 的 degraded 状态，用于显示质量降级横幅
  const [degraded, setDegraded] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    (async () => {
      try {
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

  // 空格键播放/暂停
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

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

      {/* 下半部分：3D 骨架视图 + 控制条 + 元数据 */}
      <section className="viewer-section" aria-label="3D 骨架交互视图">
        <div className="viewer-frame">
          <SkeletonViewer keyframes={keyframes} timeline={timeline} />
          <div className="viewer-hint">拖拽旋转 · 滚轮缩放 · 右键平移</div>
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
        </div>
      </section>

      {/* 视频对比：仅当 job 完成且来源为 API 时显示 */}
      {source === "api" && jobId && (
        <VideoCompare jobId={jobId} />
      )}
    </>
  );
}
