/**
 * components/VideoCompare.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 模型动图 + 解析动图同步播放（已移除原始视频展示）。
 *
 * 导出两个组件，由父组件通过 useVideoSync Hook 统一管理同步：
 *   - VideoCompare   → 主模型动图（demo_output.mp4），带播放控制
 *   - AnalysisVideo  → 解析动图（annotated_output.mp4），缩小版，放侧边栏元数据下方
 *
 * 布局根据视频宽高比自适应：横屏视频宽幅展示，竖屏视频收窄。
 */

"use client";

import { useCallback, useState, type MutableRefObject } from "react";
import { API_BASE } from "../lib/api";

/* ════════════════════════════════════════════════════════════════════════
 * 主模型动图
 * ════════════════════════════════════════════════════════════════════════ */

interface VideoCompareProps {
  jobId: string;
  modelRef: MutableRefObject<HTMLVideoElement | null>;
  playing: boolean;
  togglePlay: () => void;
  onModelPlay: () => void;
  onModelPause: () => void;
  onModelSeek: () => void;
  onModelTimeUpdate: () => void;
  onModelEnded: () => void;
}

export default function VideoCompare({
  jobId,
  modelRef,
  playing,
  togglePlay,
  onModelPlay,
  onModelPause,
  onModelSeek,
  onModelTimeUpdate,
  onModelEnded,
}: VideoCompareProps) {
  const videoUrl = `${API_BASE}/jobs/${jobId}/demo_output.mp4`;
  const [orientation, setOrientation] = useState<"landscape" | "portrait">(
    "landscape",
  );

  const handleLoadedMeta = useCallback(
    (e: React.SyntheticEvent<HTMLVideoElement>) => {
      const v = e.currentTarget;
      setOrientation(v.videoWidth > v.videoHeight ? "landscape" : "portrait");
    },
    [],
  );

  return (
    <div
      className={`vc-model vc-model--${orientation}`}
      data-orientation={orientation}
    >
      <div className="vc-model__head">
        <h2 className="section-label">模型动图 · Model Output</h2>
        <button
          className="vc-play-btn"
          onClick={togglePlay}
          aria-label={playing ? "暂停" : "播放"}
        >
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <rect x="2" y="1" width="3.5" height="12" rx="1" />
              <rect x="8.5" y="1" width="3.5" height="12" rx="1" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <path d="M3 1.5v11l9-5.5z" />
            </svg>
          )}
          <span>{playing ? "暂停" : "播放"}</span>
        </button>
      </div>
      <video
        ref={modelRef}
        src={videoUrl}
        preload="metadata"
        onLoadedMetadata={handleLoadedMeta}
        onPlay={onModelPlay}
        onPause={onModelPause}
        onSeeked={onModelSeek}
        onTimeUpdate={onModelTimeUpdate}
        onEnded={onModelEnded}
        className="vc-model__video"
        playsInline
      />
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
 * 解析动图（缩小版，侧边栏内嵌）
 * ════════════════════════════════════════════════════════════════════════ */

export function AnalysisVideo({
  jobId,
  analysisRef,
  onAnalysisPlay,
  onAnalysisPause,
  onAnalysisSeek,
}: {
  jobId: string;
  analysisRef: MutableRefObject<HTMLVideoElement | null>;
  onAnalysisPlay: () => void;
  onAnalysisPause: () => void;
  onAnalysisSeek: () => void;
}) {
  const videoUrl = `${API_BASE}/jobs/${jobId}/annotated_output.mp4`;

  return (
    <div className="vc-analysis">
      <span className="vc-analysis__label">解析输入 · Analysis Input</span>
      <video
        ref={analysisRef}
        src={videoUrl}
        preload="metadata"
        muted
        playsInline
        onPlay={onAnalysisPlay}
        onPause={onAnalysisPause}
        onSeeked={onAnalysisSeek}
        className="vc-analysis__video"
      />
    </div>
  );
}
