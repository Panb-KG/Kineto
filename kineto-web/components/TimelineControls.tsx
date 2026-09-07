/**
 * components/TimelineControls.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 播放控制条：播放/暂停按钮 + 可拖拽进度条（scrubber）+ 时间读数。
 * 进度条通过 useTimelineTime 订阅时间轴，重渲染被限制在本组件子树内。
 */

"use client";

import { useCallback } from "react";
import type { PoseTimeline } from "../lib/timeline";
import { useTimelineTime } from "../lib/useTimeline";

interface TimelineControlsProps {
  timeline: PoseTimeline;
  playing: boolean;
  durationMs: number;
  onToggle: () => void;
  fps: number;
}

function formatMs(ms: number): string {
  const total = ms / 1000;
  const m = Math.floor(total / 60);
  const s = Math.floor(total % 60);
  const cs = Math.floor((ms % 1000) / 10);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${cs.toString().padStart(2, "0")}`;
}

export default function TimelineControls({
  timeline,
  playing,
  durationMs,
  onToggle,
  fps,
}: TimelineControlsProps) {
  const ms = useTimelineTime(timeline);
  const ratio = durationMs > 0 ? ms / durationMs : 0;
  const frame = Math.round((ms / 1000) * fps);

  const handleScrub = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      timeline.seek(Number(e.target.value));
    },
    [timeline],
  );

  return (
    <div className="tl-controls">
      <button
        type="button"
        className="tl-play"
        onClick={onToggle}
        aria-label={playing ? "暂停" : "播放"}
        title={playing ? "暂停 (Space)" : "播放 (Space)"}
      >
        {playing ? (
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
            <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" />
            <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
            <path d="M8 5.5v13l11-6.5-11-6.5z" fill="currentColor" />
          </svg>
        )}
      </button>

      <span className="tl-time">{formatMs(ms)}</span>

      <input
        className="tl-scrubber"
        type="range"
        min={0}
        max={Math.max(durationMs, 1)}
        step={1}
        value={ms}
        onChange={handleScrub}
        aria-label="时间轴进度"
        style={{ ["--ratio" as string]: `${ratio * 100}%` }}
      />

      <span className="tl-time tl-frame">
        f{frame.toString().padStart(3, "0")}
      </span>
    </div>
  );
}
