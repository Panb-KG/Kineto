/**
 * components/VideoCompare.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 双视频同步对比：左侧原始输入视频，右侧解析后 demo 视频。
 * 播放/暂停/跳转同步，响应式布局（移动端上下排列）。
 */

"use client";

import { useRef, useCallback } from "react";
import { API_BASE } from "../lib/api";

interface VideoCompareProps {
  jobId: string;
}

export default function VideoCompare({ jobId }: VideoCompareProps) {
  const leftRef = useRef<HTMLVideoElement>(null);
  const rightRef = useRef<HTMLVideoElement>(null);
  const syncing = useRef(false);

  const syncTime = useCallback(
    (source: HTMLVideoElement, target: HTMLVideoElement | null) => {
      if (syncing.current || !target) return;
      syncing.current = true;
      target.currentTime = source.currentTime;
      requestAnimationFrame(() => {
        syncing.current = false;
      });
    },
    [],
  );

  const handlePlay = useCallback(
    (source: HTMLVideoElement, target: HTMLVideoElement | null) => {
      if (target) {
        target.currentTime = source.currentTime;
        target.play();
      }
    },
    [],
  );

  const handlePause = useCallback((target: HTMLVideoElement | null) => {
    if (target) target.pause();
  }, []);

  const inputUrl = `${API_BASE}/jobs/${jobId}/input.mp4`;
  const demoUrl = `${API_BASE}/jobs/${jobId}/demo_output.mp4`;

  return (
    <section className="video-compare" aria-label="视频对比">
      <h2 className="section-label">视频对比 · Video Compare</h2>
      <div className="video-compare__pair">
        <div className="video-compare__slot">
          <span className="video-compare__label">原始视频</span>
          <video
            ref={leftRef}
            src={inputUrl}
            controls
            preload="metadata"
            onPlay={() => handlePlay(leftRef.current!, rightRef.current)}
            onPause={() => handlePause(rightRef.current)}
            onSeeked={() => syncTime(leftRef.current!, rightRef.current)}
            className="video-compare__video"
          />
        </div>
        <div className="video-compare__slot">
          <span className="video-compare__label">解析视频</span>
          <video
            ref={rightRef}
            src={demoUrl}
            controls
            preload="metadata"
            onPlay={() => handlePlay(rightRef.current!, leftRef.current)}
            onPause={() => handlePause(leftRef.current)}
            onSeeked={() => syncTime(rightRef.current!, leftRef.current)}
            className="video-compare__video"
          />
        </div>
      </div>
    </section>
  );
}
