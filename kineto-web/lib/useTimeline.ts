/**
 * lib/useTimeline.ts
 * ─────────────────────────────────────────────────────────────────────────
 * 将 PoseTimeline 接入 React 的 hook。
 *
 * 只把「是否播放」这类低频状态提升为 React state；高频的当前时间不进 state，
 * 而是由各消费组件（进度条 / 3D 视图）自行订阅 timeline，避免整页每帧重渲染。
 */

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PoseTimeline, type TimelineListener } from "./timeline";
import type { Keyframe } from "./types";

export interface UseTimelineResult {
  /** 稳定的时间轴实例（在整个生命周期内不变）。 */
  timeline: PoseTimeline;
  playing: boolean;
  durationMs: number;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (ms: number) => void;
  seekRatio: (ratio: number) => void;
}

/**
 * 创建并管理一个 PoseTimeline。
 * @param keyframes 关键帧数组，用于推导总时长（取末帧 timestamp_ms）。
 * @param autoPlay  是否在挂载后自动播放（默认 true）。
 */
export function useTimeline(keyframes: Keyframe[], autoPlay = true): UseTimelineResult {
  const durationMs = useMemo(() => {
    if (keyframes.length === 0) return 0;
    return keyframes[keyframes.length - 1].timestamp_ms;
  }, [keyframes]);

  const timeline = useMemo(() => new PoseTimeline(durationMs), [durationMs]);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (autoPlay && durationMs > 0) {
      timeline.play();
      setPlaying(true);
    }
    return () => {
      timeline.dispose();
    };
    // timeline 随 durationMs 变化而重建，此处依赖足够
  }, [timeline, autoPlay, durationMs]);

  const play = useCallback(() => {
    timeline.play();
    setPlaying(true);
  }, [timeline]);

  const pause = useCallback(() => {
    timeline.pause();
    setPlaying(false);
  }, [timeline]);

  const toggle = useCallback(() => {
    if (timeline.isPlaying()) {
      timeline.pause();
      setPlaying(false);
    } else {
      timeline.play();
      setPlaying(true);
    }
  }, [timeline]);

  const seek = useCallback((ms: number) => timeline.seek(ms), [timeline]);
  const seekRatio = useCallback((r: number) => timeline.seekRatio(r), [timeline]);

  return { timeline, playing, durationMs, play, pause, toggle, seek, seekRatio };
}

/**
 * 订阅时间轴当前时间的轻量 hook，仅供进度条等小组件使用。
 * 以 rAF 频率更新局部 state，其重渲染被限制在调用组件子树内。
 */
export function useTimelineTime(timeline: PoseTimeline): number {
  const [ms, setMs] = useState(() => timeline.getMs());
  const msRef = useRef(ms);
  msRef.current = ms;

  useEffect(() => {
    const listener: TimelineListener = (next) => {
      // 仅在整毫秒变化时更新，减少无谓渲染
      if (Math.abs(next - msRef.current) >= 1) setMs(next);
    };
    const unsub = timeline.subscribe(listener);
    return unsub;
  }, [timeline]);

  return ms;
}
