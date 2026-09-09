/**
 * lib/useVideoSync.ts
 * ─────────────────────────────────────────────────────────────────────────
 * 双视频同步播放 Hook：管理 model demo 与 analysis input 的播放/暂停/跳转联动。
 * 以 model 为主时钟，analysis 跟随同步。
 */

"use client";

import { useRef, useState, useCallback, useEffect } from "react";

export function useVideoSync() {
  const modelRef = useRef<HTMLVideoElement>(null!);
  const analysisRef = useRef<HTMLVideoElement>(null!);
  const lock = useRef(false);
  const [playing, setPlaying] = useState(false);

  /* ── 同步原语 ─────────────────────────────────────────────────────── */

  const alignTime = useCallback(
    (from: HTMLVideoElement, to: HTMLVideoElement | null) => {
      if (lock.current || !to) return;
      lock.current = true;
      to.currentTime = from.currentTime;
      requestAnimationFrame(() => {
        lock.current = false;
      });
    },
    [],
  );

  const playBoth = useCallback(
    (from: HTMLVideoElement, to: HTMLVideoElement | null) => {
      if (to) {
        to.currentTime = from.currentTime;
        to.play().catch(() => {});
      }
      setPlaying(true);
    },
    [],
  );

  const pauseBoth = useCallback(
    (_from: HTMLVideoElement, to: HTMLVideoElement | null) => {
      if (to) to.pause();
      setPlaying(false);
    },
    [],
  );

  /* ── Model 事件 ──────────────────────────────────────────────────── */

  const onModelPlay = useCallback(() => {
    playBoth(modelRef.current!, analysisRef.current);
  }, [playBoth]);

  const onModelPause = useCallback(() => {
    pauseBoth(modelRef.current!, analysisRef.current);
  }, [pauseBoth]);

  const onModelSeek = useCallback(() => {
    alignTime(modelRef.current!, analysisRef.current);
  }, [alignTime]);

  const onModelTimeUpdate = useCallback(() => {
    const m = modelRef.current;
    const a = analysisRef.current;
    if (!m || !a || lock.current) return;
    if (Math.abs(m.currentTime - a.currentTime) > 0.10) {
      a.currentTime = m.currentTime;
    }
  }, []);

  const onModelEnded = useCallback(() => {
    if (analysisRef.current) {
      analysisRef.current.pause();
      analysisRef.current.currentTime = 0;
    }
    setPlaying(false);
  }, []);

  /* ── Analysis 事件（用户直接操作 analysis 时反向同步） ──────────── */

  const onAnalysisPlay = useCallback(() => {
    playBoth(analysisRef.current!, modelRef.current);
  }, [playBoth]);

  const onAnalysisPause = useCallback(() => {
    pauseBoth(analysisRef.current!, modelRef.current);
  }, [pauseBoth]);

  const onAnalysisSeek = useCallback(() => {
    alignTime(analysisRef.current!, modelRef.current);
  }, [alignTime]);

  /* ── 外部控制 ────────────────────────────────────────────────────── */

  const togglePlay = useCallback(() => {
    const m = modelRef.current;
    const a = analysisRef.current;
    if (!m) return;
    if (m.paused) {
      m.play().catch(() => {});
      if (a) {
        a.currentTime = m.currentTime;
        a.play().catch(() => {});
      }
      setPlaying(true);
    } else {
      m.pause();
      if (a) a.pause();
      setPlaying(false);
    }
  }, []);

  const seekTo = useCallback((time: number) => {
    const m = modelRef.current;
    const a = analysisRef.current;
    if (m) m.currentTime = time;
    if (a) a.currentTime = time;
  }, []);

  /* ── 清理：卸载时暂停 ───────────────────────────────────────────── */

  useEffect(() => {
    return () => {
      modelRef.current?.pause();
      analysisRef.current?.pause();
    };
  }, []);

  return {
    modelRef,
    analysisRef,
    playing,
    togglePlay,
    seekTo,
    onModelPlay,
    onModelPause,
    onModelSeek,
    onModelTimeUpdate,
    onModelEnded,
    onAnalysisPlay,
    onAnalysisPause,
    onAnalysisSeek,
  };
}
