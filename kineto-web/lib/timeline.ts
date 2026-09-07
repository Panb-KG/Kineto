/**
 * lib/timeline.ts
 * ─────────────────────────────────────────────────────────────────────────
 * 姿态时间轴引擎。
 *
 * 设计取舍（为何选用 @tweenjs/tween.js）:
 *   PROJECT_PLAN.md 任务 2.1 与项目既有技术栈均明确要求使用 Tween.js 在关键
 *   姿态之间平滑循环播放，故此处不引入 gsap / react-spring，保持与规划一致。
 *
 * 架构:
 *   - 主播放头（playhead, 单位 ms）由一个 Linear 缓动的 Tween 驱动， Tween 负责
 *     在两个关键帧时间戳之间产生连续、无抖动的时间推进；
 *   - 关节位置的插值由 sampleJoints() 完成：对给定时间在有序关键帧中二分定位
 *     相邻两帧，再逐关节线性插值。466 帧的真实样本以 60fps 渲染时，每帧仅需
 *     O(log n) 定位 + 24×3 次插值，完全无卡顿；
 *   - 时间推进与 React 渲染解耦：3D 视图在 useFrame 中直接读取 getMs()，
 *     命令式更新几何体，不触发 React 重渲染。
 */

import { Easing, Group, Tween } from "@tweenjs/tween.js";
import type { Keyframe, Vec3 } from "./types";

/** 时间轴订阅者：接收当前播放时间（ms）。 */
export type TimelineListener = (ms: number) => void;

/** 播放头代理对象，供 tween 修改。 */
interface Playhead {
  ms: number;
}

/**
 * 姿态时间轴：管理播放/暂停/拖拽，并对外暴露当前时间与订阅接口。
 * 与框架无关（纯 TS），可在 R3F 的 useFrame 或 React hook 中消费。
 */
export class PoseTimeline {
  /** 时间轴总时长（ms），取最后一个关键帧的 timestamp_ms。 */
  readonly durationMs: number;
  /** 是否循环播放。 */
  loop = true;

  private group = new Group();
  private proxy: Playhead = { ms: 0 };
  private tween: Tween<Playhead> | null = null;
  private rafId: number | null = null;
  private listeners = new Set<TimelineListener>();

  private _playing = false;
  private _ms = 0;

  constructor(durationMs: number) {
    this.durationMs = Math.max(durationMs, 1);
  }

  /** 当前播放时间（ms）。 */
  getMs(): number {
    return this._ms;
  }

  /** 归一化进度 0..1。 */
  getProgress(): number {
    return this._ms / this.durationMs;
  }

  isPlaying(): boolean {
    return this._playing;
  }

  /** 订阅时间变化（供 UI 层如进度条使用）。返回取消订阅函数。 */
  subscribe(listener: TimelineListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    for (const l of this.listeners) l(this._ms);
  }

  private setTime(ms: number): void {
    this._ms = ms;
    this.emit();
  }

  /** 从给定时间启动一段到 durationMs 的线性 tween。 */
  private startSegment(fromMs: number): void {
    this.tween?.stop();
    const remaining = Math.max(this.durationMs - fromMs, 1);
    this.proxy.ms = fromMs;
    this.tween = new Tween(this.proxy, this.group)
      .to({ ms: this.durationMs }, remaining)
      .easing(Easing.Linear.None)
      .onUpdate(() => this.setTime(this.proxy.ms))
      .onComplete(() => this.handleComplete())
      .start(performance.now());
  }

  private handleComplete(): void {
    if (this.loop) {
      // 无缝循环：回到起点并立即开始下一段。
      this.setTime(0);
      this.startSegment(0);
    } else {
      this.setTime(this.durationMs);
      this._playing = false;
      this.stopLoop();
    }
  }

  private startLoop(): void {
    if (this.rafId !== null) return;
    const tick = () => {
      this.rafId = requestAnimationFrame(tick);
      this.group.update(performance.now());
    };
    this.rafId = requestAnimationFrame(tick);
  }

  private stopLoop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  play(): void {
    if (this._playing) return;
    this._playing = true;
    // 若已停在末尾且非循环，从头开始。
    const from = !this.loop && this._ms >= this.durationMs ? 0 : this._ms;
    this.startSegment(from);
    this.startLoop();
  }

  pause(): void {
    if (!this._playing) return;
    this._playing = false;
    this.tween?.stop();
    this.tween = null;
    this.stopLoop();
  }

  toggle(): void {
    this._playing ? this.pause() : this.play();
  }

  /** 跳转到指定时间（ms），自动 clamp 到 [0, duration]。播放中会平滑续播。 */
  seek(ms: number): void {
    const clamped = Math.min(Math.max(ms, 0), this.durationMs);
    if (this._playing) {
      this.startSegment(clamped);
      this.setTime(clamped);
    } else {
      this.setTime(clamped);
    }
  }

  /** 按 0..1 进度跳转。 */
  seekRatio(ratio: number): void {
    this.seek(ratio * this.durationMs);
  }

  /** 释放资源（组件卸载时调用）。 */
  dispose(): void {
    this.pause();
    this.group.removeAll();
    this.listeners.clear();
  }
}

// ───────────────────────────────────────────────────────────────────────────
// 关键帧插值
// ───────────────────────────────────────────────────────────────────────────

/**
 * 预计算的时间索引：把关键帧时间戳抽成 Float64Array，便于二分查找与插值，
 * 避免每帧访问对象数组带来的开销。
 */
export interface TimeIndex {
  times: Float64Array;
  keyframes: Keyframe[];
  durationMs: number;
}

/** 由关键帧数组构建时间索引（要求 timestamp_ms 单调不减）。 */
export function buildTimeIndex(keyframes: Keyframe[]): TimeIndex {
  const n = keyframes.length;
  const times = new Float64Array(n);
  for (let i = 0; i < n; i++) times[i] = keyframes[i].timestamp_ms;
  const durationMs = n > 0 ? times[n - 1] : 0;
  return { times, keyframes, durationMs };
}

/** 二分查找：返回最后一个满足 times[i] <= t 的下标 i。 */
function lowerIndex(times: Float64Array, t: number): number {
  let lo = 0;
  let hi = times.length - 1;
  if (t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * 在给定时间采样 24 个关节的三维坐标，写入 out（长度需为 24×3）。
 * 对相邻两个关键帧做线性插值；时间越界时 clamp 到首/尾帧。
 * 返回 out 以便链式调用。
 */
export function sampleJoints(
  index: TimeIndex,
  timeMs: number,
  out: Float32Array | Float64Array,
): Float32Array | Float64Array {
  const { times, keyframes } = index;
  const n = keyframes.length;
  if (n === 0) return out;

  if (n === 1) {
    const j = keyframes[0].joints_3d;
    for (let k = 0; k < j.length && k * 3 + 2 < out.length; k++) {
      out[k * 3] = j[k][0];
      out[k * 3 + 1] = j[k][1];
      out[k * 3 + 2] = j[k][2];
    }
    return out;
  }

  const i = lowerIndex(times, timeMs);
  const i2 = Math.min(i + 1, n - 1);
  const a = keyframes[i];
  const b = keyframes[i2];

  const t0 = times[i];
  const t1 = times[i2];
  const span = t1 - t0;
  const frac = span > 1e-6 ? (timeMs - t0) / span : 0;

  const ja = a.joints_3d;
  const jb = b.joints_3d;
  const count = Math.min(ja.length, jb.length, out.length / 3);
  for (let k = 0; k < count; k++) {
    const pa: Vec3 = ja[k];
    const pb: Vec3 = jb[k];
    out[k * 3] = pa[0] + (pb[0] - pa[0]) * frac;
    out[k * 3 + 1] = pa[1] + (pb[1] - pa[1]) * frac;
    out[k * 3 + 2] = pa[2] + (pb[2] - pa[2]) * frac;
  }
  return out;
}

/** computeFraming 返回的帧变换（居中 + 缩放）。 */
export interface FrameTransform {
  offset: [number, number, number];
  scale: number;
}

/**
 * 计算把骨架居中并归一化到目标尺寸所需的偏移与缩放。
 * 基于所有关键帧的全局包围盒，保证整段动画使用同一变换，骨架不会漂移。
 */
export function computeFraming(
  keyframes: Keyframe[],
  targetSize = 2.4,
): FrameTransform {
  if (keyframes.length === 0) {
    return { offset: [0, 0, 0], scale: 1 };
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const kf of keyframes) {
    for (const [x, y, z] of kf.joints_3d) {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  const scale = targetSize / span;
  return { offset: [-cx, -cy, -cz], scale };
}
