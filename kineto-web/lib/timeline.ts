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

/**
 * computeFraming 返回的完整帧变换。
 * rotation 为 3×3 行主序矩阵（Float32Array(9)），用于将 SMPL 相机空间的
 * 姿态朝向校正到 Three.js Y-up 约定。
 */
export interface FrameTransform {
  offset: [number, number, number];
  scale: number;
  /** 3×3 行主序旋转矩阵，将 SMPL 坐标旋转到 Y-up 场景空间。 */
  rotation: Float32Array;
}

/**
 * 计算 SMPL → Y-up 的朝向校正旋转矩阵。
 *
 * 几何原理：
 *   SMPL 引擎输出相机空间坐标，身体长轴方向取决于拍摄姿态——仰卧时 spine
 *   沿 X 轴，站立时沿 Y 轴。Three.js 场景使用 Y-up 约定，需要把 spine 对齐
 *   到 +Y 方向。
 *
 *   算法：对所有帧的 spine 向量（joint12 − joint0，即 neck − pelvis）求平均，
 *   得到姿态主方向；然后用 Rodrigues 旋转公式构造将该方向对齐到 +Y 的矩阵。
 *   仅计算一次，不逐帧重算。
 *
 *   对已经是站立姿态（spine ≈ +Y）的视频，旋转角趋近 0，矩阵趋近单位阵，
 *   不影响现有行为。
 */
function computeOrientationRotation(keyframes: Keyframe[]): Float32Array {
  // 累积 spine 向量并取平均，比单帧更鲁棒
  let sx = 0, sy = 0, sz = 0;
  for (const kf of keyframes) {
    const j = kf.joints_3d;
    if (j.length > 12) {
      sx += j[12][0] - j[0][0];
      sy += j[12][1] - j[0][1];
      sz += j[12][2] - j[0][2];
    }
  }
  const len = Math.sqrt(sx * sx + sy * sy + sz * sz);
  if (len < 1e-6) return new Float32Array([1,0,0, 0,1,0, 0,0,1]);
  sx /= len; sy /= len; sz /= len;

  // 与 +Y 的点积 → 夹角余弦
  const dot = sy; // spine · (0,1,0)
  // 已对齐（夹角 < ~3°），直接返回单位阵
  if (dot > 0.995) return new Float32Array([1,0,0, 0,1,0, 0,0,1]);

  // 旋转轴 = spine × Y，归一化
  let ax = sz, ay = 0, az = -sx; // (sx,sy,sz) × (0,1,0)
  const axLen = Math.sqrt(ax * ax + az * az);
  if (axLen < 1e-8) {
    // spine ≈ -Y（完全倒置），绕任意垂直轴转 π
    ax = 1; az = 0;
  } else {
    ax /= axLen; az /= axLen;
  }

  // Rodrigues 旋转矩阵 R = I·cos θ + (1−cos θ)·k⊗k + sin θ·[k]×
  // 其中 cos θ = dot, sin θ = axLen（因为 k 是归一化的叉积方向）
  const c = dot;
  const s = axLen;
  const t = 1 - c;
  // 代入 k=(ax, 0, az) 化简后的 3×3 行主序矩阵
  return new Float32Array([
    c + t * ax * ax,     t * ax * ay + s * az,  t * ax * az - s * ay, // row0
    t * ay * ax - s * az, c + t * ay * ay,       t * ay * az + s * ax, // row1
    t * az * ax + s * ay, t * az * ay - s * ax,  c + t * az * az,      // row2
  ]);
}

/**
 * 计算把骨架居中并归一化到目标尺寸所需的偏移与缩放。
 * 基于所有关键帧的全局包围盒，保证整段动画使用同一变换，骨架不会漂移。
 *
 * 同时计算 SMPL → Y-up 的朝向校正旋转（见 computeOrientationRotation），
 * 使仰卧视频（spine 沿 X）也能正确直立显示。
 */
export function computeFraming(
  keyframes: Keyframe[],
  targetSize = 2.4,
): FrameTransform {
  const rotation = computeOrientationRotation(keyframes);

  if (keyframes.length === 0) {
    return { offset: [0, 0, 0], scale: 1, rotation };
  }

  // 先对全部关节施加旋转，再在旋转后的空间计算包围盒
  // 这样居中/缩放的变换与旋转后的姿态匹配，不会偏移
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const kf of keyframes) {
    for (const [x, y, z] of kf.joints_3d) {
      const rx = rotation[0]*x + rotation[1]*y + rotation[2]*z;
      const ry = rotation[3]*x + rotation[4]*y + rotation[5]*z;
      const rz = rotation[6]*x + rotation[7]*y + rotation[8]*z;
      if (rx < minX) minX = rx;
      if (ry < minY) minY = ry;
      if (rz < minZ) minZ = rz;
      if (rx > maxX) maxX = rx;
      if (ry > maxY) maxY = ry;
      if (rz > maxZ) maxZ = rz;
    }
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  const scale = targetSize / span;
  return { offset: [-cx, -cy, -cz], scale, rotation };
}
