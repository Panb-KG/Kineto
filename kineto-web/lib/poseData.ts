/**
 * lib/poseData.ts
 * ─────────────────────────────────────────────────────────────────────────
 * 姿态数据加载层。开发环境完全离线可用：
 *
 *   1. 若显式提供 jobId 且后端已配置 → 尝试 GET /jobs/{id}/pose_data.json；
 *   2. 上述任一环节失败（未配置 / 网络不可达 / 4xx / 5xx）→ 优雅回退到内置
 *      fixture: /fixtures/pose_data.sample.json；
 *   3. 未提供 jobId → 直接加载 fixture。
 *
 * 加载结果附带 source 标记（'api' | 'fixture'）与可选告警信息，供 UI 展示。
 */

import { ApiError, fetchPoseData, isApiConfigured } from "./api";
import type { PoseData } from "./types";
import { JOINT_COUNT, JOINT_ORDER_CANONICAL } from "./types";

/**
 * 内置示例数据（复制自关节序根因整改后的 canonical 产物
 * kineto-engine/output_test/pose_data.json：466 帧、joints_3d 为 SMPL canonical
 * 序、smpl_thetas 72 维、含 additive 的 cam_t / betas）。
 */
export const FIXTURE_URL = "/fixtures/pose_data.sample.json";

export type PoseDataSource = "api" | "fixture";

export interface LoadedPoseData {
  data: PoseData;
  source: PoseDataSource;
  /** 若从 API 回退到 fixture，此处记录回退原因，供 UI 提示。 */
  fallbackReason?: string;
}

/** 从静态 fixture 加载（离线默认路径）。 */
export async function loadFixture(signal?: AbortSignal): Promise<PoseData> {
  const res = await fetch(FIXTURE_URL, { signal });
  if (!res.ok) {
    throw new Error(`无法加载本地 fixture (${res.status}): ${FIXTURE_URL}`);
  }
  const data = (await res.json()) as PoseData;
  validatePoseData(data);
  return data;
}

/**
 * 加载姿态数据：优先 API（当提供 jobId 且后端已配置），失败则回退 fixture。
 * @param jobId 可选任务 ID
 */
export async function loadPoseData(
  jobId?: string,
  signal?: AbortSignal,
): Promise<LoadedPoseData> {
  if (jobId && isApiConfigured()) {
    try {
      const data = await fetchPoseData(jobId, signal);
      validatePoseData(data);
      return { data, source: "api" };
    } catch (err) {
      const reason =
        err instanceof ApiError ? err.message : `API 加载失败: ${String(err)}`;
      const data = await loadFixture(signal);
      return { data, source: "fixture", fallbackReason: reason };
    }
  }

  // 未提供 jobId：直接离线加载 fixture。
  const data = await loadFixture(signal);
  const fallbackReason = !jobId
    ? "未提供任务 ID，使用内置示例数据"
    : undefined;
  return { data, source: "fixture", fallbackReason };
}

/**
 * 最小结构校验：确保关键字段存在且关节数正确，避免渲染期崩溃。
 * 校验失败直接抛错（数据契约被破坏属于严重问题，不应静默）。
 *
 * 兼容性原则（canonical 产物 + additive 字段）：
 *  - 硬断言只针对 joints_3d：必须为 24 个 [x,y,z]（SMPL canonical 序，
 *    下标语义见 lib/skeleton.ts）；
 *  - smpl_thetas（canonical 为 72 维）/ cam_t / betas / confidence_score 均为
 *    **可选 additive** 字段：缺失一律接受，存在时仅校验形状合法（数组 /
 *    cam_t 为 3 维），绝不因维度差异或缺失而拒绝加载。
 *
 * 全帧验证（修复 #18）：
 *  所有 keyframes 均校验 joints_3d 形状（24 × [x,y,z]），而非仅首帧；
 *  首帧额外做 additive 字段（smpl_thetas/cam_t/betas）的宽松形状校验。
 *  466 帧 × 24 关节 ≈ 1.1 万次判断，开销 <1ms。
 */
export function validatePoseData(data: PoseData): void {
  if (!data || typeof data !== "object") {
    throw new Error("pose_data 结构非法：根对象缺失");
  }
  if (!data.metadata || typeof data.metadata.video_fps !== "number") {
    throw new Error("pose_data 结构非法：metadata.video_fps 缺失");
  }
  if (!Array.isArray(data.keyframes) || data.keyframes.length === 0) {
    throw new Error("pose_data 结构非法：keyframes 为空");
  }

  // ── joint_order 告警（M3）：非 canonical 序时 console.warn，UI 层另行展示 ──
  const jointOrder = data.metadata.joint_order;
  if (jointOrder !== undefined && jointOrder !== JOINT_ORDER_CANONICAL) {
    // eslint-disable-next-line no-console
    console.warn(
      `[kineto] joint_order = "${jointOrder}"（非 "${JOINT_ORDER_CANONICAL}"），` +
        "前端 skeleton.ts 镜像 canonical 序，渲染结果可能偏斜。",
    );
  }

  // ── 全帧形状硬校验（修复 #18：从只校验 keyframes[0] 改为全帧）──
  data.keyframes.forEach((kf, i) => {
    if (!Array.isArray(kf.joints_3d) || kf.joints_3d.length !== JOINT_COUNT) {
      throw new Error(
        `pose_data 结构非法：keyframes[${i}].joints_3d 应为 ${JOINT_COUNT} 个关节，` +
        `实际 ${kf.joints_3d?.length ?? "undefined"}`,
      );
    }
    if (kf.joints_3d.some((j) => !Array.isArray(j) || j.length !== 3)) {
      throw new Error(
        `pose_data 结构非法：keyframes[${i}].joints_3d 中存在非 [x,y,z] 项`,
      );
    }
  });

  // ── 首帧 additive 可选字段宽松校验（存在时校验形状，缺失则接受）──
  const first = data.keyframes[0];
  if (first.smpl_thetas !== undefined && !Array.isArray(first.smpl_thetas)) {
    throw new Error("pose_data 结构非法：smpl_thetas 存在但不是数组");
  }
  if (
    first.cam_t !== undefined &&
    (!Array.isArray(first.cam_t) || first.cam_t.length !== 3)
  ) {
    throw new Error("pose_data 结构非法：cam_t 应为 [x,y,z]（3 维）");
  }
  if (first.betas !== undefined && !Array.isArray(first.betas)) {
    throw new Error("pose_data 结构非法：betas 存在但不是数组");
  }
}
