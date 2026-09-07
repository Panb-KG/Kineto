/**
 * lib/types.ts
 * ─────────────────────────────────────────────────────────────────────────
 * 与 kineto-engine 输出的 pose_data.json 数据契约严格对应的 TypeScript 类型。
 *
 * 真实样本结构（关节序根因整改后的 canonical 产物：
 * kineto-engine/output_test/pose_data.json，466 帧，已核对，勿臆测）:
 * {
 *   "metadata": {
 *     "video_fps": 20.03,
 *     "total_frames": 466,
 *     "resolution": "1280x720",
 *     "model_version": "4dhumans-v1.0",
 *     "device": "mps",
 *     "extraction_mode": "4dhumans",
 *     "pipeline": {
 *       "max_iterations": 2,
 *       "quality_threshold": 0.6,
 *       "final_quality_score": 0.9936,
 *       "refine_applied": true
 *     }
 *   },
 *   "keyframes": [ { frame_index, timestamp_ms, state_label, joints_3d,
 *                    smpl_thetas, cam_t, confidence_score, betas }, ... ]
 * }
 *
 * 注意:
 *  - 质量分的唯一正确路径是 **metadata.pipeline.final_quality_score**；根对象与
 *    metadata 顶层**都没有** final_quality_score，切勿读错嵌套层级。
 *  - joints_3d 为 **SMPL canonical 序**（与 kineto-engine/skeleton_spec.py 一致，
 *    前端镜像于 lib/skeleton.ts），不是 HMR2 内部的 OpenPose Body-25 序。
 *  - smpl_thetas 为 **72** 维（24 关节 × axis-angle 3 维），不是 216 维。
 *  - cam_t / betas 为整改后产物的 additive 字段（旧样本可能缺失），故均为可选；
 *    类型层面对缺失保持向后兼容，不得因缺失而报错。
 */

/** 三维坐标 [x, y, z]（单位约为米，SMPL/相机空间）。 */
export type Vec3 = [number, number, number];

/** 关键帧状态标签。 */
export type StateLabel = "initial" | "active" | "final" | (string & {});

/** 姿态提取来源。仅 "4dhumans" 视为真实数据，其余触发假数据告警。 */
export type ExtractionMode = "4dhumans" | (string & {});

/** 处理流水线元信息（嵌套于 metadata.pipeline）。 */
export interface PipelineMeta {
  max_iterations?: number;
  quality_threshold?: number;
  /** 最终质量得分 0..1，用于元数据面板展示。 */
  final_quality_score?: number;
  refine_applied?: boolean;
}

/** 顶层元数据块。 */
export interface PoseMetadata {
  video_fps: number;
  total_frames: number;
  resolution: string;
  model_version?: string;
  device?: string;
  extraction_mode: ExtractionMode;
  pipeline?: PipelineMeta;
}

/**
 * 单个关键帧。
 *
 * joints_3d 为 24 个 SMPL 关节的三维坐标，顺序为 **SMPL canonical 序**
 * （0=pelvis、9=spine3、12=neck、15=head；collar 13/14 的父为 spine3 9），
 * 与 lib/skeleton.ts 的 SMPL_JOINT_NAMES / SMPL_PARENTS（镜像引擎权威源
 * kineto-engine/skeleton_spec.py）严格一致；引擎已在交付前将 HMR2 的
 * OpenPose Body-25 序归一为 canonical 序（旧根因已修），前端无需再做重排。
 *
 * smpl_thetas 为 **72** 维 axis-angle（global_orient 3 维 + body_pose 69 维
 * = 24 关节 × 3），**不是** 216 维。
 */
export interface Keyframe {
  frame_index: number;
  /** 相对视频起点的毫秒时间戳（时间轴主键）。 */
  timestamp_ms: number;
  state_label: StateLabel;
  /** 24 × [x, y, z]，SMPL canonical 序（见上方契约说明）。 */
  joints_3d: Vec3[];
  /** 72 维 SMPL 姿态 theta（24 关节 × axis-angle 3 维；本期不用于网格渲染，仅保留）。 */
  smpl_thetas?: number[];
  /** 相机平移 [x, y, z]（additive：整改后产物携带，旧样本可能缺失）。 */
  cam_t?: Vec3;
  confidence_score?: number;
  /** SMPL 体型参数 β（10 维；additive：整改后产物携带，旧样本可能缺失）。 */
  betas?: number[];
}

/** pose_data.json 根对象。 */
export interface PoseData {
  metadata: PoseMetadata;
  keyframes: Keyframe[];
}

/** 关节数量常量（SMPL 标准 24 关节）。 */
export const JOINT_COUNT = 24;

/** smpl_thetas 维度：24 关节 × axis-angle 3 维 = 72（整改前旧注释误作 216）。 */
export const THETAS_DIM = 72;

/** SMPL 体型参数 betas 维度（additive 字段，仅用于校验/未来网格渲染）。 */
export const BETAS_DIM = 10;

/** 任务状态枚举（后端契约：GET /jobs/{id} 的 state 字段）。 */
export type JobState = "queued" | "running" | "done" | "failed";

/**
 * 后端任务对象（GET /jobs/{id} 返回体）。
 *
 * 权威契约（务必严格遵守，勿自创字段）：
 *  - 规范字段是 `state`（非 status），完成值为 `done`、失败值为 `failed`；
 *  - 后端不返回 `status` / `completed` / `job_id`；
 *  - `quality_score` / `extraction_mode` 为完成后可选字段；`error` 仅失败时出现；
 *  - `degraded` / `quality_warning`：质量门禁 warn 模式下判决不达标仍交付 done，
 *    但会**仅当为真时**附加这两个字段（纯 additive，健康 job 的响应形状不变）。
 */
export interface JobStatus {
  state: JobState;
  /** 处理进度 0..1（或 0..100，以后端为准）。 */
  progress: number;
  /** 完成后由 4DHumans 管线给出的质量得分。 */
  quality_score?: number;
  /** 姿态提取来源，如 "4dhumans"。 */
  extraction_mode?: string;
  /** 降级交付（state === "done" 但质量不达标；additive，仅真时下发）。 */
  degraded?: boolean;
  /** 质量告警（与 degraded 同时出现；additive，仅真时下发）。 */
  quality_warning?: boolean;
  /** 失败原因（state === "failed" 时）。 */
  error?: string;
}
