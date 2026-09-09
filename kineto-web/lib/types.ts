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
  /**
   * 关节排列顺序标识（additive，旧产物可能缺失）。
   * 当值不为 "smpl-canonical" 时前端应显示告警，提示该产物使用了非标准关节序，
   * 与前端 skeleton.ts 的 canonical 镜像可能不一致。
   */
  joint_order?: string;
  /** 数据契约版本号（additive，旧产物可能缺失）。 */
  schema_version?: number;
  /** 输入视频 MD5 hash，用于验证输入输出一致性（additive，旧产物可能缺失）。 */
  video_md5?: string;
  /** 四宫格教学图文件名列表（additive，旧产物可能缺失）。 */
  grid_images?: string[];
  /** 四宫格教学图标签列表（additive，旧产物可能缺失）。 */
  grid_labels?: string[];
  /** 是否携带 SMPL mesh 数据（additive，旧产物可能缺失）。 */
  has_mesh?: boolean;
  /**
   * [P1 mesh 节奏贴合] SMPL 顶点二进制文件名（additive，旧产物缺失）。
   * 值为 "mesh_vertices.f32"（位于产物目录内，经 /api 代理拉取）；
   * 为 null/缺失时表示顶点走旧 JSON 嵌入格式（keyframes[].mesh_vertices）。
   */
  mesh_vertices_file?: string;
  /** [P1] 二进制顶点轨道帧数（= keyframes.length，严格 1:1）。 */
  mesh_vertices_frames?: number;
  /** [P1] 每帧顶点数（SMPL 标准 6890）。 */
  mesh_vertices_per_frame?: number;
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
  /** SMPL mesh 顶点坐标 6890 × [x, y, z]（additive，仅当 metadata.has_mesh 时存在）。 */
  mesh_vertices?: Vec3[];
}

/** pose_data.json 根对象。 */
export interface PoseData {
  metadata: PoseMetadata;
  keyframes: Keyframe[];
  /** SMPL 三角面索引 13776 × [a, b, c]（根级别，所有帧共用；additive，旧产物可能缺失）。 */
  mesh_faces?: Vec3[];
}

/** 关节数量常量（SMPL 标准 24 关节）。 */
export const JOINT_COUNT = 24;

/** smpl_thetas 维度：24 关节 × axis-angle 3 维 = 72（整改前旧注释误作 216）。 */
export const THETAS_DIM = 72;

/** SMPL 体型参数 betas 维度（additive 字段，仅用于校验/未来网格渲染）。 */
export const BETAS_DIM = 10;

/** SMPL mesh 标准顶点数。 */
export const MESH_VERTEX_COUNT = 6890;

/** SMPL mesh 标准三角面数。 */
export const MESH_FACE_COUNT = 13776;

/** 标准关节序标识值（引擎 canonical 产物应为此值）。 */
export const JOINT_ORDER_CANONICAL = "smpl-canonical";

/** 任务状态枚举（后端契约：GET /jobs/{id} 的 state 字段）。 */
export type JobState = "queued" | "running" | "done" | "failed";

/**
 * [P1 mesh 节奏贴合] SMPL 顶点二进制轨道。
 *
 * 由 mesh_vertices.f32 加载而来（帧数×vertexCount×3 float32 LE），与
 * keyframes **严格 1:1**（第 i 帧顶点对应 keyframes[i]），加载时已做
 * 相机系→世界系预翻转（y→-y, z→-z）。采样时对相邻帧线性插值——
 * J_regressor 为线性映射，顶点插值与 joints_3d 插值严格同步，
 * 叠加模式下骨架与 mesh 位置/节奏完全贴合。
 */
export interface MeshTrack {
  /** 帧数（= keyframes.length）。 */
  frameCount: number;
  /** 每帧顶点数（SMPL 标准 6890）。 */
  vertexCount: number;
  /** 预翻转后的顶点数据，长度 frameCount × vertexCount × 3。 */
  vertices: Float32Array;
}

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
