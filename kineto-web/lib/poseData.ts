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

import {
  ApiError,
  fetchMeshTrackFile,
  fetchPoseData,
  isApiConfigured,
} from "./api";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import type { BufferGeometry } from "three";
import type { Keyframe, MeshTrack, PoseData } from "./types";
import { JOINT_COUNT, JOINT_ORDER_CANONICAL, MESH_VERTEX_COUNT } from "./types";

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

// ── [P1.1] DRACO 解码器（单例；WASM 从同源 /draco/ 加载，无 CDN 依赖）────────
let dracoLoaderSingleton: DRACOLoader | null = null;
function getDracoLoader(): DRACOLoader {
  if (!dracoLoaderSingleton) {
    dracoLoaderSingleton = new DRACOLoader().setDecoderPath("/draco/");
  }
  return dracoLoaderSingleton;
}

/**
 * KDRC v2 容器头（小端）：
 * magic 4B | version u16(=2) | bits u16 | frameCount u32 | vpf u32 |
 * stride u32 | perm u32[vpf]（perm[orig_v]=dec_v，引擎编码时以 3D 最近点
 * 匹配求出）| offsets u32[frameCount] | 逐帧 DRACO blob。
 */
interface DrcHeader {
  frameCount: number;
  vertexCount: number;
  stride: number;
  /** orig→dec 顶点排列：SMPL 原始序顶点 v 的解码位置 = perm[v]。 */
  perm: Uint32Array;
  offsets: Uint32Array;
  headerLen: number;
}

function parseDrcHeader(buf: ArrayBuffer): DrcHeader {
  const bytes = new Uint8Array(buf, 0, 4);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== "KDRC") {
    throw new Error("mesh_track.drcs magic 非法（文件损坏或非本管线产物）");
  }
  const dv = new DataView(buf);
  const version = dv.getUint16(4, true);
  if (version < 2) {
    throw new Error(`KDRC v${version} 为开发期旧格式（不含顶点排列），请重新生成产物`);
  }
  const frameCount = dv.getUint32(8, true);
  const vertexCount = dv.getUint32(12, true);
  const stride = dv.getUint32(16, true);
  const permOffset = 20;
  const offsetsOffset = permOffset + 4 * vertexCount;
  const headerLen = offsetsOffset + 4 * frameCount;
  if (frameCount <= 0 || vertexCount <= 0 || buf.byteLength < headerLen) {
    throw new Error("mesh_track.drcs 容器头截断或字段非法");
  }
  const perm = new Uint32Array(buf, permOffset, vertexCount);
  // 双射校验（文件损坏不应静默渲染错序网格）
  const seen = new Set<number>();
  for (let v = 0; v < vertexCount; v++) {
    const d = perm[v];
    if (d >= vertexCount || seen.has(d)) {
      throw new Error("mesh_track.drcs perm 非双射（容器损坏）");
    }
    seen.add(d);
  }
  const offsets = new Uint32Array(buf, offsetsOffset, frameCount);
  return { frameCount, vertexCount, stride, perm, offsets, headerLen };
}

/** 解码单个 DRACO blob → 顶点位置（draco 排列）。 */
function decodeDracoPositions(
  loader: DRACOLoader,
  blob: ArrayBuffer,
): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    loader.parse(
      blob,
      (geo: BufferGeometry) => {
        try {
          const posAttr = geo.getAttribute("position");
          if (!posAttr) {
            reject(new Error("DRACO 解码缺 position 属性"));
            return;
          }
          resolve(Float32Array.from(posAttr.array as ArrayLike<number>));
        } catch (exc) {
          reject(exc instanceof Error ? exc : new Error(String(exc)));
        }
      },
      (err: unknown) => reject(err instanceof Error ? err : new Error(String(err))),
    );
  });
}

/** [P1.1] 解码 KDRC v2 容器 → MeshTrack（顶点经容器头 perm 还原 SMPL
 *  原始序 + 预翻转 + times）。面索引仍用 pose_data.json 根级 mesh_faces。 */
async function buildDracoTrack(buf: ArrayBuffer, data: PoseData): Promise<MeshTrack> {
  const hdr = parseDrcHeader(buf);
  if (hdr.vertexCount !== MESH_VERTEX_COUNT) {
    throw new Error(
      `mesh_track.drcs 顶点数 ${hdr.vertexCount} 非 SMPL 标准 ${MESH_VERTEX_COUNT}`,
    );
  }
  const keyframes = data.keyframes;

  // mesh 帧 k ↔ keyframes[k*stride] 的时间戳
  const times = new Float64Array(hdr.frameCount);
  for (let k = 0; k < hdr.frameCount; k++) {
    const ki = k * hdr.stride;
    if (ki >= keyframes.length) {
      throw new Error(`mesh 帧 ${k} 经 stride=${hdr.stride} 映射越界`);
    }
    times[k] = keyframes[ki].timestamp_ms;
  }

  const loader = getDracoLoader();
  const vertices = new Float32Array(hdr.frameCount * hdr.vertexCount * 3);
  const perm = hdr.perm;
  const lastOffset = hdr.offsets[hdr.frameCount - 1];
  for (let k = 0; k < hdr.frameCount; k++) {
    const start = hdr.offsets[k];
    const end = k + 1 < hdr.frameCount ? hdr.offsets[k + 1] : buf.byteLength;
    if (start < hdr.headerLen || end > buf.byteLength || end <= start) {
      throw new Error(`帧 ${k} blob 偏移非法（[${start}, ${end})）`);
    }
    const positions = await decodeDracoPositions(loader, buf.slice(start, end));
    if (positions.length !== hdr.vertexCount * 3) {
      throw new Error(`帧 ${k} DRACO 顶点数 ${positions.length / 3} ≠ ${hdr.vertexCount}`);
    }
    const base = k * hdr.vertexCount * 3;
    for (let v = 0; v < hdr.vertexCount; v++) {
      const di = perm[v] * 3; // orig 序 v → draco 序 perm[v]
      vertices[base + v * 3] = positions[di];
      vertices[base + v * 3 + 1] = -positions[di + 1]; // 相机系→世界系：Y 翻转
      vertices[base + v * 3 + 2] = -positions[di + 2]; // Z 翻转
    }
  }
  // 末帧偏移必须落在容器内（偏移表损坏的兜底校验）
  if (lastOffset >= buf.byteLength) {
    throw new Error("mesh_track.drcs 末帧偏移越界");
  }
  return { frameCount: hdr.frameCount, vertexCount: hdr.vertexCount, vertices, times };
}

/**
 * [P1] f32 全帧轨道组装：相机系→世界系预翻转（y→-y, z→-z，与
 * sampleJoints 同一变换），帧数必须与 keyframes 1:1。
 */
function buildF32Track(raw: Float32Array, keyframes: Keyframe[]): MeshTrack {
  const frameCount = raw.length / (MESH_VERTEX_COUNT * 3);
  if (!Number.isInteger(frameCount) || frameCount !== keyframes.length) {
    throw new Error(
      `mesh_vertices.f32 帧数不符：${frameCount} ≠ keyframes ${keyframes.length}`,
    );
  }
  const times = new Float64Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    times[i] = keyframes[i].timestamp_ms;
    const base = i * MESH_VERTEX_COUNT * 3;
    for (let k = 0; k < MESH_VERTEX_COUNT; k++) {
      raw[base + k * 3 + 1] = -raw[base + k * 3 + 1]; // Y 翻转
      raw[base + k * 3 + 2] = -raw[base + k * 3 + 2]; // Z 翻转
    }
  }
  return { frameCount, vertexCount: MESH_VERTEX_COUNT, vertices: raw, times };
}

/**
 * 拉取并解码 mesh 轨道（DRACO 压缩或 f32 回退）。
 * 设计为**不阻塞骨架展示**：由 ViewerStage 在 pose 就绪后后台调用，
 * 失败返回 undefined（mesh 模式提示加载失败，骨架不受影响）。
 */
export async function loadMeshTrack(
  jobId: string,
  data: PoseData,
  signal?: AbortSignal,
): Promise<MeshTrack | undefined> {
  const {
    mesh_vertices_file: file,
    mesh_vertices_frames: frames,
    mesh_vertices_per_frame: vpf,
    mesh_encoding: encoding,
  } = data.metadata;
  if (!file || !frames || !vpf) {
    return undefined; // 旧 JSON 嵌入格式或无 mesh 产物
  }
  if (vpf !== MESH_VERTEX_COUNT) {
    // eslint-disable-next-line no-console
    console.warn(
      `[kineto] mesh_vertices_per_frame=${vpf} 非 SMPL 标准 ` +
        `${MESH_VERTEX_COUNT}，忽略 mesh 轨道（mesh 模式回退嵌入路径）`,
    );
    return undefined;
  }
  try {
    const buf = await fetchMeshTrackFile(jobId, file, signal);
    if ((encoding ?? "").startsWith("draco")) {
      return await buildDracoTrack(buf, data);
    }
    // f32 全帧回退格式
    if (buf.byteLength !== frames * vpf * 3 * 4) {
      throw new Error(
        `mesh_vertices.f32 大小不符：${buf.byteLength}B ≠ 预期 ${frames * vpf * 3 * 4}B`,
      );
    }
    return buildF32Track(new Float32Array(buf), data.keyframes);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(`[kineto] mesh 轨道加载失败: ${String(err)}`);
    return undefined;
  }
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
      // [P1.1] mesh 轨道（DRACO 压缩，Funnel 低带宽下需数十秒）**不阻塞**
      // 首屏：pose 就绪即渲染骨架，mesh 由 ViewerStage 后台 loadMeshTrack。
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
