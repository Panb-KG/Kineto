/**
 * components/SkeletonViewer.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * React Three Fiber 3D 骨架视图（医疗极简风）。
 *
 *  - 24 个关节渲染为球体；23 根骨骼渲染为圆柱，拓扑与颜色取自
 *    lib/skeleton.ts（镜像引擎骨架 SSOT kineto-engine/skeleton_spec.py，
 *    解剖学正确：collar 13/14 挂在 spine3 9 上）；
 *  - joints_3d 为 SMPL canonical 序，下标与 SMPL_JOINT_NAMES 逐一对应，
 *    因此本组件无需任何关节重排逻辑即可解剖正确渲染；
 *  - drei OrbitControls 支持任意解剖平面的旋转 / 缩放 / 平移；
 *  - 地面网格 (drei Grid) + 坐标轴 (axesHelper) + 柔和的临床级光照；
 *  - 每帧在 useFrame 中命令式读取时间轴并插值关节，不触发 React 重渲染，
 *    466 帧真实样本亦无抖动。
 *
 * 本组件依赖浏览器 WebGL/window，需以 ssr:false 动态导入（见 app/page.tsx）。
 */

"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { PoseTimeline } from "../lib/timeline";
import {
  buildTimeIndex,
  computeFraming,
  computeSpineOrientation,
  sampleJoints,
  applyRotationToJoints,
} from "../lib/timeline";
import {
  SMPL_JOINT_NAMES,
  SMPL_SKELETON,
  boneColor,
} from "../lib/skeleton";
import type { Keyframe } from "../lib/types";
import { JOINT_COUNT } from "../lib/types";

// ── 视觉常量（医疗极简）───────────────────────────────────────────────────
const JOINT_RADIUS = 0.028;
const JOINT_COLOR = "#2b3a46"; // 关节：深石板灰，克制而清晰
const BONE_RADIUS = 0.016;
const HEAD_JOINT_INDEX = 15; // canonical 序中 15 = head；头部关节略放大，作为视觉锚点

interface SkeletonViewerProps {
  keyframes: Keyframe[];
  timeline: PoseTimeline;
  /** 目标归一化尺寸（世界单位），默认 2.4。 */
  targetSize?: number;
}

/**
 * Canvas 内部的骨架场景：持有几何体 ref，并在 useFrame 中更新姿态。
 */
function SkeletonRig({
  keyframes,
  timeline,
  targetSize = 2.4,
}: SkeletonViewerProps) {
  // 预计算：时间索引（用于插值）+ 全局居中/缩放（整段动画共用，避免漂移）
  const timeIndex = useMemo(() => buildTimeIndex(keyframes), [keyframes]);
  const framing = useMemo(
    () => computeFraming(keyframes, targetSize),
    [keyframes, targetSize],
  );
  // Spine 朝向校正四元数（横卧视频 → 直立）
  const spineQuat = useMemo(
    () => computeSpineOrientation(keyframes),
    [keyframes],
  );

  // 复用的采样缓冲与临时向量，避免每帧 GC
  const buffer = useRef<Float32Array>(new Float32Array(JOINT_COUNT * 3));
  const jointRefs = useRef<Array<THREE.Mesh | null>>([]);
  const boneRefs = useRef<Array<THREE.Mesh | null>>([]);

  const tmpA = useMemo(() => new THREE.Vector3(), []);
  const tmpB = useMemo(() => new THREE.Vector3(), []);
  const tmpDir = useMemo(() => new THREE.Vector3(), []);
  const tmpMid = useMemo(() => new THREE.Vector3(), []);
  const UP = useMemo(() => new THREE.Vector3(0, 1, 0), []);

  const { offset, scale } = framing;

  useFrame(() => {
    const ms = timeline.getMs();
    sampleJoints(timeIndex, ms, buffer.current);
    // 应用 spine 朝向校正（横卧视频 → 直立）
    applyRotationToJoints(buffer.current, spineQuat);
    const buf = buffer.current;

    // 更新关节球位置（直接应用 offset + scale，无旋转）
    for (let i = 0; i < JOINT_COUNT; i++) {
      const mesh = jointRefs.current[i];
      if (!mesh) continue;
      mesh.position.set(
        (buf[i * 3] + offset[0]) * scale,
        (buf[i * 3 + 1] + offset[1]) * scale,
        (buf[i * 3 + 2] + offset[2]) * scale,
      );
    }

    // 更新骨骼圆柱的位置/朝向/长度
    for (let b = 0; b < SMPL_SKELETON.length; b++) {
      const [ia, ib] = SMPL_SKELETON[b];
      const mesh = boneRefs.current[b];
      if (!mesh) continue;

      // 关节 A
      tmpA.set(
        (buf[ia * 3] + offset[0]) * scale,
        (buf[ia * 3 + 1] + offset[1]) * scale,
        (buf[ia * 3 + 2] + offset[2]) * scale,
      );

      // 关节 B
      tmpB.set(
        (buf[ib * 3] + offset[0]) * scale,
        (buf[ib * 3 + 1] + offset[1]) * scale,
        (buf[ib * 3 + 2] + offset[2]) * scale,
      );

      tmpDir.subVectors(tmpB, tmpA);
      const len = tmpDir.length();
      tmpMid.addVectors(tmpA, tmpB).multiplyScalar(0.5);

      mesh.position.copy(tmpMid);
      if (len > 1e-6) {
        tmpDir.multiplyScalar(1 / len);
        mesh.quaternion.setFromUnitVectors(UP, tmpDir);
        mesh.scale.set(1, len, 1);
      } else {
        mesh.scale.set(1, 1e-4, 1);
      }
    }
  });

  return (
    <group>
      {/* 关节球体 */}
      {SMPL_JOINT_NAMES.map((name, i) => (
        <mesh
          key={name}
          ref={(el) => {
            jointRefs.current[i] = el as THREE.Mesh | null;
          }}
          castShadow
        >
          <sphereGeometry
            args={[i === HEAD_JOINT_INDEX ? JOINT_RADIUS * 1.5 : JOINT_RADIUS, 20, 20]}
          />
          <meshStandardMaterial
            color={JOINT_COLOR}
            roughness={0.35}
            metalness={0.05}
          />
        </mesh>
      ))}

      {/* 骨骼圆柱（颜色取自 BONE_PART_MAP） */}
      {SMPL_SKELETON.map(([a, b], i) => (
        <mesh
          key={`${a}-${b}`}
          ref={(el) => {
            boneRefs.current[i] = el as THREE.Mesh | null;
          }}
          castShadow
        >
          {/* 单位高度圆柱（沿 +Y），运行时按骨长缩放 scale.y */}
          <cylinderGeometry args={[BONE_RADIUS, BONE_RADIUS, 1, 12, 1, false]} />
          <meshStandardMaterial
            color={boneColor(a, b)}
            roughness={0.45}
            metalness={0.08}
          />
        </mesh>
      ))}
    </group>
  );
}

/**
 * 场景灯光与环境：临床级柔和布光。
 */
function SceneEnvironment() {
  return (
    <>
      <hemisphereLight args={["#ffffff", "#dce6ef", 0.75]} />
      <ambientLight intensity={0.35} />
      <directionalLight
        position={[4, 8, 5]}
        intensity={0.85}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight position={[-5, 3, -4]} intensity={0.25} color="#cfe0f0" />

      {/* 地面网格：极淡的临床坐标网格 */}
      <Grid
        position={[0, -1.35, 0]}
        args={[20, 20]}
        cellSize={0.25}
        cellThickness={0.6}
        cellColor="#c9d6e2"
        sectionSize={1}
        sectionThickness={1}
        sectionColor="#9fb3c8"
        fadeDistance={16}
        fadeStrength={1.2}
        infiniteGrid
        followCamera={false}
      />

      {/* 坐标轴（低调、短小，仅供方位参考） */}
      <axesHelper args={[0.6]} />
    </>
  );
}

/**
 * 顶层导出的 3D 视图组件（含 Canvas）。
 */
export default function SkeletonViewer(props: SkeletonViewerProps) {
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      camera={{ position: [3.1, 1.2, 3.6], fov: 42, near: 0.1, far: 100 }}
      style={{ background: "transparent" }}
    >
      <SceneEnvironment />
      <SkeletonRig {...props} />
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.8}
        zoomSpeed={0.9}
        panSpeed={0.8}
        minDistance={1.2}
        maxDistance={12}
        target={[0, 0, 0]}
      />
    </Canvas>
  );
}
