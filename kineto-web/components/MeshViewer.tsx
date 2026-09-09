/**
 * components/MeshViewer.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * React Three Fiber SMPL mesh 渲染组件。
 *
 *  - 6890 顶点 + 13776 三角面，使用 BufferGeometry 渲染；
 *  - 每帧在 useFrame 中命令式更新顶点位置（插值），不触发 React 重渲染；
 *  - 肤色材质 + 临床级布光，与骨架渲染共享同一场景环境；
 *  - 支持与骨架切换/叠加显示。
 *
 * 本组件依赖浏览器 WebGL/window，需以 ssr:false 动态导入。
 */

"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { PoseTimeline } from "../lib/timeline";
import {
  buildTimeIndex,
  computeMeshFraming,
  computeSpineOrientation,
  sampleMeshVertices,
  applyRotationToJoints,
} from "../lib/timeline";
import type { Keyframe, Vec3 } from "../lib/types";
import { MESH_VERTEX_COUNT } from "../lib/types";

// ── 视觉常量 ─────────────────────────────────────────────────────────────
const SKIN_COLOR = "#e8c4a0"; // 肤色：温暖的浅桃色
const WIREFRAME_COLOR = "#4a6072"; // 线框色：深石板灰

interface MeshViewerProps {
  keyframes: Keyframe[];
  faces: Vec3[];
  timeline: PoseTimeline;
  /** 是否叠加显示骨架。 */
  showSkeleton?: boolean;
  /** 是否显示线框。 */
  showWireframe?: boolean;
  /** 目标归一化尺寸（世界单位）。 */
  targetSize?: number;
}

/**
 * Canvas 内部的 mesh 场景：持有几何体 ref，在 useFrame 中更新顶点。
 */
function SMPLMesh({
  keyframes,
  faces,
  timeline,
  showWireframe = false,
  targetSize = 2.4,
}: Omit<MeshViewerProps, "showSkeleton">) {
  // 预计算：时间索引 + 全局居中/缩放
  const timeIndex = useMemo(() => buildTimeIndex(keyframes), [keyframes]);
  const framing = useMemo(
    () => computeMeshFraming(keyframes, targetSize),
    [keyframes, targetSize],
  );
  // Spine 朝向校正四元数（横卧视频 → 直立）
  const spineQuat = useMemo(
    () => computeSpineOrientation(keyframes),
    [keyframes],
  );

  // 顶点缓冲区（复用，避免每帧 GC）
  const vertexBuffer = useRef<Float32Array>(
    new Float32Array(MESH_VERTEX_COUNT * 3),
  );

  // 几何体（一次性创建，后续只更新顶点）
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();

    // 顶点（初始化为第一帧数据或全零）
    const positions = new Float32Array(MESH_VERTEX_COUNT * 3);
    const firstFrame = keyframes.find((kf) => kf.mesh_vertices);
    if (firstFrame?.mesh_vertices) {
      const { offset, scale } = framing;
      const verts = firstFrame.mesh_vertices;
      // 先填充原始顶点，再应用 spine 旋转
      const tempVerts = new Float32Array(verts.length * 3);
      for (let k = 0; k < verts.length; k++) {
        tempVerts[k * 3] = verts[k][0];
        tempVerts[k * 3 + 1] = verts[k][1];
        tempVerts[k * 3 + 2] = verts[k][2];
      }
      applyRotationToJoints(tempVerts, spineQuat);
      for (let k = 0; k < verts.length; k++) {
        positions[k * 3] = (tempVerts[k * 3] + offset[0]) * scale;
        positions[k * 3 + 1] = (tempVerts[k * 3 + 1] + offset[1]) * scale;
        positions[k * 3 + 2] = (tempVerts[k * 3 + 2] + offset[2]) * scale;
      }
    }
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    // 面索引
    const indexArray = new Uint32Array(faces.length * 3);
    for (let i = 0; i < faces.length; i++) {
      indexArray[i * 3] = faces[i][0];
      indexArray[i * 3 + 1] = faces[i][1];
      indexArray[i * 3 + 2] = faces[i][2];
    }
    geo.setIndex(new THREE.BufferAttribute(indexArray, 1));

    // 计算法线
    geo.computeVertexNormals();

    return geo;
  }, [keyframes, faces, framing, spineQuat]);

  const { offset, scale } = framing;

  // 每帧更新顶点位置
  useFrame(() => {
    const ms = timeline.getMs();
    sampleMeshVertices(timeIndex, ms, vertexBuffer.current);
    // 应用 spine 朝向校正（横卧视频 → 直立）
    applyRotationToJoints(vertexBuffer.current, spineQuat);

    const posAttr = geometry.attributes.position as THREE.BufferAttribute;
    const arr = posAttr.array as Float32Array;
    const buf = vertexBuffer.current;

    for (let k = 0; k < MESH_VERTEX_COUNT; k++) {
      arr[k * 3] = (buf[k * 3] + offset[0]) * scale;
      arr[k * 3 + 1] = (buf[k * 3 + 1] + offset[1]) * scale;
      arr[k * 3 + 2] = (buf[k * 3 + 2] + offset[2]) * scale;
    }

    posAttr.needsUpdate = true;
    geometry.computeVertexNormals();
  });

  return (
    <group>
      {/* 实体 mesh */}
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          color={SKIN_COLOR}
          roughness={0.6}
          metalness={0.08}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* 可选线框叠加 */}
      {showWireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial
            color={WIREFRAME_COLOR}
            wireframe
            transparent
            opacity={0.15}
          />
        </mesh>
      )}
    </group>
  );
}

/**
 * 场景灯光与环境：与 SkeletonViewer 保持一致的临床级布光。
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

      <axesHelper args={[0.6]} />
    </>
  );
}

/**
 * 顶层导出的 3D mesh 视图组件（含 Canvas）。
 */
export default function MeshViewer(props: MeshViewerProps) {
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      camera={{ position: [3.1, 1.2, 3.6], fov: 42, near: 0.1, far: 100 }}
      style={{ background: "transparent" }}
    >
      <SceneEnvironment />
      <SMPLMesh {...props} />
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
