/**
 * components/CombinedViewer.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 叠加模式专用：在同一 Canvas 中渲染 SMPL mesh + 骨架，共享统一 framing
 * 与 OrbitControls，解决两个独立 Canvas 导致的比例不一致 / 旋转不同步问题。
 *
 *  - 使用 computeMeshFraming（基于 mesh_vertices）计算统一 offset/scale，
 *    同时应用于 mesh 顶点和骨架关节，确保两者精确对齐；
 *  - 单 Canvas + 单 OrbitControls，旋转 / 缩放 / 平移完全同步；
 *  - 单 SceneEnvironment（灯光 + 网格 + 坐标轴），避免重复渲染。
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
  sampleMeshVertices,
  sampleJoints,
} from "../lib/timeline";
import {
  SMPL_JOINT_NAMES,
  SMPL_SKELETON,
  boneColor,
} from "../lib/skeleton";
import type { Keyframe, Vec3 } from "../lib/types";
import { MESH_VERTEX_COUNT, JOINT_COUNT } from "../lib/types";

// ── 视觉常量（与 MeshViewer / SkeletonViewer 保持一致）────────────────────
const SKIN_COLOR = "#e8c4a0";
const WIREFRAME_COLOR = "#4a6072";
const JOINT_RADIUS = 0.028;
const JOINT_COLOR = "#2b3a46";
const BONE_RADIUS = 0.016;
const HEAD_JOINT_INDEX = 15;

interface CombinedViewerProps {
  keyframes: Keyframe[];
  faces: Vec3[];
  timeline: PoseTimeline;
  showWireframe?: boolean;
  targetSize?: number;
}

// ── SMPL Mesh（内部组件）───────────────────────────────────────────────────
function SMPLMesh({
  keyframes,
  faces,
  timeline,
  showWireframe = false,
  offset,
  scale,
  timeIndex,
}: {
  keyframes: Keyframe[];
  faces: Vec3[];
  timeline: PoseTimeline;
  showWireframe: boolean;
  offset: [number, number, number];
  scale: number;
  timeIndex: ReturnType<typeof buildTimeIndex>;
}) {
  const vertexBuffer = useRef<Float32Array>(
    new Float32Array(MESH_VERTEX_COUNT * 3),
  );

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();

    const positions = new Float32Array(MESH_VERTEX_COUNT * 3);
    const firstFrame = keyframes.find((kf) => kf.mesh_vertices);
    if (firstFrame?.mesh_vertices) {
      const verts = firstFrame.mesh_vertices;
      for (let k = 0; k < verts.length; k++) {
        positions[k * 3] = (verts[k][0] + offset[0]) * scale;
        positions[k * 3 + 1] = (verts[k][1] + offset[1]) * scale;
        positions[k * 3 + 2] = (verts[k][2] + offset[2]) * scale;
      }
    }
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const indexArray = new Uint32Array(faces.length * 3);
    for (let i = 0; i < faces.length; i++) {
      indexArray[i * 3] = faces[i][0];
      indexArray[i * 3 + 1] = faces[i][1];
      indexArray[i * 3 + 2] = faces[i][2];
    }
    geo.setIndex(new THREE.BufferAttribute(indexArray, 1));
    geo.computeVertexNormals();

    return geo;
  }, [keyframes, faces, offset, scale]);

  useFrame(() => {
    const ms = timeline.getMs();
    sampleMeshVertices(timeIndex, ms, vertexBuffer.current);

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
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          color={SKIN_COLOR}
          roughness={0.6}
          metalness={0.08}
          side={THREE.DoubleSide}
        />
      </mesh>
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

// ── Skeleton Rig（内部组件）─────────────────────────────────────────────────
function SkeletonRig({
  timeline,
  offset,
  scale,
  timeIndex,
}: {
  timeline: PoseTimeline;
  offset: [number, number, number];
  scale: number;
  timeIndex: ReturnType<typeof buildTimeIndex>;
}) {
  const buffer = useRef<Float32Array>(new Float32Array(JOINT_COUNT * 3));
  const jointRefs = useRef<Array<THREE.Mesh | null>>([]);
  const boneRefs = useRef<Array<THREE.Mesh | null>>([]);

  const tmpA = useMemo(() => new THREE.Vector3(), []);
  const tmpB = useMemo(() => new THREE.Vector3(), []);
  const tmpDir = useMemo(() => new THREE.Vector3(), []);
  const tmpMid = useMemo(() => new THREE.Vector3(), []);
  const UP = useMemo(() => new THREE.Vector3(0, 1, 0), []);

  useFrame(() => {
    const ms = timeline.getMs();
    sampleJoints(timeIndex, ms, buffer.current);
    const buf = buffer.current;

    for (let i = 0; i < JOINT_COUNT; i++) {
      const mesh = jointRefs.current[i];
      if (!mesh) continue;
      mesh.position.set(
        (buf[i * 3] + offset[0]) * scale,
        (buf[i * 3 + 1] + offset[1]) * scale,
        (buf[i * 3 + 2] + offset[2]) * scale,
      );
    }

    for (let b = 0; b < SMPL_SKELETON.length; b++) {
      const [ia, ib] = SMPL_SKELETON[b];
      const mesh = boneRefs.current[b];
      if (!mesh) continue;

      tmpA.set(
        (buf[ia * 3] + offset[0]) * scale,
        (buf[ia * 3 + 1] + offset[1]) * scale,
        (buf[ia * 3 + 2] + offset[2]) * scale,
      );
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

      {SMPL_SKELETON.map(([a, b], i) => (
        <mesh
          key={`${a}-${b}`}
          ref={(el) => {
            boneRefs.current[i] = el as THREE.Mesh | null;
          }}
          castShadow
        >
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

// ── 场景环境（灯光 + 网格 + 坐标轴）────────────────────────────────────────
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

// ── 顶层导出 ────────────────────────────────────────────────────────────────
export default function CombinedViewer({
  keyframes,
  faces,
  timeline,
  showWireframe = false,
  targetSize = 2.4,
}: CombinedViewerProps) {
  // 统一 framing：基于 mesh bounding box，同时应用于 mesh 和 skeleton
  const framing = useMemo(
    () => computeMeshFraming(keyframes, targetSize),
    [keyframes, targetSize],
  );
  const timeIndex = useMemo(() => buildTimeIndex(keyframes), [keyframes]);
  const { offset, scale } = framing;

  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      camera={{ position: [3.1, 1.2, 3.6], fov: 42, near: 0.1, far: 100 }}
      style={{ background: "transparent" }}
    >
      <SceneEnvironment />
      <SMPLMesh
        keyframes={keyframes}
        faces={faces}
        timeline={timeline}
        showWireframe={showWireframe}
        offset={offset}
        scale={scale}
        timeIndex={timeIndex}
      />
      <SkeletonRig
        timeline={timeline}
        offset={offset}
        scale={scale}
        timeIndex={timeIndex}
      />
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
