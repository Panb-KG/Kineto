/**
 * lib/skeleton.ts
 * ─────────────────────────────────────────────────────────────────────────
 * SMPL 24 关节标准拓扑定义，**精确镜像**引擎侧骨架单一事实源 (SSOT)
 * kineto-engine/skeleton_spec.py 的 SMPL_JOINT_NAMES / SMPL_PARENTS /
 * SMPL_SKELETON / BONE_PART_MAP；部位配色（非 SSOT 管辖）镜像
 * kineto_core.py 的 BONE_COLORS。
 *
 * canonical 序要点（关节序根因整改 P1 之后）：
 *  - joints_3d 的下标即 SMPL canonical 序（0=pelvis、9=spine3、12=neck、
 *    15=head），**不是** HMR2 内部 pred_keypoints_3d 的 OpenPose Body-25 序；
 *  - collar(13/14) 的父节点是 spine3(9)（真 kintree），不是 neck(12)：
 *    旧假边 (12,13)/(12,14) 已淘汰，改为 (9,13)/(9,14)；
 *  - 23 条边含 ankle→foot(7→10, 8→11) 与 wrist→hand(20→22, 21→23)。
 *
 * 一致性闸门：deploy/check_ssot.py（validate.sh 的 G12）逐项比对本文件与
 * SSOT（A1 关节名 / A2 父表 / A3 骨架边 / A4 部位归属）。本文件只做镜像：
 * 如需改动，先改 SSOT 再同步此处，切勿单独调整顺序或增删骨对。
 */

/**
 * 24 个 SMPL 关节名称，顺序与 joints_3d 数组下标严格一致
 * （镜像 skeleton_spec.py 的 SMPL_JOINT_NAMES，canonical 序）。
 */
export const SMPL_JOINT_NAMES = [
  "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
  "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
  "neck", "left_collar", "right_collar", "head", "left_shoulder",
  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
  "left_hand", "right_hand",
] as const;

export type JointName = (typeof SMPL_JOINT_NAMES)[number];

/**
 * SMPL canonical 运动学树父节点表（24 项，-1 = 根/骨盆），镜像
 * skeleton_spec.py 的 SMPL_PARENTS（源自 basicModel pkl 的 kintree_table）。
 * canonical 序下父索引恒小于子索引；collar(13/14) 的父为 spine3(9)。
 */
export const SMPL_PARENTS: readonly number[] = [
  -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
  13, 14, 16, 17, 18, 19, 20, 21,
] as const;

/**
 * 骨骼连接（parent, child）索引对，共 23 根骨骼 = 真 kintree 的全部父→子边，
 * 即 SMPL_PARENTS 的展开（镜像 skeleton_spec.py 的 SMPL_SKELETON）。
 */
export const SMPL_SKELETON: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [0, 2], [0, 3],        // pelvis → hips, spine1
  [1, 4], [2, 5],                // hip → knee
  [3, 6],                        // spine1 → spine2
  [4, 7], [5, 8],                // knee → ankle
  [6, 9],                        // spine2 → spine3
  [7, 10], [8, 11],              // ankle → foot
  [9, 12],                       // spine3 → neck
  [9, 13], [9, 14],              // spine3 → collars（真 kintree；旧假边 [12,13]/[12,14] 已淘汰）
  [12, 15],                      // neck → head
  [13, 16], [14, 17],            // collar → shoulder
  [16, 18], [17, 19],            // shoulder → elbow
  [18, 20], [19, 21],            // elbow → wrist
  [20, 22], [21, 23],            // wrist → hand
];

/** 身体部位分类。 */
export type BodyPart =
  | "torso"
  | "left_arm"
  | "right_arm"
  | "left_leg"
  | "right_leg"
  | "head";

/**
 * kineto_core.py 中的 BONE_COLORS 以 BGR（OpenCV 约定）存储（配色不属于骨架
 * SSOT，拓扑常量以 skeleton_spec.py 为准）。
 * 这里保留原始 BGR 值，随后统一转换为 three.js 使用的 RGB，
 * 使前端渲染出的颜色与后端可视化在视觉上完全一致。
 */
const BONE_COLORS_BGR: Record<BodyPart, [number, number, number]> = {
  torso: [200, 200, 100],
  left_arm: [100, 200, 100],
  right_arm: [100, 100, 200],
  left_leg: [200, 100, 100],
  right_leg: [100, 200, 200],
  head: [255, 200, 200],
};

/** 将 [0,255] 的 BGR 三元组转换为 RGB CSS hex（交换 R/B 通道）。 */
function bgrToHex([b, g, r]: [number, number, number]): string {
  const to = (v: number) => v.toString(16).padStart(2, "0");
  return `#${to(r)}${to(g)}${to(b)}`;
}

/** 部位 → RGB hex 颜色（供 three.js / CSS 使用）。 */
export const BONE_COLORS: Record<BodyPart, string> = {
  torso: bgrToHex(BONE_COLORS_BGR.torso),
  left_arm: bgrToHex(BONE_COLORS_BGR.left_arm),
  right_arm: bgrToHex(BONE_COLORS_BGR.right_arm),
  left_leg: bgrToHex(BONE_COLORS_BGR.left_leg),
  right_leg: bgrToHex(BONE_COLORS_BGR.right_leg),
  head: bgrToHex(BONE_COLORS_BGR.head),
};

/** 部位 → 原始 BGR 值（保留以便与后端比对）。 */
export const BONE_COLORS_RAW_BGR = BONE_COLORS_BGR;

/**
 * 骨骼部位映射，键为 `${parent}-${child}`，键集合与 SMPL_SKELETON 完全一致。
 * 镜像 skeleton_spec.py 的 BONE_PART_MAP：左右臂链首条边挂在 spine3(9)→collar
 * （(9,13)=left_arm、(9,14)=right_arm），不再是 neck(12)→collar。
 */
const BONE_PART_MAP_ENTRIES: ReadonlyArray<[readonly [number, number], BodyPart]> = [
  [[0, 3], "torso"], [[3, 6], "torso"], [[6, 9], "torso"], [[9, 12], "torso"],
  [[9, 13], "left_arm"], [[13, 16], "left_arm"], [[16, 18], "left_arm"], [[18, 20], "left_arm"], [[20, 22], "left_arm"],
  [[9, 14], "right_arm"], [[14, 17], "right_arm"], [[17, 19], "right_arm"], [[19, 21], "right_arm"], [[21, 23], "right_arm"],
  [[0, 1], "left_leg"], [[1, 4], "left_leg"], [[4, 7], "left_leg"], [[7, 10], "left_leg"],
  [[0, 2], "right_leg"], [[2, 5], "right_leg"], [[5, 8], "right_leg"], [[8, 11], "right_leg"],
  [[12, 15], "head"],
];

export const BONE_PART_MAP: ReadonlyMap<string, BodyPart> = new Map(
  BONE_PART_MAP_ENTRIES.map(([[a, b], part]) => [`${a}-${b}`, part] as const),
);

/** 查询某根骨骼的部位；未命中时回退到 torso。 */
export function bonePart(a: number, b: number): BodyPart {
  return BONE_PART_MAP.get(`${a}-${b}`) ?? "torso";
}

/** 查询某根骨骼的 RGB 颜色。 */
export function boneColor(a: number, b: number): string {
  return BONE_COLORS[bonePart(a, b)];
}

/** 关节索引 → 名称。 */
export function jointName(index: number): JointName {
  return SMPL_JOINT_NAMES[index];
}

/**
 * 开发期镜像自检（仅 DEV，只告警、**绝不抛错**，不影响生产渲染）：
 * 本文件内三份拓扑常量必须自洽，并守住 P1 整改的核心事实——
 * collar(13/14) 的父为 spine3(9)。权威闸门仍是 deploy/check_ssot.py（G12），
 * 此处只为在浏览器侧尽早暴露本地漂移。
 */
if (process.env.NODE_ENV !== "production") {
  const edgeKey = (e: readonly [number, number]) => `${e[0]}-${e[1]}`;
  const derived = new Set(
    SMPL_PARENTS.flatMap((p, c) => (p >= 0 ? [edgeKey([p, c])] : [])),
  );
  const literal = SMPL_SKELETON.map(edgeKey);
  const problems: string[] = [];

  if (literal.length !== derived.size || literal.some((k) => !derived.has(k))) {
    problems.push("SMPL_SKELETON 与 SMPL_PARENTS 的展开不一致");
  }
  if (SMPL_PARENTS[13] !== 9 || SMPL_PARENTS[14] !== 9) {
    problems.push("collar(13/14) 的父不是 spine3(9)（真 kintree）");
  }
  if (BONE_PART_MAP.size !== SMPL_SKELETON.length) {
    problems.push("BONE_PART_MAP 键数与骨架边数不一致");
  }
  if (SMPL_JOINT_NAMES.length !== 24) {
    problems.push("SMPL_JOINT_NAMES 不是 24 项");
  }

  if (problems.length > 0) {
    // eslint-disable-next-line no-console
    console.warn(
      `[kineto-web/skeleton] 骨架镜像自检异常：${problems.join("；")}` +
        "（权威源：kineto-engine/skeleton_spec.py，闸门：deploy/check_ssot.py）",
    );
  }
}
