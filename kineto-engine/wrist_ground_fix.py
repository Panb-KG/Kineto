#!/usr/bin/env python3
"""P2.2 四点支撑腕背伸修正（生产后处理模块）。

HMR2 对四点支撑/鸟狗式的撑地手臂普遍缺失腕背伸：实测腕弯折角（肘-腕-手
夹角）177°（直腕），mitten 手掌垂直戳穿地面；解剖学上手掌应平放贴地、
腕背伸约 90°。本模块在 Phase3（thetas 合并完成、keyframe 写入前）批量
修正腕关节 thetas，并同步交付关节，保证 joints_3d ↔ smpl_thetas 同源
（P2/改动 C 契约），使后续 mesh forward（compute_mesh_for_keyframes）
与骨架渲染显示同一个「平放贴地」姿态。

与 P2.1（已回退）的本质区别
─────────────────────────
P2.1 用 SMPL J_reg 回归关节替换多关节锚点、IK 反算 thetas 拟合锚点，把
肩肘拉到 SMPL 流形外变形（G14 0.06mm 但形体扭曲）。本方案：
  1. **只改腕关节（j20/j21）的 rotvec**，不动 global_orient、不动其他关节；
  2. 修正旋转是世界系 Rodrigues 最小旋转（手骨旋到与小臂精确成 90° 且朝
     身体前方向），再经 FK 世界旋转映射为腕部**局部**增量，按触地权重 slerp；
  3. 关节更新只取该旋转的真实 SMPL forward 结果（J_reg 回归，与 G14 同数学），
     实测非目标关节位移 ≤2.7mm，不存在 IK 拉伸变形。

触地检测（2026-09-11 鸟狗式 885 帧诊断标定）
───────────────────────────────────────────
• 地面：全身关节包络 p95；前方向：脊柱（spine1−pelvis）水平投影中位数；
• 三门合取：腕离地 <0.20m 且 mitten 叶点离地 <0.08m 且叶点**画面平面**
  速度 <0.15m/s。速度只取 x-y 分量：实测 cam_t.z（深度）帧间抖动 p90≈0.5m
  （HMR2 弱透视尺度跳动），三维速度会把静止撑地手误判为运动，而 T.x/T.y
  抖动仅 6-13mm；
• 几何判定统一用窗 9 中值平滑后的 cam_t（仅判定用，不改交付数据）；
• 连续 ≥5 帧成段、短间隙 ≤12 帧桥接、段端 5 帧线性 fade、窗 5 滑动平均。

仅 4dhumans 模式由 kineto_core 调用；fallback/合成姿态不触碰。
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from skeleton_spec import SMPL_JOINT_NAMES, SMPL_PARENTS

# ── 关节索引一律经 SSOT 名字解析（禁止硬编码关节常量，项目规则 §2.5）──────
_PELVIS = SMPL_JOINT_NAMES.index("pelvis")
_SPINE1 = SMPL_JOINT_NAMES.index("spine1")
_L_ELBOW = SMPL_JOINT_NAMES.index("left_elbow")
_R_ELBOW = SMPL_JOINT_NAMES.index("right_elbow")
_L_WRIST = SMPL_JOINT_NAMES.index("left_wrist")
_R_WRIST = SMPL_JOINT_NAMES.index("right_wrist")
_L_HAND = SMPL_JOINT_NAMES.index("left_hand")
_R_HAND = SMPL_JOINT_NAMES.index("right_hand")

# 每只手：(腕, 肘, 手叶点, 权重字典键)
_HAND_CHAINS = (
    (_L_WRIST, _L_ELBOW, _L_HAND, "L"),
    (_R_WRIST, _R_ELBOW, _R_HAND, "R"),
)

# ── 触地检测标定参数（鸟狗式诊断数据，改动须重新离线标定）─────────────────
GROUND_PCTL = 95      # 全身包络地面分位
WRIST_NEAR_M = 0.20   # 腕近地门（四点支撑时腕离地约一掌厚）
LEAF_NEAR_M = 0.08    # mitten 叶点近地门
LEAF_VEL_MS = 0.15    # 叶点画面平面静止门（m/s）
RUN_MIN = 5           # 最短连续支撑帧数
GAP_MERGE = 12        # 支撑段间隙 ≤ 此帧数则桥接（防止权重反复横跳）
FADE = 5              # 段端线性渐入渐出帧数
SMOOTH_WIN = 5        # 权重滑动平均窗
CAM_T_MED_WIN = 9     # cam_t 中值平滑窗（仅几何判定用）


def _theta_slice(joint: int) -> slice:
    """thetas(72) 中关节 j 的 rotvec 段：0 号 global_orient，其后 body_pose。"""
    return slice(3 + (joint - 1) * 3, 3 + joint * 3)


def _median_smooth(x: np.ndarray, win: int = CAM_T_MED_WIN) -> np.ndarray:
    """边缘填充的时序中值平滑（仅用于几何判定，不改交付数据）。"""
    pad = win // 2
    xp = np.pad(x, ((pad, pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    return np.median(np.stack([xp[i:i + len(x)] for i in range(win)]), axis=0)


def _minimal_rotations(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """批量 Rodrigues 最小旋转：把单位向量 v0_b 旋到 v1_b，返回 (B,3,3)。"""
    v0 = v0 / (np.linalg.norm(v0, axis=1, keepdims=True) + 1e-12)
    v1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12)
    v = np.cross(v0, v1)
    c = np.einsum("ij,ij->i", v0, v1)
    s = np.linalg.norm(v, axis=1)
    vx = np.zeros((len(v0), 3, 3))
    vx[:, 0, 1], vx[:, 0, 2] = -v[:, 2], v[:, 1]
    vx[:, 1, 0], vx[:, 1, 2] = v[:, 2], -v[:, 0]
    vx[:, 2, 0], vx[:, 2, 1] = -v[:, 1], v[:, 0]
    ok = s > 1e-12
    k = np.where(ok, (1 - c) / np.where(ok, s * s, 1.0), 0.0)
    rot = np.broadcast_to(np.eye(3), (len(v0), 3, 3)) + vx + \
        np.einsum("bij,b->bij", vx @ vx, k)
    # 反平行（v0≈-v1，本场景手骨朝下 vs 目标水平不会出现）：逐行确定性
    # 选一正交轴转 180°，避免向量化公式下公共轴取自其他帧。
    anti = np.where((~ok) & (c < 0))[0]
    if anti.size:
        rot = rot.copy()
        for b in anti:
            a = np.array([1.0, 0.0, 0.0])
            a -= v0[b] * (v0[b] @ a)
            a /= np.linalg.norm(a)
            rot[b] = 2 * np.outer(a, a) - np.eye(3)
    return rot


def _support_weights(Jw: np.ndarray, fps: float, ground_y: float,
                     wrist: int, leaf: int) -> np.ndarray:
    """叶点+腕联合触地门 → 成段 → 桥接/丢短段 → 段端 fade + 滑动平均。

    返回 (N,) [0,1] 权重。速度仅取画面平面 x-y 分量，理由见模块文档。
    """
    n = len(Jw)
    h_w = ground_y - Jw[:, wrist, 1]
    h_l = ground_y - Jw[:, leaf, 1]
    dleaf_xy = np.gradient(Jw[:, leaf, :2], axis=0) * fps
    v_l = np.linalg.norm(dleaf_xy, axis=1)
    raw = (h_w < WRIST_NEAR_M) & (h_l < LEAF_NEAR_M) & (v_l < LEAF_VEL_MS)

    segs: list[list[int]] = []
    i = 0
    while i < n:
        if not raw[i]:
            i += 1
            continue
        j = i
        while j < n and raw[j]:
            j += 1
        segs.append([i, j])
        i = j
    merged: list[list[int]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= GAP_MERGE:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    w = np.zeros(n, dtype=np.float64)
    for i, j in merged:
        if j - i < RUN_MIN:
            continue
        w[i:j] = 1.0
        for t in range(FADE):
            if i + t < j:
                w[i + t] = min(w[i + t], (t + 1) / (FADE + 1))
            if j - 1 - t >= i:
                w[j - 1 - t] = min(w[j - 1 - t], (t + 1) / (FADE + 1))
    if SMOOTH_WIN > 1:
        w = np.convolve(w, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")
    return np.clip(w, 0.0, 1.0)


def _fk_world_rotmats(thetas: np.ndarray, parents: list[int]) -> np.ndarray:
    """从 (N,72) thetas 求 24 关节世界旋转矩阵 (N,24,3,3)。"""
    n = thetas.shape[0]
    rl = Rotation.from_rotvec(thetas.reshape(n, 24, 3)).as_matrix()
    rw = np.empty_like(rl)
    for j in range(24):
        p = parents[j]
        rw[:, j] = rl[:, j] if p < 0 else rw[:, p] @ rl[:, j]
    return rw


def _forward_regressed_joints(thetas: np.ndarray, betas: np.ndarray,
                              smpl_model, device) -> np.ndarray:
    """逐帧 SMPL forward → J_regressor 回归的 24 个模型空间关节 (N,24,3)。

    与 compute_mesh_for_keyframes 严格同路径（bs=1、pose2rot=False、矩阵
    输入、同一 J_regressor），保证本模块写回的关节与 G14 所比 mesh 关节
    数学同源；逐帧而非批量是因为生产 smpl 层按 batch_size=1 构建。
    """
    j_reg = smpl_model.J_regressor.detach().cpu().numpy().reshape(24, 6890)
    out = np.empty((len(thetas), 24, 3), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(thetas)):
            go = Rotation.from_rotvec(
                thetas[i, :3].astype(np.float64)).as_matrix().reshape(1, 1, 3, 3)
            bp = Rotation.from_rotvec(
                thetas[i, 3:].reshape(23, 3).astype(np.float64)
            ).as_matrix().reshape(1, 23, 3, 3)
            smpl_out = smpl_model(
                betas=torch.tensor(betas[i:i + 1], dtype=torch.float32, device=device),
                global_orient=torch.tensor(go, dtype=torch.float32, device=device),
                body_pose=torch.tensor(bp, dtype=torch.float32, device=device),
                pose2rot=False)
            verts = smpl_out.vertices[0].cpu().numpy()
            out[i] = j_reg @ verts
    return out


def _bend_angles(Jw: np.ndarray) -> np.ndarray:
    """肘-腕-手弯折角（度），两手列拼接 (N,2)，顺序 [L, R]。"""
    out = []
    for wrist, elbow, leaf, _ in _HAND_CHAINS:
        a = Jw[:, elbow] - Jw[:, wrist]
        b = Jw[:, leaf] - Jw[:, wrist]
        cos = np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.stack(out, axis=1)


def apply_wrist_dorsiflexion(thetas: np.ndarray, joints: np.ndarray,
                             cam_t: np.ndarray, fps: float,
                             betas: np.ndarray, smpl_model, device,
                             parents: list[int] | None = None
                             ) -> tuple[np.ndarray, np.ndarray, dict]:
    """批量修正四点支撑触地手臂的腕背伸。

    参数
    ----
    thetas : (N,72) float32，Phase3 合并后的交付 thetas（HMR2/重算值）
    joints : (N,24,3) float32，模型空间交付关节（不含 cam_t；即 final_joints）
    cam_t  : (N,3) float32
    betas  : (N,10) float32（调用方把 None 帧填 0）
    smpl_model : smplx SMPL 层（extractor.model.smpl）

    返回
    ----
    thetas_fix : (N,72) float32（仅腕 j20/j21 的 rotvec 在支撑帧被改）
    joints_fix : (N,24,3) float32（仅 active 帧的腕/手叶 4 关节同源更新）
    info : 诊断/溯源信息（支撑帧数、bend 前后中位数等）
    """
    parents = list(parents if parents is not None else SMPL_PARENTS)
    n = len(thetas)
    thetas_fix = np.asarray(thetas, dtype=np.float32).copy()
    joints_fix = np.asarray(joints, dtype=np.float32).copy()
    info = {"applied": False, "fps": float(fps), "hands": {}}

    if n == 0 or smpl_model is None:
        return thetas_fix, joints_fix, info

    # 几何判定专用坐标：cam_t 中值平滑压深度抖动（交付数据不变）。
    Jw = joints.astype(np.float64) + _median_smooth(cam_t.astype(np.float64))[:, None, :]
    ground_y = float(np.percentile(Jw[:, :, 1], GROUND_PCTL))
    spine = np.median(Jw[:, _SPINE1] - Jw[:, _PELVIS], axis=0)
    fwd = np.array([spine[0], 0.0, spine[2]])
    fwd /= max(np.linalg.norm(fwd), 1e-9)

    th64 = thetas_fix.astype(np.float64)
    rworld = _fk_world_rotmats(th64, parents)
    weights: dict[str, np.ndarray] = {}
    bend0 = _bend_angles(Jw)
    bend1 = bend0.copy()

    for col, (wrist, elbow, leaf, tag) in enumerate(_HAND_CHAINS):
        w = _support_weights(Jw, fps, ground_y, wrist, leaf)
        weights[tag] = w
        active = w > 1e-6
        info["hands"][tag] = {
            "support_frames": int(active.sum()),
            "bend_before_median_deg": round(float(np.median(bend0[active, col])), 1)
            if active.any() else None,
        }
        if not active.any():
            continue

        # 世界系目标手骨方向：身体前方向在「小臂法平面」上的投影——与小臂
        # 精确成 90°（bend=肘-腕-手夹角）且尽量朝头前向；退化回退纯 fwd。
        v0 = Jw[:, leaf] - Jw[:, wrist]
        v0 /= np.linalg.norm(v0, axis=1, keepdims=True)
        u = Jw[:, elbow] - Jw[:, wrist]
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v1 = fwd[None] - u * np.einsum("ij,j->i", u, fwd)[:, None]
        norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
        degen = (norm1[:, 0] < 1e-6)
        v1 = v1 / np.maximum(norm1, 1e-9)
        v1[degen] = fwd[None]

        Rw_world = _minimal_rotations(v0, v1)
        # 世界系修正映射到腕局部：δ_local = R_elbow_worldᵀ · δ_world · R_elbow_world
        rp = rworld[:, elbow]
        delta_local = np.transpose(rp, (0, 2, 1)) @ Rw_world @ rp
        sl = _theta_slice(wrist)
        cur_local = Rotation.from_matrix(
            Rotation.from_rotvec(th64[:, sl]).as_matrix())
        new_local = (Rotation.from_rotvec(
            Rotation.from_matrix(delta_local).as_rotvec() * w[:, None])
            * cur_local)
        th64[:, sl] = new_local.as_rotvec()
        thetas_fix[:, sl] = th64[:, sl].astype(np.float32)

    active_any = np.zeros(n, dtype=bool)
    for w in weights.values():
        active_any |= w > 1e-6
    if not active_any.any():
        info["ground_y_m"] = round(ground_y, 4)
        return thetas_fix, joints_fix, info

    # 修后姿态的真实 SMPL 回归关节（与 mesh/G14 同数学），pelvis 锚定到交付
    # 坐标系后，active 帧写回**全部 24 关节**。原因：SMPL LBS 蒙皮使腕旋转
    # 经皮肤顶点波及对侧手关节（实测 ~1-2mm），只写回支撑手会导致未支撑侧
    # 手关节与 mesh forward 不一致、G14 出现 ~2mm 偏差。全量写回后 joints↔
    # thetas↔mesh 完全同源，G14 = 0.000mm；j0-19 蒙皮波及 ≤2.7mm 属 SMPL 真实
    # 结果，应反映在交付关节中（骨长变化 < 阈值，不触发审计）。
    idx = np.where(active_any)[0]
    fk = _forward_regressed_joints(
        thetas_fix[idx], np.asarray(betas, dtype=np.float32)[idx],
        smpl_model, device)
    t_align = joints_fix[idx, 0:1] - fk[:, 0:1]
    fk_aligned = fk + t_align
    joints_fix[idx] = fk_aligned.astype(np.float32)

    # 修后 bend（用写回关节 + 平滑 cam_t 重建世界系，仅日志/溯源用）
    Jw1 = joints_fix.astype(np.float64) \
        + _median_smooth(cam_t.astype(np.float64))[:, None, :]
    bend1 = _bend_angles(Jw1)
    for col, (_, _, _, tag) in enumerate(_HAND_CHAINS):
        active = weights[tag] > 1e-6
        if active.any():
            info["hands"][tag]["bend_after_median_deg"] = \
                round(float(np.median(bend1[active, col])), 1)
            info["hands"][tag]["weight_max"] = round(float(weights[tag].max()), 3)

    info.update({
        "applied": True,
        "ground_y_m": round(ground_y, 4),
        "params": {"wrist_near_m": WRIST_NEAR_M, "leaf_near_m": LEAF_NEAR_M,
                   "leaf_vel_ms": LEAF_VEL_MS, "run_min": RUN_MIN,
                   "gap_merge": GAP_MERGE, "fade": FADE,
                   "smooth_win": SMOOTH_WIN, "cam_t_med_win": CAM_T_MED_WIN},
    })
    return thetas_fix, joints_fix, info
