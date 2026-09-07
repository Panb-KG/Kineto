#!/usr/bin/env python3
"""
Kineto Pose Audit - 姿态解算质量自动审核
对 pose_data.json 进行多维度检查，输出质量报告 + 问题帧可视化
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 骨架解剖常量单一事实源 (SSOT)：骨对/命名/骨长界/对称对/角度 pivot 均由
# skeleton_spec 从真 SMPL kintree + basicModel pkl 派生，与 kineto_core 共享，
# 消除旧版双骨架树冲突（collar 13/14 父节点误作 12，真 kintree 为 9）、
# BONE_PAIRS 命名自 collar 之后整体错位一格、硬编码对称下标 (13,16,14,17)、
# 以及跨模块骨长界不一致（如 (13,16) 一处 0.22-0.40、另一处 0.20-0.35）。
from skeleton_spec import (
    BONE_LENGTH_BOUNDS,
    BONE_NAMES,
    JOINT_ANGLE_PIVOTS,
    JOINT_ANGLE_RANGES,
    SMPL_SKELETON,
    SYMMETRIC_BONE_PAIRS,
)

# ============================================================================
# 人体解剖学约束 (基于 SMPL 24 关节 canonical 序，均来自 skeleton_spec SSOT)
# ============================================================================

# 骨长合理范围 (米)：由真实 rest 骨长（J_regressor @ v_template）数据驱动派生，
# 覆盖全部 23 条 kintree 骨；键名由 BONE_NAMES 稳定生成。
BONE_LENGTH_RANGES = {BONE_NAMES[bone]: bounds for bone, bounds in BONE_LENGTH_BOUNDS.items()}

# 审核用骨对 (parent, child, name)：即真 kintree 全部边。
BONE_PAIRS = [(p, c, BONE_NAMES[(p, c)]) for p, c in SMPL_SKELETON]


# ============================================================================
# [P4/改动 G] 可视化开关 + 采样帧缓存（性能/IO 缓解，不改输出语义）
# ============================================================================

def _visualize_enabled():
    """审计可视化总开关（env KINETO_AUDIT_VISUALIZE）。默认开=保持当前行为；
    设 0/false/no/off 关闭，跳过 audit_frame_*.jpg 产出与其 seek/IO。可视化在
    generate_report 之后运行且不写回 audit_results，故开关**不改 score/verdict**。"""
    raw = os.environ.get("KINETO_AUDIT_VISUALIZE")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _default_cache_size():
    """采样帧缓存默认容量（env KINETO_AUDIT_CACHE_FRAMES，默认 48；<=0 禁用缓存=
    每次重新解码，等价原行为仅无复用）。有界 FIFO 防长视频缓存无界增长 OOM。"""
    raw = os.environ.get("KINETO_AUDIT_CACHE_FRAMES")
    try:
        return int(raw) if raw is not None and raw.strip() != "" else 48
    except ValueError:
        return 48


class FrameCache:
    """有界采样帧缓存：跨 check_projection_alignment / visualize_problem_frames 及
    跨 refine iteration 复用已解码帧，消除冗余 seek+decode（原每轮 proj~10 +
    visualize~20，跨 3 iteration 最多 ~90 次）。

    无回归保证：
      - 只决定“是否重新解码”，不改变被采样帧集合（sample_indices 由调用方按原逻辑
        算），也不改变解码结果——同一视频同一 idx 的 seek+read 结果确定，缓存命中
        返回的 (ret, frame) 与重新解码逐位一致。
      - 缓存 (ret, frame)：解码失败(ret=False) 也缓存，保持与原逐次 seek 一致的跳过
        语义（失败帧同样跳过，不影响 mean_in_person_ratio）。
      - visualize 在缓存帧上绘制前必须 .copy()（见其实现），避免 in-place 绘制污染
        缓存中的共享只读帧。
    max_entries<=0 时退化为每次重新解码（等价原行为，仅无复用）。
    """

    def __init__(self, video_path, max_entries=None):
        self.video_path = str(video_path)
        self.max_entries = _default_cache_size() if max_entries is None else int(max_entries)
        self._cap = cv2.VideoCapture(self.video_path)
        self._ok = self._cap.isOpened()
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._ok else 0
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._ok else 0
        self._cache: dict = {}
        self._order: list = []
        self.seek_count = 0   # 实际 seek+decode 次数（性能计数）
        self.hit_count = 0    # 缓存命中次数

    @property
    def opened(self):
        return self._ok

    def get(self, idx):
        """返回 (ret, frame)。frame 为缓存内共享只读对象，调用方若要修改须先 .copy()。"""
        idx = int(idx)
        if not self._ok:
            return False, None
        if idx in self._cache:
            self.hit_count += 1
            return self._cache[idx]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        self.seek_count += 1
        if not ret:
            frame = None
        if self.max_entries > 0:
            if len(self._cache) >= self.max_entries:
                old = self._order.pop(0)
                self._cache.pop(old, None)
            self._cache[idx] = (ret, frame)
            self._order.append(idx)
        return (ret, frame)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._ok = False
        self._cache.clear()
        self._order.clear()


# ============================================================================
# 审核检查项
# ============================================================================

def check_bone_length_consistency(joints_arr):
    """骨长一致性：同一根骨头在不同帧的长度波动应很小"""
    results = {}
    for i, j, name in BONE_PAIRS:
        lengths = np.linalg.norm(joints_arr[:, i, :] - joints_arr[:, j, :], axis=1)
        mean_len = lengths.mean()
        std_len = lengths.std()
        cv_ratio = std_len / mean_len if mean_len > 1e-6 else 999

        min_v, max_v = BONE_LENGTH_RANGES.get(name, (0, 99))
        in_range = min_v <= mean_len <= max_v

        results[name] = {
            "mean": float(mean_len),
            "std": float(std_len),
            "cv": float(cv_ratio),
            "in_range": in_range,
            "range": [min_v, max_v],
        }
    return results


def check_bone_length_validity(bone_results):
    """骨长是否在解剖学合理范围内"""
    issues = []
    for name, info in bone_results.items():
        if not info["in_range"]:
            issues.append({
                "bone": name,
                "mean": info["mean"],
                "range": info["range"],
                "severity": "high" if abs(info["mean"] - np.mean(info["range"])) > 0.3 else "medium",
            })
    return issues


def check_temporal_smoothness(joints_arr, fps):
    """帧间平滑度：相邻帧关节位移不应突变。

    [M2] spike 阈值改为**相对 + 绝对下限**：max(0.05, 3.0×mean_disp)。旧固定 0.05
    对慢动作/小幅运动序列会把正常帧间位移误判为 spike（虚高 spike_count → verdict
    被帧数级污染）；相对阈值随序列自身运动尺度自适应，绝对下限 0.05 防极静止序列
    把噪声当 spike。返回体 additive 增加 spike_ratio（spike 帧占比）与 n_frames，供
    compute_verdict 用**比率**而非**帧数**裁决（量纲一致）。
    """
    diffs = np.diff(joints_arr, axis=0)
    per_frame_disp = np.linalg.norm(diffs, axis=2).mean(axis=1)

    n_frames = int(len(per_frame_disp))
    mean_disp = float(per_frame_disp.mean()) if n_frames > 0 else 0.0
    max_disp = float(per_frame_disp.max()) if n_frames > 0 else 0.0
    # [M2] 相对 + 绝对下限阈值（mean_disp=0 时退化为绝对下限 0.05）
    threshold = max(0.05, 3.0 * mean_disp) if mean_disp > 0 else 0.05

    spike_frames = np.where(per_frame_disp > threshold)[0]
    spike_ratio = float(len(spike_frames) / n_frames) if n_frames > 0 else 0.0

    return {
        "mean_displacement": mean_disp,
        "max_displacement": max_disp,
        "spike_count": int(len(spike_frames)),
        "spike_frames": spike_frames[:20].tolist(),
        "spike_ratio": spike_ratio,
        "n_frames": n_frames,
        "threshold": float(threshold),
    }


def check_left_right_symmetry(joints_arr):
    """左右对称性：对称骨骼长度应接近（对称对由 SSOT 真 kintree 派生，
    替换旧硬编码下标 13,16,14,17 等错位写法）"""
    results = {}
    for l_bone, r_bone in SYMMETRIC_BONE_PAIRS:
        l_name = BONE_NAMES[l_bone]
        r_name = BONE_NAMES[r_bone]
        li, lj = l_bone
        ri, rj = r_bone
        l_len = np.linalg.norm(joints_arr[:, li, :] - joints_arr[:, lj, :], axis=1).mean()
        r_len = np.linalg.norm(joints_arr[:, ri, :] - joints_arr[:, rj, :], axis=1).mean()
        ratio = l_len / r_len if r_len > 1e-6 else 0
        asymmetry = abs(1 - ratio)
        results[l_name] = {
            "left": float(l_len),
            "right": float(r_len),
            "ratio": float(ratio),
            "asymmetry": float(asymmetry),
        }
    return results


def check_joint_angles(joints_arr):
    """关节角度是否在解剖学合理范围内（pivot 三元组由 SSOT 真 kintree 派生：
    (a, pivot, b) 中 (a,pivot)/(pivot,b) 均为真实骨边；旧表 l_elbow 误用
    (13,16,18)——13 是 collar 而非 shoulder，已修正为 (16,18,20)）

    [P4/改动 G] 向量化：弃逐帧 Python 循环（466 帧×8 检查 ~20ms），改批量 numpy
    （~0.26ms，~77x）。**逐位无回归**：点积用 np.matmul 批量 BLAS gemm（与逐帧
    np.dot 同一 FMA 路径），范数用 sqrt(matmul(v,v))（与逐帧 np.linalg.norm 一致）
    ——实测 float32/float64 下与逐帧参考 np.array_equal 逐位相同（max_abs_diff=0、
    out_of_range 翻转=0），故 joint_angles 及下游 verdict/score 完全不变。
    （element-wise (v1*v2).sum() 会避开 BLAS FMA 产生 ~7e-5 度偏差，故不采用。）"""
    joints_arr = np.asarray(joints_arr)

    def _batch_dot(a, b):
        # (F,1,3)@(F,3,1)->(F,)：批量 BLAS 点积，逐位等价于逐帧 np.dot(a[f], b[f])
        return np.matmul(a[:, None, :], b[:, :, None]).reshape(-1)

    angle_checks = JOINT_ANGLE_PIVOTS

    results = {}
    for name, (p, pivot_j, c) in angle_checks.items():
        pivot = joints_arr[:, pivot_j, :]
        v1 = joints_arr[:, p, :] - pivot
        v2 = joints_arr[:, c, :] - pivot
        n1 = np.sqrt(_batch_dot(v1, v1))
        n2 = np.sqrt(_batch_dot(v2, v2))
        cos = np.clip(_batch_dot(v1, v2) / (n1 * n2 + 1e-8), -1, 1)
        angles = np.degrees(np.arccos(cos))
        min_a, max_a = JOINT_ANGLE_RANGES.get(name, (0, 180))
        out_of_range = np.sum((angles < min_a) | (angles > max_a))
        results[name] = {
            "mean": float(angles.mean()),
            "std": float(angles.std()),
            "min": float(angles.min()),
            "max": float(angles.max()),
            "out_of_range_count": int(out_of_range),
            "range": [min_a, max_a],
        }
    return results


def check_pose_orientation(joints_arr):
    """检测身体朝向：卧姿 vs 直立"""
    pelvis = joints_arr[:, 0, :]
    neck = joints_arr[:, 12, :]
    head = joints_arr[:, 15, :]

    body_vec = neck - pelvis
    body_vertical = np.abs(body_vec[:, 1]).mean()
    body_horizontal = np.sqrt(np.abs(body_vec[:, 0]).mean()**2 + np.abs(body_vec[:, 2]).mean()**2)

    is_supine = body_horizontal > body_vertical * 1.5

    return {
        "body_vertical_extent": float(body_vertical),
        "body_horizontal_extent": float(body_horizontal),
        "is_supine": bool(is_supine),
        "orientation": "supine" if is_supine else "upright",
    }


def check_projection_alignment(joints_arr, video_path, output_dir, max_samples=10,
                               cam_t_arr=None, bbox_arr=None, focal_length=5000.0,
                               frame_cache=None):
    """投影对齐检查：将 3D 骨架投影回 2D，检查是否落在人物 bbox 内

    [P4/改动 G] 用 frame_cache 复用已解码帧，消除冗余 seek+decode。**不改采样**：
    sample_indices 仍按原 np.linspace 计算，被审计帧集合不变；投影数学只用
    joints/cam_t/bbox（不读 frame 像素），缓存仅替代“解码”这一步并保留原 ret 跳过
    语义（解码失败帧同样跳过），故 samples/mean_in_person_ratio 逐位不变。
    """
    if not Path(video_path).exists():
        return {"error": "video not found"}

    own_cache = frame_cache is None
    cache = frame_cache if frame_cache is not None else FrameCache(video_path)
    if not cache.opened:
        if own_cache:
            cache.release()
        return {"error": "cannot open video"}

    w = cache.width
    h = cache.height
    cx, cy = w / 2, h / 2
    # cam_t 在全图分辨率下换算，焦距需从 256 模型空间同步缩放
    focal = focal_length / 256.0 * max(w, h)

    sample_indices = np.linspace(0, len(joints_arr) - 1, max_samples, dtype=int)
    alignment_scores = []

    for idx in sample_indices:
        ret, frame = cache.get(idx)
        if not ret:
            continue

        joints = joints_arr[idx]
        cam_t = cam_t_arr[idx] if cam_t_arr is not None else np.zeros(3, dtype=np.float32)

        points_2d = []
        for joint in joints:
            J = joint + cam_t
            if abs(J[2]) < 1e-6:
                points_2d.append((0, 0))
                continue
            u = focal * J[0] / J[2] + cx
            v = focal * J[1] / J[2] + cy
            points_2d.append((int(u), int(v)))

        if bbox_arr is not None:
            x1, y1, x2, y2 = bbox_arr[idx]
            margin_x = (x2 - x1) * 0.2
            margin_y = (y2 - y1) * 0.2
            x1 -= margin_x
            y1 -= margin_y
            x2 += margin_x
            y2 += margin_y
            in_bbox = sum(1 for px, py in points_2d if x1 <= px <= x2 and y1 <= py <= y2)
            total = len(points_2d)
            alignment_scores.append({
                "frame": int(idx),
                "in_person_ratio": in_bbox / total,
                "points_in_bbox": in_bbox,
                "total_points": total,
            })
        else:
            in_frame = sum(1 for px, py in points_2d if 0 <= px < w and 0 <= py < h)
            total = len(points_2d)
            alignment_scores.append({
                "frame": int(idx),
                "in_person_ratio": in_frame / total,
                "points_in_bbox": in_frame,
                "total_points": total,
            })

    if own_cache:
        cache.release()
    return {
        "samples": alignment_scores,
        "mean_in_person_ratio": float(np.mean([s["in_person_ratio"] for s in alignment_scores]))
            if alignment_scores else 0,
    }


# ============================================================================
# 可视化：问题帧标注
# ============================================================================

def visualize_problem_frames(joints_arr, video_path, output_dir, audit_results, max_frames=20,
                             cam_t_arr=None, focal_length=5000.0, frame_cache=None):
    """将问题帧叠加骨架标注输出

    [P4/改动 G] 用 frame_cache 复用已解码帧（与 check_projection_alignment 及跨
    iteration 共享）。**关键**：绘制（cv2.line/circle/putText）会 in-place 修改帧，
    故从缓存取帧后必须先 .copy() 再绘制，避免污染缓存中的共享只读帧——保证输出的
    audit_frame_*.jpg 与“每次重新解码后绘制”逐位一致。seek 上限 max_frames(≤20) 保留。
    本函数仅产出诊断图片、不参与 score/verdict，故开关/缓存均不改审计结论。
    """
    if not Path(video_path).exists():
        return []

    own_cache = frame_cache is None
    cache = frame_cache if frame_cache is not None else FrameCache(video_path)
    if not cache.opened:
        if own_cache:
            cache.release()
        return []

    w = cache.width
    h = cache.height
    cx, cy = w / 2, h / 2
    focal = focal_length / 256.0 * max(w, h)

    problem_frames = set()

    smooth = audit_results.get("temporal_smoothness", {})
    for f in smooth.get("spike_frames", []):
        problem_frames.add(f)

    bone = audit_results.get("bone_validity", [])
    if bone:
        for f in range(0, len(joints_arr), max(1, len(joints_arr) // 5)):
            problem_frames.add(f)

    sample_indices = sorted(list(problem_frames))[:max_frames]
    if not sample_indices:
        sample_indices = [0, len(joints_arr)//4, len(joints_arr)//2, len(joints_arr)*3//4, len(joints_arr)-1]

    out_imgs = []
    for idx in sample_indices:
        ret, frame = cache.get(idx)
        if not ret:
            continue
        frame = frame.copy()  # 缓存帧共享只读；in-place 绘制前必须复制，避免污染缓存

        joints = joints_arr[idx]
        cam_t = cam_t_arr[idx] if cam_t_arr is not None else np.zeros(3, dtype=np.float32)

        points_2d = []
        for joint in joints:
            J = joint + cam_t
            if abs(J[2]) < 1e-6:
                points_2d.append((0, 0))
                continue
            u = focal * J[0] / J[2] + cx
            v = focal * J[1] / J[2] + cy
            points_2d.append((int(u), int(v)))

        skeleton_edges = SMPL_SKELETON  # SSOT 真 kintree（含 ankle→foot/wrist→hand，collar 父=9）

        for i, j in skeleton_edges:
            if i < len(points_2d) and j < len(points_2d):
                cv2.line(frame, points_2d[i], points_2d[j], (0, 0, 255), 2)

        for idx2, pt in enumerate(points_2d):
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)

        cv2.putText(frame, f"frame {idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_path = output_dir / f"audit_frame_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        out_imgs.append(str(out_path))

    if own_cache:
        cache.release()
    return out_imgs


# ============================================================================
# 机器可读判决（P3/改动 E）
# ============================================================================

def compute_verdict(audit_results):
    """基于 P1/P2 后的**诚实信号**产出机器可读判决（写入 audit_results.json）。

    [M2] 量纲重设计（整改“spike 帧数直接累加进 total_issues”的量纲混用）：
      - total_issues 只累加**同量纲的计数**：骨长越界骨数 + (spike_ratio>0.10 ? 1 : 0)。
        spike 用**比率**而非帧数（466 帧序列的 20 个 spike ≠ 20 个问题）。
      - 关节角度越界改为**信息性 angle_caveat**（不计入 total_issues、不单独降级）：
        单帧角度越界常见于真实运动（如髅关节大幅屈伸），按帧计数会污染裁决；仅记录
        越界帧占比供人工复核，不改变 pass/warn/fail。
      - fail 判据用**显式解剖/投影失效**：high 级骨长越界 / 骨长越界骨数≥5 /
        投影对齐率<0.50 / 投影证据缺失（[m1]）。
      - warn：骨长越界>0 / 投影对齐率<0.90 / spike_ratio>0.10。
    卧姿(is_supine) 为信息性 caveat，不单独降级 verdict（管线已用旋转欺骗处理横卧）。
    返回 {verdict, total_issues, failure_reason, spike_ratio, angle_caveat}。
    """
    bone_val = audit_results.get("bone_validity", []) or []
    smooth = audit_results.get("temporal_smoothness", {}) or {}
    angles = audit_results.get("joint_angles", {}) or {}
    proj = audit_results.get("projection_alignment", {}) or {}

    bone_issues = len(bone_val)
    high_sev_bones = sum(1 for b in bone_val
                         if isinstance(b, dict) and b.get("severity") == "high")

    # [M2] spike 用比率裁决（量纲一致）；兼容旧审计无 spike_ratio 时按 count/n_frames 回退
    n_frames = max(int(smooth.get("n_frames", 0) or 0), 1)
    spike_count = int(smooth.get("spike_count", 0) or 0)
    spike_ratio = smooth.get("spike_ratio")
    if not isinstance(spike_ratio, (int, float)) or isinstance(spike_ratio, bool):
        spike_ratio = spike_count / n_frames
    spike_ratio = float(spike_ratio)
    spike_issue = 1 if spike_ratio > 0.10 else 0

    # [M2] 角度越界：信息性 caveat（不进 total_issues），记录越界帧占比供复核
    angle_caveat = []
    for name, info in angles.items():
        if not isinstance(info, dict):
            continue
        oorc = int(info.get("out_of_range_count", 0) or 0)
        if oorc > 0:
            angle_caveat.append({"joint": name, "out_of_range_count": oorc,
                                 "out_of_range_frac": round(oorc / n_frames, 4)})

    proj_ratio = proj.get("mean_in_person_ratio")
    proj_ok = isinstance(proj_ratio, (int, float)) and not isinstance(proj_ratio, bool)
    # [m1] 投影证据缺失（视频不存在/打不开 → {"error": ...}，或 ratio 非数值）：与
    # compute_quality_score“缺失按 0 分”方向一致，统一判 fail（证据缺失时不得判 pass）。
    proj_missing = (not proj_ok) or ("error" in proj)

    # [M2] total_issues 分量纲：骨长越界骨数 + spike 比率问题（角度为信息性，不计入）
    total_issues = bone_issues + spike_issue

    reasons = []
    fail = False
    if high_sev_bones > 0:
        fail = True
        reasons.append(f"{high_sev_bones} 条骨长严重越界(high)")
    if bone_issues >= 5:
        fail = True
        reasons.append(f"{bone_issues} 条骨长越界 ≥ 5（解剖失效）")
    if proj_missing:
        fail = True
        reasons.append(f"投影对齐证据缺失({proj.get('error', 'no valid samples')})，不得判 pass")
    elif float(proj_ratio) < 0.5:
        fail = True
        reasons.append(f"投影对齐率 {float(proj_ratio):.2f} < 0.50（骨架未落在人物上）")

    if fail:
        verdict = "fail"
    elif bone_issues > 0 or spike_issue > 0 or (proj_ok and float(proj_ratio) < 0.9):
        verdict = "warn"
        if bone_issues:
            reasons.append(f"{bone_issues} 条骨长越界")
        if spike_issue:
            reasons.append(f"时序突变帧占比 {spike_ratio:.2f} > 0.10")
        if proj_ok and float(proj_ratio) < 0.9:
            reasons.append(f"投影对齐率 {float(proj_ratio):.2f} < 0.90")
    else:
        verdict = "pass"

    return {
        "verdict": verdict,
        "total_issues": int(total_issues),
        "failure_reason": ("; ".join(reasons) if reasons else None),
        "spike_ratio": round(spike_ratio, 4),
        "angle_caveat": angle_caveat,
    }


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(audit_results, output_dir):
    """生成文本质量报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("  Kineto Pose Audit Report")
    lines.append("=" * 60)
    lines.append("")

    orientation = audit_results.get("pose_orientation", {})
    lines.append(f"[Orientation] {orientation.get('orientation', 'unknown')}")
    lines.append(f"  vertical: {orientation.get('body_vertical_extent', 0):.4f}")
    lines.append(f"  horizontal: {orientation.get('body_horizontal_extent', 0):.4f}")
    if orientation.get("is_supine"):
        lines.append("  ⚠️  检测到卧姿 — 模型可能按直立假设解算，偏差会较大")
    lines.append("")

    bone_val = audit_results.get("bone_validity", [])
    lines.append(f"[Bone Validity] {len(bone_val)} issues")
    for b in bone_val:
        tag = "🔴" if b["severity"] == "high" else ""
        lines.append(f"  {tag} {b['bone']}: mean={b['mean']:.4f} (expected {b['range']})")
    lines.append("")

    bone_cons = audit_results.get("bone_consistency", {})
    lines.append("[Bone Consistency] (CV = std/mean, 越小越稳定)")
    for name, info in sorted(bone_cons.items(), key=lambda x: -x[1]["cv"]):
        tag = "✅" if info["cv"] < 0.05 else "⚠️" if info["cv"] < 0.15 else "🔴"
        lines.append(f"  {tag} {name}: mean={info['mean']:.4f} cv={info['cv']:.4f}")
    lines.append("")

    smooth = audit_results.get("temporal_smoothness", {})
    lines.append(f"[Temporal Smoothness]")
    lines.append(f"  mean displacement: {smooth.get('mean_displacement', 0):.6f}")
    lines.append(f"  max displacement:  {smooth.get('max_displacement', 0):.6f}")
    lines.append(f"  spike frames:      {smooth.get('spike_count', 0)}")
    if smooth.get("spike_frames"):
        lines.append(f"  spike indices:     {smooth['spike_frames'][:10]}")
    lines.append("")

    symm = audit_results.get("symmetry", {})
    lines.append("[Left-Right Symmetry]")
    for name, info in symm.items():
        tag = "✅" if info["asymmetry"] < 0.1 else "️" if info["asymmetry"] < 0.3 else "🔴"
        lines.append(f"  {tag} {name}: L={info['left']:.4f} R={info['right']:.4f} ratio={info['ratio']:.3f}")
    lines.append("")

    angles = audit_results.get("joint_angles", {})
    lines.append("[Joint Angles]")
    for name, info in angles.items():
        tag = "✅" if info["out_of_range_count"] == 0 else "⚠️" if info["out_of_range_count"] < 50 else "🔴"
        lines.append(f"  {tag} {name}: mean={info['mean']:.1f}° range=[{info['min']:.1f}, {info['max']:.1f}]° "
                      f"out={info['out_of_range_count']}/{audit_results.get('_total_frames', 0)}")
    lines.append("")

    proj = audit_results.get("projection_alignment", {})
    if "mean_in_person_ratio" in proj:
        lines.append(f"[Projection Alignment]")
        lines.append(f"  mean in-person ratio: {proj['mean_in_person_ratio']:.3f}")
        for s in proj.get("samples", []):
            tag = "✅" if s["in_person_ratio"] > 0.6 else "⚠️"
            lines.append(f"  {tag} frame {s['frame']}: {s['points_in_bbox']}/{s['total_points']} in person bbox")
        lines.append("")

    lines.append("=" * 60)
    lines.append("  Summary")
    lines.append("=" * 60)

    # [P3/改动 E] 机器可读判决：注入 audit_results（随后写入 audit_results.json），
    # 供 api._run_engine 门禁以 verdict 为主信号裁决；同时渲染进人类可读报告。
    verdict_info = compute_verdict(audit_results)
    verdict = verdict_info["verdict"]
    total_issues = verdict_info["total_issues"]
    failure_reason = verdict_info["failure_reason"]
    audit_results["verdict"] = verdict
    audit_results["total_issues"] = total_issues
    audit_results["failure_reason"] = failure_reason
    # [M2] additive：spike 比率与角度信息性 caveat 一并注入，供前端/deploy 消费
    audit_results["spike_ratio"] = verdict_info.get("spike_ratio")
    audit_results["angle_caveat"] = verdict_info.get("angle_caveat")

    _icon = {"pass": "✅", "warn": "⚠️", "fail": "🔴"}[verdict]
    _label = {"pass": "姿态质量良好", "warn": "存在质量问题，建议复查",
              "fail": "质量不达标"}[verdict]
    lines.append(f"  {_icon} verdict={verdict.upper()}  {_label}")
    lines.append(f"  total_issues={total_issues}")
    if failure_reason:
        lines.append(f"  failure_reason: {failure_reason}")

    if orientation.get("is_supine"):
        lines.append("  ⚠️  卧姿场景（信息性 caveat，不单独降级 verdict）：当前模型未针对卧姿优化，建议:")
        lines.append("     1. 使用卧姿专用模型或微调")
        lines.append("     2. 增加侧视角 2D 关键点检测作为辅助")
        lines.append("     3. 加入时序平滑后处理")

    report = "\n".join(lines)
    report_path = output_dir / "audit_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    return report


# ============================================================================
# 主流程
# ============================================================================

def audit_from_joints(joints_arr, video_path, output_dir, fps=30.0, total_frames=None, visualize=True,
                      cam_t_arr=None, bbox_arr=None, frame_cache=None):
    """核心审计函数：接受 numpy 数组，返回审计结果字典

    [P4/改动 G] frame_cache：可选的跨调用共享采样帧缓存（由 kineto_core.process_video
    创建并在各 iteration 间复用，消除最多 ~90 次冗余 seek）。None 时本函数自建局部
    缓存并在结束时释放（CLI/run_audit 路径向后兼容）。visualize 额外受 env
    KINETO_AUDIT_VISUALIZE 总开关约束（默认开=保持当前行为；关闭仅省 seek/IO，
    **不改 score/verdict**——可视化在 generate_report 之后且不写回 audit_results）。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_frames = total_frames if total_frames is not None else len(joints_arr)

    print("[Audit] 开始姿态质量审核...")

    own_cache = frame_cache is None
    cache = frame_cache if frame_cache is not None else FrameCache(video_path)

    bone_cons = check_bone_length_consistency(joints_arr)
    bone_val = check_bone_length_validity(bone_cons)
    smooth = check_temporal_smoothness(joints_arr, fps)
    symm = check_left_right_symmetry(joints_arr)
    angles = check_joint_angles(joints_arr)
    orientation = check_pose_orientation(joints_arr)
    proj = check_projection_alignment(joints_arr, video_path, out,
                                      cam_t_arr=cam_t_arr, bbox_arr=bbox_arr,
                                      frame_cache=cache)

    audit_results = {
        "_total_frames": n_frames,
        "bone_consistency": bone_cons,
        "bone_validity": bone_val,
        "temporal_smoothness": smooth,
        "symmetry": symm,
        "joint_angles": angles,
        "pose_orientation": orientation,
        "projection_alignment": proj,
    }

    report = generate_report(audit_results, out)
    print(report)

    if visualize and _visualize_enabled():
        print(f"\n[Audit] 问题帧可视化...")
        imgs = visualize_problem_frames(joints_arr, video_path, out, audit_results,
                                        cam_t_arr=cam_t_arr, frame_cache=cache)
        print(f"[Audit] 输出 {len(imgs)} 张问题帧图片到 {out}/")
    elif visualize:
        print("\n[Audit] 问题帧可视化已禁用 (KINETO_AUDIT_VISUALIZE=0)；不影响 score/verdict")

    def _json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    audit_json_path = out / "audit_results.json"
    with open(audit_json_path, "w") as f:
        json.dump(_json_safe(audit_results), f, indent=2)

    print(f"[Audit] 详细结果 → {audit_json_path}")
    print(f"[Audit] 报告 → {out / 'audit_report.txt'}")

    if own_cache:
        cache.release()
    return audit_results


def run_audit(pose_data_path, video_path, output_dir):
    """从 pose_data.json 文件加载并审计（CLI 入口）"""
    with open(pose_data_path) as f:
        data = json.load(f)

    keyframes = data["keyframes"]
    joints_arr = np.array([kf["joints_3d"] for kf in keyframes])
    fps = data["metadata"].get("video_fps", 30)

    return audit_from_joints(joints_arr, video_path, output_dir, fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Kineto Pose Audit - 姿态质量自动审核")
    parser.add_argument("--input", "-i", default="output/pose_data.json", help="pose_data.json 路径")
    parser.add_argument("--video", "-v", default="input_video.mp4", help="原始视频路径")
    parser.add_argument("--output", "-o", default="output/audit", help="审核输出目录")
    args = parser.parse_args()

    run_audit(args.input, args.video, args.output)


if __name__ == "__main__":
    main()
