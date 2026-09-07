#!/usr/bin/env python3
"""
Kineto Skeleton Validator - 骨架解析自动验证工具
================================================
读取已有的 pose_data.json + 原始视频，输出：
  1. 每帧三联对比图（原始帧 | 骨架叠加 | 置信度热力图）
  2. validation_report.html（内嵌图片 + 指标表格）
  3. metrics.json（机器可读指标）
  4. [可选] Gradio 交互 UI（--ui 参数）

用法：
    python skeleton_validator.py                              # 静态输出，默认路径
    python skeleton_validator.py --pose output/pose_data.json --video input_video.mp4
    python skeleton_validator.py --ui                         # 启动交互界面
    python skeleton_validator.py --samples 30                 # 采样 30 帧
"""

import argparse
import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np

# ============================================================================
# SMPL 骨架定义 — 常量统一来自 skeleton_spec (SSOT)
# ============================================================================
# [M4] 删除本地 SMPL_JOINT_NAMES/SMPL_SKELETON/BONE_VALIDITY 副本（含假边
# (12,13)/(12,14)——真 kintree 中 collar 13/14 的父节点是 spine3(9) 而非 neck(12)，
# 以及与真实骨架矛盾的旧手标骨长界），改从 skeleton_spec SSOT import，与
# kineto_core.py / pose_audit.py 同源。BONE_LENGTH_BOUNDS 为 pkl 派生常量（惰性加载），
# 本工具运行时需真实骨长界，故 import 即触发加载（与 pose_audit 一致）。
from skeleton_spec import (
    BONE_LENGTH_BOUNDS,
    BONE_NAMES,
    BONE_PART_MAP,
    SMPL_JOINT_NAMES,
    SMPL_SKELETON,
)

# 骨骼颜色（BGR）：由 SSOT 的 BONE_PART_MAP（身体部位归属）派生——左侧绿、右侧蓝、
# 躯干黄、头部粉。纯展示层，不入 SSOT；键集合与 SMPL_SKELETON 一致（23 条真骨）。
_PART_COLOR_BGR = {
    "left_leg": (50, 200, 50),
    "left_arm": (50, 200, 50),
    "right_leg": (200, 50, 50),
    "right_arm": (200, 50, 50),
    "torso": (50, 200, 200),
    "head": (180, 180, 255),
}
BONE_COLORS_BGR = {bone: _PART_COLOR_BGR.get(part, (200, 200, 200))
                   for bone, part in BONE_PART_MAP.items()}


# ============================================================================
# 投影工具
# ============================================================================

def project_joints(joints_3d, cam_t, img_w, img_h, focal_length=5000.0, model_size=256):
    """3D 关节 → 2D 画面像素坐标（与 kineto_core.SkeletonRenderer 完全一致）"""
    focal = focal_length / model_size * max(img_w, img_h)
    cx, cy = img_w / 2, img_h / 2
    points = []
    for joint in joints_3d:
        J = np.asarray(joint) + np.asarray(cam_t)
        if abs(J[2]) < 1e-6:
            points.append(None)
            continue
        u = int(focal * J[0] / J[2] + cx)
        v = int(focal * J[1] / J[2] + cy)
        points.append((u, v))
    return points


# ============================================================================
# 渲染函数
# ============================================================================

def draw_skeleton_overlay(frame, joints_3d, cam_t, confidence=1.0,
                           img_w=None, img_h=None):
    """在帧上叠加彩色骨架，返回叠加后的副本"""
    h, w = frame.shape[:2]
    img_w = img_w or w
    img_h = img_h or h

    overlay = frame.copy()
    pts = project_joints(joints_3d, cam_t, img_w, img_h)

    # 绘制骨骼连线
    for (i, j) in SMPL_SKELETON:
        if i >= len(pts) or j >= len(pts):
            continue
        if pts[i] is None or pts[j] is None:
            continue
        color = BONE_COLORS_BGR.get((i, j), (200, 200, 200))
        cv2.line(overlay, pts[i], pts[j], color, 2, cv2.LINE_AA)

    # 绘制关节点（头部关节用不同颜色）
    for idx, pt in enumerate(pts):
        if pt is None:
            continue
        is_head = idx in (12, 15)
        color = (255, 200, 200) if is_head else (0, 255, 255)
        radius = 5 if is_head else 3
        cv2.circle(overlay, pt, radius, color, -1, cv2.LINE_AA)

    alpha = min(max(confidence, 0.5), 1.0)
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 置信度文字
    conf_color = (0, 255, 0) if confidence > 0.7 else (0, 200, 255) if confidence > 0.4 else (0, 0, 255)
    cv2.putText(blended, f"conf: {confidence:.2f}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2, cv2.LINE_AA)
    return blended


def draw_confidence_heatmap(frame, joints_3d, cam_t, confidence=1.0,
                             bone_scores=None, img_w=None, img_h=None):
    """生成关节置信度热力图（黑底，关节按分数着色）"""
    h, w = frame.shape[:2]
    img_w = img_w or w
    img_h = img_h or h

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts = project_joints(joints_3d, cam_t, img_w, img_h)

    # 骨骼连线（灰色）
    for (i, j) in SMPL_SKELETON:
        if i >= len(pts) or j >= len(pts):
            continue
        if pts[i] is None or pts[j] is None:
            continue
        bone_name = _get_bone_name(i, j)
        score = (bone_scores or {}).get(bone_name, confidence)
        color = _score_to_color(score)
        cv2.line(canvas, pts[i], pts[j], color, 2, cv2.LINE_AA)

    # 关节点（颜色 = 置信度）
    for idx, pt in enumerate(pts):
        if pt is None:
            continue
        cv2.circle(canvas, pt, 5, _score_to_color(confidence), -1, cv2.LINE_AA)
        name = SMPL_JOINT_NAMES[idx] if idx < len(SMPL_JOINT_NAMES) else str(idx)
        cv2.putText(canvas, name[:6], (pt[0] + 4, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)

    # 图例
    cv2.putText(canvas, "Confidence Heatmap", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    _draw_colorbar(canvas, h)
    return canvas


def _score_to_color(score):
    """0.0=红 → 0.5=黄 → 1.0=绿 (BGR)"""
    score = float(np.clip(score, 0, 1))
    if score < 0.5:
        r = 255
        g = int(score * 2 * 255)
    else:
        r = int((1 - score) * 2 * 255)
        g = 255
    return (0, g, r)


def _draw_colorbar(canvas, img_h):
    """在右侧画置信度色条"""
    bar_x, bar_w = canvas.shape[1] - 20, 10
    bar_top, bar_bot = 30, img_h - 30
    for y in range(bar_top, bar_bot):
        score = 1.0 - (y - bar_top) / max(bar_bot - bar_top, 1)
        color = _score_to_color(score)
        canvas[y, bar_x:bar_x + bar_w] = color
    cv2.putText(canvas, "1.0", (bar_x - 18, bar_top + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    cv2.putText(canvas, "0.0", (bar_x - 18, bar_bot),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)


def _get_bone_name(i, j):
    """骨对 (i,j) 的稳定名称（来自 SSOT BONE_NAMES，兼容反向查询）。[M4]"""
    return BONE_NAMES.get((i, j)) or BONE_NAMES.get((j, i)) or ""


def make_triple_image(raw_frame, skel_frame, heatmap_frame, frame_idx, timestamp_ms):
    """将三张图拼成横向三联图，加标题栏"""
    h, w = raw_frame.shape[:2]
    sep = 4
    title_h = 30

    triple = np.zeros((h + title_h, w * 3 + sep * 2, 3), dtype=np.uint8)
    triple[title_h:, :w] = raw_frame
    triple[title_h:, w + sep:w * 2 + sep] = skel_frame
    triple[title_h:, w * 2 + sep * 2:] = heatmap_frame

    labels = ["Original", "Skeleton Overlay", "Confidence Heatmap"]
    offsets = [0, w + sep, w * 2 + sep * 2]
    for label, ox in zip(labels, offsets):
        cv2.putText(triple, label, (ox + 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    info = f"Frame {frame_idx}  |  {timestamp_ms:.0f} ms"
    cv2.putText(triple, info, (w + sep + 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
    return triple


# ============================================================================
# 指标计算
# ============================================================================

def compute_frame_metrics(joints_3d, cam_t, img_w, img_h, person_bbox=None):
    """计算单帧的骨长偏差率和投影覆盖率"""
    joints = np.asarray(joints_3d, dtype=np.float32)
    pts = project_joints(joints_3d, cam_t, img_w, img_h)

    # 骨长偏差率（各骨骼实际长度与解剖中值的偏差均值）
    # [M4] 遍历 SSOT BONE_LENGTH_BOUNDS（23 条真骨、真 kintree、数据驱动骨长界），
    # 名称由 BONE_NAMES 稳定生成；旧本地 BONE_VALIDITY 仅 17 项且含假边，已淘汰。
    bone_deviations = {}
    for (i, j), (lo, hi) in BONE_LENGTH_BOUNDS.items():
        name = BONE_NAMES[(i, j)]
        mid = (lo + hi) / 2
        actual = float(np.linalg.norm(joints[i] - joints[j]))
        if actual < 1e-6:
            dev = 1.0  # 退化骨骼 = 100% 偏差
        else:
            dev = abs(actual - mid) / mid
        in_range = lo <= actual <= hi
        bone_deviations[name] = {"length": actual, "deviation": dev, "in_range": in_range}

    avg_deviation = float(np.mean([v["deviation"] for v in bone_deviations.values()]))

    # 投影覆盖率：关节投影落在人物 bbox（扩展 20%）内的比例
    if person_bbox is not None and not np.all(np.asarray(person_bbox) == 0):
        x1, y1, x2, y2 = person_bbox
        mx = (x2 - x1) * 0.2
        my = (y2 - y1) * 0.2
        in_box = sum(
            1 for pt in pts
            if pt is not None and (x1 - mx) <= pt[0] <= (x2 + mx) and (y1 - my) <= pt[1] <= (y2 + my)
        )
        coverage = in_box / max(len([p for p in pts if p is not None]), 1)
    else:
        in_frame = sum(1 for pt in pts if pt is not None and 0 <= pt[0] < img_w and 0 <= pt[1] < img_h)
        coverage = in_frame / max(len([p for p in pts if p is not None]), 1)

    return {
        "bone_deviations": bone_deviations,
        "avg_bone_deviation": avg_deviation,
        "projection_coverage": float(coverage),
        "bone_validity_rate": float(sum(v["in_range"] for v in bone_deviations.values()) / len(bone_deviations)),
    }


def compute_temporal_stability(all_joints):
    """计算全序列时序稳定性（帧间位移）"""
    arr = np.array(all_joints, dtype=np.float32)
    if len(arr) < 2:
        return {"mean_displacement": 0.0, "max_displacement": 0.0, "spike_count": 0, "spike_frames": []}
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=2).mean(axis=1)
    threshold = 0.05
    spike_frames = np.where(diffs > threshold)[0].tolist()
    return {
        "mean_displacement": float(diffs.mean()),
        "max_displacement": float(diffs.max()),
        "spike_count": len(spike_frames),
        "spike_frames": spike_frames[:20],
        "threshold": threshold,
    }


# ============================================================================
# HTML 报告生成
# ============================================================================

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_html_report(frame_results, metrics_summary, output_dir, img_paths):
    """生成内嵌图片的 HTML 验证报告"""

    def score_badge(score, thresholds=(0.4, 0.7)):
        lo, hi = thresholds
        if score >= hi:
            cls, label = "badge-ok", f"{score:.2f} ✅"
        elif score >= lo:
            cls, label = "badge-warn", f"{score:.2f} ⚠️"
        else:
            cls, label = "badge-err", f"{score:.2f} 🔴"
        return f'<span class="{cls}">{label}</span>'

    rows_html = ""
    for i, (res, path) in enumerate(zip(frame_results, img_paths)):
        cov_badge = score_badge(res["projection_coverage"])
        val_badge = score_badge(res["bone_validity_rate"])
        dev_badge = score_badge(1.0 - min(res["avg_bone_deviation"], 1.0))
        conf_badge = score_badge(res.get("confidence", 0.5))
        img_b64 = img_to_base64(path)
        rows_html += f"""
        <tr>
          <td>{res['frame_index']}</td>
          <td>{res['timestamp_ms']:.0f} ms</td>
          <td>{conf_badge}</td>
          <td>{cov_badge}</td>
          <td>{val_badge}</td>
          <td>{dev_badge}</td>
          <td><img src="data:image/jpeg;base64,{img_b64}" class="thumb" onclick="showFull(this)"></td>
        </tr>"""

    stab = metrics_summary.get("temporal_stability", {})
    overall_cov = np.mean([r["projection_coverage"] for r in frame_results]) if frame_results else 0
    overall_val = np.mean([r["bone_validity_rate"] for r in frame_results]) if frame_results else 0
    overall_dev = np.mean([r["avg_bone_deviation"] for r in frame_results]) if frame_results else 1

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Kineto Skeleton Validation Report</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #111; color: #ddd; margin: 0; padding: 20px; }}
  h1 {{ color: #7ec8e3; border-bottom: 1px solid #333; padding-bottom: 10px; }}
  h2 {{ color: #a0d4b4; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
  th {{ background: #222; color: #7ec8e3; padding: 8px 12px; text-align: left; border: 1px solid #333; }}
  td {{ padding: 6px 12px; border: 1px solid #2a2a2a; vertical-align: middle; }}
  tr:nth-child(even) {{ background: #181818; }}
  tr:hover {{ background: #1e2a30; }}
  .badge-ok   {{ background: #1a3d1a; color: #6fde6f; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
  .badge-warn {{ background: #3d3000; color: #f0c040; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
  .badge-err  {{ background: #3d0000; color: #f06060; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
  .summary-card {{ display: inline-block; background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
                   padding: 16px 24px; margin: 8px; text-align: center; min-width: 160px; }}
  .summary-card .val {{ font-size: 2em; font-weight: bold; color: #7ec8e3; }}
  .summary-card .lbl {{ font-size: 0.85em; color: #888; margin-top: 4px; }}
  .thumb {{ max-width: 240px; cursor: pointer; border-radius: 4px; border: 1px solid #333; }}
  #lightbox {{ display:none; position:fixed; top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.9);
              z-index:1000; justify-content:center; align-items:center; }}
  #lightbox img {{ max-width:95vw; max-height:90vh; border-radius:6px; }}
  #lightbox.active {{ display:flex; }}
  .mono {{ font-family: monospace; color: #aaa; font-size: 0.85em; }}
</style>
</head>
<body>
<div id="lightbox" onclick="this.classList.remove('active')">
  <img id="lb-img" src="">
</div>
<h1>🦴 Kineto Skeleton Validation Report</h1>

<h2>Overall Summary</h2>
<div>
  <div class="summary-card">
    <div class="val">{overall_cov*100:.1f}%</div>
    <div class="lbl">Projection Coverage</div>
  </div>
  <div class="summary-card">
    <div class="val">{overall_val*100:.1f}%</div>
    <div class="lbl">Bone Validity Rate</div>
  </div>
  <div class="summary-card">
    <div class="val">{(1-min(overall_dev,1))*100:.1f}%</div>
    <div class="lbl">Bone Length Accuracy</div>
  </div>
  <div class="summary-card">
    <div class="val">{stab.get('spike_count', 0)}</div>
    <div class="lbl">Temporal Spikes</div>
  </div>
  <div class="summary-card">
    <div class="val">{stab.get('mean_displacement', 0)*1000:.2f}<span style="font-size:0.5em">mm</span></div>
    <div class="lbl">Mean Frame Displacement</div>
  </div>
</div>

<h2>Temporal Stability</h2>
<p class="mono">
  Mean displacement: {stab.get('mean_displacement', 0):.6f} m/frame &nbsp;|&nbsp;
  Max displacement: {stab.get('max_displacement', 0):.6f} m &nbsp;|&nbsp;
  Spike threshold: {stab.get('threshold', 0.05)} m &nbsp;|&nbsp;
  Spike frames: {stab.get('spike_frames', [])[:10]}
</p>

<h2>Per-Frame Results ({len(frame_results)} frames sampled)</h2>
<table>
  <thead>
    <tr>
      <th>Frame</th><th>Time</th><th>Confidence</th>
      <th>Proj Coverage</th><th>Bone Validity</th><th>Bone Accuracy</th>
      <th>Preview (click to enlarge)</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>

<p style="margin-top:40px;color:#555;font-size:0.8em">
  Generated by Kineto Skeleton Validator &nbsp;·&nbsp;
  {metrics_summary.get('total_frames', '?')} total frames &nbsp;·&nbsp;
  Model: {metrics_summary.get('model_version', 'unknown')}
</p>

<script>
function showFull(img) {{
  document.getElementById('lb-img').src = img.src;
  document.getElementById('lightbox').classList.add('active');
}}
</script>
</body>
</html>"""
    report_path = output_dir / "validation_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# ============================================================================
# 主验证流程（静态输出）
# ============================================================================

def run_validation(pose_data_path, video_path, output_dir, n_samples=20):
    """核心验证函数：读 JSON + 视频 → 输出三联图 + HTML 报告"""
    pose_path = Path(pose_data_path)
    vid_path = Path(video_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[Validator] 加载 pose_data: {pose_path}")
    with open(pose_path, encoding="utf-8") as f:
        pose_data = json.load(f)

    metadata = pose_data["metadata"]
    keyframes = pose_data["keyframes"]
    total_frames = len(keyframes)
    print(f"[Validator] 共 {total_frames} 帧，视频: {vid_path}")

    if not vid_path.exists():
        print(f"[Validator] ⚠️ 视频文件不存在: {vid_path}，将跳过叠加渲染")
        return None

    cap = cv2.VideoCapture(str(vid_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 均匀采样关键帧索引
    sample_indices = np.linspace(0, total_frames - 1, min(n_samples, total_frames), dtype=int)
    print(f"[Validator] 采样 {len(sample_indices)} 帧 → {out_path}/")

    frame_results = []
    img_paths = []
    all_joints = [kf["joints_3d"] for kf in keyframes]

    for idx in sample_indices:
        kf = keyframes[int(idx)]
        frame_idx = kf["frame_index"]
        timestamp_ms = kf["timestamp_ms"]
        joints_3d = np.asarray(kf["joints_3d"], dtype=np.float32)
        cam_t = np.asarray(kf.get("cam_t", [0, 0, 0]), dtype=np.float32)
        confidence = kf.get("confidence_score", 0.5)
        person_bbox = kf.get("person_bbox", None)

        # 兼容旧版 pose_data.json（cam_t 全零 + 无 bbox）：
        # 自动估算 cam_t，使骨架投影落在画面中心合理位置
        if np.allclose(cam_t, 0):
            focal = 5000.0 / 256.0 * max(img_w, img_h)
            # 用骨盆关节作为身体参考原点，估算合适的深度
            pelvis = joints_3d[0]
            # 关节坐标在归一化空间，典型范围 ±0.5m；设定 tz 使人体高度
            # 在画面中占合理比例（身高约 1.7m → tz ≈ focal * 1.7 / img_h）
            tz_est = focal * 1.7 / max(img_h, 1)
            # x/y 偏移：令骨盆投影到画面中心
            cam_t = np.array([
                -pelvis[0],
                -pelvis[1],
                tz_est,
            ], dtype=np.float32)

        # 读取对应帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, raw_frame = cap.read()
        if not ret:
            print(f"[Validator] ⚠️ 无法读取第 {frame_idx} 帧，跳过")
            continue

        # 计算骨骼指标
        metrics = compute_frame_metrics(joints_3d, cam_t, img_w, img_h, person_bbox)
        bone_scores = {name: (1.0 if v["in_range"] else 0.2) for name, v in metrics["bone_deviations"].items()}

        # 渲染三联图
        skel_frame = draw_skeleton_overlay(raw_frame, joints_3d, cam_t, confidence, img_w, img_h)
        heatmap = draw_confidence_heatmap(raw_frame, joints_3d, cam_t, confidence, bone_scores, img_w, img_h)
        triple = make_triple_image(raw_frame, skel_frame, heatmap, frame_idx, timestamp_ms)

        save_path = out_path / f"frame_{frame_idx:05d}_compare.jpg"
        cv2.imwrite(str(save_path), triple, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_paths.append(save_path)

        frame_results.append({
            "frame_index": frame_idx,
            "timestamp_ms": timestamp_ms,
            "confidence": confidence,
            **metrics,
        })

        cov_str = f"{metrics['projection_coverage']*100:.1f}%"
        val_str = f"{metrics['bone_validity_rate']*100:.1f}%"
        print(f"  [Frame {frame_idx:4d}] conf={confidence:.2f}  proj={cov_str}  bone_valid={val_str}")

    cap.release()

    # 时序稳定性
    stab = compute_temporal_stability(all_joints)

    metrics_summary = {
        "total_frames": total_frames,
        "sampled_frames": len(frame_results),
        "model_version": metadata.get("model_version", "unknown"),
        "temporal_stability": stab,
        "overall": {
            "mean_projection_coverage": float(np.mean([r["projection_coverage"] for r in frame_results])) if frame_results else 0,
            "mean_bone_validity_rate": float(np.mean([r["bone_validity_rate"] for r in frame_results])) if frame_results else 0,
            "mean_bone_deviation": float(np.mean([r["avg_bone_deviation"] for r in frame_results])) if frame_results else 1,
            "mean_confidence": float(np.mean([r["confidence"] for r in frame_results])) if frame_results else 0,
        },
    }

    # 保存 metrics.json
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**metrics_summary, "frame_results": frame_results}, f, indent=2)

    # 生成 HTML 报告
    report_path = generate_html_report(frame_results, metrics_summary, out_path, img_paths)

    print(f"\n[Validator] ✅ 完成!")
    print(f"  三联对比图  → {out_path}/frame_XXXXX_compare.jpg")
    print(f"  HTML 报告   → {report_path}")
    print(f"  指标 JSON   → {metrics_path}")
    print(f"\n  总体指标:")
    ov = metrics_summary["overall"]
    print(f"    投影覆盖率:  {ov['mean_projection_coverage']*100:.1f}%")
    print(f"    骨长合理率:  {ov['mean_bone_validity_rate']*100:.1f}%")
    print(f"    骨长准确度:  {(1-min(ov['mean_bone_deviation'],1))*100:.1f}%")
    print(f"    平均置信度:  {ov['mean_confidence']:.3f}")
    print(f"    时序突变帧:  {stab['spike_count']}")

    return metrics_summary


# ============================================================================
# Gradio 交互 UI（可选）
# ============================================================================

def launch_gradio_ui(pose_data_path, video_path):
    """启动 Gradio 交互验证界面"""
    try:
        import gradio as gr
    except ImportError:
        print("[UI] ❌ gradio 未安装，请运行：pip install gradio")
        return

    with open(pose_data_path, encoding="utf-8") as f:
        pose_data = json.load(f)
    keyframes = pose_data["keyframes"]
    total_frames = len(keyframes)

    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    def get_frame(frame_slider):
        idx = int(frame_slider)
        kf = keyframes[idx]
        joints_3d = np.asarray(kf["joints_3d"], dtype=np.float32)
        cam_t = np.asarray(kf.get("cam_t", [0, 0, 10]), dtype=np.float32)
        confidence = kf.get("confidence_score", 0.5)
        person_bbox = kf.get("person_bbox", None)

        cap2 = cv2.VideoCapture(str(video_path))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, kf["frame_index"])
        ret, raw = cap2.read()
        cap2.release()

        if not ret:
            return None, None, "无法读取该帧"

        metrics = compute_frame_metrics(joints_3d, cam_t, img_w, img_h, person_bbox)
        bone_scores = {name: (1.0 if v["in_range"] else 0.2)
                       for name, v in metrics["bone_deviations"].items()}

        skel = draw_skeleton_overlay(raw, joints_3d, cam_t, confidence, img_w, img_h)
        heat = draw_confidence_heatmap(raw, joints_3d, cam_t, confidence, bone_scores, img_w, img_h)

        # BGR → RGB for Gradio
        skel_rgb = cv2.cvtColor(skel, cv2.COLOR_BGR2RGB)
        heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        info = (
            f"**Frame {kf['frame_index']}** | {kf['timestamp_ms']:.0f} ms | "
            f"Conf: {confidence:.3f}\n\n"
            f"- 投影覆盖率: {metrics['projection_coverage']*100:.1f}%\n"
            f"- 骨长合理率: {metrics['bone_validity_rate']*100:.1f}%\n"
            f"- 骨长平均偏差: {metrics['avg_bone_deviation']*100:.1f}%\n\n"
            + "\n".join(
                f"  - {'✅' if v['in_range'] else '🔴'} {n}: {v['length']:.4f}m"
                for n, v in metrics["bone_deviations"].items()
            )
        )
        return skel_rgb, heat_rgb, info

    with gr.Blocks(title="Kineto Skeleton Validator", theme=gr.themes.Base()) as demo:
        gr.Markdown("## 🦴 Kineto Skeleton Validator")
        gr.Markdown(f"视频共 **{total_frames}** 帧，拖动滑块查看任意帧的骨架解析质量")
        with gr.Row():
            slider = gr.Slider(0, total_frames - 1, step=1, value=0, label="帧序号")
        with gr.Row():
            img_skel = gr.Image(label="骨架叠加", show_label=True)
            img_heat = gr.Image(label="置信度热力图", show_label=True)
        info_box = gr.Markdown("选择一帧查看详情")

        slider.change(fn=get_frame, inputs=slider, outputs=[img_skel, img_heat, info_box])
        demo.load(fn=lambda: get_frame(0), outputs=[img_skel, img_heat, info_box])

    print("[UI] 启动 Gradio 验证界面...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kineto Skeleton Validator - 骨架解析自动验证工具"
    )
    parser.add_argument("--pose", "-p", default="output/pose_data.json",
                        help="pose_data.json 路径 (default: output/pose_data.json)")
    parser.add_argument("--video", "-v", default="input_video.mp4",
                        help="原始视频路径 (default: input_video.mp4)")
    parser.add_argument("--output", "-o", default="output/validation",
                        help="验证结果输出目录 (default: output/validation)")
    parser.add_argument("--samples", "-n", type=int, default=20,
                        help="采样帧数 (default: 20)")
    parser.add_argument("--ui", action="store_true",
                        help="启动 Gradio 交互界面")
    args = parser.parse_args()

    print("=" * 60)
    print("  Kineto Skeleton Validator")
    print("=" * 60)

    if args.ui:
        launch_gradio_ui(args.pose, args.video)
    else:
        run_validation(args.pose, args.video, args.output, n_samples=args.samples)


if __name__ == "__main__":
    main()
