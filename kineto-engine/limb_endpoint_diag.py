#!/usr/bin/env python3
"""肢体端点（腕/踝/掌/脚）2D 对齐诊断工具。

背景（P2.1 诊断）：4D-Humans 的 2D 监督只到 COCO-17（腕/踝），SMPL 的
hand(22/23)/foot(10/11) 端点无 2D 观测；跪撑类动作手脚自遮挡，端点
重投影偏差被用户红圈标出。本脚本**只读**地量化偏差，不改动任何产物：

  1. YOLOv8-pose（COCO-17）对关键帧检测 2D 关键点（真值近似，高置信帧）；
  2. 复刻引擎投影（SkeletonRenderer: focal=5000/256·max(w,h),
     J = joint_3d + cam_t，针孔模型）把 SMPL 24 关节投回画面；
  3. 左右最小代价配对后，按关节组统计像素 / 毫米偏差分布；
  4. 端点延伸量（wrist→hand、ankle→foot 的骨长与共线度）；
  5. 接触穿插：相机系地面低分位估计，撑地手/脚低于地面的穿插深度。

用法：
  python limb_endpoint_diag.py <job_dir 或 pose_data.json> [--video PATH]
      [--stride N] [--out DIR]

需在引擎 venv 运行（ultralytics + cv2 + numpy）；权重 yolov8n-pose.pt
放引擎目录（与 yolov8n.pt 同处）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# SMPL canonical 24 关节序（skeleton_spec SSOT；此处仅索引用，不重复拓扑）
L_HIP, R_HIP = 1, 2
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
L_FOOT, R_FOOT = 10, 11
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_HAND, R_HAND = 22, 23

# COCO-17 索引
C_L_SHO, C_R_SHO = 5, 6
C_L_ELB, C_R_ELB = 7, 8
C_L_WRI, C_R_WRI = 9, 10
C_L_HIP, C_R_HIP = 11, 12
C_L_KNE, C_R_KNE = 13, 14
C_L_ANK, C_R_ANK = 15, 16

# 配对组：组名 → (SMPL(L,R), COCO(L,R))
PAIR_GROUPS = {
    "shoulder": ((L_SHOULDER, R_SHOULDER), (C_L_SHO, C_R_SHO)),
    "elbow": ((L_ELBOW, R_ELBOW), (C_L_ELB, C_R_ELB)),
    "wrist": ((L_WRIST, R_WRIST), (C_L_WRI, C_R_WRI)),
    "hip": ((L_HIP, R_HIP), (C_L_HIP, C_R_HIP)),
    "knee": ((L_KNEE, R_KNEE), (C_L_KNE, C_R_KNE)),
    "ankle": ((L_ANKLE, R_ANKLE), (C_L_ANK, C_R_ANK)),
}
SKELETON_EDGES = [
    (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def project(joints_3d: np.ndarray, cam_t: np.ndarray, w: int, h: int) -> np.ndarray:
    """复刻引擎 SkeletonRenderer 投影，返回 (24,3)：u,v,depth_m。"""
    focal = 5000.0 / 256.0 * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    J = joints_3d + cam_t
    out = np.zeros((len(J), 3))
    out[:, 0] = focal * J[:, 0] / J[:, 2] + cx
    out[:, 1] = focal * J[:, 1] / J[:, 2] + cy
    out[:, 2] = J[:, 2]
    return out


def best_pair(p_a, p_b, q_a, q_b):
    """两点对两点的最小总距离左右配对，返回 (p→q 映射 idx0, idx1)。"""
    d_straight = np.hypot(*(p_a - q_a)[:2]) + np.hypot(*(p_b - q_b)[:2])
    d_cross = np.hypot(*(p_a - q_b)[:2]) + np.hypot(*(p_b - q_a)[:2])
    if d_straight <= d_cross:
        return [(p_a, q_a), (p_b, q_b)]
    return [(p_a, q_b), (p_b, q_a)]


def pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="肢体端点 2D 对齐诊断（只读）")
    ap.add_argument("job", help="job 目录或 pose_data.json 路径")
    ap.add_argument("--video", default=None, help="原视频路径（默认 <job>/input.mp4 或 inbox/input_video.mp4）")
    ap.add_argument("--stride", type=int, default=1, help="关键帧采样步长（默认全帧）")
    ap.add_argument("--out", default="/tmp/limb_diag", help="输出目录")
    args = ap.parse_args()

    job = Path(args.job)
    pose_path = job / "pose_data.json" if job.is_dir() else job
    job_dir = pose_path.parent
    data = json.loads(pose_path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    w, h = (int(x) for x in meta["resolution"].split("x"))
    keyframes = data["keyframes"][:: args.stride]

    video = args.video
    if video is None:
        for cand in (job_dir / "input.mp4", Path("/srv/kineto/inbox/input_video.mp4")):
            if cand.exists():
                video = str(cand)
                break
    if not video or not Path(video).exists():
        print(f"[ERR] 找不到原视频（--video 显式指定）", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(Path(__file__).resolve().parent / "yolov8n-pose.pt"))

    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] video={video} frames={total_frames} {w}x{h} keyframes={len(keyframes)}")

    # 收集容器
    px_err: dict[str, list[float]] = {g: [] for g in PAIR_GROUPS}
    mm_err: dict[str, list[float]] = {g: [] for g in PAIR_GROUPS}
    px_err_conf: dict[str, list[float]] = {g: [] for g in PAIR_GROUPS}  # YOLO conf>=0.5 子集
    hand_len_m, foot_len_m = [], []
    hand_colin, foot_colin = [], []
    contact_penetration_m = []  # 撑地手穿插深度（米，>0 = 穿到地面以下）
    debug_imgs = []

    # 第一遍：投影 + 地面统计
    smpl_uv_all = []
    smpl_J_all = []
    for kf in keyframes:
        j3d = np.asarray(kf["joints_3d"], dtype=np.float64)
        cam_t = np.asarray(kf["cam_t"], dtype=np.float64)
        uv = project(j3d, cam_t, w, h)
        smpl_uv_all.append(uv)
        smpl_J_all.append(j3d + cam_t)
    # 地面：相机系 Y 向下为正。跪撑姿势下下肢（膝/踝/脚）与垫面接触，
    # 取这些点 y 的高分位作为地平面；**手不参与**（穿插的手会把地面拉低，
    # 导致穿插深度被系统性低估）。
    GROUND_JOINTS = [L_ANKLE, R_ANKLE, L_KNEE, R_KNEE, L_FOOT, R_FOOT]
    ground_y = float(np.percentile(
        np.concatenate([J[GROUND_JOINTS, 1] for J in smpl_J_all]), 90))
    print(f"[INFO] 地面估计（相机系 y，向下为正）: {ground_y:.3f} m")

    for ki, kf in enumerate(keyframes):
        fi = int(kf["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        res = model(frame, verbose=False, device="xpu")[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            continue
        # 取置信度最高的人
        kpts = res.keypoints.data.cpu().numpy()
        if kpts.ndim != 3 or kpts.shape[1] < 17:
            continue
        # 多人时取 bbox 最大者
        if res.boxes is not None and len(res.boxes) > 1:
            areas = (res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0]) * (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1])
            kpts = kpts[int(areas.argmax())][None]
        coco = kpts[0]  # (17,3) x,y,conf

        uv = smpl_uv_all[ki]
        J = smpl_J_all[ki]

        for gname, ((sl, sr), (cl, cr)) in PAIR_GROUPS.items():
            for p_smpl, q_coco in best_pair(uv[sl], uv[sr], coco[cl], coco[cr]):
                if q_coco[2] < 0.2:  # YOLO 该点不可见
                    continue
                d_px = float(np.hypot(p_smpl[0] - q_coco[0], p_smpl[1] - q_coco[1]))
                mm_per_px = float(p_smpl[2]) / float(5000.0 / 256.0 * max(w, h)) * 1000.0
                px_err[gname].append(d_px)
                mm_err[gname].append(d_px * mm_per_px)
                if q_coco[2] >= 0.5:
                    px_err_conf[gname].append(d_px)

        # 端点延伸（SMPL 模型侧）
        for w_i, h_i, e_i in ((L_WRIST, L_HAND, L_ELBOW), (R_WRIST, R_HAND, R_ELBOW)):
            v_wh = J[h_i] - J[w_i]
            v_ew = J[w_i] - J[e_i]
            hand_len_m.append(float(np.linalg.norm(v_wh)))
            n = np.linalg.norm(v_wh) * np.linalg.norm(v_ew)
            if n > 1e-9:
                hand_colin.append(float(np.dot(v_wh, v_ew) / n))
        for a_i, f_i, k_i in ((L_ANKLE, L_FOOT, L_KNEE), (R_ANKLE, R_FOOT, R_KNEE)):
            v_af = J[f_i] - J[a_i]
            v_ka = J[a_i] - J[k_i]
            foot_len_m.append(float(np.linalg.norm(v_af)))
            n = np.linalg.norm(v_af) * np.linalg.norm(v_ka)
            if n > 1e-9:
                foot_colin.append(float(np.dot(v_af, v_ka) / n))

        # 接触穿插：手在地面附近（±8cm）时，hand 低于地面的深度
        for h_i in (L_HAND, R_HAND):
            if abs(J[h_i, 1] - ground_y) < 0.08:
                contact_penetration_m.append(float(J[h_i, 1] - ground_y))

        # 可视化抽样（约每 60 个采样帧一张）
        if ki % max(1, len(keyframes) // 8) == 0:
            vis = frame.copy()
            for a, b in SKELETON_EDGES:
                cv2.line(vis, (int(uv[a, 0]), int(uv[a, 1])), (int(uv[b, 0]), int(uv[b, 1])),
                         (255, 200, 0), 2, cv2.LINE_AA)
            for j_i in (L_HAND, R_HAND, L_FOOT, R_FOOT, L_WRIST, R_WRIST, L_ANKLE, R_ANKLE):
                cv2.circle(vis, (int(uv[j_i, 0]), int(uv[j_i, 1])), 5, (255, 255, 0), -1)
            for c_i in range(17):
                if coco[c_i, 2] >= 0.3:
                    cv2.circle(vis, (int(coco[c_i, 0]), int(coco[c_i, 1])), 5, (0, 0, 255), 2)
            for gname, ((sl, sr), (cl, cr)) in PAIR_GROUPS.items():
                for p_smpl, q_coco in best_pair(uv[sl], uv[sr], coco[cl], coco[cr]):
                    if q_coco[2] >= 0.3:
                        cv2.line(vis, (int(p_smpl[0]), int(p_smpl[1])),
                                 (int(q_coco[0]), int(q_coco[1])), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"frame {fi}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            debug_imgs.append(vis)

    cap.release()

    # ── 报告 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" 肢体端点 2D 对齐诊断（黄=SMPL 重投影，红圈=YOLOv8-pose COCO-17，白线=偏差）")
    print("=" * 78)
    print(f"{'组':<10}{'n':>5}{'px p50':>9}{'px p90':>9}{'px p95':>9}{'px max':>9}"
          f"{'mm p50':>9}{'mm p90':>9}  备注")
    report = {}
    for gname in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        px = np.asarray(px_err[gname])
        mm = np.asarray(mm_err[gname])
        pxc = np.asarray(px_err_conf[gname])
        note = ""
        if gname in ("wrist", "ankle"):
            note = "← 问题关节（监督末端）" if pct(px, 90) > pct(np.asarray(px_err["elbow"] if gname == "wrist" else px_err["knee"]), 90) else ""
        print(f"{gname:<10}{len(px):>5}{pct(px,50):>9.1f}{pct(px,90):>9.1f}{pct(px,95):>9.1f}{pct(px,100):>9.1f}"
              f"{pct(mm,50):>9.1f}{pct(mm,90):>9.1f}  {note}")
        print(f"{'  (conf≥.5)':<10}{len(pxc):>5}{pct(pxc,50):>9.1f}{pct(pxc,90):>9.1f}{pct(pxc,95):>9.1f}{pct(pxc,100):>9.1f}")
        report[gname] = {"n": len(px), "px_p50": pct(px, 50), "px_p90": pct(px, 90),
                         "px_p95": pct(px, 95), "px_max": pct(px, 100),
                         "mm_p50": pct(mm, 50), "mm_p90": pct(mm, 90),
                         "conf_px_p90": pct(pxc, 90)}

    hl = np.asarray(hand_len_m) * 1000
    fl = np.asarray(foot_len_m) * 1000
    hc = np.asarray(hand_colin)
    fc = np.asarray(foot_colin)
    print(f"\n[端点延伸] wrist→hand 骨长: {hl.mean():.1f}±{hl.std():.1f} mm；"
          f"与前臂共线度 cos p50={pct(hc,50):.3f}（1.0=完全伸直延长）")
    print(f"[端点延伸] ankle→foot 骨长: {fl.mean():.1f}±{fl.std():.1f} mm；"
          f"与小腿共线度 cos p50={pct(fc,50):.3f}")
    pen = np.asarray(contact_penetration_m) * 1000
    if len(pen):
        print(f"[接触穿插] 撑地手样本 n={len(pen)}：穿插深度 p50={pct(pen,50):.1f}mm "
              f"p90={pct(pen,90):.1f}mm max={pct(pen,100):.1f}mm（>0 为穿到地面以下）")
    else:
        print("[接触穿插] 未检出手-地接触帧")

    (out_dir / "report.json").write_text(json.dumps({
        "groups": report,
        "hand_len_mm": {"mean": float(hl.mean()), "std": float(hl.std())},
        "foot_len_mm": {"mean": float(fl.mean()), "std": float(fl.std())},
        "hand_colinearity_p50": pct(hc, 50),
        "foot_colinearity_p50": pct(fc, 50),
        "contact_penetration_mm": {"n": len(pen), "p50": pct(pen, 50),
                                   "p90": pct(pen, 90), "max": pct(pen, 100)},
        "ground_y_m": ground_y,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if debug_imgs:
        cols = 2
        rows = (len(debug_imgs) + cols - 1) // cols
        th, tw = debug_imgs[0].shape[:2]
        grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
        for i, im in enumerate(debug_imgs):
            r, c = divmod(i, cols)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
        cv2.imwrite(str(out_dir / "diag_compare.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"\n[输出] 对比图: {out_dir / 'diag_compare.jpg'}；报告: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
