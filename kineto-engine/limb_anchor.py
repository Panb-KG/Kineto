#!/usr/bin/env python3
"""
P2.1 肢体端点 2D 锚定校正（只读输入视频 + joints，输出校正后 joints）

背景（诊断量化见 DEVELOPMENT_MEMORY）：HMR2/4D-Humans 的 2D 监督只到
COCO-17（腕/踝为止），且自遮挡帧（前伸臂、交叉腿）末端 3D 回归严重失真——
实测 wrist 重投影误差 p90≈300px(~590mm)、ankle/knee p90≈200px(~390mm)，
而躯干（肩/髋）p50≈31px 基线正常。

方法（不重跑 tracker，轻量后处理）：
  1. YOLOv8n-pose 逐帧检测 COCO-17 2D 关键点（多人取 bbox 最大者）；
  2. 每条四肢链（肩→肘→腕、髋→膝→踝）自根向叶逐点做**射线×骨长球面
     解析求交**：父关节为球心、解剖骨长为半径的球面，与 2D 检测点反投影
     射线的交点即校正后 3D 位置——天然保持骨长、不引入形变；叶点射线与
     中球面相距过远（HMR2 尺度/深度偏差致 2D 几何不相容）时，**级联**
     把中关节沿其射线回滑到最近可行深度（中骨微弹 ≤25%，超限放弃）；
     中点球面亦未击中时取垂足最近点（叶关节放宽到 2.0 倍骨长 + 降权，
     超出骨长交后续骨长约束收底）；
  2b. **直链反解优先**：根→中→叶在 2D 近似共线（伸直臂/腿）时 YOLO 中点
     无视觉拐点、定位沿肢干系统性偏移（实测上/前臂像素比 0.6 vs 骨长比
     1.06），此时忽略中点像素，由「根点 + 叶射线 + 两段骨长」直链铺设
     中/叶位置（骨长零误差、叶点落在射线上）；
  3. 门控防呆：关键点置信度 ≥0.6、初始重投影误差 ∈[8,700]px（过小不动、
     过大必错），其中 >350px 的大偏差段必须通过时序连续性门控（±3 帧内
     有连续同侧观测）——连续高置信轨迹上的大偏差是真实修正目标（臂/腿
     前伸段），孤立野值拒绝；左右侧按总重投影代价在「同侧/交叉」两种
     假设间择优（交叉腿帧 YOLO 偶发左右翻转）；击中/回退分别以 0.85/0.60
     权重混合（不全量吸附，保留 HMR2 3D 先验）；
  4. 校正轨迹做窗长 5 的中值滤波（20fps 康复动作下滞后 ≤2 帧可忽略，
     且对单帧检测野值鲁棒）；
  5. 叶子关节随动：hand 沿校正后前臂方向（诊断 cos=0.997，模型行为即
     手沿前臂直线延长）；foot 保留 HMR2 估计的脚踝→脚局部朝向（诊断
     cos=0.255 证明「沿小腿延长」对脚是错的），仅随修正后的踝平移。

坐标约定与引擎一致：joints 为 SMPL canonical（骨盆原点），相机系 =
joints + cam_t（Y 向下、Z 向前），焦距 focal = 5000/256*max(w,h)。
骨架常量只从 skeleton_spec 导入（项目铁律）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from skeleton_spec import BONE_LENGTH_BOUNDS

# ---------------------------------------------------------------------------
# COCO-17 关键点索引（person-centric 左右，与 SMPL canonical 同名侧一致）
# ---------------------------------------------------------------------------
COCO_L_SHOULDER, COCO_R_SHOULDER = 5, 6
COCO_L_ELBOW, COCO_R_ELBOW = 7, 8
COCO_L_WRIST, COCO_R_WRIST = 9, 10
COCO_L_HIP, COCO_R_HIP = 11, 12
COCO_L_KNEE, COCO_R_KNEE = 13, 14
COCO_L_ANKLE, COCO_R_ANKLE = 15, 16

# SMPL canonical 24 关节索引
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_HIP, R_HIP = 1, 2
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
L_HAND, R_HAND = 22, 23
L_FOOT, R_FOOT = 10, 11

# 锚定物理链：(标签, 根关节, 中关节, 叶关节, COCO 根点, COCO 中点, COCO 叶点)。
# 根（肩/髋）固定于 HMR2（躯干 p50≈31px 基线已优）；中/叶沿各自 2D 射线
# 自根向叶求解；叶与球面相距过远（尺度/深度不相容）时，中关节沿其射线
# **级联滑动**恢复可行（见主函数注释）；2D 共线伸直链走直链 IK（中点
# 检测不可信，见 _straight_chain_targets）。
CHAIN_GROUPS = [
    ("armL", L_SHOULDER, L_ELBOW, L_WRIST, COCO_L_SHOULDER, COCO_L_ELBOW, COCO_L_WRIST),
    ("armR", R_SHOULDER, R_ELBOW, R_WRIST, COCO_R_SHOULDER, COCO_R_ELBOW, COCO_R_WRIST),
    ("legL", L_HIP, L_KNEE, L_ANKLE, COCO_L_HIP, COCO_L_KNEE, COCO_L_ANKLE),
    ("legR", R_HIP, R_KNEE, R_ANKLE, COCO_R_HIP, COCO_R_KNEE, COCO_R_ANKLE),
]
# 交叉假设：同肢体左右链 COCO 索引互换（键为根/中点/叶点索引）
_COCO_PARTNER = {
    COCO_L_SHOULDER: COCO_R_SHOULDER, COCO_R_SHOULDER: COCO_L_SHOULDER,
    COCO_L_ELBOW: COCO_R_ELBOW, COCO_R_ELBOW: COCO_L_ELBOW,
    COCO_L_WRIST: COCO_R_WRIST, COCO_R_WRIST: COCO_L_WRIST,
    COCO_L_HIP: COCO_R_HIP, COCO_R_HIP: COCO_L_HIP,
    COCO_L_KNEE: COCO_R_KNEE, COCO_R_KNEE: COCO_L_KNEE,
    COCO_L_ANKLE: COCO_R_ANKLE, COCO_R_ANKLE: COCO_L_ANKLE,
}
# 交叉判定按同肢体两链整体比较（臂：两链 4 点；腿：两链 4 点）
_CROSS_ARM_IDX = [COCO_L_ELBOW, COCO_L_WRIST, COCO_R_ELBOW, COCO_R_WRIST]
_CROSS_LEG_IDX = [COCO_L_KNEE, COCO_L_ANKLE, COCO_R_KNEE, COCO_R_ANKLE]

# 叶子随动：(父, 子, 模式)；hand 沿「校正后前臂方向」，foot 保留原局部朝向
LEAF_FOLLOW = [
    (L_WRIST, L_HAND, "forearm_dir", L_ELBOW),
    (R_WRIST, R_HAND, "forearm_dir", R_ELBOW),
    (L_ANKLE, L_FOOT, "keep_dir", None),
    (R_ANKLE, R_FOOT, "keep_dir", None),
]

# 门控阈值
KP_CONF_MIN = 0.6      # YOLO 关键点置信度下限
ERR_MIN_PX = 8.0       # 初始重投影误差小于此值视为已对齐，不动
ERR_MAX_PX = 700.0     # 初始误差硬上界（超过必为检测错误/配错人）
ERR_BIG_PX = 350.0     # 大误差区起点：此值以上须通过时序一致性门控
TRACK_NEIGH_PX = 80.0  # 大误差帧：±3 帧内同侧点须有 ≤此距离的连续观测
TRACK_NEIGH_WIN = 3
TRACK_NEIGH_CONF = 0.4
MOVE_MAX_M = 0.6       # 单关节单次校正位移上限（米），防野值
DEPTH_MIN_M = 0.2      # 交点深度下限
MEDIAN_WINDOW = 5      # 校正轨迹中值滤波窗长（帧）
LEAF_FOLLOW_EPS = 0.01  # 父关节位移超此值（米）才重定位叶子
# 射线未击中球面（2D 像素几何与固定骨长/深度严格不相容，多由 HMR2 尺度/深度
# 轻微偏差引起）时的回退：沿射线把关节滑到**最近可行位置**（级联深度滑动，
# 见 anchor_limbs_to_2d），骨长最多微弹 MISS_STRETCH_MAX 倍，超限判几何
# 严重不相容（疑似检测/配对错误）放弃锚定。
MISS_STRETCH_MAX = 1.25
# 叶关节（腕/踝）回退上限放宽到 2.0 倍骨长：诊断发现部分前伸帧存在 HMR2
# 全局尺度偏差（腕射线与中关节球面相距到 1.5-1.8 倍骨长），根固定时不可
# 严格满足骨长；此时 2D 像素位置仍可信，放宽回退 + 降混合权重，超出的骨
# 长由后续骨长约束合法收底（沿肢干方向回拉部分位移，仍保留大部分校正）。
MISS_STRETCH_MAX_LEAF = 2.00
BLEND_MISS_LEAF = 0.50
# 混合权重：击中射线×球面（几何相容）时校正置信高；最近点回退（骨被微弹）
# 时像素证据仍可信但 3D 深度不可观，取较小权重，骨长交后续骨长约束收底。
BLEND_HIT = 0.85
BLEND_MISS = 0.60
# 直链 IK：根→中→叶三段 2D 近似共线（伸直臂/腿）时，YOLO 中点（肘/膝）
# 检测无视觉拐点、定位不可信（实测伸直臂 YOLO 上/前臂像素比 0.6，而解剖
# 骨长比≈1.06），此时忽略中点像素，由「根点 + 叶射线 + 两段骨长」直链
# 反解中/叶位置。门控：共线度、根点置信度、总跨度与全骨长透视一致性
# （排除纵深弯折：2D 同样共线但总跨度明显缩短）。
STRAIGHT_DETOUR_MAX = 1.08  # 中点绕行率 (n1+n2)/根→叶直线距离 上限（实测伸直肢≤1.05；中点侧向偏 40px 的真弯折更大）
STRAIGHT_SPAN_TOL = 0.15  # 总像素跨度与全骨长透视投影的相对容差（收紧：真伸直肢跨度吻合好；纵深弯折总跨度压缩 ≥15% 须排除）
STRAIGHT_RATIO_CONFLICT = 0.35  # YOLO 中/叶跨度比与骨长比的相对分歧阈值（中点失真签名）
STRAIGHT_LEAF_E0_MIN = 40.0     # 叶点初始误差 ≥ 此值才启用直链（已对齐帧不接管）
ROOT_CONF_MIN = 0.5       # 根点（肩/髋）置信度下限（仅用于直链方向判定）
STRAIGHT_BLEND = 0.85     # 直链反解混合权重（骨长精确、叶点落在 2D 射线上）
# 直链位移上限比普通射线求解放宽：直链目标几何完全约束（两段骨长精确 + 叶点
# 在射线上 + 门控严格），大位移正是对 HMR2 「整条肢塌缩到躯干」失败模式的
# 修正（实测可达 ~0.65m）；中点位移小于叶点（中点在根与叶之间）。
STRAIGHT_MOVE_MAX_MID = 0.7
STRAIGHT_MOVE_MAX_LEAF = 0.9

_ANCHOR_JOINT_IDS = [L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
                     L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
_GROUP_OF_CHILD = {
    L_ELBOW: "elbow", R_ELBOW: "elbow",
    L_WRIST: "wrist", R_WRIST: "wrist",
    L_KNEE: "knee", R_KNEE: "knee",
    L_ANKLE: "ankle", R_ANKLE: "ankle",
}


def detect_pose_2d(video_path: str, n_expected: int, device: str = "xpu") -> np.ndarray:
    """逐帧检测 COCO-17 关键点。

    返回 (n_expected,17,3) 数组 [x, y, conf]；未检出/缺帧置 conf=-1。
    多人取 bbox 面积最大者。设备推理失败一次性回退 CPU；模型文件缺失
    返回 None（调用方跳过锚定，行为退化为现状）。
    """
    model_path = Path(__file__).resolve().parent / "yolov8n-pose.pt"
    if not model_path.exists():
        print(f"[Anchor][WARNING] 未找到 {model_path.name}，跳过 2D 锚定")
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"[Anchor][WARNING] YOLOv8n-pose 加载失败（{type(e).__name__}: {e}），跳过锚定")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Anchor][WARNING] 无法打开视频 {video_path}，跳过锚定")
        return None

    kp_all = np.full((n_expected, 17, 3), -1.0, dtype=np.float32)
    use_device = device
    idx = 0
    try:
        while idx < n_expected:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                res = model(frame, verbose=False, device=use_device)[0]
            except Exception:
                # XPU/指定设备推理异常 → 一次性回退 CPU（ultralytics 自动迁移）
                if use_device != "cpu":
                    use_device = "cpu"
                    res = model(frame, verbose=False, device="cpu")[0]
                else:
                    raise
            kpts = getattr(getattr(res, "keypoints", None), "data", None)
            if kpts is not None and len(kpts) > 0:
                kpts = kpts.cpu().numpy()
                if kpts.ndim == 3 and kpts.shape[1] >= 17:
                    # 多人取 bbox 最大者；无 boxes 信息时取第 0 人
                    pick = 0
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and len(boxes) > 1:
                        xy = boxes.xyxy.cpu().numpy()
                        areas = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
                        pick = int(areas.argmax())
                    kp_all[idx, :17, :] = kpts[pick, :17, :]
            idx += 1
    finally:
        cap.release()

    print(f"[Anchor] 2D 关键点检测完成：{idx} 帧（device={use_device}）")
    return kp_all


def _bone_radius(parent: int, child: int, j_orig: np.ndarray) -> float:
    """当前帧骨长 clip 进 SSOT 骨长界（与 apply_bone_length_constraint 同约定）。"""
    length = float(np.linalg.norm(j_orig[child] - j_orig[parent]))
    lo, hi = BONE_LENGTH_BOUNDS[(parent, child)]
    return float(np.clip(length, lo, hi))


def _project(J: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    """相机系点 → 像素 (u,v)。J 形状 (...,3)。"""
    z = np.clip(J[..., 2], 1e-6, None)
    u = focal * J[..., 0] / z + cx
    v = focal * J[..., 1] / z + cy
    return np.stack([u, v], axis=-1)


def _ray_of(uv: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    """像素 (u,v) → 相机系单位射线方向（原点为相机光心）。"""
    d = np.array([(uv[0] - cx) / focal, (uv[1] - cy) / focal, 1.0], dtype=np.float64)
    return d / np.linalg.norm(d)


def _solve_on_ray(P: np.ndarray, child_orig: np.ndarray, kp: np.ndarray,
                  r: float, focal: float, cx: float, cy: float,
                  track_good: bool,
                  stretch_cap: float = MISS_STRETCH_MAX,
                  miss_w: float = BLEND_MISS) -> dict | None:
    """射线（2D 点反投影）与球心 P、半径 r 的球面求解单关节目标位置。

    成功返回 {"point": 相机系目标点, "hit": 是否严格击中（False=垂足回退）,
    "w": 混合权重}；门控不通过返回 None。门控：置信度、初差区间
    [ERR_MIN, ERR_MAX]、大误差段须时序连续、深度/位移上限、回退骨微弹
    ≤stretch_cap（叶关节放宽，见常量说明）。
    """
    if kp[2] < KP_CONF_MIN:
        return None
    uv_target = kp[:2]
    uv0 = _project(child_orig, focal, cx, cy)
    e0 = float(np.hypot(*(uv0 - uv_target)))
    if e0 < ERR_MIN_PX or e0 > ERR_MAX_PX:
        return None
    if e0 > ERR_BIG_PX and not track_good:
        return None

    d = _ray_of(uv_target, focal, cx, cy)
    t_orig = float(d @ child_orig)
    b = float(d @ P)
    disc = b * b - float(P @ P) + r * r
    if disc >= 0:
        sq = np.sqrt(disc)
        t = b - sq if abs((b - sq) - t_orig) < abs((b + sq) - t_orig) else b + sq
        w = BLEND_HIT
    else:
        # 未击中：取射线上距球心最近点（垂足），此时骨被拉长；微弹超限放弃
        t = b
        stretch = float(np.sqrt(max(float(P @ P) - b * b, 0.0)))
        if stretch > stretch_cap * r:
            return None
        w = miss_w
    if t < DEPTH_MIN_M:
        return None
    F = t * d
    if float(np.linalg.norm(F - child_orig)) > MOVE_MAX_M:
        return None
    return {"point": F, "hit": disc >= 0, "w": w}


def _apply_blend(joints: np.ndarray, J_cam: np.ndarray, J_orig_cam: np.ndarray,
                 cam_t_fi: np.ndarray, fi: int, joint_idx: int, sol: dict,
                 stats: dict, grp: str, focal: float, cx: float, cy: float,
                 uv_target: np.ndarray) -> None:
    """按 sol 权重把目标点混合写回 joints 与本帧相机系工作副本，并记台账。"""
    w = sol["w"]
    F = sol["point"]
    C_new = J_cam[joint_idx] + w * (F - J_cam[joint_idx])
    J_cam[joint_idx] = C_new
    joints[fi, joint_idx] = C_new - cam_t_fi
    stats[grp]["n"] += 1
    stats[grp]["hit" if sol["hit"] else "miss"] += 1
    stats[grp]["corr_mm"].append(float(np.linalg.norm(C_new - J_orig_cam[joint_idx])) * 1000.0)
    stats[grp]["resid_px"].append(float(np.hypot(
        *(_project(C_new, focal, cx, cy) - uv_target))))


def _track_continuity(kp2d: np.ndarray) -> np.ndarray:
    """时序连续性表 (N,17) bool：点在 ±TRACK_NEIGH_WIN 内有距离 ≤TRACK_NEIGH_PX、
    conf≥TRACK_NEIGH_CONF 的同侧观测则为 True（连续轨迹，大偏差可锚）。"""
    n = len(kp2d)
    near = np.zeros((n, 17), dtype=bool)
    for k in range(1, TRACK_NEIGH_WIN + 1):
        if n <= k:
            break
        # i ↔ i+k 对齐比较，两段近邻各写回一次（前邻/后邻）
        a0, a1 = kp2d[:n - k], kp2d[k:]
        dist = np.hypot(a0[:, :, 0] - a1[:, :, 0], a0[:, :, 1] - a1[:, :, 1])
        ok = ((a0[:, :, 2] >= KP_CONF_MIN)
              & (a1[:, :, 2] >= TRACK_NEIGH_CONF)
              & (dist <= TRACK_NEIGH_PX))
        near[:n - k] |= ok
        near[k:] |= ok
    return near


def _straight_chain_targets(J_cam: np.ndarray, J_orig_cam: np.ndarray,
                            root: int, mid: int, leaf: int,
                            kpf: np.ndarray, coco_root: int, coco_mid: int,
                            coco_leaf: int, focal: float, cx: float, cy: float,
                            leaf_track: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """直链反解：伸直肢（根→中→叶 2D 共线）时返回 (中目标点, 叶目标点)。

    伸直臂/腿在图像中没有肘/膝视觉拐点，YOLO 中点定位沿肢干方向系统性
    偏移（实测伸直臂上/前臂像素比 0.6 vs 解剖骨长比 1.06）。此时以根 3D
    点为起点、叶 2D 射线为方向、两段骨长精确铺设：中点 = 根 + r_mid·d̂，
    叶点 = 根 + (r_mid+r_leaf)·d̂（d̂ 为根指向「叶射线根深度点」的单位
    向量）——骨长零误差、叶点落在 2D 射线上。门控不通过返回 None（回退
    普通射线×球面流程）。
    """
    kpr, kpm, kpl = kpf[coco_root], kpf[coco_mid], kpf[coco_leaf]
    if kpr[2] < ROOT_CONF_MIN or kpl[2] < KP_CONF_MIN or kpm[2] < KP_CONF_MIN:
        return None
    v1 = kpm[:2] - kpr[:2]
    v2 = kpl[:2] - kpm[:2]
    n1 = float(np.hypot(*v1))
    n2 = float(np.hypot(*v2))
    if n1 < 20.0 or n2 < 20.0:
        return None
    # 直链判定主门控用「中点绕行率」：两段路径长 / 根→叶直线距离。
    # 伸直肢路径近乎直线（实测 ≤1.05，中点侧向偏移也只在中点位置不在线上）；
    # 真弯折显著 >1.08（90° 等段弯折 ≈1.41）。比逐段夹角 cos 更稳：中点检测
    # 沿肢干方向错位（失真签名）不影响绕行率，仅侧向偏移才推高绕行率。
    direct = float(np.hypot(*(kpl[:2] - kpr[:2])))
    if direct < 20.0 or (n1 + n2) / direct > STRAIGHT_DETOUR_MAX:
        return None

    r_mid = _bone_radius(root, mid, J_orig_cam)
    r_leaf = _bone_radius(mid, leaf, J_orig_cam)
    total = r_mid + r_leaf

    # 中点失真签名：YOLO 中/叶跨度比与解剖骨长比严重分歧（伸直臂实测
    # 0.60 vs 1.06）。轻度弯折（2D 共线但中点像素仍可信）比值变化小，
    # 分歧不足时不接管，交普通射线×球面流程。
    rat_yolo, rat_bone = n1 / n2, r_mid / r_leaf
    if abs(rat_yolo - rat_bone) / rat_bone < STRAIGHT_RATIO_CONFLICT:
        return None

    # 总跨度须与全骨长在根深度的透视投影一致（容差内）——2D 共线也可能是
    # 肢干在纵深方向弯折（总跨度因透视缩短），那种情形中点像素仍有信息量。
    d = _ray_of(kpl[:2], focal, cx, cy)
    t0 = float(d @ J_cam[root])
    if t0 < DEPTH_MIN_M:
        return None
    if abs((n1 + n2) - total * focal / t0) / (total * focal / t0) > STRAIGHT_SPAN_TOL:
        return None

    # 叶点初始误差门控（已对齐帧无需直链接管；>ERR_BIG_PX 须时序连续）
    e0 = float(np.hypot(*(_project(J_orig_cam[leaf], focal, cx, cy) - kpl[:2])))
    if e0 < STRAIGHT_LEAF_E0_MIN or e0 > ERR_MAX_PX:
        return None
    if e0 > ERR_BIG_PX and not leaf_track:
        return None

    W = t0 * d
    dirv = W - J_cam[root]
    L = float(np.linalg.norm(dirv))
    if not (total * 0.5 <= L <= total * 1.8):
        return None
    hat = dirv / L
    mid_t = J_cam[root] + r_mid * hat
    leaf_t = J_cam[root] + total * hat
    if (float(np.linalg.norm(mid_t - J_orig_cam[mid])) > STRAIGHT_MOVE_MAX_MID
            or float(np.linalg.norm(leaf_t - J_orig_cam[leaf])) > STRAIGHT_MOVE_MAX_LEAF):
        return None
    return mid_t, leaf_t


def _median_filter_joints(joints: np.ndarray, joint_ids: list[int],
                          window: int = MEDIAN_WINDOW) -> np.ndarray:
    """对指定关节的 3D 轨迹做边缘填充的窗长 window 中值滤波。"""
    pad = window // 2
    padded = np.pad(joints, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    shifts = np.stack([padded[k:k + len(joints)] for k in range(window)], axis=0)
    out = joints.copy()
    out[:, joint_ids, :] = np.median(shifts[:, :, joint_ids, :], axis=0)
    return out


def anchor_limbs_to_2d(joints: np.ndarray, cam_t: np.ndarray, kp2d: np.ndarray,
                       img_w: int, img_h: int,
                       focal_length: float = 5000.0, image_size: int = 256
                       ) -> tuple[np.ndarray, dict]:
    """2D 锚定校正主入口。

    joints/cam_t: (N,24,3)/(N,3)。返回 (校正后 joints 副本, 统计 dict)。
    """
    joints = np.asarray(joints, dtype=np.float64).copy()
    cam_t = np.asarray(cam_t, dtype=np.float64)
    joints_orig = joints.copy()              # Phase1 原始快照（叶子朝向/位移判定用）
    n = min(len(joints), len(kp2d))
    focal = focal_length / image_size * max(img_w, img_h)
    cx, cy = img_w / 2.0, img_h / 2.0
    straight_n = 0                           # 直链反解触发链次

    stats: dict = {g: {"n": 0, "hit": 0, "miss": 0,
                       "corr_mm": [], "resid_px": []}
                   for g in ("elbow", "wrist", "knee", "ankle")}

    # 时序连续性表：每帧每个 COCO 点在 ±WIN 内是否有距离 ≤NEIGH_PX 的可靠观测。
    # 大误差（>ERR_BIG_PX）锚定必须通过此门控——连续轨迹上的高 conf 大偏差是
    # 真实修正目标（臂/腿前伸段），孤立大偏差野值被拒绝。
    track_ok = _track_continuity(kp2d[:n])

    for fi in range(n):
        J_cam = joints[fi] + cam_t[fi]          # 相机系工作副本（逐链更新）
        J_orig_cam = J_cam.copy()               # 本帧原始相机系位置（骨长/初差用）
        kpf = kp2d[fi]

        # 左右交叉判定：臂/腿两组各按总重投影代价择优
        cross_arm = _should_cross(J_cam, kpf, [CHAIN_GROUPS[0], CHAIN_GROUPS[1]],
                                  focal, cx, cy)
        cross_leg = _should_cross(J_cam, kpf, [CHAIN_GROUPS[2], CHAIN_GROUPS[3]],
                                  focal, cx, cy)

        for ci, (_tag, root, mid, leaf, coco_root, coco_mid, coco_leaf) in enumerate(CHAIN_GROUPS):
            crossed = cross_arm if ci < 2 else cross_leg
            if crossed:
                coco_root = _COCO_PARTNER[coco_root]
                coco_mid = _COCO_PARTNER[coco_mid]
                coco_leaf = _COCO_PARTNER[coco_leaf]

            # ---- 直链反解优先：2D 共线伸直肢（中点检测不可信）----
            straight = _straight_chain_targets(
                J_cam, J_orig_cam, root, mid, leaf, kpf,
                coco_root, coco_mid, coco_leaf, focal, cx, cy,
                track_ok[fi, coco_leaf])
            if straight is not None:
                straight_n += 1
                mid_t, leaf_t = straight
                _apply_blend(joints, J_cam, J_orig_cam, cam_t[fi], fi, mid,
                             {"point": mid_t, "hit": True, "w": STRAIGHT_BLEND},
                             stats, _GROUP_OF_CHILD[mid], focal, cx, cy,
                             kpf[coco_mid][:2])
                _apply_blend(joints, J_cam, J_orig_cam, cam_t[fi], fi, leaf,
                             {"point": leaf_t, "hit": True, "w": STRAIGHT_BLEND},
                             stats, _GROUP_OF_CHILD[leaf], focal, cx, cy,
                             kpf[coco_leaf][:2])
                continue

            # ---- 中关节：射线 × (根球心, 骨长) 球面 ----
            mid_sol = _solve_on_ray(J_cam[root], J_orig_cam[mid], kpf[coco_mid],
                                    _bone_radius(root, mid, J_orig_cam),
                                    focal, cx, cy, track_ok[fi, coco_mid])
            if mid_sol is not None:
                _apply_blend(joints, J_cam, J_orig_cam, cam_t[fi], fi, mid,
                             mid_sol, stats, _GROUP_OF_CHILD[mid], focal, cx, cy,
                             kpf[coco_mid][:2])

            # ---- 叶关节：射线 × (中关节球心, 骨长) 球面；不相容则级联滑中 ----
            # 叶关节（腕/踝）放宽骨微弹门控并降混合权重（尺度偏差帧像素仍可信，
            # 超出骨长交骨长约束收底）。
            r_leaf = _bone_radius(mid, leaf, J_orig_cam)
            leaf_sol = _solve_on_ray(J_cam[mid], J_orig_cam[leaf], kpf[coco_leaf],
                                     r_leaf, focal, cx, cy,
                                     track_ok[fi, coco_leaf],
                                     stretch_cap=MISS_STRETCH_MAX_LEAF,
                                     miss_w=BLEND_MISS_LEAF)
            if leaf_sol is None and mid_sol is not None:
                # 级联：叶射线到中射线的夹角使当前中深度不可行 → 沿中射线
                # 把中关节滑到最近可行深度 tm_feas = r_leaf/sqrt(1-(d_mid·d_leaf)²)，
                # 中骨（根→中）允许微弹 ≤MISS_STRETCH_MAX。
                d_mid = _ray_of(kpf[coco_mid][:2], focal, cx, cy)
                d_leaf = _ray_of(kpf[coco_leaf][:2], focal, cx, cy)
                if d_mid is not None and d_leaf is not None:
                    cos_a = float(np.clip(d_mid @ d_leaf, -1.0, 1.0))
                    sin2 = max(1.0 - cos_a * cos_a, 1e-12)
                    tm_feas = r_leaf / np.sqrt(sin2)
                    tm_cur = float(np.linalg.norm(J_cam[mid]))   # 中当前深度（在射线方向）
                    if tm_cur > tm_feas and tm_feas > DEPTH_MIN_M:
                        M_slide = tm_feas * d_mid
                        # 根骨微弹检验
                        root_stretch = float(np.linalg.norm(M_slide - J_cam[root]))
                        r_mid = _bone_radius(root, mid, J_orig_cam)
                        if root_stretch <= MISS_STRETCH_MAX * r_mid:
                            # 中关节沿射线回滑（与中关节自己的混合已写入，这里是
                            # 同一帧内的可行域投影，直接覆盖为滑动位置）
                            J_cam[mid] = M_slide
                            joints[fi, mid] = M_slide - cam_t[fi]
                            # 叶点在新中球面上重解（此时判别式 ≥0）
                            leaf_sol = _solve_on_ray(M_slide, J_orig_cam[leaf],
                                                     kpf[coco_leaf],
                                                     r_leaf, focal, cx, cy,
                                                     track_ok[fi, coco_leaf],
                                                     stretch_cap=MISS_STRETCH_MAX_LEAF,
                                                     miss_w=BLEND_MISS_LEAF)
            if leaf_sol is not None:
                _apply_blend(joints, J_cam, J_orig_cam, cam_t[fi], fi, leaf,
                             leaf_sol, stats, _GROUP_OF_CHILD[leaf], focal, cx, cy,
                             kpf[coco_leaf][:2])

    # ---- 校正轨迹中值滤波（压 YOLO 单帧抖动/野值；康复慢速动作滞后可忽略）----
    joints = _median_filter_joints(joints, _ANCHOR_JOINT_IDS, MEDIAN_WINDOW)

    # ---- 叶子关节随动（仅父关节被实质移动的帧）----
    leaf_n = 0
    for fi in range(n):
        for (parent, child, mode, aux) in LEAF_FOLLOW:
            # 父关节相对原始位置位移 > eps 才重定位叶子（未锚定帧保持原样）
            if np.linalg.norm(joints[fi, parent] - joints_orig[fi, parent]) <= LEAF_FOLLOW_EPS:
                continue
            p_new = joints[fi, parent]
            if mode == "forearm_dir":
                # hand：沿**校正后**前臂（肘→腕）方向延长（诊断 cos=0.997）
                direction = p_new - joints[fi, aux]
            else:
                # foot：保留 HMR2 **原始**脚踝→脚局部朝向（诊断证明沿小腿
                # 延长错误：cos p50=0.255；脚有独立踝角，2D 无脚趾观测时
                # 不臆造朝向），仅随修正后的踝平移
                direction = joints_orig[fi, child] - joints_orig[fi, parent]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                continue
            direction /= norm
            r = _bone_radius(parent, child, joints_orig[fi])
            joints[fi, child] = p_new + direction * r
            leaf_n += 1

    summary = {
        "model": "yolov8n-pose",
        "method": "ray-sphere-straight-ik-v3e",
        "frames": int(n),
        "straight_chain_count": straight_n,
        "anchored_counts": {g: stats[g]["n"] for g in stats},
        "leaf_repositioned": leaf_n,
        "groups": {},
    }
    print("[Anchor] 校正统计（组/锚点数(击中/回退)/校正位移 mm p50/p90/残差 px p50）：")
    for g, s in stats.items():
        if s["corr_mm"]:
            corr = np.array(s["corr_mm"])
            resid = np.array(s["resid_px"])
            summary["groups"][g] = {
                "n": s["n"],
                "hit": s["hit"],
                "miss_fallback": s["miss"],
                "corr_mm_p50": round(float(np.percentile(corr, 50)), 1),
                "corr_mm_p90": round(float(np.percentile(corr, 90)), 1),
                "resid_px_p50": round(float(np.percentile(resid, 50)), 1),
            }
            print(f"  {g:6s} n={s['n']:4d}({s['hit']}/{s['miss']})  "
                  f"校正 {np.percentile(corr,50):6.1f}/{np.percentile(corr,90):6.1f} mm  "
                  f"残差 {np.percentile(resid,50):5.1f} px")
        else:
            print(f"  {g:6s} 无锚定点")
    print(f"[Anchor] 直链反解 {straight_n} 链次（伸直肢中点忽略 YOLO）；"
          f"叶子随动重定位 {leaf_n} 处（hand 沿前臂 / foot 保朝向）；"
          f"骨长复原最大单点修正 {restore_max_mm:.1f} mm")
    return joints, summary


def _should_cross(J_cam: np.ndarray, kp_frame: np.ndarray, chains: list,
                  focal: float, cx: float, cy: float) -> bool:
    """比较同侧/交叉两种左右配对假设的总重投影代价，取较优者。

    chains 为同肢体（臂或腿）的两条 CHAIN_GROUPS 元组；代价取两条链中关节+
    叶关节共 4 点的平均像素误差（交叉腿/臂帧 YOLO 偶发左右翻转）。
    """
    costs = [0.0, 0.0]   # [同侧, 交叉]
    counts = [0, 0]
    for chain in chains:
        _, _root, mid, leaf, _coco_root, coco_mid, coco_leaf = chain
        for j, kp_idx in ((mid, coco_mid), (leaf, coco_leaf)):
            uv0 = _project(J_cam[j], focal, cx, cy)
            for hyp, idx in ((0, kp_idx), (1, _COCO_PARTNER[kp_idx])):
                kp = kp_frame[idx]
                if kp[2] < KP_CONF_MIN:
                    continue
                costs[hyp] += float(np.hypot(*(uv0 - kp[:2])))
                counts[hyp] += 1
    if counts[0] == 0 and counts[1] == 0:
        return False
    # 按可用点数归一，避免某假设因点多而总代价虚高
    c0 = costs[0] / max(counts[0], 1)
    c1 = costs[1] / max(counts[1], 1)
    return c1 < c0
