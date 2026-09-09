#!/usr/bin/env python3
"""
Kineto Core - 视频 3D 姿态解算引擎
基于 4DHumans 提取 SMPL-X 格式三维关节参数，设备自适应 (CUDA/XPU/MPS/CPU)
"""

import argparse
import contextlib
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.lbs import vertices2joints
from tqdm import tqdm

# 骨架解剖常量单一事实源 (SSOT)：关节名/父子拓扑/骨对/骨长界/真实 rest 模板
# 均由 skeleton_spec 从真 SMPL kintree + basicModel pkl 派生，本文件不再硬编码。
from skeleton_spec import (
    BONE_LENGTH_BOUNDS,
    BONE_NAMES,
    BONE_PART_MAP,
    SMPL_CHILDREN_FULL,
    SMPL_JOINT_NAMES,  # noqa: F401  (对外重导出，供下游/调试引用)
    SMPL_PARENTS,
    SMPL_REST_JOINTS,
    SMPL_SKELETON,
)


# ============================================================================
# [P3/改动 F] STRICT 拒绝异常 + 独立退出码
# ----------------------------------------------------------------------------
# StrictModeRefused：KINETO_STRICT=1 下，引擎**该拒绝**的降级/假数据场景（缺权重、
# 回退合成关键点、detector 静默降级）抛此特定异常。它替代旧代码过宽的
# `except RuntimeError`——后者会连真实 bug（如 shape 不匹配的 RuntimeError）一并
# 吞成“STRICT 拒绝”，掩盖真因。main() 只捕获 StrictModeRefused，其余异常照常
# 冒泡（打印 traceback + 退出码 1），不再被静默。继承 RuntimeError 以兼容任何
# 既存 `except RuntimeError` 的下游（向后兼容）。
class StrictModeRefused(RuntimeError):
    """STRICT 模式下拒绝交付降级/合成/假数据的显式信号。"""


# 独立退出码：argparse 用法错误用 2、通用未捕获异常用 1，故 STRICT 拒绝用 3，
# 以便调用方（api.py 子进程 / CI / 运维脚本）区分“STRICT 主动拒绝”与“真实崩溃”。
EXIT_STRICT_REFUSED = 3


# ============================================================================
# [P1.1 mesh 传输压缩] DRACO 编码
# ----------------------------------------------------------------------------
# 背景：公网入口走 Tailscale Funnel 中继，实测带宽仅 ~40-130KB/s；P1 的全帧
# f32 mesh（466 帧 ≈ 38.5MB）经 Funnel 需 5-15 分钟，浏览器必然超时 → mesh
# 模式整体不可用。DRACO 连通性感知压缩 + 时间抽帧：
#   • 14bit 量化（bbox≈1.8m → 量化步长 ~0.03mm，实测误差 <0.1mm，视觉无损）；
#   • stride=3：mesh 6.7fps（骨架仍全帧 20fps 关键帧、前端 60fps 线性插值），
#     mesh 相邻帧 150ms 线性插值在康复动作节奏下无可见顿挫（旧痛点是 16 帧/23s
#     = 0.7fps 且不插值的「跳变」）；
#   → 466 帧视频 mesh 产物约 3MB（vs 38.5MB）。
# DRACO 连通性感知压缩会按连通性遍历**重排顶点编号**（面表也可能重排），
# 解码顺序对同一套 faces 跨帧确定性一致（实测解码 faces 逐帧完全相同）。
# 引擎在编码后立即解码首帧，用 3D 最近点匹配（量化误差 <0.1mm ≪ 最小顶点
# 间距 ~1mm）求出 orig→dec 排列 perm 并写入容器头，消费方（前端/G14）
# 直接套用 perm 还原 SMPL 原始顶点序——不依赖任何「面序/绕序保持」假设；
# 引擎同时断言三角形多重集一致（排列正确性的独立验证）。
# DracoPy 缺失/编码失败时调用方回退全帧 f32（局域网/大带宽环境仍可用）。
# ============================================================================
MESH_DRACO_QUANT_BITS = 14
MESH_FRAME_STRIDE = 3
_MESH_DRC_MAGIC = b"KDRC"
_MESH_DRC_VERSION = 2
# 最近点匹配容差（米）：14bit 量化步长 ~0.03mm、实测最大误差 0.041mm；
# SMPL 网格最小顶点间距毫米级，1mm 阈值既能容忍量化又足以拒绝错误匹配。
_MESH_PERM_MATCH_TOL_M = 1e-3


def encode_mesh_track_drcs(
    vertices: np.ndarray,
    faces: np.ndarray,
    stride: int = MESH_FRAME_STRIDE,
    quant_bits: int = MESH_DRACO_QUANT_BITS,
) -> tuple[bytes, int]:
    """全帧顶点 (F,V,3) + 面索引 → KDRC 容器字节、mesh 帧数。

    容器布局 v2（小端）：
      magic 4B(b"KDRC") | version u16(=2) | quant_bits u16 |
      frame_count u32 | verts_per_frame u32 | stride u32 |
      perm u32[verts_per_frame]（perm[orig_v] = dec_v，orig→dec 排列）|
      offsets u32[frame_count]（自文件起始的字节偏移）|
      DRACO blob 顺序拼接。
    mesh 帧 k 对应原始 keyframes[k*stride]。
    """
    import DracoPy  # 惰性导入：缺失时由调用方回退 f32

    frame_idx = list(range(0, vertices.shape[0], stride))
    faces_u32 = np.ascontiguousarray(faces, dtype=np.uint32)
    faces_i64 = np.asarray(faces, dtype=np.int64)
    vpf = int(vertices.shape[1])

    blobs: list[bytes] = []
    for fi in frame_idx:
        blob = DracoPy.encode(
            points=np.ascontiguousarray(vertices[fi], dtype=np.float64),
            faces=faces_u32,
            quantization_bits=quant_bits,
            compression_level=7,
        )
        blobs.append(bytes(blob))

    # ── 求 orig→dec 顶点排列（仅首帧；跨帧确定性一致）──
    dec0 = DracoPy.decode(blobs[0])
    pts_dec = np.asarray(dec0.points, dtype=np.float64)
    faces_dec = np.asarray(dec0.faces, dtype=np.int64)
    if pts_dec.shape != (vpf, 3):
        raise RuntimeError(f"DRACO 首帧解码顶点形状异常: {pts_dec.shape} != ({vpf}, 3)")
    pts_orig0 = np.asarray(vertices[frame_idx[0]], dtype=np.float64)

    perm = np.empty(vpf, dtype=np.int64)
    nearest = np.empty(vpf, dtype=np.float64)
    chunk = 512  # 分块计算距离矩阵，避免 6890²×3 的临时内存
    for s in range(0, vpf, chunk):
        e = min(s + chunk, vpf)
        d = np.linalg.norm(pts_orig0[s:e, None, :] - pts_dec[None, :, :], axis=2)
        idx = d.argmin(axis=1)
        perm[s:e] = idx
        nearest[s:e] = d[np.arange(e - s), idx]
    if float(nearest.max()) > _MESH_PERM_MATCH_TOL_M:
        raise RuntimeError(
            f"DRACO 顶点排列匹配残差 {nearest.max() * 1000:.3f}mm "
            f"> 容差 {_MESH_PERM_MATCH_TOL_M * 1000:.1f}mm（量化异常或网格错配）"
        )
    if len(set(perm.tolist())) != vpf:
        raise RuntimeError("DRACO 顶点排列非双射（最近点匹配出现重复目标）")

    # 独立验证：orig 面表经 perm 映射后的三角形多重集必须与解码面表一致
    def _canon_tris(tris: np.ndarray) -> np.ndarray:
        s = np.sort(tris, axis=1)
        return s[np.lexsort((s[:, 2], s[:, 1], s[:, 0]))]

    mapped_faces = perm[faces_i64]
    if not np.array_equal(_canon_tris(mapped_faces), _canon_tris(faces_dec)):
        raise RuntimeError("DRACO 排列自验失败：映射后面表与解码面表不一致")

    header_len = 4 + 2 + 2 + 4 + 4 + 4 + 4 * vpf + 4 * len(blobs)
    offsets: list[int] = []
    cursor = header_len
    for b in blobs:
        offsets.append(cursor)
        cursor += len(b)

    parts = [
        _MESH_DRC_MAGIC,
        np.uint16(_MESH_DRC_VERSION).astype("<u2").tobytes(),
        np.uint16(quant_bits).astype("<u2").tobytes(),
        np.uint32(len(blobs)).astype("<u4").tobytes(),
        np.uint32(vpf).astype("<u4").tobytes(),
        np.uint32(stride).astype("<u4").tobytes(),
        np.asarray(perm, dtype="<u4").tobytes(),
        np.asarray(offsets, dtype="<u4").tobytes(),
    ]
    parts.extend(blobs)
    return b"".join(parts), len(blobs)


# ============================================================================
# 设备自适应检测
# ============================================================================

def detect_device():
    """自动检测最优计算设备: CUDA > XPU (Intel Arc) > MPS > CPU"""
    # Intel Arc (XPU): 导入 IPEX 以注册/优化 xpu 后端；缺失时静默跳过（torch>=2.5 原生支持 XPU）
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        # [Fix #15] 前缀改为 [IPEX]，保证 stdout 第一条 [Device] 行是真实设备行
        print(f"[IPEX] intel-extension-for-pytorch {ipex.__version__} 已加载")
    except Exception:
        pass

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] NVIDIA CUDA: {gpu_name} ({vram:.1f} GB)")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        try:
            name = torch.xpu.get_device_name(0)
        except Exception:
            name = "Intel GPU"
        print(f"[Device] Intel XPU: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (推理较慢)")
    return device


def clear_device_memory(device):
    """每帧推理后释放显存/内存"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ============================================================================
# SMPL 骨架定义 (24 关节标准拓扑) — 常量统一来自 skeleton_spec (SSOT)
# ============================================================================
# SMPL_JOINT_NAMES / SMPL_SKELETON / BONE_PART_MAP / SMPL_PARENTS /
# SMPL_CHILDREN_FULL / SMPL_REST_JOINTS / BONE_LENGTH_BOUNDS 均已在文件头部
# 从 skeleton_spec 导入。旧版本在此硬编码的骨架树与 SSOT 存在冲突
# （collar 13/14 父节点误作 12，真 kintree 为 9），已淘汰。

# 身体部位颜色 (BGR) — 纯展示层，不入 SSOT
BONE_COLORS = {
    "torso": (200, 200, 100),
    "left_arm": (100, 200, 100),
    "right_arm": (100, 100, 200),
    "left_leg": (200, 100, 100),
    "right_leg": (100, 200, 200),
    "head": (255, 200, 200),
}


# ============================================================================
# 4DHumans 模型加载
# ============================================================================

def load_4dhumans_model(device):
    """加载 4DHumans 预训练模型"""
    try:
        import sys
        import inspect
        from pathlib import Path

        # Python 3.11+ 移除了 inspect.getargspec，chumpy 依赖它
        if not hasattr(inspect, 'getargspec'):
            inspect.getargspec = inspect.getfullargspec

        # 添加 4D-Humans 到 Python 路径
        humans_path = Path("4D-Humans")
        if humans_path.exists():
            sys.path.insert(0, str(humans_path))

        from hmr2.models import load_hmr2

        print("[Model] 加载 4DHumans HMR2...")
        model, cfg = load_hmr2()
        model = model.to(device)
        model.eval()
        print(f"[Model] 4DHumans HMR2 加载成功 (backbone: {cfg.MODEL.BACKBONE.TYPE})")
        return model, cfg

    except (FileNotFoundError, ImportError, ValueError) as e:
        # [m19] 配置/依赖类异常**直接上抛**，不静默降级产假数据：这些异常来自
        # skeleton_spec（缺 SMPL pkl 的 FileNotFoundError、缺依赖的 ImportError、pkl
        # 形状/kintree 校验失败的 ValueError）或 4DHumans 配置/权重/依赖缺失，代表
        # **部署/环境损坏**而非“模型可选缺失”。回退简化模型会产出看似正常实则错误
        # 的姿态，故无论 STRICT 与否都响亮失败（与 C2 惰性加载的 FileNotFoundError 一致）。
        print(
            f"\n[Model][FATAL] 4DHumans 加载遇配置/依赖类异常 ({type(e).__name__}: {e})，"
            f"判定为部署/环境损坏，拒绝静默回退假数据。请检查 SMPL pkl / 依赖 / 配置完整性。",
            file=sys.stderr, flush=True,
        )
        raise
    except Exception as e:
        # [Fix #13] 捕获其余运行时异常（含 torch>=2.6 的 pickle.UnpicklingError、
        # RuntimeError、OSError 等）。旧版仅捕获 (ImportError, FileNotFoundError)
        # 导致 4DHumans ckpt 反序列化失败时异常逃逸，进程直接崩溃。
        print(
            "\n" + "!" * 72 + "\n"
            f"[Model][WARNING] 4DHumans 加载失败 ({type(e).__name__}: {e})\n"
            "[Model][WARNING] 将回退到内置简化模型，输出姿态精度显著下降，非真实 4DHumans 结果！\n"
            "[Model][WARNING] 生产环境请设置 KINETO_STRICT=1 使该回退直接报错退出。\n"
            + "!" * 72,
            file=sys.stderr, flush=True,
        )
        if os.environ.get("KINETO_STRICT") == "1":
            raise StrictModeRefused(
                f"KINETO_STRICT=1: 4DHumans 模型加载失败 ({type(e).__name__}: {e})，拒绝回退到简化模型"
            ) from e
        print(f"[Model] 4DHumans 未就绪 ({e})，使用内置简化模型")
        return None, None


def load_fallback_model(device):
    """内置简化姿态估计 (基于 OpenCV DNN + 启发式 3D 提升)"""
    print("[Model] 使用 OpenCV 内置人体关键点检测 + 3D 提升")
    return "fallback"


# ============================================================================
# 姿态解算核心
# ============================================================================

# 旋转欺骗（横卧人体）：画面旋转 ↔ 相机坐标变换
# 画面逆时针旋转 90°，等价于相机坐标做 Rz(-90°) 变换（p_rot = R @ p_orig）；
# 其转置即逆变换，用于把解算结果还原到原始帧坐标系。
ROT90_CCW_CAM = np.array([[0, 1, 0],
                          [-1, 0, 0],
                          [0, 0, 1]], dtype=np.float32)
ROT90_CW_CAM = ROT90_CCW_CAM.T.copy()


class PoseExtractor:
    def __init__(self, device):
        self.device = device
        self.model, self.cfg = load_4dhumans_model(device)
        self.detector = None

        if self.model is None:
            self.mode = "fallback"
        else:
            self.mode = "4dhumans"
            self._init_detector()

        self.focal_length = 5000.0
        self.image_size = 256
        # 旋转欺骗缓存：已验证的横卧人体画面旋转方向（cv2.ROTATE_90_*）
        self._rot_direction = None
        # 合成关键点告警去重标记（避免逐帧刷屏）
        self._synthetic_warned = False

    def _init_detector(self):
        try:
            from ultralytics import YOLO
            self.detector = YOLO("yolov8n.pt")
            print("[Detector] YOLOv8n 人体检测器已加载")
        except Exception as e:
            # [P3/改动 F.4] detector 静默降级守卫：YOLO 加载失败会使 detect_person
            # 退化为全图默认框（降低裁剪/相机平移精度）。此处只**观测/判定**降级、
            # 不改动推理路径本身；STRICT=1 下拒绝静默降级（抛 StrictModeRefused）。
            if os.environ.get("KINETO_STRICT") == "1":
                raise StrictModeRefused(
                    f"KINETO_STRICT=1: YOLOv8n 人体检测器加载失败 ({e})，"
                    "拒绝静默降级到全图默认框"
                ) from e
            print(f"[Detector] YOLOv8n 加载失败 ({e})，将使用全图默认框")
            self.detector = None

    def detect_person(self, frame):
        if self.detector is None:
            h, w = frame.shape[:2]
            return np.array([0, 0, w, h], dtype=np.float32)

        results = self.detector(frame, classes=[0], verbose=False, conf=0.25)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            h, w = frame.shape[:2]
            return np.array([0, 0, w, h], dtype=np.float32)

        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        return box

    def preprocess_frame(self, frame, box_center=None, box_size=None):
        """预处理单帧图像为模型输入
        HMR2 需要人体裁剪图作为输入：以 box_center 为中心、边长 box_size 的正方形裁剪，
        越界部分用黑色填充（与官方 ViTDetDataset 一致），再缩放到 256x256"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if box_center is not None and box_size is not None:
            half = box_size / 2.0
            size = self.image_size
            src = np.float32([
                [box_center[0], box_center[1]],
                [box_center[0], box_center[1] + half],
                [box_center[0] + half, box_center[1]],
            ])
            dst = np.float32([
                [size / 2, size / 2],
                [size / 2, size],
                [size, size / 2],
            ])
            M = cv2.getAffineTransform(src, dst)
            img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).float().to(self.device)

    def extract_pose(self, frame):
        """从单帧提取 3D 关节坐标和 SMPL 参数"""
        if self.mode == "4dhumans":
            person_bbox = self.detect_person(frame)
            return self._extract_4dhumans(frame, person_bbox)
        else:
            return self._extract_fallback(frame)

    def _extract_4dhumans(self, frame, person_bbox):
        """使用 4DHumans 模型解算，包含正确的相机参数
        流程与官方 demo 一致：人体裁剪图送入模型，再用
        cam_crop_to_full 换算全图相机平移（焦距按全图分辨率缩放）"""
        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = person_bbox
        bw, bh = x2 - x1, y2 - y1

        # ── 旋转欺骗触发判定 ──────────────────────────────────────────
        # 条件 A：检测框宽度 ≥ 高度 × 1.0（非明显竖向即视为横向，覆盖跪姿/仰卧等）
        # 条件 B：bbox 需有一定面积（过小可能是误检噪声，跳过）
        # 条件 C：排除"竖向全帧"——只有 bbox 既大（≥95%）又是竖向（宽高比≤1）
        #         才认为是全身竖构图而跳过旋转；横向 bbox 即使占满画面也需旋转
        bbox_area = bw * bh
        frame_area = img_w * img_h
        is_horizontal = (bh > 1e-3) and (bw / bh >= 1.0)
        is_valid_size = bbox_area > 0.005 * frame_area  # 至少占画面 0.5%
        is_vertical_full_frame = (bbox_area >= 0.95 * frame_area) and (bw / bh <= 1.0)

        if is_horizontal and is_valid_size and not is_vertical_full_frame:
            return self._extract_4dhumans_rotated(frame, person_bbox)

        result = self._solve_hmr2(frame, person_bbox)
        return {
            "joints_3d": result["joints_3d"],
            "smpl_thetas": result["global_orient"] + result["body_pose"],
            "cam_t": result["cam_t"],
            "person_bbox": person_bbox,
            "confidence": 0.5,
            "betas": result["betas"],
        }

    def _solve_hmr2(self, frame, person_bbox):
        """在给定（可能已旋转的）帧上运行 HMR2，返回该帧相机坐标系下的结果"""
        img_h, img_w = frame.shape[:2]

        x1, y1, x2, y2 = person_bbox
        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

        # 将检测框扩展到模型的 192:256 长宽比（官方 BBOX_SHAPE）
        bw, bh = x2 - x1, y2 - y1
        w_t, h_t = 192, 256
        if bh / max(bw, 1e-6) < h_t / w_t:
            bh_e, bw_e = max(bw * h_t / w_t, bh), bw
        else:
            bh_e, bw_e = bh, max(bh * w_t / h_t, bw)
        box_size = max(bw_e, bh_e)

        img_tensor = self.preprocess_frame(frame, box_center, box_size)
        batch = {"img": img_tensor}

        with torch.no_grad():
            output = self.model(batch)

        pred_cam = output["pred_cam"][0].cpu().numpy()
        pred_smpl_params = output["pred_smpl_params"]

        # ── 关节序归一（根因修复，P1/改动 A）────────────────────────────
        # 旧实现取 output["pred_keypoints_3d"][:24]，但该键在 vendored
        # smpl_wrapper 里经 joint_map=smpl_to_openpose 重排为 OpenPose Body-25 序，
        # 与引擎全部解剖常量/审计/前端的 SMPL canonical 序解释错配。
        # 现改用 HMR2 同一次 forward 已算出、未被重排的 pred_vertices，经模型
        # 自带 J_regressor (24×6890, SMPLLayer 注册的 buffer) 做一次
        # vertices2joints 矩阵乘（~1 MFLOP/帧，不重新推理、不碰 vendored 代码），
        # 得到 canonical SMPL 24 关节。坐标空间（[m12] 更正旧“root-relative”误述）：
        #   · SMPL **模型绝对空间**（apply_trans=False）：pelvis 不在原点，含 rest
        #     偏移 [-0.0018,-0.2233,0.0282] 并随 global_orient 旋转；**未加 cam_t**。
        #   · pred_keypoints_3d[:,:24] 来自 batch_rigid_transform(FK 关节)，本处 joints
        #     来自 vertices2joints(J_regressor @ vertices)——两者实测差 mean 7.8mm /
        #     max 26mm（非同一数组）。逆置换回 FK 关节不可行且会碰 vendored；regressor
        #     路径是唯一不碰 vendored 的 canonical 关节源，与 mesh/SMPL_REST_JOINTS 自洽，
        #     与 smpl_thetas(FK) 近似(~1cm)。mesh 渲染须统一取一侧，勿混用两源。
        #   下游 joint+cam_t 投影约定不变。
        pred_vertices = output["pred_vertices"]                       # [1, 6890, 3]
        j_regressor = self.model.smpl.J_regressor                     # [24, 6890]
        joints_canonical = vertices2joints(j_regressor, pred_vertices)  # [1, 24, 3]
        joints_3d = joints_canonical[0].float().cpu().numpy()           # [24, 3] canonical

        s, tx, ty = pred_cam
        # 焦距需从 256 模型空间缩放到全图分辨率（官方: FOCAL_LENGTH / IMAGE_SIZE * max(img_size)）
        focal = self.focal_length / self.image_size * max(img_w, img_h)

        bs = box_size * s + 1e-9
        tz = 2 * focal / bs
        cam_t_x = 2 * (box_center[0] - img_w / 2.0) / bs + tx
        cam_t_y = 2 * (box_center[1] - img_h / 2.0) / bs + ty
        cam_t = np.array([cam_t_x, cam_t_y, tz], dtype=np.float32)

        # 模型输出为旋转矩阵 (N,3,3)，转为轴角向量保持 smpl_thetas 语义
        global_orient_mat = pred_smpl_params["global_orient"][0].cpu().numpy().reshape(-1, 3, 3)
        body_pose_mat = pred_smpl_params["body_pose"][0].cpu().numpy().reshape(-1, 3, 3)
        global_orient = Rotation.from_matrix(global_orient_mat.astype(np.float64)).as_rotvec().astype(np.float32).flatten().tolist()
        body_pose = Rotation.from_matrix(body_pose_mat.astype(np.float64)).as_rotvec().astype(np.float32).flatten().tolist()

        # betas（体型参数，hmr2.py:127 产出）不再丢弃：additive 透传给下游
        # （P2 将用其做个体化 rest 模板/尺度），并随 keyframe 写入 pose_data.json。
        betas = pred_smpl_params["betas"][0].cpu().numpy().astype(np.float32).flatten().tolist()

        return {
            "joints_3d": joints_3d,
            "cam_t": cam_t,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "betas": betas,
        }

    def compute_mesh_for_keyframes(self, keyframes: list) -> dict:
        """对关键帧执行 SMPL forward，获取 mesh 顶点和 faces"""
        if not self.model or not hasattr(self.model, 'smpl'):
            return {"has_mesh": False, "mesh_vertices": [], "faces": []}

        try:
            # [P1 mesh 节奏贴合] 预分配 (帧数, 6890, 3) float32 缓冲：全帧 mesh
            # 下顶点直接写 ndarray，避免 Python 嵌套 list 的数 GB 峰值内存。
            all_vertices = np.empty(
                (len(keyframes), 6890, 3), dtype=np.float32)
            faces = self.model.smpl.faces  # numpy array, shared across frames

            for fi, kf in enumerate(keyframes):
                betas = kf.get("betas", [0] * 10)
                smpl_thetas = kf.get("smpl_thetas", [0] * 72)

                # 拆分 thetas
                global_orient_aa = np.array(smpl_thetas[:3], dtype=np.float32)
                body_pose_aa = np.array(smpl_thetas[3:72], dtype=np.float32)

                # 轴角 → 旋转矩阵（smplx 0.1.28 pose2rot=False 要求 global_orient
                # 为 (B,1,3,3) 与 body_pose 的 (B,23,3,3) 同维拼接，传 (B,3,3) 会
                # 在 torch.cat 处 RuntimeError → mesh 整体静默降级）
                global_orient_mat = Rotation.from_rotvec(global_orient_aa).as_matrix().reshape(1, 1, 3, 3)
                body_pose_mat = Rotation.from_rotvec(body_pose_aa.reshape(-1, 3)).as_matrix().reshape(1, 23, 3, 3)

                # 转 torch
                betas_t = torch.tensor([betas], dtype=torch.float32, device=self.device)
                go_t = torch.tensor(global_orient_mat, dtype=torch.float32, device=self.device)
                bp_t = torch.tensor(body_pose_mat, dtype=torch.float32, device=self.device)

                # SMPL forward
                with torch.no_grad():
                    smpl_out = self.model.smpl(betas=betas_t, body_pose=bp_t, global_orient=go_t, pose2rot=False)

                vertices = smpl_out.vertices[0].cpu().numpy()  # (6890, 3)

                # [P0 mesh↔joints 对齐] SMPL forward 输出在模型空间（pelvis 位于
                # R(global_orient)@rest 偏移处），而交付的 joints_3d 位于原始帧
                # 坐标系（refine/重算 thetas 后两者相差一个**每帧恒定平移**，实测
                # |t|≈0.31-0.32m，各关节 std=0、去 t 后残差=0）。以 pelvis 关节
                # （canonical 序 0）为锚点把 mesh 平移到 joints_3d 坐标系，使叠加
                # 模式下骨架与 mesh 位置精确重合。J_regressor 与 _solve_hmr2 的
                # canonical 关节同源，回归数学一致，故对齐是精确而非近似。
                joints_ref = kf.get("joints_3d")
                if joints_ref is not None and len(joints_ref) == 24:
                    j_reg = self.model.smpl.J_regressor.detach().cpu().numpy().reshape(24, 6890)
                    fk_joints = j_reg @ vertices                      # (24, 3) 模型空间
                    t_align = np.asarray(joints_ref[0], dtype=np.float32) - fk_joints[0]
                    vertices = vertices + t_align[np.newaxis, :]

                all_vertices[fi] = vertices

            return {
                "has_mesh": True,
                "mesh_vertices": all_vertices,  # (帧数, 6890, 3) float32，全帧
                "faces": faces.tolist(),  # 13776 × 3
                "mesh_vertices_per_frame": 6890,
            }
        except Exception as exc:
            print(f"[Mesh] ⚠️  SMPL mesh 计算失败（不影响主流程）: {exc}", file=sys.stderr)
            return {"has_mesh": False, "mesh_vertices": [], "faces": []}

    def _extract_4dhumans_rotated(self, frame, person_bbox):
        """旋转欺骗：对横卧人体分别尝试 90°CCW 和 90°CW 旋转后推理，
        用多维直立质量分选出最优方向，再将 3D 结果逆旋转回原始坐标系。

        设计要点
        ─────────
        • 不再 break-early：两个方向**都**推理，对比质量分后选胜者。
          （旧逻辑用 head-ankle 比较并提前退出，死虫动作脚踝在空中时必然选错）
        • 使用 _pose_upright_score：基于颈-骨盆-脊柱多关节投票，
          对抬腿、屈髋等姿势保持稳定。
        • 记忆上一帧选定的旋转方向（_rot_direction），连续视频通常
          方向一致，可以减少切换日志噪声。
        • 若两个方向的得分均低（< 0.25），说明此帧推理严重失败，
          仍返回得分较高的那个，并打印警告供调试。
        """
        img_h, img_w = frame.shape[:2]

        DIRECTIONS = [
            (cv2.ROTATE_90_COUNTERCLOCKWISE, "逆时针 90°", ROT90_CCW_CAM),
            (cv2.ROTATE_90_CLOCKWISE,        "顺时针 90°", ROT90_CW_CAM),
        ]

        # 若已知方向，优先尝试该方向（降低不必要的切换日志）
        if self._rot_direction == cv2.ROTATE_90_CLOCKWISE:
            DIRECTIONS = list(reversed(DIRECTIONS))

        candidates = []
        for flag, dir_name, R_flag in DIRECTIONS:
            frame_rot  = cv2.rotate(frame, flag)
            bbox_rot   = self._rotate_bbox(person_bbox, img_w, img_h, flag)
            raw        = self._solve_hmr2(frame_rot, bbox_rot)
            score      = self._pose_upright_score(raw["joints_3d"])
            candidates.append((score, flag, dir_name, R_flag, raw))
            print(f"  [RotHack] {dir_name}  直立得分={score:.2f}")

        # 按得分降序，选最优
        candidates.sort(key=lambda x: -x[0])
        best_score, chosen_flag, chosen_name, chosen_R_flag, chosen = candidates[0]

        if best_score < 0.25:
            print(f"  [RotHack] ⚠️  两方向得分均低（最高={best_score:.2f}），"
                  "推理结果可能不可靠，建议检查 YOLO 检测框是否正确")
        elif best_score < 0.45:
            print(f"  [RotHack] ⚠️  方向={chosen_name}，得分偏低={best_score:.2f}，"
                  "人体可能处于高难度姿态（极度侧卧/折叠）")

        if self._rot_direction != chosen_flag:
            print(f"  [RotHack] ✅ 选定旋转方向：{chosen_name}（得分={best_score:.2f}）")
        self._rot_direction = chosen_flag

        # ── 3D 结果逆旋转回原始帧坐标系 ──────────────────────────────
        # 画面旋转 R_flag：p_rotated = R_flag @ p_original
        # 逆变换（正交矩阵）：p_original = R_flag.T @ p_rotated
        R_inv = chosen_R_flag.T

        joints_3d = (R_inv @ chosen["joints_3d"].T).T.astype(np.float32)
        cam_t     = (R_inv @ chosen["cam_t"]).astype(np.float32)

        # global_orient 需要复合逆旋转，body_pose（相对关节角）不受影响
        R_fix      = Rotation.from_matrix(R_inv.astype(np.float64))
        R_global   = R_fix * Rotation.from_rotvec(
                        np.asarray(chosen["global_orient"], dtype=np.float64))
        global_orient = R_global.as_rotvec().astype(np.float32).tolist()

        return {
            "joints_3d":   joints_3d,
            "smpl_thetas": global_orient + chosen["body_pose"],
            "cam_t":       cam_t,
            "person_bbox": person_bbox,
            "confidence":  float(best_score),   # 把直立质量分透传为置信度
            "betas":       chosen["betas"],     # 体型参数不受画面旋转影响，直接透传
        }

    def _rotate_bbox(self, bbox, img_w, img_h, flag):
        """将原帧检测框映射到旋转 90° 后的画面坐标"""
        x1, y1, x2, y2 = bbox
        if flag == cv2.ROTATE_90_COUNTERCLOCKWISE:
            # (u, v) → (v, W - u)
            return np.array([y1, img_w - x2, y2, img_w - x1], dtype=np.float32)
        # (u, v) → (H - v, u)
        return np.array([img_h - y2, x1, img_h - y1, x2], dtype=np.float32)

    def _pose_upright_score(self, joints_3d) -> float:
        """返回 [0, 1] 直立质量分，越高说明姿态越符合'人直立'的相机坐标约定。

        **不使用脚踝**：死虫(dead bug)/腿上举等动作脚踝抬在空中，
        头-脚踝比较会给出错误结论。改用以下四项多数投票：

          ① 颈(12) y < 骨盆(0) y   权重 0.45  最关键，代表上半身朝上
          ② 头(15) y < 颈(12) y    权重 0.25  头在颈上方
          ③ 脊柱向量(颈-骨盆) y分量占主导  权重 0.20  躯干接近竖直
          ④ 双肩(16,17) 高度接近   权重 0.10  避免把极端侧卧误判为直立

        阈值说明：返回 ≥ 0.45 即可认为"基本直立"。
        """
        j = joints_3d
        score = 0.0

        # ① 颈部在骨盆上方（相机 y 轴向下，所以 neck_y < pelvis_y 为"高"）
        neck_y, pelvis_y = float(j[12][1]), float(j[0][1])
        if neck_y < pelvis_y:
            score += 0.45

        # ② 头在颈上方
        head_y = float(j[15][1])
        if head_y < neck_y:
            score += 0.25

        # ③ 脊柱方向：颈-骨盆向量的 |y| 分量 > |x| 分量 × 0.5（放宽，允许前倾/弯腰）
        spine = j[12] - j[0]
        if abs(spine[1]) > abs(spine[0]) * 0.5:
            score += 0.20

        # ④ 双肩 y 坐标差值 < 躯干长度的 40%（避免侧卧被误判为直立）
        torso_len = abs(pelvis_y - neck_y) + 1e-6
        shoulder_diff = abs(float(j[16][1]) - float(j[17][1]))
        if shoulder_diff < torso_len * 0.4:
            score += 0.10

        return score

    def _extract_fallback(self, frame):
        """
        简化姿态估计：
        1. 使用 OpenCV DNN 检测 2D 人体关键点
        2. 通过深度启发式提升为 3D 坐标
        3. 估算 cam_t 使投影对齐人物 bbox 中心
        """
        # [Fix #15-5] CLI 直跑兜底：KINETO_STRICT=1 时，整条回退管线（深度启发式
        # nz=-y/norm*0.5 + COCO17→SMPL24 插值 + smpl_thetas 全零的合成/假姿态）必须
        # 响亮失败，不得静默写出假数据。与 load_4dhumans_model / _detect_2d_keypoints
        # 已有的 STRICT 语义一致，作为整个 fallback 路径的显式入口守卫。
        # 未设或 0 时，本方法以下完全保留现有回退行为（向后兼容）。
        if os.environ.get("KINETO_STRICT") == "1":
            raise StrictModeRefused(
                "KINETO_STRICT=1: 拒绝使用简化回退管线（2D 启发式深度提升 + "
                "COCO17→SMPL24 插值 + smpl_thetas 全零的合成/假姿态数据）。"
                "请确保 4DHumans HMR2 与 OpenPose 权重就绪。"
            )
        h, w = frame.shape[:2]

        # 先用 YOLO 检测人物 bbox（若可用），用于估算 cam_t
        person_bbox = self.detect_person(frame)
        x1, y1, x2, y2 = person_bbox
        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        box_size = max(x2 - x1, y2 - y1)

        joints_2d = self._detect_2d_keypoints(frame)
        joints_3d = self._lift_2d_to_3d(joints_2d, w, h)
        smpl_thetas = self._estimate_smpl_thetas(joints_3d)
        confidence = self._compute_confidence(joints_2d)

        # 用与 4DHumans 路径相同的 cam_crop_to_full 约定估算 cam_t：
        # 令 s=1（单位缩放），tz = 2*focal/box_size
        focal = self.focal_length / self.image_size * max(w, h)
        s = 1.0
        bs = box_size * s + 1e-9
        tz = 2.0 * focal / bs
        cam_t_x = 2.0 * (box_center[0] - w / 2.0) / bs
        cam_t_y = 2.0 * (box_center[1] - h / 2.0) / bs
        cam_t = np.array([cam_t_x, cam_t_y, tz], dtype=np.float32)

        return {
            "joints_3d": joints_3d,
            "smpl_thetas": smpl_thetas,
            "cam_t": cam_t,
            "person_bbox": person_bbox,
            "confidence": confidence,
        }

    def _detect_2d_keypoints(self, frame):
        """OpenCV 内置人体关键点检测 (基于 COCO 17 点骨架)"""
        proto_file = "checkpoints/pose_deploy_lite.prototxt"
        weights_file = "checkpoints/pose_iter_440000.caffemodel"

        if not Path(weights_file).exists():
            if os.environ.get("KINETO_STRICT") == "1":
                raise StrictModeRefused(
                    f"KINETO_STRICT=1: OpenPose 权重缺失 ({weights_file})，"
                    "拒绝生成合成关键点"
                )
            if not self._synthetic_warned:
                self._synthetic_warned = True
                print(
                    "\n" + "!" * 72 + "\n"
                    f"[Pose][WARNING] OpenPose 权重缺失: {weights_file}\n"
                    "[Pose][WARNING] 将生成基于画面中心的合成关键点（假数据，仅流程验证，非真实推理）！\n"
                    "[Pose][WARNING] 生产环境请设置 KINETO_STRICT=1 使该回退直接报错退出。\n"
                    + "!" * 72,
                    file=sys.stderr, flush=True,
                )
            return self._generate_synthetic_keypoints(frame)

        net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368),
                                      (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward()

        keypoints = []
        body_parts = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
            "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
            "REar", "LEar", "Background",
        ]

        for i in range(18):
            heat_map = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heat_map)
            x = (w * point[0]) / output.shape[3]
            y = (h * point[1]) / output.shape[2]
            if conf > 0.1:
                keypoints.append((x, y, conf))
            else:
                keypoints.append((0, 0, 0))

        return keypoints

    def _generate_synthetic_keypoints(self, frame):
        """当模型权重不可用时，生成基于画面中心的合成关键点（用于流程验证）

        注意：告警已由 _detect_2d_keypoints 在调用本方法前统一发出（self._synthetic_warned），
        此处不再重复打印。
        """
        if os.environ.get("KINETO_STRICT") == "1":
            # [m2] 抛 StrictModeRefused（而非裸 RuntimeError）：与 load_4dhumans_model /
            # _init_detector 的 STRICT 拒绝统一异常类型，使 main() 的 except
            # StrictModeRefused 能正确捕获并以 EXIT_STRICT_REFUSED(3) 退出（裸
            # RuntimeError 会被当成真实崩溃 → 退出码 1，语义混淆）。
            raise StrictModeRefused(
                "KINETO_STRICT=1: 拒绝返回合成关键点（假数据），"
                "请检查 checkpoints/ 下 OpenPose 权重是否就绪"
            )
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        scale = min(w, h) * 0.3

        synthetic_2d = [
            (cx, cy - scale * 0.9, 0.8),           # 0: Nose
            (cx, cy - scale * 0.5, 0.9),            # 1: Neck
            (cx - scale * 0.3, cy - scale * 0.5, 0.8),  # 2: RShoulder
            (cx - scale * 0.5, cy - scale * 0.1, 0.7),  # 3: RElbow
            (cx - scale * 0.6, cy + scale * 0.2, 0.6),  # 4: RWrist
            (cx + scale * 0.3, cy - scale * 0.5, 0.8),  # 5: LShoulder
            (cx + scale * 0.5, cy - scale * 0.1, 0.7),  # 6: LElbow
            (cx + scale * 0.6, cy + scale * 0.2, 0.6),  # 7: LWrist
            (cx - scale * 0.15, cy + scale * 0.1, 0.8), # 8: RHip
            (cx - scale * 0.15, cy + scale * 0.5, 0.7), # 9: RKnee
            (cx - scale * 0.15, cy + scale * 0.9, 0.6), # 10: RAnkle
            (cx + scale * 0.15, cy + scale * 0.1, 0.8), # 11: LHip
            (cx + scale * 0.15, cy + scale * 0.5, 0.7), # 12: LKnee
            (cx + scale * 0.15, cy + scale * 0.9, 0.6), # 13: LAnkle
            (cx - scale * 0.05, cy - scale * 0.95, 0.7), # 14: REye
            (cx + scale * 0.05, cy - scale * 0.95, 0.7), # 15: LEye
            (cx - scale * 0.1, cy - scale * 0.9, 0.5),   # 16: REar
            (cx + scale * 0.1, cy - scale * 0.9, 0.5),   # 17: LEar
        ]
        return synthetic_2d

    def _lift_2d_to_3d(self, joints_2d, img_w, img_h):
        """2D → 3D 启发式提升：用 y 坐标估算深度，归一化到 SMPL 24 关节空间"""
        cx = img_w / 2
        cy = img_h / 2
        norm_factor = max(img_w, img_h)

        joints_17 = []
        for x, y, conf in joints_2d:
            nx = (x - cx) / norm_factor
            ny = (y - cy) / norm_factor
            nz = -y / norm_factor * 0.5
            joints_17.append([nx, ny, nz])

        joints_24 = self._map_coco17_to_smpl24(joints_17)
        return np.array(joints_24, dtype=np.float32)

    def _map_coco17_to_smpl24(self, joints_17):
        """COCO 17 点 → SMPL 24 关节映射 (插值补充缺失关节)"""
        j = joints_17
        smpl = [[0, 0, 0]] * 24

        smpl[0] = j[11] if len(j) > 11 else [0, 0, 0]  # pelvis ≈ mid-hip
        smpl[1] = j[11] if len(j) > 11 else [0, 0, 0]  # left_hip
        smpl[2] = j[8] if len(j) > 8 else [0, 0, 0]   # right_hip
        smpl[3] = self._mid(j[1], j[11])                 # spine1
        smpl[4] = j[12] if len(j) > 12 else [0, 0, 0]  # left_knee
        smpl[5] = j[9] if len(j) > 9 else [0, 0, 0]    # right_knee
        smpl[6] = self._mid(j[1], j[5])                  # spine2
        smpl[7] = j[13] if len(j) > 13 else [0, 0, 0]  # left_ankle
        smpl[8] = j[10] if len(j) > 10 else [0, 0, 0]  # right_ankle
        smpl[9] = j[1]                                    # spine3
        smpl[10] = j[13] if len(j) > 13 else [0, 0, 0] # left_foot
        smpl[11] = j[10] if len(j) > 10 else [0, 0, 0] # right_foot
        smpl[12] = j[1]                                    # neck
        smpl[13] = self._mid(j[1], j[5])                  # left_collar
        smpl[14] = self._mid(j[1], j[2])                  # right_collar
        smpl[15] = j[0]                                    # head
        smpl[16] = j[5] if len(j) > 5 else [0, 0, 0]   # left_shoulder
        smpl[17] = j[2] if len(j) > 2 else [0, 0, 0]   # right_shoulder
        smpl[18] = j[6] if len(j) > 6 else [0, 0, 0]   # left_elbow
        smpl[19] = j[3] if len(j) > 3 else [0, 0, 0]   # right_elbow
        smpl[20] = j[7] if len(j) > 7 else [0, 0, 0]   # left_wrist
        smpl[21] = j[4] if len(j) > 4 else [0, 0, 0]   # right_wrist
        smpl[22] = j[7] if len(j) > 7 else [0, 0, 0]   # left_hand
        smpl[23] = j[4] if len(j) > 4 else [0, 0, 0]   # right_hand

        return smpl

    @staticmethod
    def _mid(a, b):
        return [(a[i] + b[i]) / 2 for i in range(len(a))]

    def _estimate_smpl_thetas(self, joints_3d):
        """从 3D 关节位置估算 SMPL 旋转参数 (简化版)"""
        thetas = []
        for i in range(24):
            thetas.extend([0.0, 0.0, 0.0])
        return thetas

    def _compute_confidence(self, joints_2d):
        """基于可见关键点数量计算置信度"""
        visible = sum(1 for _, _, c in joints_2d if c > 0.1)
        return min(visible / len(joints_2d), 1.0)


# ============================================================================
# 3D 骨架渲染器
# ============================================================================

class SkeletonRenderer:
    """将 3D 关节投影到 2D 画面并绘制骨架"""

    def __init__(self, img_w, img_h, focal_length=5000.0, image_size=256):
        self.img_w = img_w
        self.img_h = img_h
        # cam_t 由 cam_crop_to_full 在全图分辨率下换算，焦距需同步缩放
        self.focal_length = focal_length / image_size * max(img_w, img_h)

    def project_3d_to_2d(self, joints_3d, cam_t):
        """透视投影: 3D → 2D 画面坐标
        joints_3d 以骨盆为原点，需加上 cam_t 平移至相机坐标系"""
        points_2d = []
        cx, cy = self.img_w / 2, self.img_h / 2
        focal = self.focal_length

        for joint in joints_3d:
            J = joint + cam_t
            if abs(J[2]) < 1e-6:
                points_2d.append((0, 0))
                continue
            u = focal * J[0] / J[2] + cx
            v = focal * J[1] / J[2] + cy
            points_2d.append((int(u), int(v)))

        return points_2d

    def draw_skeleton(self, frame, joints_3d, cam_t, confidence=1.0):
        """在原帧上绘制 3D 骨架"""
        overlay = frame.copy()
        points_2d = self.project_3d_to_2d(joints_3d, cam_t)

        alpha = min(confidence, 1.0)

        for i, j in SMPL_SKELETON:
            if i >= len(points_2d) or j >= len(points_2d):
                continue
            part = BONE_PART_MAP.get((i, j), "torso")
            color = BONE_COLORS.get(part, (200, 200, 200))
            cv2.line(overlay, points_2d[i], points_2d[j], color, 2, cv2.LINE_AA)

        for idx, pt in enumerate(points_2d):
            color = (0, 255, 255) if idx < 15 else (255, 255, 0)
            cv2.circle(overlay, pt, 3, color, -1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# ============================================================================
# 后处理修正模块
# ============================================================================
# BONE_LENGTH_BOUNDS（全部 23 条 kintree 骨，界由真实 rest 骨长数据驱动派生）
# 已在文件头部从 skeleton_spec (SSOT) 导入；旧版在此硬编码的子集（缺
# ankle→foot/wrist→hand/spine3→collar，且 collar 相关界因命名错位与真实
# 骨长矛盾）已淘汰。

# ============================================================================
# [P2/改动 C] joints_3d ↔ smpl_thetas 逐帧一致性（整改 Felix #1）
# ----------------------------------------------------------------------------
# 背景：Phase2 只修正 joints_3d（时序平滑 / 骨长约束），smpl_thetas 若不同步
# 则写盘时两者不再描述同一姿态。Felix #1 旧实现用**全片布尔** refine_modified
# + **逐帧** Kabsch，且该布尔几乎恒真 → 每帧都把 HMR2 精确 thetas 覆盖成几何近似。
#
# P1 归一后的新认知：canonical joints_3d 与 HMR2 原始 thetas（pred_smpl_params
# 的 global_orient+body_pose，72 维）对**未被 refine 改动的帧本就自洽**（同源一次
# SMPL forward）。故 P2 策略：
#   ① 默认**沿用 HMR2 原始精确 thetas**（不再全片覆盖）。
#   ② **仅对 refine 实质改动了 joints 的帧**（逐帧改动检测：单关节最大位移 >
#      REFINE_CHANGE_EPS），从改动后的 canonical joints 重算 thetas，使交付
#      joints↔thetas 逐帧一致（正是用户原始诉求）。微小平滑抖动不触发有损重算。
#   ③ 旋转求解按子骨数 K 分派（[C1]）：K==1（17/24 关节单子骨）用**最小旋转
#      (Rodrigues)**——Kabsch 在秩 1 的 H 上 twist 不可观测、SVD 零空间基由 LAPACK
#      任意选定 → 逐帧跳变噪声；K>=2 用**批量 Kabsch**（一次 np.linalg.svd 处理
#      (N,3,3) 堆叠）。叶子关节(K=0)保留 HMR2 原值（见 _THETA_OBSERVABLE_JOINTS）。
#
# 质量权衡（明确认知）：joints→thetas 的旋转反算是**几何近似**（每关节单一旋转
# 对齐子骨朝向，忽略 LBS/pose-shape blendshape 与运动链耦合），故只对实质改动帧
# 使用；干净帧保留 HMR2 原值（真值姿态参数）。rest 用 SSOT 真实模板
# SMPL_REST_JOINTS——反算只用骨向量（rest 差分），平移无关，故与其 root-中心化
# 版 SMPL_REST_JOINTS_CENTERED 等价；betas 已随帧透传（P2 之后可用于个体化 rest）。
# ============================================================================

# 逐帧改动检测阈值（米）：refined vs original 关节的**单关节最大位移**超过此值才
# 判定为“实质改动”、触发有损的 joints→thetas 重算；低于此值的微小平滑抖动保留
# HMR2 原始精确 thetas。取值远小于典型骨长(0.1–0.5m)、大于干净帧的平滑数值噪声
# （骨长约束对已在界内的骨做 target=clip(L,lo,hi)=L 的恒等重建，残差仅 ~1e-7，
# 不会误触发）。
REFINE_CHANGE_EPS = 5e-3

# [C1] 旋转“真被观测决定”的关节 = 至少有一条子骨的关节（K>=1）。仅这些关节的
# thetas 会从改动后的 canonical joints 重算；叶子关节（K=0：10,11,15,22,23）的
# 局部旋转几何上不可观测（无子骨约束），**保留 HMR2 原值**，绝不覆盖/归零。
# canonical 序下可观测关节 = [0,1,2,3,4,5,6,7,8,9,12,13,14,16,17,18,19,20,21]（19 个）。
_THETA_OBSERVABLE_JOINTS = np.array(
    [i for i in range(24) if len(SMPL_CHILDREN_FULL.get(i, [])) >= 1], dtype=np.int64)


def _kabsch_rotation(src, dst):
    """闭式求最优旋转 R 使 R @ src_k ≈ dst_k（Kabsch/Procrustes，3×3 SVD）。
    src/dst 形状 (K,3)。含反射修正，保证返回的是纯旋转 (det=+1)。
    单样本版；批量路径见 _kabsch_rotation_batch。"""
    H = np.asarray(src, dtype=np.float64).T @ np.asarray(dst, dtype=np.float64)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def _kabsch_rotation_batch(src, dst):
    """批量 Kabsch：对 B 个样本各求最优旋转 R_b 使 R_b @ src_k ≈ dst_k。
    src 形状 (K,3)（各样本共享的 rest 骨朝向）或 (B,K,3)；dst 形状 (B,K,3)。
    返回 (B,3,3)。np.linalg.svd 对堆叠 (B,3,3) 一次性分解，弃逐帧 Python SVD 循环。
    与 _kabsch_rotation 数学等价（同样含 det=+1 反射修正）。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if dst.ndim == 2:
        dst = dst[None]                                # [m11] 单样本容忍 (K,3)→(1,K,3)
    if src.ndim == 2:
        src = np.broadcast_to(src, dst.shape)          # (B,K,3)
    # [m11] 输入维度断言：批量 Kabsch 要求 src/dst 同为 (B,K,3) 且末维=3。
    if src.ndim != 3 or dst.ndim != 3 or src.shape != dst.shape or src.shape[2] != 3:
        raise ValueError(
            f"_kabsch_rotation_batch: src/dst 须为 (B,K,3) 且形状一致，"
            f"实际 src={src.shape}, dst={dst.shape}")
    # H_b = src_bᵀ @ dst_b → (B,3,3)
    H = np.einsum('bki,bkj->bij', src, dst)
    U, _S, Vt = np.linalg.svd(H)                        # 各 (B,3,3)
    Ut = U.transpose(0, 2, 1)                           # Uᵀ
    V = Vt.transpose(0, 2, 1)                           # V
    d = np.sign(np.linalg.det(V @ Ut))                  # (B,)
    d[d == 0] = 1.0
    D = np.zeros((dst.shape[0], 3, 3), dtype=np.float64)
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    return V @ D @ Ut                                   # (B,3,3) = Vtᵀ @ D @ Uᵀ


def _min_rotation_batch(src, dst):
    """[C1] 最小旋转（Rodrigues）：求把单位向量 src 旋到各 dst_b 的**最小角度**旋转。

    用于 K==1（单子骨）关节：此时 Kabsch 的 H = srcᵀ·dst 秩 1，正交 Procrustes 解是
    绕 src 轴的 1 参数族（**twist 不可观测**），np.linalg.svd 的零空间基由 LAPACK 任意
    选定 → 对输入不连续、逐帧跳变（噪声）。最小旋转取该解集中 twist=0 的成员：确定、
    连续（除反平行奇异）、物理最小；src==dst 时精确等于单位阵（rest 骨架 thetas≈0）。

    src 形状 (3,)（rest 子骨朝向，各帧共享）；dst 形状 (B,3)（观测子骨朝向）。
    返回 (B,3,3)。反平行（src≈-dst，180°）为奇异点：绕任一与 src 正交的轴转 180° 皆
    可，此处确定性地选一个单位正交轴 a，返回 R = 2·a·aᵀ - I（绕 a 转 180°，正确公式）。
    """
    src = np.asarray(src, dtype=np.float64).reshape(3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    B = dst.shape[0]
    # v = src × dst（旋转轴方向，未归一）；c = cosθ = src·dst；s = sinθ = |v|
    v = np.cross(np.broadcast_to(src, (B, 3)), dst)     # (B,3)
    c = dst @ src                                        # (B,)
    s = np.linalg.norm(v, axis=1)                        # (B,)
    # 叉积矩阵 vx（B,3,3）；Rodrigues: R = I + vx + vx² · (1-c)/s²
    vx = np.zeros((B, 3, 3), dtype=np.float64)
    vx[:, 0, 1], vx[:, 0, 2] = -v[:, 2], v[:, 1]
    vx[:, 1, 0], vx[:, 1, 2] = v[:, 2], -v[:, 0]
    vx[:, 2, 0], vx[:, 2, 1] = -v[:, 1], v[:, 0]
    ok = s > 1e-12                                       # sinθ>0：非平行、非反平行
    k = np.where(ok, (1.0 - c) / np.where(ok, s * s, 1.0), 0.0)
    R = np.broadcast_to(np.eye(3), (B, 3, 3)) + vx + np.einsum('bij,b->bij', vx @ vx, k)
    # 平行同向（ok 且 v=0）：vx=0,k=0 → R=I（精确单位阵）✓
    # 反平行（~ok 且 c<0，180°）：确定性选与 src 正交的单位轴 a，R = 2aaᵀ - I
    anti = (~ok) & (c < 0)
    if anti.any():
        a = np.array([1.0, 0.0, 0.0]) if abs(src[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
        a = a - src * float(src @ a)                     # Gram-Schmidt 去 src 分量 → a⊥src
        a = a / np.linalg.norm(a)                        # 单位化
        R = R.copy()
        R[anti] = 2.0 * np.outer(a, a) - np.eye(3)       # 绕 a 转 180°（R@src = -src = dst ✓）
    return R


def thetas_from_joints_batch(joints_arr):
    """批量：从 (N,24,3) canonical 关节反算 (N,72) smpl_thetas。

    沿运动学树自根向叶一次线性扫描（canonical 序下父索引恒小于子索引）。对关节 idx：
    用 rest 子骨朝向 src(K,3) 与观测子骨朝向 dst(N,K,3) 求全局旋转 R_global[idx]；
    局部旋转 = R_global_parentᵀ @ R_global[idx]，转轴角即 thetas[:,idx]。

    [C1] 旋转求解按子骨数 K 分派：
      · K==1（17/24 关节为单子骨）：Kabsch 的 H=srcᵀ·dst 秩 1，正交 Procrustes 解是
        绕 src 轴的 1 参数族——**twist 不可观测**，np.linalg.svd 的零空间基由 LAPACK
        任意选定，对输入不连续 → 逐帧跳变噪声。改用**最小旋转（Rodrigues）**
        _min_rotation_batch：取解集中 twist=0 的成员，确定、连续、物理最小，src==dst
        时精确为单位阵（rest 骨架 thetas≈0，max|theta|~1e-15）。
      · K>=2（pelvis/spine3 等多子骨）：子骨朝向共同约束旋转、twist 可观测，用批量
        Kabsch（一次 np.linalg.svd 处理 (N,3,3) 堆叠）。
    叶子关节（K=0：10,11,15,22,23）无子骨、局部旋转几何不可观测，本函数内置 0；
    **调用方只覆盖可观测关节**（见 _THETA_OBSERVABLE_JOINTS），叶子关节保留 HMR2 原值。

    全程无迭代优化、无 SMPL 模型依赖，4dhumans 与 fallback 模式均可用且确定性可复现。
    返回 (N,72) float32。
    """
    J = np.asarray(joints_arr, dtype=np.float64)
    if J.ndim == 2:
        J = J[None]
    # [m10] 形状断言 + 空输入(N=0)安全返回
    if J.ndim != 3 or J.shape[1:] != (24, 3):
        raise ValueError(
            f"thetas_from_joints_batch: 输入须为 (N,24,3)，实际 {J.shape}")
    N = J.shape[0]
    if N == 0:
        return np.zeros((0, 72), dtype=np.float32)
    eye = np.broadcast_to(np.eye(3), (N, 3, 3))
    R_global = [None] * 24
    thetas = np.zeros((N, 24, 3), dtype=np.float64)

    for idx in range(24):
        parent = SMPL_PARENTS[idx]
        R_parent = R_global[parent] if parent >= 0 else eye
        children = SMPL_CHILDREN_FULL.get(idx, [])

        src_list, dst_list = [], []
        for c in children:
            r = SMPL_REST_JOINTS[c] - SMPL_REST_JOINTS[idx]
            nr = float(np.linalg.norm(r))
            if nr < 1e-9:
                continue  # 退化 rest 骨，不参与对齐
            u = J[:, c, :] - J[:, idx, :]                          # (N,3)
            nu = np.linalg.norm(u, axis=1)                         # (N,)
            src_list.append(r / nr)                                # (3,)
            dst_list.append(u / np.where(nu < 1e-9, 1.0, nu)[:, None])  # (N,3)

        if not src_list:
            # 无可观测子骨：局部旋转取单位，全局旋转沿用父节点
            R_global[idx] = R_parent
            thetas[:, idx, :] = 0.0
            continue

        src = np.asarray(src_list)                                 # (K,3)
        dst = np.nan_to_num(np.stack(dst_list, axis=1),            # (N,K,3)
                            nan=0.0, posinf=0.0, neginf=0.0)
        # [C1] K==1 用最小旋转（twist 不可观测，Kabsch 秩 1 病态）；K>=2 用批量 Kabsch
        if src.shape[0] == 1:
            R_g = _min_rotation_batch(src[0], dst[:, 0, :])        # (N,3,3)
        else:
            R_g = _kabsch_rotation_batch(src, dst)                 # (N,3,3)
        R_global[idx] = R_g
        # 局部旋转 = R_parentᵀ @ R_g（einsum: (R_parentᵀ R_g)[i,j]=Σ_k R_parent[k,i]R_g[k,j]）
        R_local = np.einsum('bki,bkj->bij', R_parent, R_g)
        thetas[:, idx, :] = Rotation.from_matrix(R_local).as_rotvec()

    return thetas.reshape(N, 72).astype(np.float32)               # (N,72)


def thetas_from_joints(joints):
    """单帧封装：从 (24,3) canonical 关节反算 72 维 smpl_thetas（内部走批量实现）。
    保留供调试/单帧调用；生产路径用 thetas_from_joints_batch。"""
    return thetas_from_joints_batch(np.asarray(joints, dtype=np.float64)[None])[0].tolist()


class PoseRefiner:
    """对 joints_3d 数组做后处理修正，不重新推理"""

    def __init__(self, joints):
        self.joints = joints.copy()
        # [P2/改动 D] 骨长约束 clip 台账：记录**被约束触及**的骨（哪根、夹了多少帧、
        # 越界幅度），供 compute_quality_score 诚实惩罚，杜绝“夹进边界即满分”的自我
        # 实现。键为骨对 (i,j)，值 {clip_frac, rel_violation, max_violation}。
        self.clip_info = {}

    def apply_temporal_smooth(self, window=5):
        kernel = np.ones(window) / window
        for j in range(self.joints.shape[1]):
            for d in range(3):
                self.joints[:, j, d] = np.convolve(
                    self.joints[:, j, d], kernel, mode='same')

    def apply_bone_length_constraint(self):
        # [M1] 在**历史台账基础上累积**（不再每次清空）：refiner 跨 iteration 继承
        # clip_info，本方法合并本轮新夹取、每根骨保留历史最严重值，使质量分看到累计
        # 被夹的骨——杜绝“每轮清零→跨轮虚高”。__init__ 时 self.clip_info 可为继承值。
        clip_info = dict(self.clip_info)
        for (i, j), (lo, hi) in BONE_LENGTH_BOUNDS.items():
            lengths = np.linalg.norm(
                self.joints[:, i, :] - self.joints[:, j, :], axis=1)
            lengths = np.clip(lengths, 1e-6, None)
            direction = (self.joints[:, j, :] - self.joints[:, i, :]) / lengths[:, None]
            target = np.clip(lengths, lo, hi)
            # clip 台账：violation=每帧被夹的绝对位移(米)；仅被夹帧(>0)计入。
            # rel_violation 以骨长界跨度 (hi-lo) 归一，度量“越界有多严重”。
            span = max(hi - lo, 1e-6)
            violation = np.abs(lengths - target)
            clipped = violation > 1e-6
            if clipped.any():
                # [M1] rel_violation 只对**被夹帧**求均值：旧写法对全部帧求均值
                # （含未夹帧的 0）再与 clip_frac 相乘 → O(clip_frac²) 双重稀释，
                # 轻微夹取被平方级低估。改为被夹帧均值，度量“被夹时夹得多狠”。
                new_info = {
                    "clip_frac": float(clipped.mean()),
                    "rel_violation": float((violation[clipped] / span).mean()),
                    "max_violation": float(violation.max()),
                }
                old = clip_info.get((i, j))
                if old is None:
                    clip_info[(i, j)] = new_info
                else:
                    # 合并：每根骨保留历史最严重值（clip_frac/rel_violation/max 取大）
                    clip_info[(i, j)] = {
                        "clip_frac": max(old.get("clip_frac", 0.0), new_info["clip_frac"]),
                        "rel_violation": max(old.get("rel_violation", 0.0), new_info["rel_violation"]),
                        "max_violation": max(old.get("max_violation", 0.0), new_info["max_violation"]),
                    }
            self.joints[:, j, :] = self.joints[:, i, :] + direction * target[:, None]
        self.clip_info = clip_info


def compute_quality_score(audit_results, clip_info=None):
    """质量分 ∈ [0,1]，对**交付的 canonical joints**（真实输出）诚实评分。

    [P2/改动 D] 整改 Felix #3：**移除 baseline 冻结**。Felix 版把骨长派生项冻结在
    refine 前基线（baseline_audit）上，既破坏 refine 收敛（分数对修正无响应），又用
    改动前的 joints 去裁决改动后的交付物。P2 改为直接评交付 joints。

    同时**杜绝 clip 反向抬分**：apply_bone_length_constraint 用 np.clip 把越界骨长
    夹进硬边界，夹完后 bone_validity 平凡为 0（自我实现满分）。故引入 clip_info
    （骨长约束台账，来自 PoseRefiner.clip_info）：被夹过的骨按 clip_frac×rel_violation
    折算为“等效违规骨数”，与审计违规骨数一并扣分——约束无法把分数反向抬高，
    bone_score 成为真实姿态质量的诚实信号。
    """
    bone_val = audit_results.get("bone_validity", [])
    total_bones = len(BONE_LENGTH_BOUNDS)

    # clip 惩罚：被骨长约束触及的骨，按“夹了多少帧 × 越界幅度(归一到界跨度)”
    # 累加为等效违规骨数。clip_frac∈[0,1]、rel_violation clamp 到 [0,1]，故单根
    # 骨惩罚≤ 1，与 len(bone_val) 量纲一致。
    clip_penalty = 0.0
    if clip_info:
        for info in clip_info.values():
            clip_penalty += info.get("clip_frac", 0.0) * min(info.get("rel_violation", 0.0), 1.0)

    violations = len(bone_val) + clip_penalty
    bone_score = 1.0 - violations / total_bones if total_bones > 0 else 0.0
    bone_score = max(bone_score, 0.0)

    bone_cons = audit_results.get("bone_consistency", {})
    cvs = [info["cv"] for info in bone_cons.values() if isinstance(info, dict)]
    avg_cv = np.mean(cvs) if cvs else 1.0
    smooth_score = 1.0 - min(avg_cv, 1.0)

    symm = audit_results.get("symmetry", {})
    asymms = [info["asymmetry"] for info in symm.values() if isinstance(info, dict)]
    avg_asymm = np.mean(asymms) if asymms else 1.0
    symm_score = 1.0 - min(avg_asymm, 1.0)

    proj = audit_results.get("projection_alignment", {})
    proj_score = proj.get("mean_in_person_ratio", 0)

    score = 0.25 * bone_score + 0.20 * smooth_score + 0.20 * symm_score + 0.35 * proj_score
    return float(min(max(score, 0.0), 1.0))


def compute_frame_confidence(joints_arr, frame_idx, bone_ranges=None):
    n_frames = len(joints_arr)
    scores = []

    bone_ok = 0
    bone_total = 0
    for (i, j), (lo, hi) in (bone_ranges or {}).items():
        if i >= joints_arr.shape[1] or j >= joints_arr.shape[1]:
            continue
        length = np.linalg.norm(joints_arr[frame_idx, i] - joints_arr[frame_idx, j])
        if lo <= length <= hi:
            bone_ok += 1
        bone_total += 1
    if bone_total > 0:
        scores.append(bone_ok / bone_total)

    if 0 < frame_idx < n_frames - 1:
        disp = np.linalg.norm(
            joints_arr[frame_idx] - joints_arr[frame_idx - 1], axis=1).mean()
        scores.append(max(0, 1.0 - disp * 20))

    l_len = np.linalg.norm(joints_arr[frame_idx, 1] - joints_arr[frame_idx, 4])
    r_len = np.linalg.norm(joints_arr[frame_idx, 2] - joints_arr[frame_idx, 5])
    if l_len > 1e-6 and r_len > 1e-6:
        ratio = min(l_len, r_len) / max(l_len, r_len)
        scores.append(ratio)

    return float(np.mean(scores)) if scores else 0.5


# ============================================================================
# 姿态分类（基于 SMPL 24 关节）
# ============================================================================

def classify_pose_type(joints_3d: np.ndarray) -> dict:
    """
    基于 SMPL 24 关节分类姿态类型。

    返回:
    {
        "pose_type": "standing" | "supine" | "kneeling" | "sitting" | "unknown",
        "spine_direction": [x, y, z],  # 归一化 spine 向量
        "confidence": float  # 分类置信度 0-1
    }
    """
    # SMPL 关节索引（参考 skeleton_spec.py）
    pelvis = joints_3d[0]
    neck = joints_3d[12]
    left_knee = joints_3d[4]
    right_knee = joints_3d[5]

    # 1. 计算 spine 向量
    spine_vec = neck - pelvis
    spine_norm = np.linalg.norm(spine_vec)
    if spine_norm < 1e-6:
        return {"pose_type": "unknown", "spine_direction": [0, 0, 0], "confidence": 0.0}

    spine_dir = spine_vec / spine_norm

    # 2. 计算特征
    spine_y_dominance = abs(spine_dir[1]) / max(abs(spine_dir[0]), abs(spine_dir[1]), abs(spine_dir[2]), 1e-6)

    # Y 轴平坦度（身体是否水平铺开）
    y_range = joints_3d[:, 1].max() - joints_3d[:, 1].min()
    x_range = joints_3d[:, 0].max() - joints_3d[:, 0].min()
    z_range = joints_3d[:, 2].max() - joints_3d[:, 2].min()
    max_range = max(x_range, y_range, z_range)
    y_flatness = y_range / max_range if max_range > 1e-6 else 1.0

    # 膝盖相对骨盆高度
    knee_height = (left_knee[1] + right_knee[1]) / 2 - pelvis[1]

    # 3. 分类逻辑
    if spine_y_dominance > 0.7 and knee_height < -0.15:
        pose_type = "standing"
        confidence = min(1.0, spine_y_dominance)
    elif spine_y_dominance < 0.2 and y_flatness < 0.5:
        pose_type = "supine"  # 仰卧/俯卧
        confidence = 1.0 - spine_y_dominance
    elif spine_y_dominance > 0.5 and knee_height > -0.10:
        # 区分 kneeling 和 sitting
        if knee_height > 0.0:
            pose_type = "kneeling"
        else:
            pose_type = "sitting"
        confidence = spine_y_dominance
    else:
        pose_type = "unknown"
        confidence = 0.5

    return {
        "pose_type": pose_type,
        "spine_direction": spine_dir.tolist(),
        "confidence": float(confidence)
    }


# ============================================================================
# 四宫格教学图生成
# ============================================================================

# 关键帧阶段标签（与前端 KeyframeCards 语义对齐）
_GRID_LABELS = ["起始位 Setup", "离心 Descent", "转折点 Amortization", "向心 Ascent"]


def _generate_grid_images(video_path: str, keyframes: list, output_dir: Path,
                          focal_length: float = 5000.0, image_size: int = 256) -> dict:
    """生成四宫格教学图：关键帧截图 + 骨骼标注。

    对最多 4 个关键帧：
    1. 读取完整视频帧
    2. 根据人体 2D 投影 bbox 裁剪（带 20% padding）
    3. 叠加骨骼标注
    4. 所有图片 pad 到统一画布尺寸，确保四张大小一致

    失败不影响主流程（调用方须 try/except）。

    返回 {"grid_images": [filename, ...], "grid_labels": [label, ...]}
    """
    labels = _GRID_LABELS
    grid_images: list[str] = []
    grid_labels: list[str] = []

    if not keyframes:
        return {"grid_images": grid_images, "grid_labels": grid_labels}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[Grid] ⚠️  无法打开视频，跳过四宫格生成", file=sys.stderr)
        return {"grid_images": grid_images, "grid_labels": grid_labels}

    try:
        # 探测视频尺寸
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            print("[Grid] ⚠️  无法探测视频尺寸，跳过四宫格生成", file=sys.stderr)
            return {"grid_images": grid_images, "grid_labels": grid_labels}

        frame_aspect = w / h  # 原始帧长宽比

        # 用于 bbox 投影的临时渲染器（全帧分辨率）
        bbox_renderer = SkeletonRenderer(w, h, focal_length=focal_length,
                                         image_size=image_size)

        # ---- 第一遍：收集裁剪后的帧 ----
        PAD_RATIO = 0.35          # bbox 四周留白比例（加大以避免裁剪头脚）
        TARGET_H = 512            # 统一输出高度
        cropped_frames = []       # (annotated_bgr, label)

        for idx, kf in enumerate(keyframes[:4]):
            frame_idx = kf.get("frame_index", 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"[Grid] ⚠️  帧 {frame_idx} 读取失败，跳过", file=sys.stderr)
                continue

            joints_3d = np.array(kf["joints_3d"], dtype=np.float32)
            cam_t = np.array(kf.get("cam_t", [0, 0, 0]), dtype=np.float32)
            conf = float(kf.get("confidence_score", 1.0))

            # 投影 3D → 2D 以计算人体 bbox
            points_2d = bbox_renderer.project_3d_to_2d(joints_3d, cam_t)
            xs = [p[0] for p in points_2d]
            ys = [p[1] for p in points_2d]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            bw = max(x_max - x_min, 1)
            bh = max(y_max - y_min, 1)
            pad_x = bw * PAD_RATIO
            pad_y = bh * PAD_RATIO

            crop_x1 = max(0, int(x_min - pad_x))
            crop_y1 = max(0, int(y_min - pad_y))
            crop_x2 = min(w, int(x_max + pad_x))
            crop_y2 = min(h, int(y_max + pad_y))

            # 保持原始长宽比：扩展裁剪区域以匹配帧宽高比
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            target_crop_h = int(crop_w / frame_aspect)
            if target_crop_h > crop_h:
                # 需要增加高度（上下均匀扩展）
                extra = target_crop_h - crop_h
                crop_y1 = max(0, crop_y1 - extra // 2)
                crop_y2 = crop_y1 + target_crop_h
                if crop_y2 > h:
                    crop_y2 = h
                    crop_y1 = max(0, crop_y2 - target_crop_h)
            else:
                # 需要增加宽度（左右均匀扩展）
                target_crop_w = int(crop_h * frame_aspect)
                extra = target_crop_w - crop_w
                crop_x1 = max(0, crop_x1 - extra // 2)
                crop_x2 = crop_x1 + target_crop_w
                if crop_x2 > w:
                    crop_x2 = w
                    crop_x1 = max(0, crop_x2 - target_crop_w)

            # 裁剪原帧
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            if cropped.size == 0:
                print(f"[Grid] ⚠️  帧 {frame_idx} 裁剪为空，跳过", file=sys.stderr)
                continue

            # 在裁剪帧上绘制骨骼（直接用已投影的 2D 点偏移到裁剪坐标系）
            cw, ch = cropped.shape[1], cropped.shape[0]
            shifted = [(px - crop_x1, py - crop_y1) for px, py in points_2d]
            overlay = cropped.copy()
            alpha = min(conf, 1.0)
            for i, j in SMPL_SKELETON:
                if i >= len(shifted) or j >= len(shifted):
                    continue
                part = BONE_PART_MAP.get((i, j), "torso")
                color = BONE_COLORS.get(part, (200, 200, 200))
                cv2.line(overlay, shifted[i], shifted[j], color, 2, cv2.LINE_AA)
            for pt_idx, pt in enumerate(shifted):
                color = (0, 255, 255) if pt_idx < 15 else (255, 255, 0)
                cv2.circle(overlay, pt, 3, color, -1, cv2.LINE_AA)
            annotated = cv2.addWeighted(overlay, alpha, cropped, 1 - alpha, 0)

            # 添加阶段标签（白字黑底半透明条）
            label = labels[idx] if idx < len(labels) else f"关键帧 {idx + 1}"
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (cw, 56), (0, 0, 0), -1)
            annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)
            cv2.putText(annotated, label, (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                        cv2.LINE_AA)

            cropped_frames.append((annotated, label))

        if not cropped_frames:
            return {"grid_images": grid_images, "grid_labels": grid_labels}

        # ---- 第二遍：统一画布尺寸（保持原始长宽比）----
        # 按目标高度等比缩放每张图，宽度按比例；然后 pad 到最大宽度
        max_canvas_w = 0
        resized = []
        for img, label in cropped_frames:
            ih, iw = img.shape[:2]
            scale = TARGET_H / ih
            new_w = int(iw * scale)
            scaled = cv2.resize(img, (new_w, TARGET_H),
                                interpolation=cv2.INTER_AREA)
            max_canvas_w = max(max_canvas_w, new_w)
            resized.append((scaled, label))

        canvas_w = max_canvas_w

        for idx, (img, label) in enumerate(resized):
            ih, iw = img.shape[:2]
            if iw < canvas_w:
                # 居中放置，两侧填黑
                canvas = np.zeros((TARGET_H, canvas_w, 3), dtype=np.uint8)
                x_off = (canvas_w - iw) // 2
                canvas[:, x_off:x_off + iw] = img
            else:
                canvas = img

            # 保存 JPEG（质量 92%）
            filename = f"grid_{idx + 1:02d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # 检查文件大小（硬约束 200KB）
            file_size = filepath.stat().st_size
            if file_size > 200 * 1024:
                cv2.imwrite(str(filepath), canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])
                print(f"[Grid] {filename} 超 200KB ({file_size // 1024}KB)，"
                      f"降至 Q75 ({filepath.stat().st_size // 1024}KB)")

            grid_images.append(filename)
            grid_labels.append(label)
            print(f"[Grid] ✅ {filename}: {label} ({canvas_w}x{TARGET_H})")

    finally:
        cap.release()

    return {"grid_images": grid_images, "grid_labels": grid_labels}


# ============================================================================
# ffmpeg H.264 转码（确保浏览器兼容）
# ============================================================================

def _transcode_to_h264(output_file: Path) -> None:
    """使用 ffmpeg 将视频转码为 H.264，确保浏览器可播放。
    如果 ffmpeg 不可用或转码失败，保留原始文件不做处理。
    """
    import subprocess
    h264_output = output_file.with_name(output_file.stem + "_h264.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(output_file),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(h264_output),
            ],
            check=True,
            capture_output=True,
        )
        output_file.unlink()
        h264_output.rename(output_file)
        print(f"[ffmpeg] ✅ H.264 转码成功: {output_file.name}")
    except Exception as e:
        print(f"[ffmpeg] ⚠️  转码失败（使用原始编码）: {e}", file=sys.stderr)
        if h264_output.exists():
            h264_output.unlink()


# ============================================================================
# 全帧骨骼标注视频生成
# ============================================================================

def _generate_annotated_video(video_path: str, keyframes: list, output_path: Path,
                              focal_length: float = 5000.0, image_size: int = 256) -> str:
    """生成全帧骨骼标注视频。

    对视频全部帧逐帧处理，叠加 2D 骨骼投影。
    输出为 annotated_output.mp4（H.264 编码）。

    返回输出文件名（相对路径）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Annotated] ⚠️  无法打开视频，跳过标注视频生成", file=sys.stderr)
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        print("[Annotated] ⚠️  无法探测视频尺寸，跳过标注视频生成", file=sys.stderr)
        cap.release()
        return ""

    # 创建输出视频（H.264 编码，浏览器兼容）
    output_file = output_path / "annotated_output.mp4"
    writer = None
    for codec_tag, codec_label in [("avc1", "H.264"), ("mjpg", "MJPEG")]:
        _f = cv2.VideoWriter_fourcc(*codec_tag)
        _w = cv2.VideoWriter(str(output_file), _f, fps, (width, height))
        if _w.isOpened():
            writer = _w
            print(f"[Annotated] 使用 {codec_label} ({codec_tag}) 编码输出标注视频")
            break
        _w.release()
    if writer is None:
        writer = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print("[Annotated] ⚠️  H.264/MJPEG 均不可用，回退 mp4v")

    # 构建 frame_idx → keyframe 的快速索引
    kf_by_frame = {kf.get("frame_index"): kf for kf in keyframes}

    renderer = SkeletonRenderer(width, height, focal_length=focal_length, image_size=image_size)

    processed = 0
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        kf = kf_by_frame.get(frame_idx)
        if kf and "joints_3d" in kf:
            joints_3d = np.array(kf["joints_3d"], dtype=np.float32)
            cam_t = np.array(kf.get("cam_t", [0, 0, 0]), dtype=np.float32)
            conf = float(kf.get("confidence_score", 1.0))
            frame = renderer.draw_skeleton(frame, joints_3d, cam_t, conf)

        writer.write(frame)
        processed += 1

        if processed % 100 == 0:
            print(f"[Annotated] 进度: {processed}/{total_frames} 帧")

    cap.release()
    writer.release()

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"[Annotated] ✅ 生成完成: {output_file.name} ({processed} 帧, {file_size_mb:.1f}MB)")

    # [Fix #67] ffmpeg 转码为 H.264，确保浏览器可播放
    _transcode_to_h264(output_file)

    return "annotated_output.mp4"


# ============================================================================
# 主流程: 视频逐帧解算
# ============================================================================

def process_video(input_path, output_dir, device, max_iterations=3,
                  quality_threshold=0.6, no_refine=False):
    """闭环姿态解算：推理 → 审计 → 修正 → 再审计，直至收敛"""
    # [m6] ExitStack 托管全部 I/O 资源（cap/pbar/frame_cache/cap2/writer）：无论正常
    # 返回、StrictModeRefused 还是任何异常路径，退出 with 时统一释放，杜绝句柄泄漏。
    # 各资源仍保留阶段末的显式 release（提前释放、行为不变），callback 仅为异常兜底
    # （release/close 均幂等，重复调用无害）。实际逻辑在 _process_video_impl。
    with contextlib.ExitStack() as stack:
        return _process_video_impl(
            stack, input_path, output_dir, device,
            max_iterations, quality_threshold, no_refine)


def _process_video_impl(stack, input_path, output_dir, device,
                        max_iterations, quality_threshold, no_refine):
    cap = cv2.VideoCapture(str(input_path))
    stack.callback(cap.release)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {input_path}")
    print(f"[Video] {width}x{height} @ {fps:.1f}fps, {total_frames} 帧")
    print(f"[Pipeline] max_iter={max_iterations}, quality_thresh={quality_threshold}, no_refine={no_refine}")

    # [P4/改动 G] 长视频架构级天花板声明 + 告警（不改架构，仅观测）。
    # 当前架构对超长视频存在线性膨胀上限：pose_data.json/mp4 随帧数线性增长
    # （实测 ~9.16KB/帧 → 54000 帧 ≈ 494MB JSON + ~1.9GB mp4），前端全量 fetch+parse
    # 大 JSON 可能卡顿/OOM；api 子进程超时已按帧数缩放（见 api._engine_timeout）。
    # 声明“支持帧数上限”（env KINETO_SUPPORTED_FRAME_CEILING，默认 9000），超限时
    # 响亮告警（stderr）但不阻断——是否继续由调用方/门禁决定。
    # 未来缓解项（不在本阶段）：分块/流式交付、帧率降采样（默认 off）。
    supported_ceiling = int(os.environ.get("KINETO_SUPPORTED_FRAME_CEILING", "9000") or 9000)
    if total_frames > supported_ceiling:
        est_json_mb = total_frames * 9.16 / 1000  # ~9.16KB/帧（实测线性膨胀）
        print(
            f"[Video][WARNING] 帧数 {total_frames} 超过架构级支持上限 {supported_ceiling}："
            f"pose_data.json 预计 ~{est_json_mb:.0f}MB（~9.16KB/帧线性膨胀），"
            f"前端全量 fetch+parse 可能卡顿；子进程超时已按帧数缩放(api._engine_timeout)。"
            f"建议分段处理。调整上限 KINETO_SUPPORTED_FRAME_CEILING=<帧数>；"
            f"分块/流式交付、帧率降采样列为未来缓解项。",
            file=sys.stderr, flush=True)

    extractor = PoseExtractor(device)

    # ---- 计算输入视频 MD5（用于前端追踪输入输出一致性）----
    def _compute_video_md5(path: str) -> str:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    video_md5 = _compute_video_md5(input_path)
    print(f"[Video] MD5: {video_md5}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- 第 1 步：逐帧推理，收集原始数据（不缓存帧图像）----
    print("\n[Phase 1] 逐帧推理...")
    joints_list = []
    smpl_thetas_list = []
    cam_t_list = []
    bbox_list = []
    betas_list = []   # additive：HMR2 体型参数（fallback 模式无，保持 None）

    pbar = tqdm(total=total_frames, desc="推理进度", unit="帧")
    stack.callback(pbar.close)
    frame_idx = 0
    # [P4/改动 G] clear_device_memory 节流步长（env KINETO_CLEAR_MEM_EVERY，默认 30；
    # =1 恢复逐帧原行为）。clear_device_memory 只调 torch.*.empty_cache() 释放缓存，
    # **不改任何计算结果**，故对 joints/thetas/score 逐位无影响；节流仅降低释放开销。
    clear_mem_every = max(1, int(os.environ.get("KINETO_CLEAR_MEM_EVERY", "30") or 30))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = extractor.extract_pose(frame)
        joints_list.append(result["joints_3d"])
        smpl_thetas_list.append(result["smpl_thetas"])
        cam_t_list.append(result.get("cam_t", np.zeros(3, dtype=np.float32)))
        bbox_list.append(result.get("person_bbox", np.array([0, 0, width, height], dtype=np.float32)))
        betas_list.append(result.get("betas"))
        frame_idx += 1
        pbar.update(1)
        # 节流：每 clear_mem_every 帧释放一次（原逐帧）。默认 30 稳妥控显存峰值不 OOM。
        if frame_idx % clear_mem_every == 0:
            clear_device_memory(device)
        if frame_idx % 50 == 0:
            gc.collect()
    pbar.close()
    cap.release()
    clear_device_memory(device)  # [P4/改动 G] 循环后补一次，确保 Phase1 结束显存回落

    joints_arr = np.array(joints_list, dtype=np.float32)
    cam_t_arr = np.array(cam_t_list, dtype=np.float32)
    bbox_arr = np.array(bbox_list, dtype=np.float32)
    print(f"[Phase 1] 推理完成: {len(joints_arr)} 帧")

    # ---- 第 2 步：闭环审计-修正 ----
    # [P4/改动 G] 跨 iteration/跨 check 共享的采样帧缓存：审计每轮 proj(~10 seek) +
    # visualize(~20 seek)，3 iteration 最多 ~90 次冗余 seek+decode。FrameCache 有界
    # FIFO 复用已解码帧，**不改被审计帧集合**（sample_indices 仍按原逻辑算），只消除
    # 冗余 IO；容量 env KINETO_AUDIT_CACHE_FRAMES（默认 48，<=0 禁用复用=每次重解码）。
    from pose_audit import FrameCache
    frame_cache = FrameCache(str(input_path))
    stack.callback(frame_cache.release)
    if no_refine:
        # [P3/改动 F.3] --no-refine 仍跑**真实审计**得诚实质量分（弃 final_score=0 必死
        # 路径：旧写法使 no-refine 恒判 0 分，与 api 门禁/前端展示矛盾）。不施加任何
        # 修正，clip_info=None（无骨长约束烘焙）；verdict 仍写入 audit_iter0/audit_results.json。
        print("\n[Phase 2] 跳过修正 (--no-refine)，仍跑真实审计以得诚实质量分")
        final_joints = joints_arr
        from pose_audit import audit_from_joints
        audit = audit_from_joints(
            final_joints, str(input_path), str(output_path / "audit_iter0"),
            fps=fps, total_frames=total_frames, visualize=True,
            cam_t_arr=cam_t_arr, bbox_arr=bbox_arr, frame_cache=frame_cache)
        final_score = compute_quality_score(audit, clip_info=None)
        print(f"[Phase 2] --no-refine 诚实质量分: {final_score:.4f} "
              f"(verdict={audit.get('verdict')})")
    else:
        print("\n[Phase 2] 闭环审计-修正...")
        final_joints = joints_arr
        prev_score = -1.0
        score = 0.0
        # [P2/改动 D] clip 台账跨 iteration 传递：iteration k 的 audit 作用于
        # iteration k-1 refine 产出的 final_joints，故评分时用 k-1 那轮的 clip_info
        # （描述已烘焙进当前 joints 的骨长夹取）。iteration 0 无前置 clip → None。
        clip_info = None

        for iteration in range(max_iterations):
            iter_dir = output_path / f"audit_iter{iteration}"
            from pose_audit import audit_from_joints
            audit = audit_from_joints(
                final_joints, str(input_path), str(iter_dir),
                fps=fps, total_frames=total_frames, visualize=True,
                cam_t_arr=cam_t_arr, bbox_arr=bbox_arr, frame_cache=frame_cache)

            score = compute_quality_score(audit, clip_info=clip_info)
            print(f"\n[Iter {iteration}] 质量分: {score:.4f}")

            if score >= quality_threshold:
                print(f"[Iter {iteration}] 质量达标 (≥{quality_threshold})，停止迭代")
                break
            if iteration > 0 and score <= prev_score + 0.01:
                print(f"[Iter {iteration}] 改进停滞 (Δ={score-prev_score:.4f})，停止迭代")
                break
            prev_score = score

            # [M1] 末轮跳过 refine：已达最大迭代次数则不再修正——保证“交付物 == 被
            # 审计物”（final_joints 即最后一次 audit 评的对象），杜绝“改了 joints 却
            # 用改前的 score/verdict 交付”的错配。
            if iteration == max_iterations - 1:
                print(f"[Iter {iteration}] 已达 max_iterations={max_iterations}，"
                      f"跳过 refine（保证交付物==被审计物）(#M1)")
                break

            refiner = PoseRefiner(final_joints)
            # [M1] 继承上一轮 clip 台账：骨长约束跨轮累积，refiner 在历史台账上合并
            # 本轮，使质量分看到“累计被夹过的骨”，而非每轮清零导致跨轮虚高。
            refiner.clip_info = dict(clip_info or {})
            smooth = audit.get("temporal_smoothness", {})
            bone_val = audit.get("bone_validity", [])

            applied = []
            # [Fix #15-2] 删除死守卫 `or True`：原写法使平滑被无条件应用。
            # 恢复真实意图——仅当审计检测到时序突变(spike_count>0)时才平滑。
            if smooth.get("spike_count", 0) > 0:
                refiner.apply_temporal_smooth(window=5)
                applied.append("temporal_smooth")
            if bone_val:
                refiner.apply_bone_length_constraint()
                applied.append("bone_length_constraint")

            final_joints = refiner.joints
            # [M1] 捕获本轮累积后的 clip 台账（apply_bone_length_constraint 已在历史
            # 台账上合并、每根骨保留最严重值），供下一 iteration 评分诚实惩罚。
            clip_info = refiner.clip_info
            print(f"[Iter {iteration}] 已应用修正: {', '.join(applied) or '无'}")

        final_score = score

    # [P4/改动 G] 释放共享采样帧缓存（Phase2 结束；Phase3 重新独立读取视频渲染）。
    frame_cache.release()

    # ---- 第 3 步：重算置信度 + 渲染输出（重新读取视频）----
    print("\n[Phase 3] 重算置信度 + 渲染输出...")
    bone_ranges = {k: v for k, v in BONE_LENGTH_BOUNDS.items()}

    # [P2/改动 C] joints_3d ↔ smpl_thetas 逐帧一致（整改 Felix #1）：
    #   ① 默认沿用 HMR2 原始精确 thetas（与 canonical joints 同源一次 SMPL forward，
    #      对未被 refine 改动的帧本就自洽）。
    #   ② 仅对 refine **实质改动**了 joints 的帧（逐帧改动检测：单关节最大位移 >
    #      REFINE_CHANGE_EPS），从改动后的 canonical joints 批量重算 thetas，使交付
    #      joints↔thetas 逐帧一致；微小平滑抖动不触发有损重算。
    # cam_t 是相机平移而非姿态，refinement 不改变全局平移语义，故保持 Phase1 原值。
    thetas_arr = np.asarray(smpl_thetas_list, dtype=np.float32).copy()   # (N,72) HMR2 原值
    n_frames_out = len(final_joints)
    n_changed = 0
    if (not no_refine) and thetas_arr.shape[0] == n_frames_out:
        delta = np.linalg.norm(final_joints - joints_arr, axis=2)         # (N,24) 每关节位移
        changed_mask = delta.max(axis=1) > REFINE_CHANGE_EPS              # (N,) 单关节最大位移超阈
        n_changed = int(changed_mask.sum())
        if n_changed > 0:
            # [C1] 只覆盖“旋转真被观测决定”的关节（K>=1，有子骨）；叶子关节
            # （K=0：10,11,15,22,23）局部旋转几何不可观测，**保留 HMR2 原值**，绝不
            # 覆盖/归零。recomputed 与 orig 同为 (n_changed,24,3)，按可观测关节下标
            # 选择性写回，其余维严格保持 HMR2 原始 thetas（未改帧整段不动）。
            recomputed = thetas_from_joints_batch(
                final_joints[changed_mask].astype(np.float64)).reshape(-1, 24, 3)
            orig = thetas_arr[changed_mask].reshape(-1, 24, 3).copy()
            orig[:, _THETA_OBSERVABLE_JOINTS, :] = recomputed[:, _THETA_OBSERVABLE_JOINTS, :]
            thetas_arr[changed_mask] = orig.reshape(-1, 72)
            print(f"[Phase 3] {n_changed}/{n_frames_out} 帧被 refine 实质改动，批量重算 "
                  f"smpl_thetas（仅 {len(_THETA_OBSERVABLE_JOINTS)} 个可观测关节；叶子关节"
                  f"及其余帧保留 HMR2 原值）(#C1/#P2-C)")
        else:
            print("[Phase 3] refine 未实质改动任何帧，全部沿用 HMR2 原始 thetas (#P2-C)")
    else:
        print("[Phase 3] 沿用 HMR2 原始 thetas（no_refine 或帧数不匹配）(#P2-C)")

    renderer = SkeletonRenderer(width, height, focal_length=extractor.focal_length,
                                image_size=extractor.image_size)
    # [Fix #61] 视频编码改为 web 兼容：mp4v (MPEG-4 Part 2) 不被现代浏览器支持，
    # 导致 <video> 显示黑屏。优先尝试 H.264 (avc1)，不可用时回退 MJPEG (mjpg)。
    video_out_path = output_path / "demo_output.mp4"
    fourcc, writer = None, None
    for codec_tag, codec_label in [("avc1", "H.264"), ("mjpg", "MJPEG")]:
        _f = cv2.VideoWriter_fourcc(*codec_tag)
        _w = cv2.VideoWriter(str(video_out_path), _f, fps, (width, height))
        if _w.isOpened():
            fourcc, writer = _f, _w
            print(f"[Video] 使用 {codec_label} ({codec_tag}) 编码输出视频")
            break
        _w.release()
    if writer is None:
        # 最后兜底：原始 mp4v（浏览器可能不支持，但至少产出文件）
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
        print("[Video] ⚠️  H.264/MJPEG 均不可用，回退 mp4v（浏览器可能无法播放）")
    stack.callback(writer.release)

    cap2 = cv2.VideoCapture(str(input_path))
    stack.callback(cap2.release)
    keyframes = []
    confidences = []
    for idx in tqdm(range(len(final_joints)), desc="渲染进度", unit="帧"):
        conf = compute_frame_confidence(final_joints, idx, bone_ranges)
        confidences.append(conf)

        # [P2/改动 C] 逐帧 thetas：干净帧为 HMR2 原值，实质改动帧为批量重算值
        thetas_out = thetas_arr[idx].tolist()

        keyframe = {
            "frame_index": idx,
            "timestamp_ms": round((idx / fps) * 1000, 2),
            "state_label": _classify_state(idx, total_frames),
            "joints_3d": final_joints[idx].tolist(),
            "smpl_thetas": thetas_out,
            "cam_t": cam_t_arr[idx].tolist(),
            "confidence_score": round(conf, 4),
        }
        # additive 新增字段：betas（仅 4dhumans 模式产出；schema 兼容，旧消费方
        # 不受影响，P2 用其做个体化 rest/尺度）
        if idx < len(betas_list) and betas_list[idx] is not None:
            keyframe["betas"] = betas_list[idx]
        keyframes.append(keyframe)

        ret, frame = cap2.read()
        if ret:
            rendered = renderer.draw_skeleton(frame, final_joints[idx], cam_t_arr[idx], conf)
            writer.write(rendered)

    cap2.release()
    writer.release()

    # [Fix #67] ffmpeg 转码为 H.264，确保浏览器可播放
    _transcode_to_h264(video_out_path)

    # ---- 第 4 步：生成全帧骨骼标注视频 ----
    try:
        annotated_video = _generate_annotated_video(
            str(input_path), keyframes, output_path,
            focal_length=extractor.focal_length, image_size=extractor.image_size)
        if annotated_video:
            print(f"[Annotated] 生成骨骼标注视频：{annotated_video}")
    except Exception as exc:
        print(f"[Annotated] ⚠️  标注视频生成失败（不影响主流程）: {exc}", file=sys.stderr)

    # ---- 第 5 步：生成四宫格教学图（关键帧截图 + 骨骼叠加）----
    grid_result = {"grid_images": [], "grid_labels": []}
    # 从全部帧中均匀采样 4 个代表性关键帧（而非取前 4 个连续帧）
    n_total = len(keyframes)
    if n_total >= 4:
        _sample_indices = np.linspace(0, n_total - 1, 4, dtype=int).tolist()
        grid_keyframes = [keyframes[i] for i in _sample_indices]
    else:
        grid_keyframes = keyframes

    try:
        grid_result = _generate_grid_images(
            str(input_path), grid_keyframes, output_path,
            focal_length=extractor.focal_length, image_size=extractor.image_size)
        print(f"[Grid] 生成 {len(grid_result['grid_images'])} 张教学图")
    except Exception as exc:
        print(f"[Grid] ⚠️  四宫格生成失败（不影响主流程）: {exc}", file=sys.stderr)

    # 姿态分类（使用第一帧关键帧）
    if keyframes and len(keyframes) > 0:
        first_frame_joints = np.array(keyframes[0]["joints_3d"], dtype=np.float32)
        pose_info = classify_pose_type(first_frame_joints)
    else:
        pose_info = {"pose_type": "unknown", "spine_direction": [0, 0, 0], "confidence": 0.0}

    # ---- SMPL Mesh 计算（全帧 forward；交付用 DRACO 压缩二进制）----
    # [P1 mesh 节奏贴合] 全帧 SMPL forward（J_regressor 线性性保证：顶点相邻帧
    # 线性插值与 joints_3d 线性插值严格同步）。
    # [P1.1 传输压缩] 交付编码：DRACO 14bit + stride3（mesh_track.drcs，466 帧
    # 视频约 3MB，适配 Funnel 中继 ~40-130KB/s 的现实带宽）；DracoPy 缺失时
    # 回退全帧 f32（~38MB，局域网环境可用）。两种格式 JSON 均不嵌入顶点。
    mesh_result = extractor.compute_mesh_for_keyframes(keyframes)
    mesh_file_name: str | None = None
    mesh_encoding: str | None = None
    mesh_frames = 0
    mesh_vertex_count = 0
    mesh_frame_stride = 1
    if mesh_result["has_mesh"]:
        verts_arr = np.asarray(mesh_result["mesh_vertices"], dtype=np.float32)
        full_mesh_frames, mesh_vertex_count = int(verts_arr.shape[0]), int(verts_arr.shape[1])
        faces_arr = np.asarray(mesh_result["faces"], dtype=np.uint32)
        try:
            buf, drc_frames = encode_mesh_track_drcs(verts_arr, faces_arr)
            mesh_file_name = "mesh_track.drcs"
            (output_path / mesh_file_name).write_bytes(buf)
            mesh_encoding = f"draco{MESH_DRACO_QUANT_BITS}"
            mesh_frame_stride = MESH_FRAME_STRIDE
            mesh_frames = drc_frames  # 抽帧后的 mesh 帧数
            print(f"[Mesh] ✓ DRACO 编码完成：{drc_frames} mesh 帧"
                  f"（stride={MESH_FRAME_STRIDE}，{MESH_DRACO_QUANT_BITS}bit）"
                  f"× {mesh_vertex_count} 顶点, {len(faces_arr)} 面 → {mesh_file_name}"
                  f"（{len(buf) / 1e6:.1f}MB / 全帧 f32 {verts_arr.nbytes / 1e6:.1f}MB）")
        except Exception as exc:
            print(f"[Mesh] ⚠️  DRACO 编码失败（回退全帧 f32）: {exc}", file=sys.stderr)
            mesh_file_name = "mesh_vertices.f32"
            verts_arr.tofile(output_path / mesh_file_name)
            mesh_encoding = "f32"
            mesh_frame_stride = 1
            mesh_frames = full_mesh_frames
            print(f"[Mesh] ✓ 全帧 f32：{mesh_frames} 帧 × {mesh_vertex_count} 顶点"
                  f" → {mesh_file_name}（{verts_arr.nbytes / 1e6:.1f}MB）")
    else:
        print("[Mesh] 无 mesh（fallback 模式或计算失败）")

    metadata = {
        "video_fps": round(fps, 2),
        "total_frames": total_frames,
        "resolution": f"{width}x{height}",
        "model_version": "4dhumans-v1.0" if extractor.mode == "4dhumans" else "fallback-v1.0",
        "device": str(device),
        "extraction_mode": extractor.mode,
        # [M3] additive 判别位（前端/deploy 按此消费，确切值不可改）：
        #   joint_order="smpl-canonical"：joints_3d/thetas 均为 SMPL canonical 24 序
        #     （P1 归一后；旧产物 joints_3d 曾为 OpenPose Body-25 序）。
        #   schema_version=2：pose_data schema 第 2 版（canonical 关节序 + 本判别位）。
        "joint_order": "smpl-canonical",
        "schema_version": 2,
        "video_md5": video_md5,
        # 姿态分类结果（additive，不影响现有推理路径）
        "pose_type": pose_info["pose_type"],
        "spine_direction": pose_info["spine_direction"],
        "pose_classification_confidence": pose_info["confidence"],
        # 四宫格教学图
        "grid_images": grid_result["grid_images"],
        "grid_labels": grid_result["grid_labels"],
        # SMPL mesh（[P1.1] 顶点在独立压缩二进制，JSON 不嵌入）：
        #   mesh_vertices_file：产物文件名（mesh_track.drcs | mesh_vertices.f32）
        #   mesh_encoding："draco14"（DRACO 压缩，mesh 帧按 stride 抽帧）| "f32"
        #   mesh_vertices_frames：mesh 帧数（draco 时为抽帧后数量）
        #   mesh_frame_stride：mesh 帧 k ↔ keyframes[k*stride]（f32 时为 1）
        "has_mesh": mesh_result["has_mesh"],
        "mesh_vertices_per_frame": mesh_vertex_count,
        "mesh_vertices_frames": mesh_frames,
        "mesh_vertices_file": mesh_file_name,
        "mesh_encoding": mesh_encoding,
        "mesh_frame_stride": mesh_frame_stride,
        "pipeline": {
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
            "final_quality_score": round(final_score, 4),
            "refine_applied": not no_refine,
        },
    }

    pose_data = {
        "metadata": metadata,
        "keyframes": keyframes,
        "mesh_faces": mesh_result["faces"] if mesh_result["has_mesh"] else [],
    }
    json_path = output_path / "pose_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, indent=2, ensure_ascii=False)

    avg_conf = np.mean(confidences)
    print(f"\n[Output] pose_data.json → {json_path}")
    print(f"[Output] demo_output.mp4 → {video_out_path}")
    print(f"[Stats] {len(keyframes)} 帧, 平均置信度: {avg_conf:.3f}, 质量分: {final_score:.4f}")

    return pose_data


def _classify_state(frame_idx, total_frames):
    """根据帧位置标记动作阶段"""
    ratio = frame_idx / max(total_frames - 1, 1)
    if ratio < 0.1:
        return "initial"
    elif ratio > 0.9:
        return "final"
    else:
        return "active"


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kineto Core - 3D 姿态解算引擎")
    parser.add_argument("--input", "-i", default="input_video.mp4", help="输入视频路径")
    parser.add_argument("--output", "-o", default="output", help="输出目录")
    parser.add_argument("--max-iter", type=int, default=3, help="最大迭代修正次数 (default: 3)")
    parser.add_argument("--quality-thresh", type=float, default=0.6, help="质量阈值 (default: 0.6)")
    parser.add_argument("--no-refine", action="store_true", help="跳过修正，单次输出（兼容旧行为）")
    args = parser.parse_args()

    print("=" * 60)
    print("  Kineto Core - 3D 姿态解算引擎")
    print("=" * 60)

    device = detect_device()
    start_time = time.time()

    try:
        process_video(
            args.input, args.output, device,
            max_iterations=args.max_iter,
            quality_threshold=args.quality_thresh,
            no_refine=args.no_refine,
        )
    except StrictModeRefused as exc:
        # [P3/改动 F] 仅捕获 STRICT 主动拒绝（假数据/缺权重/detector 降级），用独立
        # 退出码 EXIT_STRICT_REFUSED(3) 区别于 argparse 用法错误(2)/真实崩溃(1)。
        # 其余真实异常不再被过宽的 except RuntimeError 吞掉，照常冒泡（traceback+退出码 1）。
        # 因在 Phase1 报错，不会写出含合成/假数据的 pose_data.json。
        print(f"\n[FATAL][STRICT] {exc}", file=sys.stderr, flush=True)
        sys.exit(EXIT_STRICT_REFUSED)

    elapsed = time.time() - start_time
    print(f"\n[Done] 总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
