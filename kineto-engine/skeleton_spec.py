#!/usr/bin/env python3
"""
Kineto Skeleton Spec — SMPL 24 关节骨架单一事实源 (SSOT)
========================================================
本模块是引擎侧全部骨架解剖常量的**唯一权威源**：关节名、父子拓扑(kintree)、
骨架骨对、骨长上下界、对称骨对、关节角度检查 pivot、真实 canonical rest 模板。
kineto_core.py 与 pose_audit.py 一律从这里 import，不得再各自硬编码。

设计依据（关节序根因整改 P1 / 改动 B）
--------------------------------------
1. **canonical 序**：全部索引按 SMPL canonical 24 关节序（与
   ``vertices2joints(J_regressor, vertices)`` 的输出序一致）。注意 HMR2 的
   ``pred_keypoints_3d`` 是经 smpl_wrapper 重排后的 OpenPose Body-25 序，
   与本模块索引**不可**混用；kineto_core._solve_hmr2 已将交付的 joints_3d
   归一为 canonical 序。
2. **真 kintree**：父表来自 basicModel pkl 的 kintree_table（加载时交叉校验）。
   消除了旧代码的双骨架树冲突——collar(13/14) 的父节点统一为 spine3(9)
   （旧 SMPL_SKELETON/BONE_PART_MAP/SMPL_CHILDREN/pose_audit.BONE_PAIRS
   误作 neck(12)）。
3. **真实 rest 模板**：SMPL_REST_JOINTS = J_regressor @ v_template
   （betas=0、pose=0），替换旧 kineto_core 中硬编码的假模板。
4. **骨长界数据驱动**：BONE_LENGTH_BOUNDS 由真实 rest 骨长按统一比例
   [0.75×, 1.35×] 派生（覆盖 betas 个体差异与估计误差），替换旧手标值——
   旧值与真实骨架明显矛盾（如 spine2→spine3 实长 0.059 却标 (0.10,0.20)、
   spine3→neck 实长 0.218 却标 (0.08,0.18)、collar→shoulder 实长 ~0.099
   却标 (0.22,0.40)，系命名整体错位一格所致），在 canonical 数据上会产生
   恒假阳/假阴审计信号并把骨长约束夹向错误目标。

前端 kineto-web/lib/skeleton.ts 应镜像本模块（一致性闸门属后续阶段）。
"""

import inspect as _inspect
import os as _os
import pickle as _pickle
import warnings as _warnings
from pathlib import Path as _Path

import numpy as np

# [n3] chumpy 兼容 shim：Python 3.11+ 移除了 inspect.getargspec，而 SMPL pkl 内的
# chumpy 对象在反序列化时依赖它。此处于**模块导入期一次性安装并保持**（与
# kineto_core.load_4dhumans_model 的策略统一），消除“skeleton_spec 必须先于任何
# chumpy 触碰被 import”的隐式顺序依赖——kineto_core 在模块级 import 本模块，故
# 引擎内任何 chumpy 路径运行前 shim 必已就位。幂等：已存在则不覆盖。
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec

__all__ = [
    "SMPL_MODEL_PKL",
    "SMPL_JOINT_NAMES",
    "SMPL_PARENTS",
    "SMPL_CHILDREN_FULL",
    "SMPL_SKELETON",
    "BONE_NAMES",
    "BONE_PART_MAP",
    "BONE_LENGTH_BOUNDS",
    "SYMMETRIC_BONE_PAIRS",
    "JOINT_ANGLE_PIVOTS",
    "JOINT_ANGLE_RANGES",
    "SMPL_REST_JOINTS",
    "SMPL_REST_JOINTS_CENTERED",
    "SMPL_V_TEMPLATE",
    "SMPL_J_REGRESSOR",
    "bone_name",
    "rest_bone_lengths",
    "validate_self_consistency",
]

# ============================================================================
# 基础拓扑（canonical SMPL 24）
# ============================================================================

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

NUM_JOINTS = 24

_SHORT = {
    "pelvis": "pelvis", "left_hip": "l_hip", "right_hip": "r_hip",
    "spine1": "spine1", "spine2": "spine2", "spine3": "spine3",
    "left_knee": "l_knee", "right_knee": "r_knee",
    "left_ankle": "l_ankle", "right_ankle": "r_ankle",
    "left_foot": "l_foot", "right_foot": "r_foot",
    "neck": "neck", "head": "head",
    "left_collar": "l_collar", "right_collar": "r_collar",
    "left_shoulder": "l_shoulder", "right_shoulder": "r_shoulder",
    "left_elbow": "l_elbow", "right_elbow": "r_elbow",
    "left_wrist": "l_wrist", "right_wrist": "r_wrist",
    "left_hand": "l_hand", "right_hand": "r_hand",
}


def bone_name(parent: int, child: int) -> str:
    """骨对 (parent, child) 的稳定命名，如 (9, 13) → 'spine3_l_collar'。"""
    return f"{_SHORT[SMPL_JOINT_NAMES[parent]]}_{_SHORT[SMPL_JOINT_NAMES[child]]}"


# SMPL 24 关节标准运动学树父节点（-1 表示根/骨盆）。canonical 顺序下父节点索引
# 恒小于子节点。加载 pkl 时会与 kintree_table 交叉校验（见 _load_smpl_template）。
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                13, 14, 16, 17, 18, 19, 20, 21]

# 由父表派生的完整子节点表（含脚踝→脚、手腕→手）。
SMPL_CHILDREN_FULL = {j: [] for j in range(NUM_JOINTS)}
for _child, _parent in enumerate(SMPL_PARENTS):
    if _parent >= 0:
        SMPL_CHILDREN_FULL[_parent].append(_child)

# 骨架骨对 = 真 kintree 的全部父→子边（23 条，覆盖 ankle→foot、wrist→hand，
# collar 父节点为 9）。旧 SMPL_SKELETON 中 (12,13)/(12,14) 等假边已被淘汰。
SMPL_SKELETON = [(SMPL_PARENTS[c], c) for c in range(1, NUM_JOINTS)]

# 骨对 → 稳定名称（audit JSON / 报告键名的事实源）
BONE_NAMES = {(p, c): bone_name(p, c) for p, c in SMPL_SKELETON}

# 身体部位归属（渲染着色用；键集合与 SMPL_SKELETON 一致）
_TORSO_CHAIN = [(0, 3), (3, 6), (6, 9), (9, 12)]
_LEFT_ARM = [(9, 13), (13, 16), (16, 18), (18, 20), (20, 22)]
_RIGHT_ARM = [(9, 14), (14, 17), (17, 19), (19, 21), (21, 23)]
_LEFT_LEG = [(0, 1), (1, 4), (4, 7), (7, 10)]
_RIGHT_LEG = [(0, 2), (2, 5), (5, 8), (8, 11)]

BONE_PART_MAP = {}
BONE_PART_MAP.update({b: "torso" for b in _TORSO_CHAIN})
BONE_PART_MAP.update({b: "left_arm" for b in _LEFT_ARM})
BONE_PART_MAP.update({b: "right_arm" for b in _RIGHT_ARM})
BONE_PART_MAP.update({b: "left_leg" for b in _LEFT_LEG})
BONE_PART_MAP.update({b: "right_leg" for b in _RIGHT_LEG})
BONE_PART_MAP[(12, 15)] = "head"

# 左右对称骨对：(左骨, 右骨)，索引与命名均由真 kintree 派生
# （替换 pose_audit 旧硬编码 13,16,14,17 等错位下标）。
SYMMETRIC_BONE_PAIRS = [
    ((0, 1), (0, 2)),     # pelvis→hip
    ((1, 4), (2, 5)),     # hip→knee (大腿)
    ((4, 7), (5, 8)),     # knee→ankle (小腿)
    ((7, 10), (8, 11)),   # ankle→foot
    ((9, 13), (9, 14)),   # spine3→collar
    ((13, 16), (14, 17)), # collar→shoulder
    ((16, 18), (17, 19)), # shoulder→elbow (上臂)
    ((18, 20), (19, 21)), # elbow→wrist (前臂)
    ((20, 22), (21, 23)), # wrist→hand
]

# 关节角度检查 pivot：joint → (a, pivot, b)，(a,pivot) 与 (pivot,b) 均为 kintree 边。
# 由真 kintree 派生（修正 pose_audit 旧表：l_elbow/r_elbow 曾误作 (13,16,18)/
# (14,17,19)——13/14 是 collar 而非 shoulder；l_shoulder/r_shoulder 旧表缺失，
# 现按 (13,16,18)/(14,17,19) 补全，度量 collar→shoulder→elbow 夹角）。
JOINT_ANGLE_PIVOTS = {
    "l_hip":      (0, 1, 4),
    "r_hip":      (0, 2, 5),
    "l_knee":     (1, 4, 7),
    "r_knee":     (2, 5, 8),
    "l_shoulder": (13, 16, 18),
    "r_shoulder": (14, 17, 19),
    "l_elbow":    (16, 18, 20),
    "r_elbow":    (17, 19, 21),
}

# 关节角度合理范围（度）：0°=完全折叠，180°=伸直。膝/肘为铰链关节，
# 过伸超出生理范围即判异常；髋/肩活动度大。
JOINT_ANGLE_RANGES = {
    "l_knee":     (0, 160),
    "r_knee":     (0, 160),
    "l_elbow":    (0, 160),
    "r_elbow":    (0, 160),
    "l_hip":      (0, 130),
    "r_hip":      (0, 130),
    "l_shoulder": (0, 180),
    "r_shoulder": (0, 180),
}

# ============================================================================
# 真实 rest 模板（从 basicModel pkl 读取 J_regressor / v_template / kintree）
# ============================================================================

SMPL_MODEL_PKL = (_Path(__file__).resolve().parent
                  / "4D-Humans" / "data"
                  / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")


# [m7] 白名单：仅这些类在 scipy.sparse 顶层命名空间稳定可寻址。旧 scipy 序列化的
# 内部子模块路径（如 scipy.sparse._csr._csr_matrix）不在白名单 → 保持原路径，避免
# 把“仅 DeprecationWarning”的情形升级成 AttributeError 硬失败。
_SPARSE_TOPLEVEL_OK = {
    "csc_matrix", "csr_matrix", "coo_matrix", "dia_matrix",
    "lil_matrix", "dok_matrix", "bsr_matrix", "spmatrix",
}


class _UnpicklerRemapSparse(_pickle.Unpickler):
    """把旧 scipy 序列化的 ``scipy.sparse.<sub>`` 类路径重映射到非弃用的
    ``scipy.sparse`` 顶层命名空间，避免加载 basicModel pkl 时的
    ``scipy.sparse.csc is deprecated`` DeprecationWarning（走非弃用导入路径）。
    仅对白名单内、顶层确可寻址的类做重映射；其余保持原路径（最多一条弃用告警），
    杜绝一刀切重映射在异构 scipy 版本上抛 AttributeError。"""

    def find_class(self, module, name):
        if module.startswith("scipy.sparse.") and name in _SPARSE_TOPLEVEL_OK:
            try:
                return super().find_class("scipy.sparse", name)
            except AttributeError:
                pass  # 顶层不可寻址 → 回落原路径（只多一条 DeprecationWarning）
        return super().find_class(module, name)


# [C2] pkl 解析候选（按优先级）：① env KINETO_SMPL_PKL 显式覆盖；② vendored
# basicModel（随 4D-Humans 数据分发）；③ 4DHumans 缓存回退 SMPL_NEUTRAL.pkl
# （实测其 J_regressor/v_template/kintree_table 与 basicModel 逐位相同 max|diff|=0.0，
# 且在所有受支持部署中均已挂载：compose /root/.cache、systemd HOME=/srv/kineto）。
_SMPL_PKL_FALLBACK = (_Path.home() / ".cache" / "4DHumans"
                      / "data" / "smpl" / "SMPL_NEUTRAL.pkl")


def _resolve_pkl_path():
    """返回首个存在的 SMPL pkl 路径；全部缺失则抛 FileNotFoundError（**惰性**触发，
    绝不在 import 期）。env KINETO_SMPL_PKL 优先，便于部署显式指定/覆盖。"""
    env = _os.environ.get("KINETO_SMPL_PKL")
    candidates = [p for p in (env, str(SMPL_MODEL_PKL), str(_SMPL_PKL_FALLBACK)) if p]
    for c in candidates:
        if _Path(c).exists():
            return c
    raise FileNotFoundError(
        "[skeleton_spec] 未找到任何 SMPL basicModel pkl，尝试过：\n  - "
        + "\n  - ".join(candidates)
        + "\n真实 rest 模板/骨长界依赖该文件。请确认部署完整性"
          "（deploy/transfer_models.sh），或设 KINETO_SMPL_PKL 指向有效 pkl。"
    )


def _load_smpl_template(pkl_path=None):
    """读取 SMPL basicModel pkl，返回 (J_regressor(24,6890), v_template(6890,3))。

    - pkl_path=None 时经 _resolve_pkl_path() 解析（env 覆盖 + 缓存回退）。
    - pkl 内含 chumpy 对象：inspect.getargspec shim 已于模块导入期一次性安装
      （见文件头 [n3]），此处不再临时装卸。
    - J_regressor 在 pkl 中为 scipy sparse (24,6890)，转为稠密 float64。
      该 pkl 由旧版 scipy 序列化，pickle 内部记录的类路径为已弃用的
      ``scipy.sparse.csc.csc_matrix`` 等子模块；直接 load 会触发
      ``DeprecationWarning: scipy.sparse.csc namespace is deprecated``。
      故用自定义 Unpickler 把白名单内 ``scipy.sparse.<sub>`` 类路径重映射到
      非弃用的 ``scipy.sparse`` 顶层命名空间，彻底走非弃用路径、消除告警，
      且不改动 pkl 本身。
    - 加载时与硬编码 SMPL_PARENTS 交叉校验 kintree_table，任何不一致直接
      抛错（SSOT 自证一致性）。
    """
    if pkl_path is None:
        pkl_path = _resolve_pkl_path()
    if not _Path(pkl_path).exists():
        raise FileNotFoundError(
            f"[skeleton_spec] SMPL basicModel 缺失: {pkl_path}\n"
            "真实 rest 模板依赖该文件（随 4D-Humans vendored 数据分发），"
            "请确认部署完整性（deploy/transfer_models.sh）。"
        )

    # csc_matrix 类路径已由 _UnpicklerRemapSparse 重映射到非弃用命名空间；
    # 此处再作用域受限地抑制 DeprecationWarning，仅为屏蔽 chumpy 库
    # 自身 import 链触发的 scipy.sparse.linalg.interface 弃用告警
    # （第三方遗留依赖内部导入，无法重定向，与本模块无关）。
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", DeprecationWarning)
        with open(pkl_path, "rb") as f:
            data = _UnpicklerRemapSparse(f, encoding="latin1").load()

    j_regressor = data["J_regressor"]
    if hasattr(j_regressor, "toarray"):          # scipy sparse
        j_regressor = j_regressor.toarray()
    j_regressor = np.asarray(j_regressor, dtype=np.float64)
    v_template = np.asarray(data["v_template"], dtype=np.float64)

    if j_regressor.shape != (NUM_JOINTS, 6890) or v_template.shape != (6890, 3):
        raise ValueError(
            f"[skeleton_spec] pkl 形状异常: J_regressor={j_regressor.shape}, "
            f"v_template={v_template.shape}")

    # kintree 交叉校验：kintree_table[0, c] 即关节 c 的父节点（根为 uint32 溢出值）
    kintree = np.asarray(data["kintree_table"]).astype(np.int64)
    pkl_parents = [int(kintree[0, c]) if kintree[0, c] < NUM_JOINTS else -1
                   for c in range(NUM_JOINTS)]
    if pkl_parents != SMPL_PARENTS:
        raise ValueError(
            f"[skeleton_spec] kintree 校验失败:\n"
            f"  pkl      = {pkl_parents}\n"
            f"  SMPL_PARENTS = {SMPL_PARENTS}")

    return j_regressor, v_template


# ============================================================================
# [C2] pkl 派生常量惰性加载（绝不在 import 期触发；首次访问时才读 pkl）
# ----------------------------------------------------------------------------
# 下列常量依赖真实 SMPL pkl（J_regressor/v_template）：SMPL_J_REGRESSOR、
# SMPL_V_TEMPLATE、SMPL_REST_JOINTS、SMPL_REST_JOINTS_CENTERED、BONE_LENGTH_BOUNDS。
# 旧实现于模块级无条件 _load_smpl_template() → 缺 pkl 时 `import skeleton_spec`
# 直接崩（Docker/XPU 部署 100% 失败，且 /healthz 仍报 healthy → 静默全面故障）。
# 现改为 PEP 562 模块级 __getattr__ 惰性触发：纯拓扑常量（关节名/父表/骨架/
# 骨对名/部位/对称对/角度 pivot）import 期即就绪；pkl 派生常量首次被访问（如
# kineto_core 的 `from skeleton_spec import BONE_LENGTH_BOUNDS`）时才加载。**导出符号名不变**。
_LAZY_PKL_NAMES = frozenset({
    "SMPL_J_REGRESSOR", "SMPL_V_TEMPLATE", "SMPL_REST_JOINTS",
    "SMPL_REST_JOINTS_CENTERED", "BONE_LENGTH_BOUNDS",
})

# 骨长上下界比例（数据驱动：真实 rest 骨长 × [0.75, 1.35]）
_BONE_LO_RATIO, _BONE_HI_RATIO = 0.75, 1.35


def _rest_bone_lengths_from(rest_joints):
    """给定 rest 关节数组，返回 {(parent, child): 骨长(米)}，覆盖全部 23 条边。"""
    J = np.asarray(rest_joints)
    return {(p, c): float(np.linalg.norm(J[c] - J[p])) for p, c in SMPL_SKELETON}


def _ensure_smpl_loaded():
    """惰性加载并缓存 pkl 派生常量到模块 globals（幂等：已加载则直接返回）。
    缺 pkl 时抛 FileNotFoundError——但只在**首次访问派生常量**时，绝不在 import 期。"""
    if "SMPL_REST_JOINTS" in globals():
        return
    j_reg, v_tpl = _load_smpl_template()
    # canonical rest 关节（betas=0, pose=0 的 T-pose，24×3，米，SMPL 模型坐标系：
    # Y 向上、左侧为 +X）。即 J_regressor @ v_template，与 pkl 内置 'J' 一致。
    rest = j_reg @ v_tpl
    globals()["SMPL_J_REGRESSOR"] = j_reg
    globals()["SMPL_V_TEMPLATE"] = v_tpl
    globals()["SMPL_REST_JOINTS"] = rest
    # root(骨盆)中心化版本：便于直接做骨向量/尺度分析（P2 个体化 rest 用）。
    globals()["SMPL_REST_JOINTS_CENTERED"] = rest - rest[0]
    globals()["BONE_LENGTH_BOUNDS"] = {
        bone: (round(lo * _BONE_LO_RATIO, 4), round(lo * _BONE_HI_RATIO, 4))
        for bone, lo in _rest_bone_lengths_from(rest).items()
    }


def __getattr__(name):
    """PEP 562：拦截 pkl 派生常量的首次访问，触发惰性加载后返回真实值。
    拓扑常量与函数均在模块 __dict__ 中，不会走到这里。"""
    if name in _LAZY_PKL_NAMES:
        _ensure_smpl_loaded()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def rest_bone_lengths(rest_joints=None):
    """返回 {(parent, child): 真实 rest 骨长(米)}，覆盖全部 23 条 kintree 边。
    rest_joints=None 时用惰性加载的 SMPL_REST_JOINTS（触发 pkl 读取）。"""
    if rest_joints is None:
        _ensure_smpl_loaded()
        J = globals()["SMPL_REST_JOINTS"]
    else:
        J = np.asarray(rest_joints)
    return {(p, c): float(np.linalg.norm(J[c] - J[p])) for p, c in SMPL_SKELETON}


# ============================================================================
# 自检（不在 import 期执行；由 check_ssot 闸门 / 单测 / __main__ 显式调用）
# ============================================================================

# [m8] 解剖镜像映射：左右对称关节下标互换（中线关节自映射）。用于校验
# SYMMETRIC_BONE_PAIRS 确为真镜像（防手改一侧静默通过 → symm_score 失真）。
_JOINT_MIRROR = {
    0: 0, 3: 3, 6: 6, 9: 9, 12: 12, 15: 15,
    1: 2, 2: 1, 4: 5, 5: 4, 7: 8, 8: 7, 10: 11, 11: 10,
    13: 14, 14: 13, 16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20, 22: 23, 23: 22,
}


def validate_self_consistency():
    """SSOT 不变量自检，返回问题列表（空 = 通过）。供单测/闸门(check_ssot)调用。
    会触发 pkl 派生常量的惰性加载（校验 BONE_LENGTH_BOUNDS/rest 骨长需真模板）。"""
    _ensure_smpl_loaded()
    bounds = globals()["BONE_LENGTH_BOUNDS"]
    rest_joints = globals()["SMPL_REST_JOINTS"]
    problems = []

    if len(SMPL_JOINT_NAMES) != NUM_JOINTS:
        problems.append("joint names != 24")
    if len(SMPL_PARENTS) != NUM_JOINTS or SMPL_PARENTS[0] != -1:
        problems.append("parents malformed")
    for c in range(1, NUM_JOINTS):
        if not (0 <= SMPL_PARENTS[c] < c):
            problems.append(f"parent[{c}]={SMPL_PARENTS[c]} violates canonical order")
    if SMPL_PARENTS[13] != 9 or SMPL_PARENTS[14] != 9:
        problems.append("collar 13/14 parent != 9 (真 kintree)")

    skeleton_set = set(SMPL_SKELETON)
    if len(skeleton_set) != NUM_JOINTS - 1:
        problems.append("skeleton edges != 23 or duplicated")
    if set(BONE_PART_MAP) != skeleton_set:
        problems.append("BONE_PART_MAP keys != SMPL_SKELETON")
    if set(bounds) != skeleton_set:
        problems.append("BONE_LENGTH_BOUNDS keys != SMPL_SKELETON")
    for bone, (lo, hi) in bounds.items():
        if not (0 < lo < hi):
            problems.append(f"bad bounds {bone}: ({lo},{hi})")

    # 真实 rest 骨长必须落在界内（构造即成立，防未来手改破坏）
    for bone, length in _rest_bone_lengths_from(rest_joints).items():
        lo, hi = bounds[bone]
        if not (lo <= length <= hi):
            problems.append(f"rest length {bone}={length:.4f} outside ({lo},{hi})")

    # [m8] 对称骨对必须互为真解剖镜像（不止“两端都在 skeleton 里”）
    for lb, rb in SYMMETRIC_BONE_PAIRS:
        if lb not in skeleton_set or rb not in skeleton_set:
            problems.append(f"symmetric pair not in skeleton: {lb}/{rb}")
            continue
        expect = (_JOINT_MIRROR[lb[0]], _JOINT_MIRROR[lb[1]])
        if tuple(rb) != expect:
            problems.append(f"symmetric pair {lb} 的镜像应为 {expect}，实际 {tuple(rb)}")

    # [m8] 角度 pivot 与 range 键集合必须完全一致，且覆盖 8 个可检关节
    for name, (p, pivot, c) in JOINT_ANGLE_PIVOTS.items():
        if (p, pivot) not in skeleton_set or (pivot, c) not in skeleton_set:
            problems.append(f"angle pivot {name} edges not in skeleton")
    if set(JOINT_ANGLE_PIVOTS) != set(JOINT_ANGLE_RANGES):
        problems.append("JOINT_ANGLE_PIVOTS 与 JOINT_ANGLE_RANGES 键集合不一致")
    if len(JOINT_ANGLE_PIVOTS) != 8:
        problems.append(f"JOINT_ANGLE_PIVOTS 应覆盖 8 个关节，实际 {len(JOINT_ANGLE_PIVOTS)}")

    if rest_joints.shape != (NUM_JOINTS, 3) or not np.isfinite(rest_joints).all():
        problems.append("rest joints malformed")

    return problems


if __name__ == "__main__":
    _ensure_smpl_loaded()
    _problems = validate_self_consistency()
    if _problems:  # SSOT 损坏时响亮失败（仅 __main__ 调试入口）
        raise SystemExit("[skeleton_spec] SSOT 自检失败:\n  - " + "\n  - ".join(_problems))
    print(f"[skeleton_spec] SSOT 自检通过（{len(SMPL_SKELETON)} 骨, "
          f"{len(SYMMETRIC_BONE_PAIRS)} 对称对, {len(JOINT_ANGLE_PIVOTS)} 角度 pivot）")
    for _b, (_lo, _hi) in BONE_LENGTH_BOUNDS.items():
        _p, _c = _b
        print(f"  {_b} {BONE_NAMES[_b]:>22s}  rest={rest_bone_lengths()[_b]:.4f}  "
              f"bounds=({_lo:.4f},{_hi:.4f})")
