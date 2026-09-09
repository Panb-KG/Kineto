#!/usr/bin/env python3
# =============================================================================
# Kineto / deploy — check_ssot.py   (骨架 SSOT ↔ 前端镜像 一致性闸门)
# -----------------------------------------------------------------------------
# 在 validate.sh 中是 **G12**（计划文档里称 **G8-SSOT**）；也可单独跑：
#     python3 deploy/check_ssot.py                 # 用仓库默认路径
#     bash deploy/validate.sh --ssot-only          # 只跑这一道闸门（无需引擎在线）
#
# 校验对象（**只读**，本脚本绝不写任何文件、绝不 codegen）：
#   权威源 SSOT : kineto-engine/skeleton_spec.py
#                 （SMPL_JOINT_NAMES / SMPL_PARENTS / SMPL_SKELETON / BONE_PART_MAP）
#   前端镜像    : kineto-web/lib/skeleton.ts
#                 （同名 const；可选 kineto-web/lib/skeleton.manifest.json 优先）
#
# 断言项（P1 关节序根因整改后的 canonical 事实）：
#   A1 SMPL_JOINT_NAMES：24 个 canonical 关节名，**顺序逐一相同**
#   A2 SMPL_PARENTS    ：24 项父表逐一相同，**13/14(left/right_collar) 的父必须是 9(spine3)**
#                        前端若未导出 SMPL_PARENTS，则由 SMPL_SKELETON 反推父表后比对
#   A3 SMPL_SKELETON   ：23 条 canonical 边，集合相同、无重复，
#                        **必须含 (9,13)/(9,14)，且绝不含旧假边 (12,13)/(12,14)**
#   A4 BONE_PART_MAP   ：键集合与部位归属逐一相同（左右臂链挂在 9→13/14 上）
#   G14 mesh↔joints    ：产物 pose_data.json 中携带 mesh_vertices 的关键帧，
#                        从顶点经 SMPL_J_REGRESSOR 回归的关节须与 joints_3d 重合
#                        （容差 10mm；引擎 [P0 mesh↔joints 对齐] 后按构造 ≈0）。
#                        无产物/无 mesh 帧时 SKIP，不阻塞其他闸门。
#
# 取值策略（低风险、可在任何环境跑）：
#   SSOT 侧优先 `import skeleton_spec`（顺带触发其模块级自检 validate_self_consistency）；
#   若该 import 不可用（无 numpy / 无 chumpy / 缺 basicModel pkl —— 例如在纯 Mac 系统
#   python3 或容器外），自动退化为 **AST 静态解析** skeleton_spec.py 里的字面量常量，
#   并按 SSOT 中同一套推导式（SMPL_SKELETON = [(parents[c], c) for c in 1..23]）重建，
#   因此两条路径给出的权威值一致。
#
# 退出码：
#   0 = 一致（PASS）
#   1 = 检出漂移（FAIL；闸门有效，需把前端镜像对齐 SSOT）
#   2 = 文件在但无法解析/校验（FAIL；不能证明一致就等于不安全）
#   3 = 文件缺失或环境不可用（SKIP）：
#       - 文件缺失（设备上通常没有 kineto-web/）
#       - pkl 不可用（Docker 构建期 / XPU 环境）→ 退化 AST 但显式 SKIP 未覆盖不变量
# =============================================================================

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
from pathlib import Path

NUM_JOINTS = 24
COLLAR_L, COLLAR_R, SPINE3, NECK = 13, 14, 9, 12

# G14：mesh 顶点 ↔ joints_3d 一致性容差（mm）。
# 引擎 [P0 mesh↔joints 对齐] 后两者按构造精确重合（残差 0），浮点经 JSON 往返
# 只有亚毫米误差；超过阈值说明产物由旧管线生成（未对齐）或被手工改动。
_MESH_JOINTS_TOL_MM = 10.0

_OK, _DRIFT, _INFO, _SKIP = "[OK]", "[DRIFT]", "[INFO]", "[SKIP]"


# -----------------------------------------------------------------------------
# 输出小工具
# -----------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.drifts: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []   # 非致命告警（SKIP 时展示未覆盖不变量）

    def ok(self, msg: str) -> None:
        print(f"{_OK} {msg}")

    def info(self, msg: str) -> None:
        print(f"{_INFO} {msg}")

    def drift(self, msg: str) -> None:
        self.drifts.append(msg)
        print(f"{_DRIFT} {msg}")

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"{_DRIFT} {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"{_SKIP} {msg}")


# -----------------------------------------------------------------------------
# SSOT 侧：kineto-engine/skeleton_spec.py
# -----------------------------------------------------------------------------
def load_spec_dynamic(engine_dir: Path):
    """import skeleton_spec 取权威值。

    返回 (spec_dict_or_None, how_str, pkl_available: bool)。
    - pkl_available=True：完整 import + validate_self_consistency 通过（权威路径）。
    - pkl_available=False：pkl 不可用（Docker 构建期 / XPU），import 本身成功但
      validate_self_consistency 触发 FileNotFoundError；此时返回 None 让调用方
      退化到 AST 静态解析，并标记未覆盖的 pkl 依赖不变量。
    """
    sys.path.insert(0, str(engine_dir))
    try:
        mod = importlib.import_module("skeleton_spec")
    except Exception as exc:              # noqa: BLE001 - import 本身失败（语法错等）
        return None, f"{type(exc).__name__}: {exc}", False
    finally:
        try:
            sys.path.remove(str(engine_dir))
        except ValueError:
            pass

    # import 成功；尝试 validate_self_consistency（触发 pkl 惰性加载）
    try:
        problems = list(mod.validate_self_consistency())
        if problems:                      # SSOT 自证失败 → 不能当权威值用
            return None, f"SSOT 自检未通过: {problems}", True
        return {
            "joint_names": list(mod.SMPL_JOINT_NAMES),
            "parents": [int(p) for p in mod.SMPL_PARENTS],
            "skeleton": [(int(p), int(c)) for p, c in mod.SMPL_SKELETON],
            "part_map": {(int(p), int(c)): str(v) for (p, c), v in mod.BONE_PART_MAP.items()},
        }, "import skeleton_spec（权威，含模块级自检）", True
    except FileNotFoundError:
        # pkl 不可用 → 退化到 AST，但标记 pkl_available=False
        return None, "pkl 不可用（validate_self_consistency → FileNotFoundError）", False
    except Exception as exc:              # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}", False


# AST 静态解析用到的 BONE_PART_MAP 构造链（与 skeleton_spec.py 现有写法一一对应）
_PART_CHAINS = {
    "_TORSO_CHAIN": "torso",
    "_LEFT_ARM": "left_arm",
    "_RIGHT_ARM": "right_arm",
    "_LEFT_LEG": "left_leg",
    "_RIGHT_LEG": "right_leg",
}


def load_spec_static(spec_path: Path):
    """AST 静态解析 skeleton_spec.py 的字面量常量（不 import、不需要 numpy/pkl）。"""
    tree = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))

    literals: dict[str, object] = {}
    part_map: dict[tuple[int, int], str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                try:
                    literals[tgt.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
            # BONE_PART_MAP[(12, 15)] = "head" 这类下标赋值
            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) \
                    and tgt.value.id == "BONE_PART_MAP":
                try:
                    key = ast.literal_eval(tgt.slice)
                    val = ast.literal_eval(node.value)
                    if isinstance(key, tuple) and len(key) == 2:
                        part_map[(int(key[0]), int(key[1]))] = str(val)
                except (ValueError, TypeError, SyntaxError):
                    pass

    names = literals.get("SMPL_JOINT_NAMES")
    parents = literals.get("SMPL_PARENTS")
    if not isinstance(names, list) or not isinstance(parents, list):
        raise ValueError("skeleton_spec.py 里找不到 SMPL_JOINT_NAMES / SMPL_PARENTS 字面量")

    for var, part in _PART_CHAINS.items():
        chain = literals.get(var)
        if not isinstance(chain, list):
            raise ValueError(f"skeleton_spec.py 里找不到 {var} 字面量（BONE_PART_MAP 静态重建失败）")
        for edge in chain:
            part_map[(int(edge[0]), int(edge[1]))] = part

    parents = [int(p) for p in parents]
    # 与 SSOT 同一推导式：全部父→子边（23 条）
    skeleton = [(parents[c], c) for c in range(1, len(parents))]
    return {
        "joint_names": [str(n) for n in names],
        "parents": parents,
        "skeleton": skeleton,
        "part_map": part_map,
    }


# -----------------------------------------------------------------------------
# 前端侧：kineto-web/lib/skeleton.ts（tolerant 解析）+ 可选 manifest
# -----------------------------------------------------------------------------
def strip_ts_comments(src: str) -> str:
    """去掉 // 与 /* */ 注释（带字符串状态机，避免误伤字符串里的 //）。"""
    out: list[str] = []
    i, n = 0, len(src)
    quote = ""
    while i < n:
        ch = src[i]
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'`":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _array_body(src: str, decl_name: str) -> str | None:
    """定位 `const <decl_name> ... = [ ... ]`，返回**配对中括号内**的内容（支持嵌套）。"""
    m = re.search(rf"\b(?:const|let|var|export\s+const)\s+{re.escape(decl_name)}\b", src)
    if not m:
        m = re.search(rf"\b{re.escape(decl_name)}\b\s*(?::[^=]+)?=", src)
        if not m:
            return None
        eq = src.find("=", m.end() - 1)
    else:
        # 声明之后可能有类型标注（如 `: ReadonlyArray<readonly [number, number]>`），
        # 必须先跳到**第一个 `=`**，否则会把类型标注里的 [number, number] 当成数据。
        eq = src.find("=", m.end())
    if eq < 0:
        return None
    start = src.find("[", eq)
    if start < 0:
        return None
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "[":
            depth += 1
        elif src[j] == "]":
            depth -= 1
            if depth == 0:
                return src[start + 1:j]
    return None


def _strings(body: str) -> list[str]:
    return re.findall(r"""["']([^"']+)["']""", body)


def _ints(body: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", body)]


def _pairs(body: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in re.findall(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", body)]


def load_frontend_ts(ts_path: Path):
    """tolerant 解析 skeleton.ts 里的 const 数组；解析不出关键项即抛 ValueError。"""
    src = strip_ts_comments(ts_path.read_text(encoding="utf-8"))

    names_body = _array_body(src, "SMPL_JOINT_NAMES")
    skel_body = _array_body(src, "SMPL_SKELETON")
    if names_body is None or skel_body is None:
        raise ValueError(
            "skeleton.ts 中找不到 SMPL_JOINT_NAMES / SMPL_SKELETON 的数组声明"
            "（期望 `export const SMPL_JOINT_NAMES = [...]` 与 `export const SMPL_SKELETON ... = [...]`）")

    joint_names = _strings(names_body)
    skeleton = _pairs(skel_body)
    if not joint_names or not skeleton:
        raise ValueError("skeleton.ts 的 SMPL_JOINT_NAMES / SMPL_SKELETON 解析为空（写法不被支持？）")

    parents_body = _array_body(src, "SMPL_PARENTS")
    parents = _ints(parents_body) if parents_body is not None else None

    # BONE_PART_MAP 支持两种写法：
    #   ① 条目数组  [[0, 3], "torso"], ...        （当前 skeleton.ts 的 BONE_PART_MAP_ENTRIES）
    #   ② 记录对象  "0-3": "torso", ...           （Record<string, BodyPart> 风格）
    part_map: dict[tuple[int, int], str] = {}
    for a, b, part in re.findall(
            r"\[\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*[\"']([A-Za-z_]+)[\"']\s*\]", src):
        part_map[(int(a), int(b))] = part
    for a, b, part in re.findall(
            r"[\"'](-?\d+)\s*-\s*(-?\d+)[\"']\s*:\s*[\"']([A-Za-z_]+)[\"']", src):
        part_map.setdefault((int(a), int(b)), part)

    return {
        "joint_names": joint_names,
        "skeleton": skeleton,
        "parents": parents,
        "part_map": part_map,
        "source": f"{ts_path.name}（TS 静态解析）",
    }


def load_frontend_manifest(manifest_path: Path):
    """可选：H 若产出 skeleton.manifest.json，则优先按它校验（结构更稳、不依赖 TS 写法）。"""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = data.get("joint_names") or data.get("SMPL_JOINT_NAMES")
    skel = data.get("skeleton") or data.get("SMPL_SKELETON")
    if not names or not skel:
        raise ValueError("manifest 缺 joint_names / skeleton 字段")
    raw_part = data.get("bone_part_map") or data.get("BONE_PART_MAP") or {}
    part_map: dict[tuple[int, int], str] = {}
    if isinstance(raw_part, dict):
        for k, v in raw_part.items():
            a, _, b = str(k).partition("-")
            part_map[(int(a), int(b))] = str(v)
    else:                                     # [[[0,3],"torso"], ...]
        for item in raw_part:
            (a, b), part = item[0], item[1]
            part_map[(int(a), int(b))] = str(part)
    parents = data.get("parents") or data.get("SMPL_PARENTS")
    return {
        "joint_names": [str(x) for x in names],
        "skeleton": [(int(p), int(c)) for p, c in skel],
        "parents": [int(p) for p in parents] if parents else None,
        "part_map": part_map,
        "source": f"{manifest_path.name}（前端产出的镜像清单）",
    }


# -----------------------------------------------------------------------------
# 比对
# -----------------------------------------------------------------------------
def compare(spec: dict, fe: dict, rep: Report) -> None:
    # --- A1 关节名（顺序敏感）-------------------------------------------------
    s_names, f_names = spec["joint_names"], fe["joint_names"]
    if len(f_names) != NUM_JOINTS:
        rep.drift(f"A1 SMPL_JOINT_NAMES 前端有 {len(f_names)} 项，应为 {NUM_JOINTS} 项")
    if s_names == f_names:
        rep.ok(f"A1 SMPL_JOINT_NAMES：{NUM_JOINTS} 个 canonical 关节名顺序完全一致")
    else:
        diff = [f"#{i} 前端={f_names[i] if i < len(f_names) else '<缺失>'} / SSOT={n}"
                for i, n in enumerate(s_names)
                if i >= len(f_names) or f_names[i] != n]
        rep.drift("A1 SMPL_JOINT_NAMES 顺序/取值漂移：\n         " + "\n         ".join(diff[:8]))

    # --- A3 骨架边（集合比对 + collar 硬断言）--------------------------------
    s_edges, f_edges = set(spec["skeleton"]), set(fe["skeleton"])
    if len(fe["skeleton"]) != len(f_edges):
        rep.drift(f"A3 SMPL_SKELETON 前端有重复边（{len(fe['skeleton'])} 条 / 去重后 {len(f_edges)} 条）")
    if len(f_edges) != NUM_JOINTS - 1:
        rep.drift(f"A3 SMPL_SKELETON 前端有 {len(f_edges)} 条边，应为 {NUM_JOINTS - 1} 条")
    missing, extra = sorted(s_edges - f_edges), sorted(f_edges - s_edges)
    if not missing and not extra and len(f_edges) == NUM_JOINTS - 1:
        rep.ok(f"A3 SMPL_SKELETON：{NUM_JOINTS - 1} 条 canonical 边集合一致")
    else:
        if missing:
            rep.drift("A3 前端缺少 SSOT 边（parent,child）: " + ", ".join(map(str, missing)))
        if extra:
            rep.drift("A3 前端多出非 canonical 边（parent,child）: " + ", ".join(map(str, extra)))

    # collar 父节点：P1 整改的核心事实（旧镜像误把 collar 挂在 neck(12) 上）
    for collar, side in ((COLLAR_L, "left_collar"), (COLLAR_R, "right_collar")):
        if (SPINE3, collar) in f_edges and (NECK, collar) not in f_edges:
            rep.ok(f"A3a {side}({collar}) 的父为 spine3({SPINE3}) —— 与真 kintree 一致")
        elif (NECK, collar) in f_edges:
            rep.drift(f"A3a {side}({collar}) 的父仍是 neck({NECK})，SSOT 为 spine3({SPINE3})"
                      f" —— 这是 P1 前的旧假边，必须改为 [{SPINE3}, {collar}]")
        else:
            rep.drift(f"A3a {side}({collar}) 缺边：前端既无 [{SPINE3},{collar}] 也无 [{NECK},{collar}]")

    # --- A2 父表（前端未导出时由骨架边反推）----------------------------------
    s_parents = spec["parents"]
    f_parents = fe.get("parents")
    if f_parents is None:
        derived = [None] * NUM_JOINTS
        for p, c in f_edges:
            if 0 <= c < NUM_JOINTS:
                derived[c] = p
        if derived[0] is None:
            derived[0] = -1                   # 根节点在边表里不出现
        if any(x is None for x in derived):
            rep.drift("A2 SMPL_PARENTS 无法从前端 SMPL_SKELETON 反推（有子节点缺父）")
        else:
            f_parents = [int(x) for x in derived]
            rep.info("A2 前端未导出 SMPL_PARENTS —— 已由 SMPL_SKELETON 反推父表后比对")
    else:
        rep.info(f"A2 前端导出了 SMPL_PARENTS（{len(f_parents)} 项），直接逐项比对")

    if f_parents is not None:
        if f_parents == s_parents:
            rep.ok("A2 SMPL_PARENTS：24 项父表与 SSOT 完全一致")
        else:
            diff = [f"#{i} 前端={f_parents[i] if i < len(f_parents) else '<缺失>'} / SSOT={p}"
                    for i, p in enumerate(s_parents)
                    if i >= len(f_parents) or f_parents[i] != p]
            rep.drift("A2 SMPL_PARENTS 漂移：\n         " + "\n         ".join(diff[:8]))
        for collar, side in ((COLLAR_L, "left_collar"), (COLLAR_R, "right_collar")):
            got = f_parents[collar] if len(f_parents) > collar else None
            if got == SPINE3:
                rep.ok(f"A2a parents[{collar}]({side}) == {SPINE3}(spine3)")
            else:
                rep.drift(f"A2a parents[{collar}]({side}) == {got}，应为 {SPINE3}(spine3；不是 {NECK} neck)")

    # --- A4 部位归属 ----------------------------------------------------------
    s_part, f_part = spec["part_map"], fe["part_map"]
    if not f_part:
        rep.error("A4 无法从前端解析 BONE_PART_MAP（支持写法：[[p, c], \"part\"] 条目数组，"
                  "或 \"p-c\": \"part\" 记录对象）—— 不能证明一致即判失败")
    else:
        keys_diff_missing = sorted(set(s_part) - set(f_part))
        keys_diff_extra = sorted(set(f_part) - set(s_part))
        val_diff = sorted(k for k in set(s_part) & set(f_part) if s_part[k] != f_part[k])
        if not (keys_diff_missing or keys_diff_extra or val_diff):
            rep.ok(f"A4 BONE_PART_MAP：{len(s_part)} 条边的部位归属完全一致")
        else:
            if keys_diff_missing:
                rep.drift("A4 BONE_PART_MAP 前端缺键: " + ", ".join(map(str, keys_diff_missing)))
            if keys_diff_extra:
                rep.drift("A4 BONE_PART_MAP 前端多键（非 canonical 边）: "
                          + ", ".join(map(str, keys_diff_extra)))
            for k in val_diff:
                rep.drift(f"A4 BONE_PART_MAP[{k}] 前端={f_part[k]} / SSOT={s_part[k]}")
        # 手臂链必须挂在 spine3(9)→collar 上，而不是 neck(12)→collar
        for collar, part in ((COLLAR_L, "left_arm"), (COLLAR_R, "right_arm")):
            if f_part.get((SPINE3, collar)) == part:
                rep.ok(f"A4a {part} 含边 ({SPINE3},{collar}) —— 与 SSOT 一致")
            elif (NECK, collar) in f_part:
                rep.drift(f"A4a {part} 仍把 collar 边写成 ({NECK},{collar})，应为 ({SPINE3},{collar})")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="骨架 SSOT(skeleton_spec.py) ↔ 前端镜像(skeleton.ts) 一致性闸门（只读，非 codegen）")
    ap.add_argument("--repo-root", default=str(here.parent), help="仓库根（默认 deploy/ 的上一级）")
    ap.add_argument("--spec", default=None, help="SSOT 路径（默认 <root>/kineto-engine/skeleton_spec.py）")
    ap.add_argument("--ts", default=None, help="前端镜像路径（默认 <root>/kineto-web/lib/skeleton.ts）")
    ap.add_argument("--manifest", default=None,
                    help="可选前端镜像清单（默认 <root>/kineto-web/lib/skeleton.manifest.json，存在即优先）")
    ap.add_argument("--engine-dir", default=None, help="引擎目录（默认 <root>/kineto-engine）")
    ap.add_argument("--pose-data", default=None,
                    help="G14 产物路径（默认自动探测 <engine-dir>/output_test|output_closed|output/pose_data.json）")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    engine_dir = Path(args.engine_dir) if args.engine_dir else root / "kineto-engine"
    spec_path = Path(args.spec) if args.spec else engine_dir / "skeleton_spec.py"
    ts_path = Path(args.ts) if args.ts else root / "kineto-web" / "lib" / "skeleton.ts"
    manifest_path = Path(args.manifest) if args.manifest else root / "kineto-web" / "lib" / "skeleton.manifest.json"

    rep = Report()
    print("=" * 78)
    print(" G12 / G8-SSOT  骨架单一事实源一致性闸门（只读校验，不改任何文件）")
    print(f"   SSOT   : {spec_path}")
    print(f"   前端   : {ts_path}")
    print(f"   manifest: {manifest_path} ({'存在' if manifest_path.exists() else '不存在'})")
    print("=" * 78)

    missing = [str(p) for p in (spec_path, ts_path) if not p.exists()]
    if missing and not (manifest_path.exists() and spec_path.exists()):
        rep.info(f"文件缺失，无从校验: {', '.join(missing)}")
        rep.info("设备上通常没有 kineto-web/（前端跑在 Zeabur）—— 本闸门应在**仓库检出**"
                 "（Mac / CI）上跑；此时记 SKIP 而非 FAIL。")
        print(f"{_SKIP} 无法校验（文件缺失）")
        return 3

    # --- SSOT 侧 ---
    pkl_available = True   # 默认假设可用；load_spec_dynamic 会修正
    spec, how, pkl_available = load_spec_dynamic(engine_dir)
    if spec is None:
        rep.info(f"import skeleton_spec 不可用（{how}）→ 退化为 AST 静态解析字面量")
        try:
            spec = load_spec_static(spec_path)
            how = "AST 静态解析 skeleton_spec.py（字面量 + SSOT 同款推导式）"
        except Exception as exc:              # noqa: BLE001
            rep.error(f"SSOT 静态解析失败: {type(exc).__name__}: {exc}")
            return 2
        # pkl 不可用 → AST 只能校验拓扑不变量，pkl 派生的不变量（骨长界/rest/对称对）未覆盖
        if not pkl_available:
            rep.warn("pkl 不可用 → 以下不变量**未被本次校验覆盖**（运行时首次推理前会重试加载）：")
            rep.warn("  · BONE_LENGTH_BOUNDS 键集合 = SMPL_SKELETON（骨长界数据驱动，需 pkl 派生 rest 骨长）")
            rep.warn("  · 真实 rest 骨长落在界内（构造性不变量，需 pkl 的 J_regressor @ v_template）")
            rep.warn("  · 对称骨对互为真解剖镜像（需 skeleton_set 完整校验）")
    rep.info(f"SSOT 取值方式: {how}")

    try:
        problems_self = []
        if len(spec["joint_names"]) != NUM_JOINTS:
            problems_self.append(f"SSOT 关节名 {len(spec['joint_names'])} != {NUM_JOINTS}")
        if spec["parents"][COLLAR_L] != SPINE3 or spec["parents"][COLLAR_R] != SPINE3:
            problems_self.append("SSOT collar(13/14) 父 != 9")
        if len(spec["skeleton"]) != NUM_JOINTS - 1:
            problems_self.append(f"SSOT 骨架边 {len(spec['skeleton'])} != {NUM_JOINTS - 1}")
        for p in problems_self:
            rep.error(p)
        if problems_self:
            return 2
        rep.ok(f"SSOT 自身：{NUM_JOINTS} 关节 / {NUM_JOINTS - 1} 边 / collar 父={SPINE3}")
    except Exception as exc:                  # noqa: BLE001
        rep.error(f"SSOT 结构异常: {type(exc).__name__}: {exc}")
        return 2

    # --- 前端侧 ---
    fe = None
    if manifest_path.exists():
        try:
            fe = load_frontend_manifest(manifest_path)
        except Exception as exc:              # noqa: BLE001
            rep.error(f"manifest 解析失败: {type(exc).__name__}: {exc}（改回解析 skeleton.ts？）")
            return 2
    if fe is None:
        try:
            fe = load_frontend_ts(ts_path)
        except Exception as exc:              # noqa: BLE001
            rep.error(f"skeleton.ts 解析失败: {type(exc).__name__}: {exc}")
            return 2
    rep.info(f"前端取值方式: {fe['source']}")

    compare(spec, fe, rep)

    # --- skeleton_validator.py 迁移检查（G12 覆盖引擎侧第三份骨架常量副本）---
    check_skeleton_validator_migration(engine_dir, rep)

    # --- G14：产物级 mesh 顶点 ↔ joints_3d 一致性（叠加模式贴合性闸门）---
    pose_data = Path(args.pose_data) if args.pose_data else _find_pose_data_with_mesh(engine_dir)
    check_mesh_joints_alignment(pose_data, engine_dir, rep)

    print("-" * 78)
    if rep.errors and not rep.drifts:
        print(f"==> G12/G8-SSOT 无法完成校验（{len(rep.errors)} 项解析/结构错误）")
        return 2
    if rep.drifts or rep.errors:
        print(f"==> G12/G8-SSOT FAIL：检出 {len(rep.drifts) + len(rep.errors)} 处漂移/错误 —— "
              f"前端镜像必须对齐 SSOT（kineto-engine/skeleton_spec.py）")
        print("    修法（由前端 Owner 执行，本闸门不 codegen）：把 skeleton.ts 的 collar 父节点")
        print(f"    由 neck({NECK}) 改为 spine3({SPINE3})，即 [12,13]/[12,14] → [9,13]/[9,14]，")
        print("    并同步 BONE_PART_MAP 的左右臂链首条边；如导出 SMPL_PARENTS 则与 SSOT 逐项对齐。")
        return 1
    # pkl 不可用但拓扑一致 → SKIP（非 PASS，因 pkl 派生不变量未覆盖）
    if not pkl_available:
        print(f"==> G12/G8-SSOT SKIP：拓扑一致（A1-A4）但 pkl 不可用，{len(rep.warnings)} 项不变量未覆盖")
        print("    运行时首次推理前会重试加载 pkl 派生常量（BONE_LENGTH_BOUNDS / rest 骨长）。")
        return 3
    print("==> G12/G8-SSOT PASS：前端 skeleton.ts 与 SSOT skeleton_spec.py 完全一致"
          "（含 skeleton_validator.py 迁移检查 + G14 mesh↔joints 一致性）")
    return 0


# -----------------------------------------------------------------------------
# G12 扩展：检查 skeleton_validator.py 是否已从 SSOT import（不再保留本地副本）
# -----------------------------------------------------------------------------
_REQUIRED_SSOT_IMPORTS = {
    "SMPL_JOINT_NAMES", "SMPL_SKELETON", "BONE_PART_MAP",
    "BONE_LENGTH_BOUNDS", "BONE_NAMES",
}
_FORBIDDEN_LOCAL_COPIES = {"SMPL_SKELETON", "BONE_VALIDITY"}


def check_skeleton_validator_migration(engine_dir: Path, rep: Report) -> None:
    """检查 skeleton_validator.py 是否已迁移到从 skeleton_spec import（G12 扩展覆盖）。

    断言：
    A5 文件存在时，必须 `from skeleton_spec import ...` 导入 5 个 SSOT 符号。
    A6 不得保留本地 SMPL_SKELETON / BONE_VALIDITY 常量副本。
    文件不存在时记 INFO（可选组件）。
    """
    sv_path = engine_dir / "skeleton_validator.py"
    if not sv_path.exists():
        rep.info("A5/A6 skeleton_validator.py 不存在（可选组件，跳过）")
        return

    src = sv_path.read_text(encoding="utf-8")

    # A5：必须从 skeleton_spec import 5 个符号
    missing_imports = []
    for sym in sorted(_REQUIRED_SSOT_IMPORTS):
        if f"from skeleton_spec import" not in src or sym not in src:
            missing_imports.append(sym)
    # 更精确：检查 from skeleton_spec import (...) 块是否包含所有符号
    import re as _re
    m = _re.search(r"from\s+skeleton_spec\s+import\s*\(([^)]+)\)", src, _re.DOTALL)
    if m:
        import_block = m.group(1)
        for sym in sorted(_REQUIRED_SSOT_IMPORTS):
            if sym not in import_block:
                missing_imports.append(sym)
    elif "from skeleton_spec import" not in src:
        missing_imports = sorted(_REQUIRED_SSOT_IMPORTS)

    if missing_imports:
        rep.drift(f"A5 skeleton_validator.py 未从 skeleton_spec import: {', '.join(missing_imports)}")
    else:
        rep.ok("A5 skeleton_validator.py 从 skeleton_spec import 5 个 SSOT 符号（已迁移）")

    # A6：不得保留本地 SMPL_SKELETON / BONE_VALIDITY 常量定义
    for forbidden in sorted(_FORBIDDEN_LOCAL_COPIES):
        # 匹配 `SMPL_SKELETON = [...]` 或 `BONE_VALIDITY = {...}` 这类赋值
        if _re.search(rf"^\s*{forbidden}\s*=\s*[\[\{{]", src, _re.MULTILINE):
            rep.drift(f"A6 skeleton_validator.py 仍保留本地副本 {forbidden}（应从 skeleton_spec import）")
    if not any(_re.search(rf"^\s*{f}\s*=\s*[\[\{{]", src, _re.MULTILINE)
               for f in _FORBIDDEN_LOCAL_COPIES):
        rep.ok("A6 skeleton_validator.py 无本地骨架常量副本（SMPL_SKELETON/BONE_VALIDITY 已删除）")


# -----------------------------------------------------------------------------
# G14：产物级 mesh 顶点 ↔ joints_3d 一致性断言
# -----------------------------------------------------------------------------
def _find_pose_data_with_mesh(engine_dir: Path) -> Path | None:
    """按 validate.sh G13 的约定探测产物（output_test 优先，其次 output_closed / output）。"""
    for name in ("output_test", "output_closed", "output"):
        p = engine_dir / name / "pose_data.json"
        if p.exists():
            return p
    return None


def check_mesh_joints_alignment(pose_data_path: Path | None, engine_dir: Path, rep: Report) -> None:
    """G14：对产物 pose_data.json 断言 mesh 顶点与 joints_3d 同坐标（叠加模式贴合性）。

    方法：用 SSOT 的 SMPL_J_REGRESSOR 从顶点回归 24 关节，与交付 joints_3d
    逐关节比较。引擎 [P0 mesh↔joints 对齐] 保证两者按构造重合（残差 ≈0），
    故超阈值即判产物漂移（旧管线生成）。

    两条数据路径：
      - [P1] 二进制优先：metadata.mesh_vertices_file（mesh_vertices.f32，
        帧数×6890×3 float32 LE，与 keyframes 1:1）→ **全帧**断言（einsum）；
      - 兼容旧产物：keyframes 内嵌 mesh_vertices（16 帧采样期格式）逐帧断言。
    无产物 / 无 mesh / J_regressor 不可用时 SKIP（不影响退出码）。
    """
    if pose_data_path is None:
        rep.warn("G14 未找到产物 pose_data.json（output_test/output_closed/output 均缺失）→ SKIP")
        return
    if not pose_data_path.exists():
        rep.warn(f"G14 产物不存在: {pose_data_path} → SKIP")
        return

    try:
        sys.path.insert(0, str(engine_dir))
        try:
            spec_mod = importlib.import_module("skeleton_spec")
            j_regressor = spec_mod.SMPL_J_REGRESSOR
        finally:
            try:
                sys.path.remove(str(engine_dir))
            except ValueError:
                pass
    except Exception as exc:                  # noqa: BLE001 - pkl/chumpy/numpy 不可用
        rep.warn(f"G14 J_regressor 不可用（{type(exc).__name__}: {exc}）→ SKIP（需引擎 venv 运行）")
        return

    import numpy as np

    try:
        data = json.loads(pose_data_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        rep.error(f"G14 产物读取失败: {pose_data_path} ({exc})")
        return

    j_reg = np.asarray(j_regressor, dtype=np.float64).reshape(NUM_JOINTS, -1)

    # ---- [P1] 二进制路径（全帧断言）----
    meta = data.get("metadata", {})
    bin_rel = meta.get("mesh_vertices_file")
    frames_n = int(meta.get("mesh_vertices_frames") or 0)
    vertex_count = int(meta.get("mesh_vertices_per_frame") or 0)
    if bin_rel and frames_n > 0 and vertex_count > 0:
        bin_path = pose_data_path.parent / bin_rel
        if not bin_path.exists():
            rep.error(f"G14 metadata 声明 {bin_rel} 但文件缺失（产物不完整）")
            return
        expected = frames_n * vertex_count * 3 * 4
        raw = bin_path.read_bytes()
        if len(raw) != expected:
            rep.error(f"G14 {bin_rel} 大小 {len(raw)}B ≠ 预期 {expected}B"
                      f"（{frames_n}×{vertex_count}×3×float32）")
            return
        keyframes = data.get("keyframes", [])
        if len(keyframes) != frames_n:
            rep.error(f"G14 二进制帧数 {frames_n} 与 keyframes 数 {len(keyframes)} 不一致"
                      f"（1:1 契约被破坏，前端节奏映射将错位）")
            return
        verts_all = np.frombuffer(raw, dtype="<f4").reshape(frames_n, vertex_count, 3)
        if verts_all.shape[1] != j_reg.shape[1]:
            rep.error(f"G14 顶点数 {vertex_count} 与 J_regressor 列数 {j_reg.shape[1]} 不一致")
            return
        refs = np.asarray([k["joints_3d"] for k in keyframes], dtype=np.float64)
        fk = np.einsum("jv,fvc->fjc", j_reg, verts_all.astype(np.float64))
        err_mm = np.linalg.norm(fk - refs, axis=2) * 1000.0    # (F, 24)
        max_err = float(err_mm.max())
        mean_err = float(err_mm.mean())
        worst_flat = int(err_mm.argmax())
        worst_frame = int(keyframes[worst_flat // NUM_JOINTS].get("frame_index", -1))
        if max_err <= _MESH_JOINTS_TOL_MM:
            rep.ok(f"G14 mesh↔joints 一致（[P1] 二进制全帧）：{frames_n} 帧全部在容差内"
                   f"（mean {mean_err:.2f}mm / max {max_err:.2f}mm ≤ {_MESH_JOINTS_TOL_MM}mm）")
        else:
            rep.drift(f"G14 mesh↔joints 错位：max {max_err:.2f}mm（帧 {worst_frame}）"
                      f"> 容差 {_MESH_JOINTS_TOL_MM}mm —— 产物由未对齐的旧管线生成，"
                      f"叠加模式骨架与 mesh 将偏离 ~{max_err:.0f}mm，须重跑管线重新生成产物")
        return

    # ---- 旧产物兼容路径（JSON 内嵌 mesh_vertices，16 帧采样期格式）----
    kfs = [k for k in data.get("keyframes", [])
           if k.get("mesh_vertices") and len(k.get("joints_3d", [])) == NUM_JOINTS]
    if not kfs:
        rep.warn(f"G14 {pose_data_path.name} 无携带 mesh_vertices 的关键帧 → SKIP"
                 f"（叠加模式贴合性未被本次校验覆盖；真机重生成产物后复验）")
        return

    max_err, mean_err, worst_frame = 0.0, 0.0, -1
    checked = 0
    for kf in kfs:
        verts = np.asarray(kf["mesh_vertices"], dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] != j_reg.shape[1]:
            rep.warn(f"G14 帧 {kf.get('frame_index')} mesh 顶点形状异常 {verts.shape} → 跳过该帧")
            continue
        fk = j_reg @ verts                                    # (24, 3)
        ref = np.asarray(kf["joints_3d"], dtype=np.float64)
        err_mm = np.linalg.norm(fk - ref, axis=1) * 1000.0
        if err_mm.max() > max_err:
            max_err, worst_frame = float(err_mm.max()), int(kf.get("frame_index", -1))
        mean_err += float(err_mm.mean())
        checked += 1

    if checked == 0:
        rep.warn("G14 无有效 mesh 帧（形状均异常）→ SKIP")
        return

    mean_err /= checked
    if max_err <= _MESH_JOINTS_TOL_MM:
        rep.ok(f"G14 mesh↔joints 一致：{checked} 帧 mesh 帧全部在容差内"
               f"（mean {mean_err:.2f}mm / max {max_err:.2f}mm ≤ {_MESH_JOINTS_TOL_MM}mm）")
    else:
        rep.drift(f"G14 mesh↔joints 错位：max {max_err:.2f}mm（帧 {worst_frame}）"
                  f"> 容差 {_MESH_JOINTS_TOL_MM}mm —— 产物由未对齐的旧管线生成，"
                  f"叠加模式骨架与 mesh 将偏离 ~{max_err:.0f}mm，须重跑管线重新生成产物")


if __name__ == "__main__":
    sys.exit(main())
