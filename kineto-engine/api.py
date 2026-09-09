#!/usr/bin/env python3
"""
Kineto Engine HTTP Task API
===========================
FastAPI 服务：接收视频 → 排队 → 以子进程方式调用 kineto_core.py → 产出姿态数据与演示视频。

设计要点：
- 串行后台 worker（并发恒为 1）：边缘设备绝不允许并行推理。
- 引擎零改动：kineto_core.py 通过 subprocess 调用，cwd 必须设为 ENGINE_DIR
  （kineto_core.py 内部使用相对路径，见其 113/170/464/813 行）。
- 上线门禁：extraction_mode != '4dhumans' 时判定 failed，拒绝提供合成/假姿态数据。

生产启动: uvicorn api:app --workers 1  （必须单 worker）
本地测试: python api.py
"""

from __future__ import annotations

import asyncio
import fcntl
import hmac
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("kineto.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ----------------------------------------------------------------------------
# [m4] 统一 env 数值读取：未设/空串 → default；非数值 → warning + default。
# 消除散落各处的 `int(os.environ.get(K, D) or 0)` 空串吞噬 bug（空串曾被 `or`
# 吞成 0，与本意“未设走默认”相悖）。WEB_CONCURRENCY 因需 raw 值构造错误信息，
# 保留其独立解析（见下）。
# ----------------------------------------------------------------------------
def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r 非整数，回退默认 %s", name, raw, default)
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r 非数值，回退默认 %s", name, raw, default)
        return default


# ----------------------------------------------------------------------------
# [Fix #11] 单 worker 强制检查
# 状态存储为进程内内存 dict + 单 worker 线程。
# 多 worker 会导致 job 状态丢失且并发推理 OOM。
# ----------------------------------------------------------------------------
_web_concurrency = os.environ.get("WEB_CONCURRENCY", "1")
try:
    _wc_int = int(_web_concurrency)
except ValueError:
    _wc_int = 1
if _wc_int > 1:
    raise RuntimeError(
        f"WEB_CONCURRENCY={_web_concurrency} detected. "
        "Kineto Engine MUST run with a single worker (WEB_CONCURRENCY=1). "
        "Multi-worker causes: (1) in-memory job state loss across processes, "
        "(2) concurrent GPU inference leading to OOM on edge devices. "
        "Fix: set WEB_CONCURRENCY=1 or unset it, and use --workers 1 with uvicorn."
    )

ENGINE_DIR = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("KINETO_JOBS_DIR", ENGINE_DIR / "jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# [Fix #1] Inbox 根目录：JSON video_path 必须在此目录内，防止路径穿越
INBOX_ROOT = Path(os.environ.get("KINETO_INBOX", ENGINE_DIR / "inbox"))
INBOX_ROOT.mkdir(parents=True, exist_ok=True)

# [Fix #2] fail-closed 鉴权配置
API_KEY: Optional[str] = os.environ.get("KINETO_API_KEY")
ALLOW_NO_AUTH: bool = os.environ.get("KINETO_ALLOW_NO_AUTH", "0") == "1"
if API_KEY is None and ALLOW_NO_AUTH:
    logger.warning(
        "[DEV MODE] KINETO_API_KEY not set but KINETO_ALLOW_NO_AUTH=1 — "
        "auth is DISABLED. DO NOT use in production!"
    )

# [Fix #4] 上传大小限制 (MB)
MAX_UPLOAD_MB: int = _int_env("KINETO_MAX_UPLOAD_MB", 200)

# [Fix #6] 有界队列
QUEUE_MAX: int = _int_env("KINETO_QUEUE_MAX", 8)

# [Fix #7] 磁盘剩余空间护栏 (GB)
MIN_FREE_GB: float = _float_env("KINETO_MIN_FREE_GB", 5.0)

# [Fix #8] 最大保留 job 数
MAX_JOBS: int = _int_env("KINETO_MAX_JOBS", 50)

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "KINETO_CORS_ORIGINS", "https://kineto-web.zeabur.app"
    ).split(",")
    if o.strip()
]
REQUIRED_MODE = "4dhumans"  # HARD GO-LIVE GATE：仅接受真实 4DHumans 推理结果

# ----------------------------------------------------------------------------
# [P3/改动 E] 质量门禁配置（与 KINETO_STRICT 彻底解耦）
#   KINETO_QUALITY_GATE = off | warn | fail（默认 warn）
#     off  → 永远 done，且不加质量告警
#     warn → 判决不达标/score<阈值 时仍 done，但附 degraded/quality_warning（默认；
#            即便 deploy 文件尚未在 P5-I 更新也安全：绝不因 STRICT=1 而 mass fail）
#     fail → 不达标才 failed（failed 分支绝不带 quality_score，守前端 JobStatus 契约）
#   KINETO_QUALITY_THRESHOLD：final_quality_score 辅助阈值（默认 0.6，稳妥值——
#     P1/P2 后干净数据诚实分实测 ~0.99 >> 0.6，不会误判好数据；重标定依据
#     quality_log.jsonl 累积日志离线做，属运维动作）。
#   verdict（audit_results.json）为主信号，score 为辅助信号。
# ----------------------------------------------------------------------------
QUALITY_GATE_MODE: str = os.environ.get("KINETO_QUALITY_GATE", "warn").strip().lower()
if QUALITY_GATE_MODE not in ("off", "warn", "fail"):
    logger.warning("KINETO_QUALITY_GATE=%r 非法，回退默认 'warn'", QUALITY_GATE_MODE)
    QUALITY_GATE_MODE = "warn"


def _quality_threshold(default: float = 0.6) -> float:
    raw = os.environ.get("KINETO_QUALITY_THRESHOLD")
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("KINETO_QUALITY_THRESHOLD=%r 非数值，回退默认 %s", raw, default)
        return default


QUALITY_THRESHOLD: float = _quality_threshold()
# [m16] quality_log 默认落 JOBS_DIR（KINETO_JOBS_DIR，持久卷）而非 ENGINE_DIR：
# 容器重建时 ENGINE_DIR 属镜像层/易失，累积日志（数据驱动阈值重标定依据）会丢；
# JOBS_DIR 通常挂持久卷，日志得以跨部署保留。env KINETO_QUALITY_LOG 仍可覆盖。
QUALITY_LOG_PATH = Path(os.environ.get("KINETO_QUALITY_LOG", JOBS_DIR / "quality_log.jsonl"))

# ----------------------------------------------------------------------------
# [P4/改动 G] 长视频天花板缓解：子进程超时按帧数缩放 + quality_log 轮转
# ----------------------------------------------------------------------------
# 【架构级天花板声明】当前架构对超长视频（≥ KINETO_SUPPORTED_FRAME_CEILING，默认
# 9000 帧）存在线性膨胀上限：pose_data.json/mp4 随帧数线性增长（~9.16KB/帧 →
# 54000 帧 ≈ 494MB JSON + ~1.9GB mp4）+ 前端全量 fetch+parse + 子进程耗时。
# 引擎侧已在 kineto_core.process_video 超限响亮告警（stderr）；api 侧的核心轻量缓解
# 是超时按帧数缩放（防长视频误超时失败）。分块/流式交付、帧率降采样为未来项。
#
# 超时公式：timeout = clamp(FLOOR, BASE + n*PER_FRAME, MAX)：
#   - 短视频（n 小）→ 命中 FLOOR=7200s，**与当前行为逐位一致**（守无回归底线）；
#   - 长视频 → 按帧数线性放宽，避免误超时。帧数经 cv2 探测；探测失败回退 FLOOR。
# 全部 env 可配。
TIMEOUT_FLOOR_SEC = _int_env("KINETO_TIMEOUT_FLOOR_SEC", 7200)
TIMEOUT_BASE_SEC = _int_env("KINETO_TIMEOUT_BASE_SEC", 1800)
TIMEOUT_PER_FRAME_SEC = _float_env("KINETO_TIMEOUT_PER_FRAME_SEC", 0.8)
TIMEOUT_MAX_SEC = _int_env("KINETO_TIMEOUT_MAX_SEC", 21600)

# quality_log.jsonl 轮转上限（Chris 观察项：单文件无限增长）。best-effort：超过
# KINETO_QUALITY_LOG_MAX_BYTES（默认 5MB）时轮转为 .jsonl.1（仅保留一份备份），
# 绝不因轮转失败影响 job 结局。<=0 关闭轮转（保持旧的无限增长行为）。
# [m4] 用 _int_env 统一空串语义：未设/空串 → 默认 5MB；显式 "0" → 0（关轮转）。
# 旧 `int(os.environ.get(K, "5242880") or 0)` 会把空串吞成 0（误关轮转），已修正。
QUALITY_LOG_MAX_BYTES = _int_env("KINETO_QUALITY_LOG_MAX_BYTES", 5 * 1024 * 1024)

# [m17] 架构级支持帧数上限（与 kineto_core.process_video 告警阈值同源，默认 9000）。
# 超限 job 在 create_job 入队前拒绝（413），避免线性膨胀产物拖垮前端/磁盘。
SUPPORTED_FRAME_CEILING = _int_env("KINETO_SUPPORTED_FRAME_CEILING", 9000)


def _probe_frame_count(video_path: Path) -> Optional[int]:
    """用 cv2 探测视频帧数（best-effort）。失败返回 None（调用方回退保守超时）。
    延迟 import cv2，避免 api 常驻进程无谓引入重依赖（cv2 缺失时也安全回退）。
    [m3] try/finally 确保 VideoCapture 释放——旧写法 cap.get 抛异常时句柄泄漏
    （cv2 import 失败时 cap 仍为 None，finally 安全跳过）。"""
    cap = None
    try:
        import cv2  # 延迟导入
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        if cap is not None:
            cap.release()


def _engine_timeout(video_path: Path) -> int:
    """按帧数缩放子进程超时（秒）。短视频命中 FLOOR（=当前 7200s 行为），长视频放宽。"""
    n = _probe_frame_count(video_path)
    if not n:
        return TIMEOUT_FLOOR_SEC
    scaled = TIMEOUT_BASE_SEC + n * TIMEOUT_PER_FRAME_SEC
    return int(max(TIMEOUT_FLOOR_SEC, min(TIMEOUT_MAX_SEC, scaled)))


def _rotate_quality_log_if_needed() -> None:
    """quality_log.jsonl 轮转（best-effort）：超过 QUALITY_LOG_MAX_BYTES 时改名为
    .jsonl.1（原子覆盖旧备份，仅保留一份），再让调用方新建空文件继续追加。绝不抛出。"""
    if QUALITY_LOG_MAX_BYTES <= 0:
        return
    try:
        if QUALITY_LOG_PATH.exists() and QUALITY_LOG_PATH.stat().st_size > QUALITY_LOG_MAX_BYTES:
            backup = Path(str(QUALITY_LOG_PATH) + ".1")
            os.replace(QUALITY_LOG_PATH, backup)  # 原子覆盖旧备份
            logger.info("quality_log 轮转：%s → %s（>%.1fMB）",
                        QUALITY_LOG_PATH.name, backup.name, QUALITY_LOG_MAX_BYTES / 1e6)
    except Exception as exc:  # noqa: BLE001
        logger.warning("quality_log 轮转失败（不影响 job）: %s", exc)


# ----------------------------------------------------------------------------
# Job 持久化：从磁盘恢复已完成 job 的注册表（引擎重启后 job 不丢失）
# ----------------------------------------------------------------------------
def _restore_jobs_from_disk(jobs_dir: Path) -> dict[str, dict[str, Any]]:
    """扫描 JOBS_DIR，恢复已完成 job 的注册表项。

    恢复策略：
    - 有 pose_data.json → 已完成（done），从 metadata 重建状态
    - 有 input.mp4 但无 pose_data.json → 中断（failed），标记失败
    - 其他情况 → 跳过（可能是临时目录或无关文件）

    恢复的 job 为只读状态，不会重新入队处理。
    """
    restored: dict[str, dict[str, Any]] = {}
    if not jobs_dir.exists():
        return restored

    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
        # 跳过隐藏文件/锁文件等
        if job_dir.name.startswith("."):
            continue

        job_id = job_dir.name
        pose_data_path = job_dir / "pose_data.json"

        if pose_data_path.exists():
            # ---- 已完成 job：从 pose_data.json 重建状态 ----
            try:
                with open(pose_data_path, encoding="utf-8") as f:
                    pose_data = json.load(f)
                metadata = pose_data.get("metadata", {})
                pipeline = metadata.get("pipeline", {})

                restored[job_id] = {
                    "id": job_id,
                    "state": "done",
                    "progress": 1.0,
                    "extraction_mode": metadata.get("extraction_mode"),
                    "quality_score": pipeline.get("final_quality_score",
                                                   metadata.get("final_quality_score")),
                    "total_frames": metadata.get("total_frames"),
                    "degraded": metadata.get("degraded", False),
                    "quality_warning": metadata.get("quality_warning", False),
                    "video_md5": metadata.get("video_md5"),
                    "video_path": str(job_dir / "input.mp4"),
                    "jobdir": str(job_dir),
                    "created_at": job_dir.stat().st_ctime,
                    "error": None,
                    "_restored": True,  # 标记为恢复 job，防止误操作
                }
            except Exception as exc:
                logger.warning("恢复 job %s 失败: %s", job_id, exc)
        elif (job_dir / "input.mp4").exists():
            # ---- 中断 job：有输入但无产物，标记 failed 并清理不完整文件 ----
            logger.warning("job %s: 检测到中断 job（有 input.mp4 无 pose_data.json），标记 failed", job_id)

            # 清理不完整的 job 文件（input.mp4 等），释放磁盘空间
            cleaned_files: list[str] = []
            try:
                for f in job_dir.iterdir():
                    if f.is_file():
                        f.unlink()
                        cleaned_files.append(f.name)
                # 尝试删除空目录本身
                job_dir.rmdir()
                logger.info("job %s: 已清理中断 job 文件: %s", job_id, ", ".join(cleaned_files) or "(空目录)")
            except Exception as cleanup_exc:
                logger.warning("job %s: 清理中断 job 文件部分失败: %s", job_id, cleanup_exc)
                # 清理失败仍注册 job 为 failed（内存态占位，evict 机制会处理磁盘残留）

            restored[job_id] = {
                "id": job_id,
                "state": "failed",
                "progress": 0.0,
                "extraction_mode": None,
                "quality_score": None,
                "total_frames": None,
                "degraded": False,
                "quality_warning": False,
                "video_path": None,
                "jobdir": str(job_dir),
                "created_at": job_dir.stat().st_ctime if job_dir.exists() else time.time(),
                "error": "engine restarted before job completed",
                "_restored": True,
                "_cleaned_files": cleaned_files,
            }

    return restored


# ----------------------------------------------------------------------------
# 任务状态存储（内存 dict + 磁盘 job 目录；有界 queue.Queue 防 OOM）
# ----------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue(maxsize=QUEUE_MAX)  # [Fix #6]
_last_extraction_mode: Optional[str] = None


def _set_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
    return job


def _evict_old_jobs() -> None:
    """[Fix #8] 保留期清理：超过 MAX_JOBS 时按最旧优先删除 done/failed job。
    绝不裁剪 queued/running 状态的 job。
    [MJ5] rmtree 在锁外执行，避免阻塞事件循环。"""
    dirs_to_remove: list[Path] = []
    with _jobs_lock:
        if len(_jobs) <= MAX_JOBS:
            return
        # 收集可安全删除的终态 job，按 created_at 升序
        evictable = [
            (jid, j) for jid, j in _jobs.items()
            if j.get("state") in ("done", "failed")
        ]
        evictable.sort(key=lambda x: x[1].get("created_at", 0))
        to_remove = len(_jobs) - MAX_JOBS
        removed = 0
        jobs_root = JOBS_DIR.resolve()
        for jid, j in evictable:
            if removed >= to_remove:
                break
            # [m5] 删除前校验 jobdir 为绝对路径且在 JOBS_DIR 内——消除
            # Path("")→rmtree(cwd) 或路径穿越误删隐患。非法路径只从内存摘除、不删盘。
            jd = j.get("jobdir")
            if jd:
                p = Path(jd)
                if p.is_absolute() and p.resolve().is_relative_to(jobs_root):
                    dirs_to_remove.append(p)
                else:
                    logger.error("evict: 拒绝删除非法 jobdir（非绝对路径或越界）: %r", jd)
            del _jobs[jid]
            removed += 1
        if removed:
            logger.info("evicted %d old done/failed jobs (limit=%d)", removed, MAX_JOBS)
    # 锁外执行磁盘删除
    for d in dirs_to_remove:
        shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------
# [P3/改动 E] 质量门禁辅助（verdict 主信号 + score 辅信号，与 STRICT 解耦）
# ----------------------------------------------------------------------------
def _load_audit_verdict(jobdir: Path) -> Optional[dict]:
    """读取最高 iteration 的 audit_results.json 机器可读判决（verdict 主门禁信号）。
    找不到/解析失败返回 None（门禁退化为仅 score 辅助信号）。"""
    def _iter_key(p: Path) -> int:
        try:
            return int(p.name.replace("audit_iter", ""))
        except ValueError:
            return -1
    for d in sorted(jobdir.glob("audit_iter*"), key=_iter_key, reverse=True):
        f = d / "audit_results.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and "verdict" in data:
            return {
                "verdict": data.get("verdict"),
                "total_issues": data.get("total_issues"),
                "failure_reason": data.get("failure_reason"),
            }
    return None


def _decide_quality_gate(verdict: Optional[str], score: Any, gate_mode: str,
                         threshold: float) -> tuple[str, bool, bool, Optional[str]]:
    """纯函数门禁裁决（与 KINETO_STRICT 完全解耦，便于单测行为矩阵）。
    主信号 verdict（pass|warn|fail|None），辅信号 score<threshold。
    返回 (outcome, degraded, quality_warning, reason)，outcome ∈ {'done','failed'}。
      off / 达标            → ('done', False, False, None)
      不达标 + warn（默认） → ('done', True, True, reason)     # 仍 done 但显式暴露
      不达标 + fail          → ('failed', False, False, reason) # failed 不带质量标志/分数
    “不达标” = verdict ∈ {warn,fail} 或 score<threshold（绝不静默放行）。
    """
    verdict_bad = verdict in ("warn", "fail")
    score_bad = (isinstance(score, (int, float)) and not isinstance(score, bool)
                 and float(score) < float(threshold))
    quality_bad = verdict_bad or score_bad

    reasons = []
    if verdict == "fail":
        reasons.append("audit verdict=fail")
    elif verdict == "warn":
        reasons.append("audit verdict=warn")
    if score_bad:
        reasons.append(f"final_quality_score={score} < threshold={threshold}")
    reason = "; ".join(reasons) if reasons else None

    if gate_mode == "off" or not quality_bad:
        return "done", False, False, None
    if gate_mode == "fail":
        return "failed", False, False, reason
    return "done", True, True, reason   # warn（默认）


def _append_quality_log(job_id: str, score: Any, verdict_info: Optional[dict],
                        gate_mode: str, threshold: float, outcome: str,
                        extraction_mode: Optional[str]) -> None:
    """quality_log.jsonl 追加：每 job 记 final_quality_score + verdict + 关键信号 +
    门禁裁决，为数据驱动阈值重标定（P10/P50）提供依据。best-effort：任何异常
    只告警、绝不影响 job 结局。"""
    entry = {
        "ts": round(time.time(), 3),
        "job_id": job_id,
        "extraction_mode": extraction_mode,
        "final_quality_score": score,
        "verdict": (verdict_info or {}).get("verdict"),
        "total_issues": (verdict_info or {}).get("total_issues"),
        "failure_reason": (verdict_info or {}).get("failure_reason"),
        "gate_mode": gate_mode,
        "threshold": threshold,
        "outcome": outcome,
    }
    try:
        _rotate_quality_log_if_needed()
        with open(QUALITY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("quality_log 追加失败（不影响 job）: %s", exc)


# ----------------------------------------------------------------------------
# 串行 worker：消费队列，逐个跑推理子进程
# ----------------------------------------------------------------------------
def _run_engine(job_id: str, video_path: Path, jobdir: Path) -> None:
    global _last_extraction_mode
    cmd = [sys.executable, "kineto_core.py", "--input", str(video_path), "--output", str(jobdir)]
    _set_job(job_id, state="running", progress=0.1, started_at=time.time())
    logger.info("job %s: subprocess start: %s (cwd=%s)", job_id, " ".join(cmd), ENGINE_DIR)
    # [P4/改动 G] 超时按帧数缩放：短视频命中 FLOOR(=原 7200s)，长视频放宽防误超时。
    _timeout = _engine_timeout(video_path)
    logger.info("job %s: engine timeout=%ss（帧数缩放，floor=%ss）", job_id, _timeout, TIMEOUT_FLOOR_SEC)
    try:
        proc = subprocess.run(cmd, cwd=ENGINE_DIR, capture_output=True, text=True, timeout=_timeout)
    except subprocess.TimeoutExpired:
        _set_job(job_id, state="failed", progress=0.0,
                 error=f"engine subprocess timed out (>{_timeout}s)")
        return
    except Exception as exc:  # noqa: BLE001
        _set_job(job_id, state="failed", progress=0.0, error=f"failed to launch engine: {exc}")
        return

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        _set_job(job_id, state="failed", progress=0.0,
                 error=f"engine exited with code {proc.returncode}: {tail}")
        return

    # 解析产物元数据（注意：final_quality_score 位于 metadata.pipeline 下）
    pose_json = jobdir / "pose_data.json"
    if not pose_json.exists():
        _set_job(job_id, state="failed", progress=0.0, error="engine finished but pose_data.json is missing")
        return
    try:
        meta = json.loads(pose_json.read_text(encoding="utf-8")).get("metadata", {})
    except Exception as exc:  # noqa: BLE001
        _set_job(job_id, state="failed", progress=0.0, error=f"pose_data.json unreadable: {exc}")
        return

    mode = meta.get("extraction_mode")
    pipeline = meta.get("pipeline") or {}
    quality = pipeline.get("final_quality_score", meta.get("final_quality_score"))
    total_frames = meta.get("total_frames")
    _last_extraction_mode = mode

    if mode != REQUIRED_MODE:
        # 上线门禁：引擎静默回退到合成关键点时拒绝交付假数据
        err = ("engine fell back to non-4dhumans mode (likely missing weights); "
               f"refusing to serve fake pose data (extraction_mode={mode!r})")
        logger.error("job %s: %s", job_id, err)
        _set_job(job_id, state="failed", progress=0.0, extraction_mode=mode, error=err)
        return

    # ------------------------------------------------------------------
    # [P3/改动 E] 质量门禁：以审计判决 verdict 为主信号 + final_quality_score 为辅，
    # 由独立的 KINETO_QUALITY_GATE（off|warn|fail，默认 warn）裁决。
    # **与 KINETO_STRICT 彻底解耦**：STRICT 不再触发任何质量失败（其语义收敛为
    # kineto_core 侧假数据/缺权重 fail-fast，见 F）。这消除了 deploy 烘了
    # KINETO_STRICT=1 时 done→failed 的大面积回归——默认 warn 下即便 deploy 文件
    # 尚未更新也安全。不变量：交付结局不得与审计判决静默矛盾（warn 模式必附
    # degraded/quality_warning 显式暴露给前端）。
    # ------------------------------------------------------------------
    verdict_info = _load_audit_verdict(jobdir)
    verdict = (verdict_info or {}).get("verdict")
    outcome, degraded, warning, reason = _decide_quality_gate(
        verdict, quality, QUALITY_GATE_MODE, QUALITY_THRESHOLD)

    _append_quality_log(job_id, quality, verdict_info, QUALITY_GATE_MODE,
                        QUALITY_THRESHOLD, outcome, mode)

    if outcome == "failed":
        # failed 分支绝不带 quality_score（前端 JobStatus 契约：quality_score 是完成后字段）
        err = f"quality gate (KINETO_QUALITY_GATE=fail): {reason}"
        logger.error("job %s: %s", job_id, err)
        _set_job(job_id, state="failed", progress=0.0, extraction_mode=mode, error=err)
        return

    _set_job(job_id, state="done", progress=1.0, quality_score=quality,
             extraction_mode=mode, total_frames=total_frames,
             degraded=degraded, quality_warning=warning,
             finished_at=time.time())
    logger.info("job %s: done (mode=%s, verdict=%s, quality=%s, gate=%s, degraded=%s, frames=%s)",
                job_id, mode, verdict, quality, QUALITY_GATE_MODE, degraded, total_frames)


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        # [Fix #12] 在锁内快照读取 job 字段，与 _set_job/_get_job 加锁纪律一致
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                _queue.task_done()
                continue
            video_path_str = job["video_path"]
            jobdir_str = job["jobdir"]
        try:
            _run_engine(job_id, Path(video_path_str), Path(jobdir_str))
        except Exception as exc:  # noqa: BLE001 — worker 永不因单个任务崩溃
            logger.exception("job %s: worker error", job_id)
            _set_job(job_id, state="failed", progress=0.0, error=f"worker error: {exc}")
        finally:
            _queue.task_done()


# ----------------------------------------------------------------------------
# [MJ7] Worker 单实例守卫：flock 防止多进程重复处理任务
# WEB_CONCURRENCY 检查覆盖不了 `uvicorn --workers N` 场景，flock 互补。
# ----------------------------------------------------------------------------
_worker_lock_path = JOBS_DIR / ".worker.lock"
_worker_lock_fd = open(_worker_lock_path, "w")  # noqa: SIM115
try:
    fcntl.flock(_worker_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except (OSError, BlockingIOError):
    raise RuntimeError(
        f"Another Kineto Engine worker already holds {_worker_lock_path}. "
        "Only ONE engine process may run at a time (single GPU, in-memory state). "
        "Kill the existing process or remove the stale lock file."
    )
_worker_lock_fd.write(str(os.getpid()))
_worker_lock_fd.flush()

_worker = threading.Thread(target=_worker_loop, name="kineto-engine-worker", daemon=True)
_worker.start()  # 唯一 worker，保证并发 = 1

# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
def _prewarm_cv2() -> None:
    """[m17] cv2 启动预热：api 常驻进程默认延迟 import cv2（见 _probe_frame_count），
    首个 job 的帧数探测会付一次 cv2 import/后端初始化成本。启动时在线程池预热，
    把该成本移出请求路径。best-effort：预热失败只告警，不影响服务（探测会安全回退）。"""
    try:
        import cv2  # noqa: PLC0415
        cap = cv2.VideoCapture()  # 触发 cv2 视频后端初始化
        cap.release()
        logger.info("cv2 预热完成 (version=%s)", getattr(cv2, "__version__", "?"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("cv2 预热失败（不影响服务，_probe_frame_count 会安全回退）: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # [m17] 启动预热 cv2（线程池，不阻塞事件循环）；关闭时无需清理。
    await asyncio.to_thread(_prewarm_cv2)

    # ---- Job 持久化恢复：从磁盘扫描已完成/中断的 job ----
    def _do_restore() -> dict[str, dict[str, Any]]:
        return _restore_jobs_from_disk(JOBS_DIR)

    restored = await asyncio.to_thread(_do_restore)
    if restored:
        with _jobs_lock:
            _jobs.update(restored)
        done_count = sum(1 for j in restored.values() if j["state"] == "done")
        failed_count = sum(1 for j in restored.values() if j["state"] == "failed")
        logger.info(
            "从磁盘恢复了 %d 个 job（done=%d, failed=%d）",
            len(restored), done_count, failed_count,
        )
    else:
        logger.info("磁盘上无可恢复的 job")

    yield


app = FastAPI(title="Kineto Engine API", version="1.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """[Fix #2][#3] fail-closed 鉴权 + hmac 常量时间比较。

    逻辑：
    - KINETO_API_KEY 未设置时：
      - 若 KINETO_ALLOW_NO_AUTH=="1" → 放行（dev 模式，启动时已警告）
      - 否则 → 503 "auth not configured"（fail-closed，杜绝静默全放行）
    - KINETO_API_KEY 已设置时：用 hmac.compare_digest 做常量时间比较，防时序攻击
    """
    if API_KEY is None:
        if ALLOW_NO_AUTH:
            return  # 显式 opt-in 的 dev 模式
        raise HTTPException(
            status_code=503,
            detail="auth not configured: KINETO_API_KEY is not set. "
                   "Set KINETO_API_KEY or KINETO_ALLOW_NO_AUTH=1 for dev.",
        )
    # [Fix #3] 使用 hmac.compare_digest 防时序侧信道攻击
    if not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _detect_device() -> tuple[str, bool]:
    """惰性 import torch，避免 /health 因 torch 加载慢而卡死。"""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda", True
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu", True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", True
        return "cpu", True
    except Exception:  # noqa: BLE001
        return "unknown", False


def _check_model_loaded(torch_ok: bool) -> bool:
    """[MJ6] 检查 4DHumans HMR2 模型权重是否完整就绪。

    仅当以下三项全部存在时返回 True：
    1. .ckpt 权重文件（4DHumans 缓存目录下）
    2. model_config.yaml（模型配置）
    3. SMPL body model .pkl（SMPL 数据）

    不再用 yolov8n.pt 兜底（已拆分为 detector_loaded）。
    """
    if not torch_ok:
        return False

    cache_dir = Path(os.environ.get("CACHE_DIR_4DHUMANS", Path.home() / ".cache" / "4DHumans"))
    if not cache_dir.is_dir():
        return False

    # 4DHumans 典型布局: logs/train/multiruns/hmr2/.../checkpoints/epoch%3D35-step%3D1000000.ckpt
    has_ckpt = any(cache_dir.rglob("*.ckpt"))
    has_config = any(cache_dir.rglob("model_config.yaml"))
    smpl_dir = Path(os.environ.get("SMPL_DATA_DIR", Path.home() / ".cache" / "4DHumans" / "data"))
    has_smpl = any(smpl_dir.rglob("*.pkl")) if smpl_dir.is_dir() else False

    return has_ckpt and has_config and has_smpl


def _check_detector_loaded() -> bool:
    """[MJ6] 检查 YOLOv8n 人体检测器权重是否存在。"""
    return (ENGINE_DIR / "yolov8n.pt").exists()


# [Fix #9] /healthz — 公开极简存活探针，无敏感信息，无需鉴权
@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# [Fix #9] /health — 详细拓扑，需鉴权
@app.get("/health", dependencies=[Depends(require_api_key)])
def health() -> JSONResponse:
    device, torch_ok = _detect_device()
    # [MJ6] model_loaded 仅反映 4DHumans 权重完整性；detector_loaded 反映 YOLO
    model_loaded = _check_model_loaded(torch_ok)
    detector_loaded = _check_detector_loaded()
    return JSONResponse({
        "status": "ok" if torch_ok else "degraded",
        "extraction_mode": _last_extraction_mode,
        "device": device,
        "model_loaded": model_loaded,
        "detector_loaded": detector_loaded,
        "queue_depth": _queue.qsize(),
    })


@app.post("/jobs", dependencies=[Depends(require_api_key)])
async def create_job(request: Request) -> JSONResponse:
    """接收 multipart 'video' 文件或 JSON {video_path}；立即返回 job_id，绝不阻塞推理。"""

    # [MJ5] 先 evict 释放空间 → 再查磁盘；若仍不足 → 再 evict + 复查
    _evict_old_jobs()

    disk_usage = shutil.disk_usage(str(JOBS_DIR))
    free_gb = disk_usage.free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        # 再尝试一轮清理后复查
        _evict_old_jobs()
        await asyncio.to_thread(lambda: None)  # yield 让删除 I/O 有机会完成
        disk_usage = shutil.disk_usage(str(JOBS_DIR))
        free_gb = disk_usage.free / (1024 ** 3)
        if free_gb < MIN_FREE_GB:
            raise HTTPException(
                status_code=507,
                detail=f"insufficient disk space: {free_gb:.1f}GB free < {MIN_FREE_GB}GB required",
            )

    # [Fix #6] 有界队列：满时立即拒绝
    if _queue.full():
        raise HTTPException(
            status_code=429,
            detail=f"job queue is full (max={QUEUE_MAX}). Retry later.",
            headers={"Retry-After": "30"},
        )

    job_id = uuid.uuid4().hex
    jobdir: Optional[Path] = None  # [Fix #5] 延迟创建，仅在 video 校验通过后

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("video")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="multipart field 'video' (mp4) is required")

        # [Fix #5] video 校验通过后才创建 jobdir
        jobdir = JOBS_DIR / job_id
        jobdir.mkdir(parents=True, exist_ok=True)
        local_video = jobdir / "input.mp4"

        # [Fix #4] 流式写入 + 字节计数，超限则 413
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        written = 0
        try:
            with open(local_video, "wb") as f:
                while chunk := await upload.read(1 << 20):
                    written += len(chunk)
                    if written > max_bytes:
                        f.close()
                        # 删除已写部分文件
                        local_video.unlink(missing_ok=True)
                        shutil.rmtree(jobdir, ignore_errors=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload exceeds {MAX_UPLOAD_MB}MB limit "
                                   f"(received {written / 1024 / 1024:.1f}MB so far)",
                        )
                    f.write(chunk)
        except HTTPException:
            raise
        except Exception as exc:
            # 写入过程中发生其他错误 → 清理孤儿目录
            shutil.rmtree(jobdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"failed to save upload: {exc}")
    else:
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400,
                                detail="send multipart file field 'video' or JSON {video_path}")
        video_path = (payload or {}).get("video_path")
        if not video_path:
            raise HTTPException(status_code=400,
                                detail="send multipart file field 'video' or JSON {video_path}")

        # [Fix #1] 路径穿越防护：video_path 必须解析到 INBOX_ROOT 内
        resolved = (ENGINE_DIR / video_path).resolve()
        inbox_resolved = INBOX_ROOT.resolve()
        if not resolved.is_relative_to(inbox_resolved):
            raise HTTPException(
                status_code=403,
                detail=f"video_path must be within inbox directory ({INBOX_ROOT}). "
                       f"Resolved path '{resolved}' is outside allowed root.",
            )
        if not resolved.exists():
            raise HTTPException(status_code=400, detail=f"video_path not found: {resolved}")

        # [Fix #5] video 校验通过后才创建 jobdir
        jobdir = JOBS_DIR / job_id
        jobdir.mkdir(parents=True, exist_ok=True)
        local_video = resolved  # 本地路径直接使用，不复制

    # [MN7] 显式检查代替 assert（python -O 会剥离 assert）
    if jobdir is None:
        raise HTTPException(status_code=500, detail="internal error: jobdir not initialized")

    # [m17] 超长视频入队前拒绝：帧数 > SUPPORTED_FRAME_CEILING（默认 9000）的 job
    # 会在引擎侧产出线性膨胀的 pose_data.json/mp4（~9.16KB/帧），拖垮前端全量
    # fetch+parse 与磁盘。入队前用 cv2 探测帧数、超限即拒绝（413），与 kineto_core
    # 的响亮告警形成 api 侧前置门禁。探测失败(None)不拒绝（回退引擎侧告警），
    # 避免误杀无法探测帧数的合法视频。
    n_frames = _probe_frame_count(local_video)
    if n_frames is not None and n_frames > SUPPORTED_FRAME_CEILING:
        shutil.rmtree(jobdir, ignore_errors=True)
        raise HTTPException(
            status_code=413,
            detail=f"video has {n_frames} frames, exceeds supported ceiling "
                   f"{SUPPORTED_FRAME_CEILING}. Split into shorter segments.",
        )

    with _jobs_lock:
        _jobs[job_id] = {
            "state": "queued", "progress": 0.0, "quality_score": None,
            "extraction_mode": None, "error": None,
            "degraded": False, "quality_warning": False,
            "video_path": str(local_video), "jobdir": str(jobdir),
            "created_at": time.time(),
        }

    # [Fix #6] 入队；理论上前面已检查 full()，但用 put_nowait 做双保险
    try:
        _queue.put_nowait(job_id)
    except queue.Full:
        # 极端竞态：回滚 job 状态
        with _jobs_lock:
            _jobs.pop(job_id, None)
        shutil.rmtree(jobdir, ignore_errors=True)
        raise HTTPException(
            status_code=429,
            detail=f"job queue is full (max={QUEUE_MAX}). Retry later.",
            headers={"Retry-After": "30"},
        )

    logger.info("job %s: queued (queue_depth=%d)", job_id, _queue.qsize())
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def job_status(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    body = {
        "state": job["state"],
        "progress": job["progress"],
        "extraction_mode": job.get("extraction_mode"),
    }
    # [P3/改动 E] quality_score 是完成后字段：仅当非 None（done）时下发，守前端
    # JobStatus 契约——failed/queued/running 绝不带 quality_score。
    if job.get("quality_score") is not None:
        body["quality_score"] = job["quality_score"]
    # [Fix #15-4] 附加降级标志（仅当为真时下发，不改变健康 job 的响应形状）。
    # 前端轮询契约仅读 state/progress/quality_score，新增字段为纯附加、向后兼容。
    # [n2] 分别按各自存储值下发 degraded / quality_warning，不再硬编码 quality_warning=True。
    # 二者语义可分离（degraded=交付降级；quality_warning=审计告警），读实际值更诚实。
    if job.get("degraded"):
        body["degraded"] = True
    if job.get("quality_warning"):
        body["quality_warning"] = True
    if job.get("error"):
        body["error"] = job["error"]
    return JSONResponse(body)


def _serve_artifact(job_id: str, name: str) -> FileResponse:
    job = _get_job(job_id)
    if job["state"] != "done":
        raise HTTPException(status_code=409, detail=f"job is {job['state']}, artifact not available")
    path = Path(job["jobdir"]) / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found for job {job_id}")
    return FileResponse(path)


@app.get("/jobs/{job_id}/pose_data.json", dependencies=[Depends(require_api_key)])
def get_pose_data(job_id: str) -> FileResponse:
    return _serve_artifact(job_id, "pose_data.json")


@app.get("/jobs/{job_id}/mesh_vertices.f32", dependencies=[Depends(require_api_key)])
def get_mesh_vertices(job_id: str) -> FileResponse:
    """SMPL 顶点二进制（[P1 mesh 节奏贴合]：帧数×6890×3 float32 LE，与 keyframes 1:1）。

    帧数/顶点数以 pose_data.json metadata 的 mesh_vertices_frames /
    mesh_vertices_per_frame 为准。
    """
    return _serve_artifact(job_id, "mesh_vertices.f32")


@app.get("/jobs/{job_id}/demo_output.mp4", dependencies=[Depends(require_api_key)])
def get_demo_video(job_id: str) -> FileResponse:
    return _serve_artifact(job_id, "demo_output.mp4")


@app.get("/jobs/{job_id}/annotated_output.mp4", dependencies=[Depends(require_api_key)])
def get_annotated_video(job_id: str) -> FileResponse:
    """获取骨骼标注视频"""
    return _serve_artifact(job_id, "annotated_output.mp4")


@app.get("/jobs/{job_id}/input.mp4", dependencies=[Depends(require_api_key)])
def get_input_video(job_id: str) -> FileResponse:
    """返回原始上传视频"""
    job_dir = JOBS_DIR / job_id
    input_path = job_dir / "input.mp4"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="input video not found")
    return FileResponse(input_path, media_type="video/mp4")


@app.get("/jobs/{job_id}/{image_name}", dependencies=[Depends(require_api_key)])
def get_grid_image(job_id: str, image_name: str) -> FileResponse:
    """获取四宫格教学图（grid_01.jpg ~ grid_04.jpg）"""
    import re
    if not re.match(r'^grid_\d{2}\.jpg$', image_name):
        # 不是 grid 图片，返回 404
        raise HTTPException(status_code=404, detail=f"unknown artifact: {image_name}")
    return _serve_artifact(job_id, image_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
