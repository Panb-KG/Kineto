#!/usr/bin/env bash
# =============================================================================
# Kineto / deploy — validate.sh   (端到端验收)
# -----------------------------------------------------------------------------
# 运行处 : 设备上（默认，走 127.0.0.1），或 Mac 上（走 Cloudflare Tunnel 公网地址）
# 作用   : 把「引擎真的能用」拆成可判定的门禁，逐条给出 PASS/FAIL：
#            G1  GET  /healthz + /health → /healthz 公开存活；/health 需 X-API-Key，200 且含 extraction_mode
#            G1a device 字段            → Arc 上必须为 xpu；device=cpu 判 **FAIL**（api.py 已支持 xpu）
#            G2  XPU 可用性              → torch.xpu.is_available() == True（尽力而为）
#            G3  POST /jobs              → 202 且拿到 job_id
#            G4  轮询 /jobs/{id}         → 最终 state == done（并计时）
#            G5  pose_data.json          → metadata.extraction_mode == "4dhumans"
#            G6  质量分                  → metadata.pipeline.final_quality_score >= 阈值
#                                          （默认 0.6，与引擎 KINETO_QUALITY_THRESHOLD 同源）
#            G7  demo_output.mp4         → 可下载且非空
#            G8  无 OOM                  → dmesg/journal 中无本次运行期间的 oom-kill
#            G9  重启持久化              → 服务 enabled / restart policy 正确
#            G10 MoFang 回归             → 见 --mofang-mode
#            G11 Zeabur 前端可达性       → 见 --web-base（新代理架构：浏览器不再直连引擎）
#            G12 骨架 SSOT 一致性        → kineto-engine/skeleton_spec.py（SSOT）↔
#                                          kineto-web/lib/skeleton.ts（前端镜像）逐项一致：
#                                          SMPL_JOINT_NAMES(24,顺序) / SMPL_PARENTS(13,14→9) /
#                                          SMPL_SKELETON(23 边,collar 父=9) / BONE_PART_MAP。
#                                          ★ 计划文档里称 **G8-SSOT**；因本脚本既有 G8=OOM 门禁，
#                                            为不破坏既有编号与文档引用，编为 G12 并保留该别名。
#                                          ★ **一致性检查，不是 codegen**：漂移即 FAIL（由前端 Owner 对齐）。
#                                          ★ 静态、只读、不需要引擎在线：可单跑 `--ssot-only`。
#                                            设备上通常没有 kineto-web/ → 记 SKIP（请在仓库检出上跑）。
# 安全性 : 只发 HTTP 请求 + 只读查询；不启停服务、不改配置、不删文件、不生成代码。
#          唯一写入是 /tmp 下的临时产物（校验用 pose_data.json / mp4），退出时清理。
# 幂等性 : 可重复运行；每次会新建一个 job（这是 API 的设计，无副作用）。
#
# 用法:
#   # 设备上（systemd 主路径）
#   KINETO_API_KEY=xxx bash deploy/validate.sh
#   bash deploy/validate.sh --base http://127.0.0.1:8000 --video /opt/kineto/kineto-engine/input_video.mp4
#
#   # Mac 上穿隧道验收（multipart 上传，顺带验证 100MB 限制与 CORS 之外的链路）
#   bash deploy/validate.sh --base https://kineto-api.example.com --upload \
#        --video ~/Projects/Kineto/videos/input_video.mp4 --api-key xxx
#
#   # 已按 OPTION (i) 剥离 MoFang 时（回归检查转为“确认真的停了”）
#   bash deploy/validate.sh --mofang-mode stripped
#
#   # 只跑静态骨架一致性闸门（G12/G8-SSOT）：不需要引擎在线、不发任何 HTTP 请求
#   bash deploy/validate.sh --ssot-only
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
BASE="${KINETO_BASE:-http://127.0.0.1:8000}"
API_KEY="${KINETO_API_KEY:-}"
VIDEO="${KINETO_VIDEO:-}"
POLL_INTERVAL="${KINETO_POLL_INTERVAL:-10}"
TIMEOUT="${KINETO_TIMEOUT:-3600}"
# 质量分门禁阈值：与引擎侧 KINETO_QUALITY_THRESHOLD **同源**（引擎默认 0.6）。
# 优先级：--min-quality > KINETO_MIN_QUALITY > KINETO_QUALITY_THRESHOLD > 0.6。
# 注意语义差别：引擎侧 KINETO_QUALITY_GATE 默认 **warn**（不达标仍 done，只附
# degraded/quality_warning），而本脚本 G6 是**上线验收**判据，不达标一律 FAIL。
MIN_QUALITY="${KINETO_MIN_QUALITY:-${KINETO_QUALITY_THRESHOLD:-0.6}}"
MOFANG_BASE="${MOFANG_BASE:-https://192.168.1.107}"
MOFANG_MODE="coexist"        # coexist | stripped | skip
WEB_BASE="${KINETO_WEB_BASE:-}"       # Zeabur 前端基址（G11），如 https://kineto-web.zeabur.app
WEB_ORIGIN="${KINETO_WEB_ORIGIN:-}"   # 仅当仍保留浏览器→引擎直连的 CORS 预检时才用
FORCE_UPLOAD=0
VENV_PY="${KINETO_VENV_PY:-/opt/kineto/venv/bin/python}"
ENGINE_HOME="${KINETO_ENGINE_HOME:-/srv/kineto}"
ENGINE_USER="${KINETO_ENGINE_USER:-kineto}"
TMPDIR_LOCAL="${TMPDIR:-/tmp}/kineto-validate.$$"

# --- G12 / G8-SSOT 骨架一致性闸门（静态、只读；实现在 deploy/check_ssot.py）---
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
REPO_ROOT="${KINETO_REPO_ROOT:-$(cd "$_SELF_DIR/.." 2>/dev/null && pwd)}"
SSOT_CHECKER="${KINETO_SSOT_CHECKER:-$_SELF_DIR/check_ssot.py}"
SSOT_PY="${KINETO_SSOT_PY:-}"        # 留空则自动挑：引擎 venv（有 numpy）> python3
DO_SSOT_CHECK=1
SSOT_ONLY=0

usage() {
    cat <<'EOF'
用法: bash deploy/validate.sh [选项]

  --base URL            API 地址（默认 http://127.0.0.1:8000；穿隧道时给 https://kineto-api.<domain>）
  --api-key KEY         X-API-Key（也可用环境变量 KINETO_API_KEY，或自动读 /etc/kineto/kineto-engine.env）
  --video PATH          测试视频（默认自动探测设备上的 input_video.mp4 / inbox 里的第一个 mp4）
  --upload              强制用 multipart 上传视频（默认：本机可达时用 JSON {video_path}，更省带宽）
  --timeout SEC         轮询总超时（默认 3600）
  --interval SEC        轮询间隔（默认 10）
  --min-quality F       final_quality_score 门禁（默认取 KINETO_QUALITY_THRESHOLD，否则 0.6）
  --ssot-only           只跑 G12/G8-SSOT 骨架一致性闸门（静态只读，不需引擎在线）
  --no-ssot-check       跳过 G12/G8-SSOT 骨架一致性闸门
  --repo-root DIR       仓库根（G12 用；默认取本脚本上一级目录）
  --ssot-py PATH        G12 用的 python 解释器（默认：引擎 venv，其次 python3）
  --mofang-mode MODE    coexist(默认，要求 MoFang 仍可用) | stripped(要求已停用) | skip
  --mofang-base URL     MoFang 站点地址（默认 https://192.168.1.107）
  --web-base URL        Zeabur 前端基址（G11 前端可达性 + 同源代理链路；如 https://kineto-web.zeabur.app）
  --web-origin ORIGIN   仅当仍保留浏览器→引擎**直连**时，用该 origin 做 CORS 预检断言（新代理架构下通常不需要）
  --no-oom-check        跳过 dmesg OOM 检查（无 sudo 时）
  -h, --help            帮助
EOF
}

DO_OOM_CHECK=1
while [ $# -gt 0 ]; do
    case "$1" in
        --base)          BASE="${2:-}"; shift 2 ;;
        --api-key)       API_KEY="${2:-}"; shift 2 ;;
        --video)         VIDEO="${2:-}"; shift 2 ;;
        --upload)        FORCE_UPLOAD=1; shift ;;
        --timeout)       TIMEOUT="${2:-3600}"; shift 2 ;;
        --interval)      POLL_INTERVAL="${2:-10}"; shift 2 ;;
        --min-quality)   MIN_QUALITY="${2:-0.6}"; shift 2 ;;
        --ssot-only)     SSOT_ONLY=1; shift ;;
        --no-ssot-check) DO_SSOT_CHECK=0; shift ;;
        --repo-root)     REPO_ROOT="${2:-$REPO_ROOT}"; shift 2 ;;
        --ssot-py)       SSOT_PY="${2:-}"; shift 2 ;;
        --mofang-mode)   MOFANG_MODE="${2:-coexist}"; shift 2 ;;
        --mofang-base)   MOFANG_BASE="${2:-}"; shift 2 ;;
        --web-base)      WEB_BASE="${2:-}"; shift 2 ;;
        --web-origin)    WEB_ORIGIN="${2:-}"; shift 2 ;;
        --no-oom-check)  DO_OOM_CHECK=0; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) printf '未知参数: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done

C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_CYN=$'\033[36m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_RED=""; C_GRN=""; C_YEL=""; C_CYN=""; C_OFF=""; }
ok()   { printf '%s[PASS]%s %s\n' "$C_GRN" "$C_OFF" "$1"; }
warn() { printf '%s[WARN]%s %s\n' "$C_YEL" "$C_OFF" "$1"; }
bad()  { printf '%s[FAIL]%s %s\n' "$C_RED" "$C_OFF" "$1"; }
info() { printf '      %s\n' "$1"; }
step() { printf '\n%s── %s%s\n' "$C_CYN" "$1" "$C_OFF"; }

RESULTS=""      # "G1|PASS|说明"
record() { RESULTS="$RESULTS$1|$2|$3
"; }

TMP="$(mktemp -d "${TMPDIR_LOCAL}.XXXXXX" 2>/dev/null || echo "/tmp/kineto-validate.$$")"
mkdir -p "$TMP" 2>/dev/null
cleanup() { rm -rf "$TMP" 2>/dev/null || true; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
# 自动从 /etc/kineto/kineto-engine.env 取 key（若存在且可读）
if [ -z "$API_KEY" ] && [ -r /etc/kineto/kineto-engine.env ]; then
    API_KEY="$(sed -n 's/^KINETO_API_KEY=//p' /etc/kineto/kineto-engine.env | head -n1 | tr -d '"'"'"'')"
fi

curl_api() {  # curl_api <method> <path> [额外 curl 参数...]
    local m="$1" p="$2"; shift 2
    local args=(-sS -m 120 -X "$m" "${BASE}${p}")
    # API_KEY 为空时仍发送空值头：服务端若启用鉴权会回 401（预期行为），
    # 避开 bash 3.2 下空数组 + set -u 的 "unbound variable" 坑。
    args+=(-H "X-API-Key: ${API_KEY}")
    curl "${args[@]}" "$@"
}
curl_code() { # curl_code <method> <path> [额外参数...] → 只输出 HTTP 状态码
    local m="$1" p="$2"; shift 2
    local args=(-s -o /dev/null -m 120 -w '%{http_code}' -X "$m" "${BASE}${p}")
    args+=(-H "X-API-Key: ${API_KEY}")
    curl "${args[@]}" "$@" 2>/dev/null
}
# JSON 取值：jq 优先，其次 python3，最后 grep 兜底
json_get() {  # json_get <file> <python 表达式，变量名 d>
    local f="$1" expr="$2"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json,sys
try:
    d=json.load(open('$f'))
except Exception as e:
    print(''); sys.exit(0)
try:
    v=($expr)
except Exception:
    v=''
print('' if v is None else v)
" 2>/dev/null
    else
        grep -o "\"$(printf '%s' "$expr" | sed 's/.*\[.//')\"[^,}]*" "$f" 2>/dev/null | head -n1
    fi
}
fmt_dur() {  # fmt_dur <秒>
    local s="$1" h m
    h=$((s / 3600)); m=$(( (s % 3600) / 60 )); s=$((s % 60))
    if [ "$h" -gt 0 ]; then printf '%dh%02dm%02ds' "$h" "$m" "$s"
    elif [ "$m" -gt 0 ]; then printf '%dm%02ds' "$m" "$s"
    else printf '%ds' "$s"; fi
}

command -v curl >/dev/null 2>&1 || { bad "缺少 curl，无法验收"; exit 1; }
command -v python3 >/dev/null 2>&1 || warn "缺少 python3 —— JSON 断言会退化为文本匹配（建议装 python3）"

printf '%s\n' "=============================================================================="
printf ' Kineto Engine 端到端验收  —  %s\n' "$(date '+%F %T %Z')"
printf ' API: %s\n 视频: %s\n 门禁: extraction_mode==4dhumans 且 final_quality_score>=%s\n' \
       "$BASE" "${VIDEO:-<自动探测>}" "$MIN_QUALITY"
printf ' MoFang 回归模式: %s (%s)\n' "$MOFANG_MODE" "$MOFANG_BASE"
printf ' 密钥: %s\n' "$([ -n "$API_KEY" ] && echo '已提供' || echo '未提供（若服务端启用了鉴权，POST 会 401）')"
printf '%s\n' "=============================================================================="
printf '\n%s本脚本只发 HTTP 请求与只读查询，不改动设备状态、不生成代码。%s\n' "$C_GRN" "$C_OFF"

# ---------------------------------------------------------------------------
# G12 / G8-SSOT  骨架单一事实源一致性闸门
#   权威源 = kineto-engine/skeleton_spec.py（P1 关节序整改后的 canonical SSOT）；
#   前端镜像 = kineto-web/lib/skeleton.ts（或前端产出的 skeleton.manifest.json）。
#   校验逻辑全部在 deploy/check_ssot.py 里（只读、**非 codegen**），本函数只负责
#   选解释器、转发输出、把退出码翻译成 PASS/FAIL/SKIP：
#     0=一致(PASS)  1=漂移(FAIL)  2=无法解析/校验(FAIL)  3=文件缺失(SKIP)
# ---------------------------------------------------------------------------
run_ssot_gate() {
    step "G12  骨架 SSOT 一致性闸门（skeleton_spec.py ↔ skeleton.ts；计划编号 G8-SSOT）"
    if [ "$DO_SSOT_CHECK" != "1" ]; then
        warn "G12 已跳过（--no-ssot-check）"
        record "G12 SSOT骨架" "SKIP" "用户跳过"
        return 0
    fi
    if [ ! -f "$SSOT_CHECKER" ]; then
        warn "G12 跳过：找不到校验器 $SSOT_CHECKER"
        record "G12 SSOT骨架" "SKIP" "check_ssot.py 缺失"
        return 0
    fi
    # 解释器：显式指定 > 引擎 venv（有 numpy/chumpy → 可直接 import SSOT）> python3
    # （SSOT 侧 import 不可用时，check_ssot.py 会自动退化为 AST 静态解析字面量）
    local py="$SSOT_PY"
    if [ -z "$py" ]; then
        if [ -x "$VENV_PY" ]; then py="$VENV_PY"; else py="$(command -v python3 2>/dev/null || true)"; fi
    fi
    if [ -z "$py" ] || [ ! -x "$py" ]; then
        warn "G12 跳过：没有可用的 python 解释器（用 --ssot-py / KINETO_SSOT_PY 指定）"
        record "G12 SSOT骨架" "SKIP" "无 python 解释器"
        return 0
    fi
    info "解释器: $py"
    info "仓库根: $REPO_ROOT"
    local out rc
    out="$("$py" "$SSOT_CHECKER" --repo-root "$REPO_ROOT" 2>&1)"; rc=$?
    printf '%s\n' "$out" | sed 's/^/      /'
    case "$rc" in
        0) ok "G12 骨架 SSOT 一致（前端已镜像 canonical 拓扑：collar 13/14 的父=spine3 9）"
           record "G12 SSOT骨架" "PASS" "SSOT ↔ skeleton.ts 一致" ;;
        1) bad "G12 骨架 SSOT **漂移**（详见上面 [DRIFT] 行）—— 前端镜像必须对齐 SSOT"
           info "    整改事实：P1 后 collar(13/14) 的父是 spine3(9)，不再是 neck(12)；"
           info "              SMPL_SKELETON 为 23 条真 kintree 边，BONE_PART_MAP 键集合与之相同。"
           info "    修法属前端 Owner（本闸门不 codegen）：改 kineto-web/lib/skeleton.ts 后重跑 --ssot-only。"
           record "G12 SSOT骨架" "FAIL" "SSOT ↔ skeleton.ts 漂移" ;;
        3) # SKIP：可能是文件缺失（非仓库检出）或 pkl 不可用（Docker 构建期/XPU）
           if printf '%s' "$out" | grep -q 'pkl 不可用'; then
               warn "G12 SKIP：pkl 不可用（Docker 构建期/XPU 环境）→ 拓扑一致但 pkl 派生不变量未覆盖"
               info "    运行时首次推理前会重试加载 pkl 派生常量（BONE_LENGTH_BOUNDS / rest 骨长）。"
               info "    未覆盖不变量：骨长界键集合/rest 骨长落界/对称骨对镜像。"
               record "G12 SSOT骨架" "SKIP" "pkl 不可用（拓扑一致，pkl 派生未覆盖）"
           else
               warn "G12 无法校验：缺 skeleton_spec.py 或 skeleton.ts（设备上通常没有 kineto-web/）"
               info "    请在**仓库检出**（Mac/CI）上跑：bash deploy/validate.sh --ssot-only"
               record "G12 SSOT骨架" "SKIP" "文件缺失（非仓库检出？）"
           fi ;;
        *) bad "G12 校验无法完成（退出码 $rc：解析/结构错误）—— 不能证明一致即视为不安全"
           record "G12 SSOT骨架" "FAIL" "校验器退出码 $rc" ;;
    esac
    return 0
}

# --ssot-only：只跑静态闸门后直接汇总退出（不触网、不建 job、不查设备状态）
if [ "$SSOT_ONLY" = "1" ]; then
    SSOT_T0=$(date +%s)
    run_ssot_gate
    SSOT_FAILS="$(printf '%s' "$RESULTS" | grep -c '|FAIL|')"
    printf '\n%s\n' "=============================================================================="
    if [ "$SSOT_FAILS" -eq 0 ]; then
        printf '%s==> G12/G8-SSOT 通过（耗时 %s）。%s\n' "$C_GRN" \
               "$(fmt_dur "$(( $(date +%s) - SSOT_T0 ))")" "$C_OFF"
    else
        printf '%s==> G12/G8-SSOT 未通过：%d 项 FAIL —— 前端 skeleton.ts 需对齐 SSOT。%s\n' "$C_RED" \
               "$SSOT_FAILS" "$C_OFF"
    fi
    printf '%s\n' "=============================================================================="
    exit "$([ "$SSOT_FAILS" -eq 0 ] && echo 0 || echo 1)"
fi

# ---------------------------------------------------------------------------
# G1 /health（现需鉴权）+ /healthz（公开存活探针）
# ---------------------------------------------------------------------------
step "G1  GET /healthz (存活) + GET /health (详情，需 X-API-Key)"
# 先用公开的 /healthz 判断“引擎进程是否活着”，把“没起”与“鉴权/详情异常”区分开
HZ_CODE="$(curl -s -o /dev/null -m 15 -w '%{http_code}' "${BASE}/healthz" 2>/dev/null)"
info "GET /healthz → HTTP ${HZ_CODE:-000}（公开存活探针，无需密钥）"
if [ "$HZ_CODE" != "200" ]; then
    warn "G0 /healthz = ${HZ_CODE:-000} —— 引擎可能未启动/端口不对/隧道未通（或旧版 api.py 尚无 /healthz）"
fi

HEALTH="$TMP/health.json"
# /health 现需鉴权：必须带 X-API-Key（否则 401）
CODE="$(curl -s -o "$HEALTH" -m 30 -w '%{http_code}' -H "X-API-Key: ${API_KEY}" "${BASE}/health" 2>/dev/null)"
info "GET /health → HTTP ${CODE:-000}"
info "$(cat "$HEALTH" 2>/dev/null | head -c 600)"
if [ "$CODE" = "200" ]; then
    if grep -q '"extraction_mode"' "$HEALTH" 2>/dev/null; then
        ok "G1 /health 返回 200 且含 extraction_mode 字段"
        record "G1 /health" "PASS" "200 + extraction_mode 字段存在"
    else
        bad "G1 /health 返回体缺少 extraction_mode 字段（api.py 版本可能不匹配）"
        record "G1 /health" "FAIL" "缺少 extraction_mode 字段"
    fi
elif [ "$CODE" = "401" ]; then
    bad "G1 /health HTTP 401 —— /health 现需鉴权，X-API-Key 缺失或与服务端不一致（用 --api-key / KINETO_API_KEY 提供）"
    record "G1 /health" "FAIL" "401 鉴权失败"
else
    bad "G1 /health HTTP ${CODE}（引擎未启动？端口不对？隧道未通？）"
    record "G1 /health" "FAIL" "HTTP $CODE"
fi
# 附带解读
H_DEVICE="$(json_get "$HEALTH" "d.get('device','')")"
H_MODEL="$(json_get "$HEALTH" "d.get('model_loaded','')")"
H_DETECT="$(json_get "$HEALTH" "d.get('detector_loaded','')")"
H_QUEUE="$(json_get "$HEALTH" "d.get('queue_depth','')")"
H_STATUS="$(json_get "$HEALTH" "d.get('status','')")"
info "status=$H_STATUS device=$H_DEVICE model_loaded=$H_MODEL detector_loaded=$H_DETECT queue_depth=$H_QUEUE"
# G1a：api.py 的 _detect_device() 已支持 CUDA>XPU>MPS>CPU。在 Intel Arc 机器上，
#      /health 报 device=cpu 是**真故障**（不再是“已知无害缺口”）——硬 FAIL。
if [ "$CODE" = "200" ] && [ "$H_DEVICE" = "cpu" ]; then
    bad "G1a /health 报告 device=cpu —— Intel Arc 上这是真故障（api.py 已支持 xpu，_detect_device 为 CUDA>XPU>MPS>CPU）"
    info "    排查：① kineto 服务用户是否在 render/video 组（id kineto；sudo usermod -aG render,video kineto 后重启服务）"
    info "          ② /dev/dri/renderD128 是否存在且属组 render（ls -l /dev/dri）"
    info "          ③ torch 是否被 requirements.txt 换成了 CPU 轮子（torch.__version__ 应带 +xpu；见 G2 / §9 XPU-3）"
    record "G1a device" "FAIL" "Arc 上 device=cpu（应为 xpu）"
elif [ "$CODE" = "200" ] && [ "$H_DEVICE" = "xpu" ]; then
    ok "G1a /health 报告 device=xpu —— 引擎确实跑在 Intel Arc 上"
    record "G1a device" "PASS" "device=xpu"
elif [ "$CODE" = "200" ]; then
    warn "G1a /health 报告 device=$H_DEVICE —— 非 xpu（若本机不是 Arc 可忽略；Arc 上必须为 xpu）"
    record "G1a device" "WARN" "device=${H_DEVICE:-unknown}"
else
    record "G1a device" "SKIP" "/health 非 200，无法读 device"
fi
# G1b：model_loaded 现**仅**反映 4DHumans 权重（.ckpt + model_config.yaml + SMPL .pkl 三者齐全才 True），
#      detector_loaded 单独反映 YOLOv8n（yolov8n.pt）。二者拆分后，model_loaded 不再被 yolo 架空 ——
#      它如实代表 4DHumans 是否就位；False 会导致 job 回退合成关键点并被 api.py 硬判 FAILED。
if [ "$CODE" = "200" ]; then
    if [ "$H_MODEL" = "True" ]; then
        ok "G1b model_loaded=True —— 4DHumans 权重齐全（上线硬门禁 extraction_mode=4dhumans 的前提）"
        record "G1b model" "PASS" "model_loaded=True(4dhumans)"
    else
        bad "G1b model_loaded=$H_MODEL —— 4DHumans 权重/缓存未就位（\$HOME/.cache/4DHumans 或软链）"
        info "    会导致 job 回退合成关键点并被 api.py 按 extraction_mode!='4dhumans' 硬判 FAILED。见 §5.7 / §9 MODEL-1。"
        record "G1b model" "FAIL" "model_loaded=$H_MODEL(4dhumans 未齐全)"
    fi
    if [ "$H_DETECT" = "True" ]; then
        info "    detector_loaded=True（YOLOv8n 人体检测器就位）"
    else
        warn "    detector_loaded=$H_DETECT —— yolov8n.pt 缺失（人体检测器；见 §5.7）"
    fi
fi

# ---------------------------------------------------------------------------
# G2 XPU 可用性（尽力而为；Docker 路径自动跳过）
# ---------------------------------------------------------------------------
step "G2  Intel XPU 可用性"
XPU_OK=""
if [ -x "$VENV_PY" ]; then
    XPU_OUT="$(sudo -n -u "$ENGINE_USER" env HOME="$ENGINE_HOME" "$VENV_PY" -c \
        'import torch;print("avail=",torch.xpu.is_available());print("name=",torch.xpu.get_device_name(0) if torch.xpu.is_available() else "n/a");print("ver=",torch.__version__)' 2>&1)" \
        || XPU_OUT="$("$VENV_PY" -c \
        'import torch;print("avail=",torch.xpu.is_available());print("name=",torch.xpu.get_device_name(0) if torch.xpu.is_available() else "n/a");print("ver=",torch.__version__)' 2>&1)"
    info "$(printf '%s' "$XPU_OUT" | tr '\n' ' ')"
    if printf '%s' "$XPU_OUT" | grep -q 'avail= True'; then
        XPU_OK="yes"; ok "G2 torch.xpu.is_available() == True（$(printf '%s' "$XPU_OUT" | sed -n 's/^name= //p')）"
        record "G2 XPU" "PASS" "torch.xpu 可用"
    else
        XPU_OK="no"; bad "G2 torch.xpu.is_available() != True —— 检查驱动/组权限(render,video)/torch 是否为 +xpu 轮子"
        record "G2 XPU" "FAIL" "torch.xpu 不可用"
    fi
else
    warn "G2 跳过：未找到 ${VENV_PY}（容器路径请在容器内跑：docker exec kineto-engine python -c 'import torch;print(torch.xpu.is_available())'）"
    record "G2 XPU" "SKIP" "venv 不存在（容器路径？）"
fi

# ---------------------------------------------------------------------------
# 选定测试视频
# ---------------------------------------------------------------------------
step "G3  POST /jobs"
if [ -z "$VIDEO" ]; then
    for cand in /opt/kineto/kineto-engine/input_video.mp4 \
                /srv/kineto/inbox/*.mp4 \
                /opt/kineto/kineto-engine/4D-Humans/example_data/videos/gymnasts.mp4 \
                "$TMP/nope"; do
        [ -f "$cand" ] && { VIDEO="$cand"; break; }
    done
fi
USE_MULTIPART=0
if [ "$FORCE_UPLOAD" = "1" ]; then
    USE_MULTIPART=1
elif [ -n "$VIDEO" ] && [ ! -f "$VIDEO" ]; then
    bad "指定的视频不存在: $VIDEO"; record "G3 POST /jobs" "FAIL" "视频缺失"; VIDEO=""
fi
case "$BASE" in
    *127.0.0.1*|*localhost*) [ -n "$VIDEO" ] && [ -f "$VIDEO" ] && USE_MULTIPART=0 ;;
    *)                       USE_MULTIPART=1 ;;   # 远端地址 → 只能上传
esac

JOB_ID=""
if [ -n "$VIDEO" ]; then
    VSIZE="$(du -h "$VIDEO" 2>/dev/null | awk '{print $1}')"
    info "测试视频: $VIDEO ($VSIZE)"
    if [ "$USE_MULTIPART" = "1" ]; then
        info "提交方式: multipart 上传（字段名 video）"
        RESP="$TMP/job.json"
        CODE="$(curl -s -o "$RESP" -m 600 -w '%{http_code}' -X POST "${BASE}/jobs" \
                -H "X-API-Key: ${API_KEY}" -F "video=@${VIDEO}")"
    else
        info "提交方式: JSON {video_path}（设备本地路径，api.py 直接引用不复制）"
        RESP="$TMP/job.json"
        CODE="$(curl -s -o "$RESP" -m 120 -w '%{http_code}' -X POST "${BASE}/jobs" \
                -H "X-API-Key: ${API_KEY}" -H 'Content-Type: application/json' \
                -d "{\"video_path\":\"$VIDEO\"}")"
    fi
    info "HTTP $CODE  body: $(head -c 400 "$RESP" 2>/dev/null)"
    JOB_ID="$(json_get "$RESP" "d.get('job_id','')")"
    if [ "$CODE" = "202" ] && [ -n "$JOB_ID" ]; then
        ok "G3 任务已入队: job_id=$JOB_ID"
        record "G3 POST /jobs" "PASS" "202 + job_id=$JOB_ID"
    elif [ "$CODE" = "401" ]; then
        bad "G3 401 未授权 —— KINETO_API_KEY 与服务端不一致（或未提供）"
        record "G3 POST /jobs" "FAIL" "401 鉴权失败"
    elif [ "$CODE" = "413" ]; then
        bad "G3 413 请求体过大 —— 命中 Cloudflare 100MB 上限；改用设备本地 video_path 方式（见 cloudflared/README.md §6）"
        record "G3 POST /jobs" "FAIL" "413 上传超限"
    else
        bad "G3 提交失败 HTTP $CODE"
        record "G3 POST /jobs" "FAIL" "HTTP $CODE"
    fi
else
    bad "G3 未找到可用测试视频（用 --video 指定一个 mp4）"
    record "G3 POST /jobs" "FAIL" "无测试视频"
fi

# ---------------------------------------------------------------------------
# G4 轮询
# ---------------------------------------------------------------------------
step "G4  轮询 GET /jobs/{id}"
STATE=""; ELAPSED=0; T0=$(date +%s)
if [ -n "$JOB_ID" ]; then
    while :; do
        S="$TMP/status.json"
        curl_api GET "/jobs/$JOB_ID" > "$S" 2>/dev/null
        STATE="$(json_get "$S" "d.get('state','')")"
        PROG="$(json_get "$S" "d.get('progress','')")"
        NOW=$(date +%s); ELAPSED=$((NOW - T0))
        printf '      [%s] state=%s progress=%s\n' "$(fmt_dur "$ELAPSED")" "$STATE" "$PROG"
        case "$STATE" in
            done)   ok "G4 任务完成，耗时 $(fmt_dur "$ELAPSED")"
                    # [P3/改动 E] 质量门禁 KINETO_QUALITY_GATE 默认 **warn**：质量不达标时
                    # job 仍 done，但会带 degraded/quality_warning 显式暴露（绝不静默放行）。
                    # 与 KINETO_STRICT 彻底解耦：STRICT 只守假数据/缺权重/detector 降级（退出码 3）。
                    DEG="$(json_get "$S" "d.get('degraded','')")"
                    QWARN="$(json_get "$S" "d.get('quality_warning','')")"
                    if [ "$DEG" = "True" ] || [ "$QWARN" = "True" ]; then
                        warn "G4a done 但带 degraded/quality_warning=True —— 引擎质量门禁为 warn（默认）：不失败但如实暴露"
                        info "    引擎侧 KINETO_QUALITY_GATE=${KINETO_QUALITY_GATE:-warn}（off|warn|fail）；不达标原因见"
                        info "    <jobdir>/audit_iter*/audit_results.json 的 verdict/total_issues/failure_reason 与 quality_log.jsonl"
                        record "G4a degraded" "WARN" "done + degraded/quality_warning"
                    else
                        record "G4a degraded" "PASS" "done，无质量降级标记"
                    fi
                    record "G4 job done" "PASS" "耗时 $(fmt_dur "$ELAPSED")"
                    break ;;
            failed) ERR="$(json_get "$S" "d.get('error','')")"
                    bad "G4 任务 FAILED（耗时 $(fmt_dur "$ELAPSED")）: ${ERR:0:400}"
                    record "G4 job done" "FAIL" "${ERR:0:120}"
                    break ;;
            queued|running|"")
                    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
                        bad "G4 超时（>${TIMEOUT}s，最后状态 ${STATE:-unknown}）"
                        record "G4 job done" "FAIL" "超时 ${TIMEOUT}s"
                        break
                    fi
                    sleep "$POLL_INTERVAL" ;;
            *)      bad "G4 未知状态: $STATE"; record "G4 job done" "FAIL" "state=$STATE"; break ;;
        esac
    done
else
    warn "G4 跳过（无 job_id）"
    record "G4 job done" "SKIP" "无 job_id"
fi

# ---------------------------------------------------------------------------
# G5/G6/G7 产物断言
# ---------------------------------------------------------------------------
step "G5/G6/G7  产物断言 (pose_data.json / demo_output.mp4)"
if [ "$STATE" = "done" ] && [ -n "$JOB_ID" ]; then
    POSE="$TMP/pose_data.json"
    PCODE="$(curl_api GET "/jobs/$JOB_ID/pose_data.json" -o "$POSE" -w '%{http_code}' 2>/dev/null | tail -n1)"
    # 上一行同时写了文件与状态码；curl -w 的输出在 -o 之后
    PSIZE="$(wc -c < "$POSE" 2>/dev/null | tr -d ' ')"
    info "pose_data.json: HTTP ${PCODE:-?}  ${PSIZE:-0} bytes"

    MODE="$(json_get "$POSE" "(d.get('metadata') or {}).get('extraction_mode','')")"
    QUAL="$(json_get "$POSE" "((d.get('metadata') or {}).get('pipeline') or {}).get('final_quality_score','')")"
    FRAMES="$(json_get "$POSE" "(d.get('metadata') or {}).get('total_frames','')")"
    info "metadata.extraction_mode = ${MODE:-<empty>}"
    info "metadata.pipeline.final_quality_score = ${QUAL:-<empty>}"
    info "metadata.total_frames = ${FRAMES:-<empty>}"

    if [ "$MODE" = "4dhumans" ]; then
        ok "G5 extraction_mode == '4dhumans'（真实推理，非合成回退）"
        record "G5 extraction_mode" "PASS" "4dhumans"
    else
        bad "G5 extraction_mode == '${MODE:-<empty>}' != '4dhumans' —— 权重缺失或 XPU 不可用导致引擎回退；上线门禁必须拦住"
        record "G5 extraction_mode" "FAIL" "${MODE:-empty}"
    fi

    if [ -n "$QUAL" ] && python3 -c "import sys;sys.exit(0 if float('$QUAL')>=$MIN_QUALITY else 1)" 2>/dev/null; then
        ok "G6 final_quality_score=$QUAL >= $MIN_QUALITY（阈值与引擎 KINETO_QUALITY_THRESHOLD 同源）"
        record "G6 quality" "PASS" "$QUAL"
    elif [ -n "$QUAL" ]; then
        bad "G6 final_quality_score=$QUAL < $MIN_QUALITY"
        info "    注意语义：引擎侧默认 KINETO_QUALITY_GATE=warn → 该 job 仍会是 done（只附 degraded/"
        info "    quality_warning），**不是**引擎故障；但本脚本是上线验收判据，故仍判 FAIL（不得上线）。"
        info "    排查：audit_iter*/audit_results.json 的 verdict/failure_reason；确需放宽则 --min-quality / KINETO_QUALITY_THRESHOLD。"
        record "G6 quality" "FAIL" "$QUAL"
    else
        bad "G6 读不到 metadata.pipeline.final_quality_score（注意：它在 metadata.pipeline 下，不在 metadata 顶层）"
        record "G6 quality" "FAIL" "字段缺失"
    fi

    DCODE="$(curl_code GET "/jobs/$JOB_ID/demo_output.mp4")"
    if [ "$DCODE" = "200" ]; then
        curl_api GET "/jobs/$JOB_ID/demo_output.mp4" -o "$TMP/demo.mp4" >/dev/null 2>&1
        DSIZE="$(wc -c < "$TMP/demo.mp4" 2>/dev/null | tr -d ' ')"
        if [ "${DSIZE:-0}" -gt 10000 ]; then
            ok "G7 demo_output.mp4 可下载（${DSIZE} bytes）"
            record "G7 demo mp4" "PASS" "${DSIZE}B"
        else
            bad "G7 demo_output.mp4 过小（${DSIZE:-0} bytes）"
            record "G7 demo mp4" "FAIL" "${DSIZE:-0}B"
        fi
    else
        warn "G7 demo_output.mp4 HTTP ${DCODE}（若引擎以 --skip-demo 运行则属正常）"
        record "G7 demo mp4" "WARN" "HTTP $DCODE"
    fi
else
    warn "G5/G6/G7 跳过（任务未 done）"
    record "G5 extraction_mode" "SKIP" "任务未 done"
    record "G6 quality" "SKIP" "任务未 done"
    record "G7 demo mp4" "SKIP" "任务未 done"
fi

# ---------------------------------------------------------------------------
# G8 OOM
# ---------------------------------------------------------------------------
step "G8  OOM / 内核告警"
if [ "$DO_OOM_CHECK" = "1" ]; then
    OOM=""
    DMESG_RAN=0          # dmesg 是否真的被读到（无 sudo 时为 0，不能算“已检查”）
    JK_RAN=0             # journalctl -k 是否可用
    if command -v dmesg >/dev/null 2>&1; then
        if [ "$(id -u)" -eq 0 ] || sudo -n true >/dev/null 2>&1; then
            # dmesg 默认前缀 [秒.微秒] 是「自启动以来」的时间；用 /proc/uptime 算出「30 分钟前」的阈值，
            # 只保留近期条目 —— 否则历史 OOM 会让本次验收假红（与下面 journalctl --since -30min 对齐）。
            UP="$(awk '{print int($1)}' /proc/uptime 2>/dev/null || echo 0)"
            CUT=$(( UP - 1800 )); [ "$CUT" -lt 0 ] && CUT=0
            RAW="$([ "$(id -u)" -eq 0 ] && dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null)"
            OOM="$(printf '%s\n' "$RAW" \
                | awk -v cut="$CUT" 'match($0,/^\[[[:space:]]*[0-9]+\.[0-9]+\]/){ts=substr($0,2,RLENGTH-2)+0; if(ts>=cut)print}' \
                | grep -iE 'out of memory|oom-kill|killed process' | tail -n 10)"
            DMESG_RAN=1
        else
            warn "G8 dmesg 需要 sudo（无免密 sudo）—— 请以 root 重跑；否则本项只能给 SKIP 而不是 PASS"
        fi
    fi
    JK=""
    if command -v journalctl >/dev/null 2>&1; then
        JK="$(journalctl -k --since "-30min" --no-pager 2>/dev/null | grep -iE 'oom-kill|out of memory' | tail -n 5)"
        # journalctl 在无权限/无 journal 时会空输出，用退出态判断是否真的读到了
        journalctl -k -n 1 --no-pager >/dev/null 2>&1 && JK_RAN=1
    fi
    if [ "$DMESG_RAN" = "0" ] && [ "$JK_RAN" = "0" ]; then
        # 两个数据源都读不到 → 不能谎报 PASS
        warn "G8 无法读取 dmesg 与 journalctl —— 本项未真正检查，记为 SKIP"
        warn "    请手动确认：sudo dmesg | grep -iE 'oom-kill|out of memory'"
        record "G8 OOM" "SKIP" "dmesg/journal 不可读（需 sudo）"
    elif [ -z "$OOM" ] && [ -z "$JK" ]; then
        ok "G8 无 OOM 记录（数据源：$([ "$DMESG_RAN" = 1 ] && printf 'dmesg ')$([ "$JK_RAN" = 1 ] && printf 'journalctl')）"
        record "G8 OOM" "PASS" "无 oom-kill"
    else
        bad "G8 检测到 OOM 迹象:"; info "${OOM}${JK}"
        record "G8 OOM" "FAIL" "存在 oom-kill 记录"
    fi
    # 服务是否在推理中被重启过（Restart=always 会掩盖 OOM）
    if command -v systemctl >/dev/null 2>&1; then
        NR="$(systemctl show kineto-engine -p NRestarts --value 2>/dev/null)"
        if [ -n "$NR" ] && [ "$NR" != "[not set]" ]; then
            info "kineto-engine NRestarts=${NR}（>0 说明进程曾崩溃/被杀，需查 journalctl -u kineto-engine）"
        fi
    fi
else
    warn "G8 已跳过（--no-oom-check）"
    record "G8 OOM" "SKIP" "用户跳过"
fi

# ---------------------------------------------------------------------------
# G9 重启持久化
# ---------------------------------------------------------------------------
step "G9  重启持久化（reboot-persistence）"
if command -v systemctl >/dev/null 2>&1 && systemctl cat kineto-engine >/dev/null 2>&1; then
    EN="$(systemctl is-enabled kineto-engine 2>/dev/null)"
    AC="$(systemctl is-active kineto-engine 2>/dev/null)"
    if [ "$EN" = "enabled" ]; then
        ok "G9 kineto-engine: enabled + $AC —— 断电重启后会自动拉起"
        record "G9 持久化" "PASS" "systemd enabled ($AC)"
    else
        bad "G9 kineto-engine is-enabled=$EN —— 执行: sudo systemctl enable kineto-engine"
        record "G9 持久化" "FAIL" "is-enabled=$EN"
    fi
    CFEN="$(systemctl is-enabled cloudflared 2>/dev/null)"
    if [ -n "$CFEN" ]; then
        [ "$CFEN" = "enabled" ] && ok "G9b cloudflared: enabled —— 隧道会随机器自启" \
                                || warn "G9b cloudflared is-enabled=$CFEN —— 隧道不会自启，重启后前端会失联"
    fi
elif command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^kineto-engine$'; then
    RP="$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' kineto-engine 2>/dev/null)"
    if [ "$RP" = "unless-stopped" ] || [ "$RP" = "always" ]; then
        ok "G9 容器 kineto-engine restart policy=$RP —— 重启后自动恢复（需 docker 服务本身 enabled）"
        record "G9 持久化" "PASS" "docker restart=$RP"
    else
        bad "G9 容器 restart policy=${RP:-unknown} —— 应为 unless-stopped"
        record "G9 持久化" "FAIL" "docker restart=$RP"
    fi
    # G9c 容器健康状态：HEALTHCHECK 必须打公开 /healthz；若误打需鉴权的 /health，
    #     在强制注入 KINETO_API_KEY 时会恒 401 → 容器 unhealthy → cloudflared
    #     depends_on: service_healthy 永不满足 → 隧道永不启动（Docker 路径公网入口失效）。
    HS="$(docker inspect -f '{{.State.Health.Status}}' kineto-engine 2>/dev/null)"
    if [ -n "$HS" ]; then
        if [ "$HS" = "healthy" ]; then
            ok "G9c 容器 kineto-engine 健康状态=healthy（HEALTHCHECK /healthz 通过）"
            record "G9c 容器健康" "PASS" "health=healthy"
        else
            bad "G9c 容器 kineto-engine 健康状态=$HS —— HEALTHCHECK 未通过；cloudflared 的 depends_on: service_healthy 会永不满足 → 隧道不启动"
            info "    排查：docker inspect -f '{{json .State.Health}}' kineto-engine   # 看最近几次探测的输出/退出码"
            info "          确认 HEALTHCHECK 打的是公开 /healthz（非需鉴权的 /health，否则恒 401→unhealthy）。见 §9 DOCKER-1。"
            record "G9c 容器健康" "FAIL" "health=$HS"
        fi
    else
        warn "G9c 容器 kineto-engine 无 Health 状态（镜像未定义 HEALTHCHECK？）—— 跳过健康断言"
        record "G9c 容器健康" "SKIP" "无 .State.Health"
    fi
    docker info --format 'Docker 服务自启: {{.ServerVersion}}' >/dev/null 2>&1 \
        && info "另请确认: systemctl is-enabled docker → 应为 enabled"
else
    warn "G9 跳过：既没发现 kineto-engine.service，也没发现 kineto-engine 容器"
    record "G9 持久化" "SKIP" "未检测到服务"
fi
printf '\n%s重启后必须复验（这是硬性要求，不是可选项）:%s\n' "$C_YEL" "$C_OFF"
cat <<'EOF'
      sudo reboot
      # 重新 SSH 后：
      systemctl is-active kineto-engine cloudflared docker      # 都应是 active/running
      curl -fsS http://127.0.0.1:8000/healthz                   # 本地存活探针（公开）
      curl -fsS https://kineto-api.<YOUR_DOMAIN>/healthz         # 公网链路存活（/health 需带 X-API-Key）
      bash deploy/validate.sh                                     # 重跑一次完整验收
      # 特别注意：重启后 /dev/dri/renderD128 的组权限是否仍是 render
      ls -l /dev/dri && id kineto
EOF

# ---------------------------------------------------------------------------
# G10 MoFang 回归
# ---------------------------------------------------------------------------
step "G10 MoFang 回归检查（mode=${MOFANG_MODE}）"
mofang_check() {
    # 返回 HTTP 状态码；自签名证书 → -k
    curl -sk -o "$TMP/mofang_body" -m 20 -w '%{http_code}' "$1" 2>/dev/null
}
if [ "$MOFANG_MODE" = "skip" ]; then
    warn "G10 已跳过（--mofang-mode skip）"
    record "G10 MoFang" "SKIP" "用户跳过"
else
    MF_CODE="$(mofang_check "${MOFANG_BASE}/next/MoFang.html")"
    BR_CODE="$(mofang_check "${MOFANG_BASE}/bridge/v2/bootstrap")"
    info "GET ${MOFANG_BASE}/next/MoFang.html      → HTTP ${MF_CODE:-000}"
    info "GET ${MOFANG_BASE}/bridge/v2/bootstrap   → HTTP ${BR_CODE:-000}"
    BR_BODY="$(head -c 800 "$TMP/mofang_body" 2>/dev/null)"
    info "bootstrap body(截断): $BR_BODY"

    if [ "$MOFANG_MODE" = "coexist" ]; then
        # 共存路线（OPTION ii）：MoFang 必须完全不受影响
        MF_OK=1
        [ "$MF_CODE" = "200" ] || MF_OK=0
        if [ "$MF_OK" = "1" ]; then
            ok "G10a /next/MoFang.html 仍为 HTTP 200（配对 UI 未受影响）"
        else
            bad "G10a /next/MoFang.html = ${MF_CODE:-000}（期望 200）—— 共存路线下这是回归，必须回滚"
        fi
        if [ "$BR_CODE" = "200" ] && printf '%s' "$BR_BODY" | grep -qiE 'openclaw'; then
            ok "G10b /bridge/v2/bootstrap HTTP 200 且 agentHealth 含 openclaw"
        elif [ "$BR_CODE" = "200" ]; then
            warn "G10b /bridge/v2/bootstrap HTTP 200，但未在响应里匹配到 openclaw/rag —— 请人工核对 agentHealth 字段结构"
            info "期望同时看到 openclaw 与 rag 两个 agent 的健康状态为 ok"
        else
            bad "G10b /bridge/v2/bootstrap = ${BR_CODE:-000}（期望 200）"
        fi
        if printf '%s' "$BR_BODY" | grep -qiE '"rag"|rag'; then
            ok "G10c bootstrap 响应中出现 rag agent"
        else
            warn "G10c 未匹配到 rag —— 人工确认（不同固件版本字段名可能不同）"
        fi
        if [ "$MF_OK" = "1" ] && [ "$BR_CODE" = "200" ]; then
            record "G10 MoFang" "PASS" "共存无回归"
        else
            record "G10 MoFang" "FAIL" "MoFang=$MF_CODE bootstrap=$BR_CODE"
        fi
    else
        # 剥离路线（OPTION i）：期望 MoFang 上层应用已停用
        if [ "$MF_CODE" = "200" ]; then
            warn "G10 /next/MoFang.html 仍为 200 —— 若你的目标是剥离 MoFang，说明 disable 未生效或 nginx 仍在发静态页"
            record "G10 MoFang" "WARN" "剥离模式下仍 200"
        else
            ok "G10 MoFang 上层应用已停用（HTTP ${MF_CODE:-000}）—— 与剥离预期一致"
            record "G10 MoFang" "PASS" "已按预期停用 (${MF_CODE:-000})"
        fi
        info "需要恢复 MoFang 时（可逆）：sudo systemctl enable --now <discover_device.sh §10 列出的单元>"
    fi
fi

# ---------------------------------------------------------------------------
# G11 Zeabur 前端可达性（新代理架构：浏览器→Zeabur 同源 /api/*，Zeabur 服务端→引擎）
# ---------------------------------------------------------------------------
step "G11 Zeabur 前端可达性 + 同源代理链路"
if [ -z "$WEB_BASE" ]; then
    warn "G11 跳过（未提供 --web-base）—— 前端部署与验证见根目录 DEPLOY_ZEABUR.md"
    record "G11 前端" "SKIP" "未提供 --web-base"
else
    WB="${WEB_BASE%/}"
    # ① 前端站点本身可达
    WEB_CODE="$(curl -s -o /dev/null -m 30 -w '%{http_code}' "${WB}/" 2>/dev/null)"
    info "GET ${WB}/            → HTTP ${WEB_CODE:-000}"
    # ② 经**服务端代理**访问引擎（浏览器侧无需密钥；代理在服务端注入 X-API-Key）
    PROXY_CODE="$(curl -s -o "$TMP/web_health.json" -m 30 -w '%{http_code}' "${WB}/api/health" 2>/dev/null)"
    info "GET ${WB}/api/health  → HTTP ${PROXY_CODE:-000}（同源代理，密钥在服务端注入）"
    info "代理返回体(截断): $(head -c 300 "$TMP/web_health.json" 2>/dev/null)"
    G11_OK=1
    case "$WEB_CODE" in
        2*|3*) ok "G11a 前端站点可达（HTTP ${WEB_CODE}）" ;;
        *)     bad "G11a 前端站点不可达（HTTP ${WEB_CODE:-000}）—— 核对 Zeabur 域名/部署状态"; G11_OK=0 ;;
    esac
    if [ "$PROXY_CODE" = "200" ]; then
        ok "G11b 同源代理 /api/health → 200（前端→Zeabur→引擎链路通，且已配 ENGINE_API_BASE/KINETO_API_KEY）"
    elif [ "$PROXY_CODE" = "503" ]; then
        bad "G11b 代理返回 503 —— Zeabur 控制台未配置 ENGINE_API_BASE / KINETO_API_KEY（见 DEPLOY_ZEABUR.md）"; G11_OK=0
    elif [ "$PROXY_CODE" = "502" ]; then
        bad "G11b 代理返回 502 —— Zeabur 服务端连不上引擎（ENGINE_API_BASE 错？隧道未通？）"; G11_OK=0
    elif [ "$PROXY_CODE" = "404" ]; then
        warn "G11b 代理 /api/health → 404（旧版前端无代理路由？或引擎无 /health）—— 人工核对"
    else
        warn "G11b 代理 /api/health → HTTP ${PROXY_CODE:-000}（非 200；核对 ENGINE_API_BASE 与隧道）"
    fi
    if [ "$G11_OK" = "1" ]; then
        record "G11 前端" "PASS" "site=$WEB_CODE proxy=$PROXY_CODE"
    else
        record "G11 前端" "FAIL" "site=$WEB_CODE proxy=$PROXY_CODE"
    fi

    # ③ CORS 预检：仅当**仍保留浏览器→引擎直连**（旧架构 / 纵深防御）时才断言。
    #    新代理架构下浏览器只与 Zeabur 同源通信，引擎 CORS 对浏览器已非必需（KINETO_CORS_ORIGINS 降为可选）。
    if [ -n "$WEB_ORIGIN" ]; then
        info "带 Origin 的 CORS 预检：OPTIONS ${BASE}/health  Origin=${WEB_ORIGIN}"
        PRE_CODE="$(curl -s -o /dev/null -D "$TMP/cors_hdr.txt" -m 20 -w '%{http_code}' -X OPTIONS \
            -H "Origin: ${WEB_ORIGIN}" -H 'Access-Control-Request-Method: GET' \
            -H 'Access-Control-Request-Headers: x-api-key' "${BASE}/health" 2>/dev/null)"
        ACAO="$(grep -i '^access-control-allow-origin:' "$TMP/cors_hdr.txt" 2>/dev/null | head -n1 | tr -d '\r')"
        info "OPTIONS → HTTP ${PRE_CODE:-000}；${ACAO:-<无 Access-Control-Allow-Origin 头>}"
        case "$PRE_CODE" in
            2*)
                if [ -n "$ACAO" ]; then
                    ok "G11c CORS 预检通过（引擎回 Access-Control-Allow-Origin）"
                    record "G11c CORS" "PASS" "OPTIONS $PRE_CODE + ACAO"
                else
                    bad "G11c 预检 2xx 但无 Access-Control-Allow-Origin —— KINETO_CORS_ORIGINS 未含 ${WEB_ORIGIN}"
                    record "G11c CORS" "FAIL" "无 ACAO 头"
                fi ;;
            *)
                bad "G11c CORS 预检 HTTP ${PRE_CODE:-000}（期望 2xx）—— 若走新代理架构可不设 --web-origin 跳过本项"
                record "G11c CORS" "FAIL" "OPTIONS ${PRE_CODE:-000}"
                ;;
        esac
    else
        info "未提供 --web-origin：跳过 CORS 预检（新代理架构下浏览器不直连引擎，引擎 CORS 为可选纵深防御）"
        record "G11c CORS" "SKIP" "新代理架构，浏览器不直连引擎"
    fi
fi

# ---------------------------------------------------------------------------
# G12 / G8-SSOT  骨架单一事实源一致性（静态、只读；函数定义在脚本头部）
# ---------------------------------------------------------------------------
run_ssot_gate

# ---------------------------------------------------------------------------
# G13  黄金样本基线（静态、只读；断言 output_test 的 audit verdict=pass）
#   引擎侧整改后，黄金样本 output_test/audit_iter0/audit_results.json 必须
#   verdict=pass（而非 warn）。这是部署前的基线断言，把 warn 通胀挡在部署前。
#   文件不存在时记 SKIP（非仓库检出 / 引擎未重生成）。
# ---------------------------------------------------------------------------
step "G13  黄金样本基线（output_test audit verdict=pass）"
GOLDEN_DIR="${REPO_ROOT}/kineto-engine/output_test"
GOLDEN_AUDIT="${GOLDEN_DIR}/audit_iter0/audit_results.json"
if [ ! -f "$GOLDEN_AUDIT" ]; then
    warn "G13 跳过：黄金样本不存在（$GOLDEN_AUDIT）"
    info "    请在仓库检出上重生成：cd kineto-engine && python kineto_core.py --refine --output-dir output_test"
    record "G13 黄金样本" "SKIP" "audit_results.json 不存在"
else
    GOLDEN_VERDICT="$(json_get "$GOLDEN_AUDIT" "d.get('verdict','')")"
    GOLDEN_ISSUES="$(json_get "$GOLDEN_AUDIT" "d.get('total_issues','')")"
    GOLDEN_SCORE="$(json_get "$GOLDEN_AUDIT" "d.get('final_quality_score','')")"
    info "verdict=${GOLDEN_VERDICT:-<empty>}  total_issues=${GOLDEN_ISSUES:-<empty>}  final_quality_score=${GOLDEN_SCORE:-<empty>}"
    if [ "$GOLDEN_VERDICT" = "pass" ]; then
        ok "G13 黄金样本 verdict=pass（warn 通胀已修复，基线干净）"
        record "G13 黄金样本" "PASS" "verdict=pass"
    else
        bad "G13 黄金样本 verdict=${GOLDEN_VERDICT:-<empty>} != pass —— warn 通胀未修复或引擎未重生成"
        info "    期望：引擎侧 compute_verdict 整改后，干净数据必须 verdict=pass。"
        info "    排查：kineto-engine/output_test/audit_iter0/audit_results.json 的 total_issues/failure_reason"
        info "    修复：引擎整改后重跑 python kineto_core.py --refine --output-dir output_test"
        record "G13 黄金样本" "FAIL" "verdict=${GOLDEN_VERDICT:-empty}"
    fi
fi

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
printf '\n%s\n' "=============================================================================="
printf ' 验收汇总  (总耗时 %s)\n' "$(fmt_dur "$(( $(date +%s) - T0 ))")"
printf '%s\n' "------------------------------------------------------------------------------"
printf ' %-22s %-6s %s\n' "门禁" "结果" "说明"
printf '%s\n' "------------------------------------------------------------------------------"
FAIL_COUNT=0
printf '%s' "$RESULTS" | while IFS='|' read -r NAME ST DESC; do
    [ -z "$NAME" ] && continue
    case "$ST" in
        PASS) printf ' %s%-22s %-6s%s %s\n' "$C_GRN" "$NAME" "$ST" "$C_OFF" "$DESC" ;;
        FAIL) printf ' %s%-22s %-6s%s %s\n' "$C_RED" "$NAME" "$ST" "$C_OFF" "$DESC" ;;
        *)    printf ' %s%-22s %-6s%s %s\n' "$C_YEL" "$NAME" "$ST" "$C_OFF" "$DESC" ;;
    esac
done
# while 在子 shell 中，重新统计一次 FAIL 数
FAIL_COUNT="$(printf '%s' "$RESULTS" | grep -c '|FAIL|')"
WARN_COUNT="$(printf '%s' "$RESULTS" | grep -c '|WARN|')"
printf '%s\n' "------------------------------------------------------------------------------"
if [ "$FAIL_COUNT" -eq 0 ]; then
    printf '%s==> 验收通过（%d 个 WARN 需人工看一眼）。可以把公网地址交给 Zeabur 前端联调。%s\n' "$C_GRN" "$WARN_COUNT" "$C_OFF"
else
    printf '%s==> 验收未通过：%d 项 FAIL。请按 DEPLOY_MOFANG.md §8 的排障表处理，不要上线。%s\n' "$C_RED" "$FAIL_COUNT" "$C_OFF"
    if printf '%s' "$RESULTS" | grep -q '^G12 SSOT骨架|FAIL|'; then
        printf '%s    └─ 含 G12/G8-SSOT 骨架漂移：把 kineto-web/lib/skeleton.ts 对齐 kineto-engine/skeleton_spec.py%s\n' "$C_RED" "$C_OFF"
        printf '       （collar 13/14 的父 = spine3 9），然后单跑复验：bash deploy/validate.sh --ssot-only\n'
    fi
fi
printf '%s\n' "=============================================================================="
exit "$([ "$FAIL_COUNT" -eq 0 ] && echo 0 || echo 1)"
