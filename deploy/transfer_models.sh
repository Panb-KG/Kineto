#!/usr/bin/env bash
# =============================================================================
# Kineto / deploy — transfer_models.sh
# -----------------------------------------------------------------------------
# 运行处 : **Mac 本机**（模型权重与 SMPL 授权文件都在 Mac 上，设备上没有）
# 作用   : 把 4DHumans 缓存 + YOLOv8n + SMPL 基础模型 rsync 到设备的
#          /srv/kineto/models，并给出（可选执行）让缓存被正确解析的软链接布线。
# 安全性 : - 不使用 rsync --delete：**只增不删**，绝不会删掉设备上已有文件
#          - 不做任何 sudo 操作，除非显式传入 --wire 且二次确认
#          - 传输前打印完整计划并要求 [y/N] 确认
#          - 幂等：重复运行只会补齐差异（rsync 按 size+mtime 判断）
#
# 目标布局（设备端）:
#   /srv/kineto/models/
#   ├── 4DHumans/                      ← 4DHumans 缓存的**规范副本**
#   │   ├── logs/train/multiruns/hmr2/0/
#   │   │   ├── checkpoints/epoch=35-step=1000000.ckpt   (2.5 GB)
#   │   │   ├── model_config.yaml
#   │   │   └── dataset_config.yaml
#   │   └── data/
#   │       ├── smpl/SMPL_NEUTRAL.pkl  (37 MB)
#   │       ├── SMPL_to_J19.pkl
#   │       └── smpl_mean_params.npz
#   └── engine/                        ← kineto_core.py 用相对路径引用的权重
#       ├── yolov8n.pt                 (6.3 MB)
#       └── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl  (37 MB)
#
#   为什么是这个布局？hmr2/configs/__init__.py 里
#       CACHE_DIR = os.path.join(os.environ["HOME"], ".cache")
#       CACHE_DIR_4DHUMANS = CACHE_DIR + "/4DHumans"
#   所以缓存位置**完全由 HOME 决定**：
#     · systemd 主路径 : HOME=/srv/kineto  → /srv/kineto/.cache/4DHumans
#                        → 由 --wire 建软链指向 /srv/kineto/models/4DHumans
#     · Docker 备选路径 : 卷挂载 /srv/kineto/models:/root/.cache，HOME=/root
#                        → /root/.cache/4DHumans == /srv/kineto/models/4DHumans
#   两条路径共用同一份 2.5GB 权重，不重复占盘。
#
#   ⚠️ 明确排除 hmr2_data.tar.gz（2.5GB）：它只是 logs/ + data/ 的打包来源，
#      已解压过，内容重复。传它等于白烧 2.5GB 磁盘和一倍传输时间。
#      注意：hmr2/models/download_models() 会在该文件缺失时尝试重新下载，
#      但 kineto_core.py 走的是 load_hmr2()，**不会**触发下载。
#
# 用法:
#   bash deploy/transfer_models.sh --device kineto@192.168.1.107
#   bash deploy/transfer_models.sh --device kineto@192.168.1.107 --port 22 --dry-run
#   bash deploy/transfer_models.sh --device kineto@192.168.1.107 --wire     # 传完顺便布线
#   bash deploy/transfer_models.sh --device kineto@192.168.1.107 --quick    # 只校验大小
#
# 依赖: rsync (macOS 自带), ssh, shasum|sha256sum
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# 参数与默认值
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEVICE="${KINETO_DEVICE:-}"
SSH_PORT="${KINETO_SSH_PORT:-22}"
REMOTE_DIR="${KINETO_REMOTE_MODELS:-/srv/kineto/models}"
LOCAL_CACHE="${KINETO_LOCAL_CACHE:-$HOME/.cache/4DHumans}"
ENGINE_DIR="$REPO_ROOT/kineto-engine"

DO_WIRE=0
DRY_RUN=0
QUICK_VERIFY=0
ASSUME_YES=0

usage() {
    cat <<'EOF'
用法: bash deploy/transfer_models.sh --device <user@host> [选项]

必填:
  --device USER@HOST      设备 SSH 目标（也可用环境变量 KINETO_DEVICE）

选项:
  --port N                SSH 端口（默认 22）
  --remote-dir PATH       设备端模型根目录（默认 /srv/kineto/models）
  --local-cache PATH      Mac 上的 4DHumans 缓存（默认 ~/.cache/4DHumans）
  --wire                  传输后在设备上创建软链接/目录并修正属主（需 sudo，会二次确认）
  --dry-run               只打印 rsync 将做什么，不实际传输
  --quick                 校验只看文件大小（跳过 2.5GB 的 sha256，省 ~2 分钟）
  --yes                   跳过确认提示（仅在 CI/脚本化场景使用；仍不会执行破坏性操作）
  -h, --help              显示本帮助

环境变量: KINETO_DEVICE / KINETO_SSH_PORT / KINETO_REMOTE_MODELS / KINETO_LOCAL_CACHE
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --device)      DEVICE="${2:-}"; shift 2 ;;
        --port)        SSH_PORT="${2:-22}"; shift 2 ;;
        --remote-dir)  REMOTE_DIR="${2:-}"; shift 2 ;;
        --local-cache) LOCAL_CACHE="${2:-}"; shift 2 ;;
        --wire)        DO_WIRE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --quick)       QUICK_VERIFY=1; shift ;;
        --yes)         ASSUME_YES=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             printf '未知参数: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done

C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_CYN=$'\033[36m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_RED=""; C_GRN=""; C_YEL=""; C_CYN=""; C_OFF=""; }
ok()   { printf '%s[PASS]%s %s\n' "$C_GRN" "$C_OFF" "$1"; }
warn() { printf '%s[WARN]%s %s\n' "$C_YEL" "$C_OFF" "$1"; }
bad()  { printf '%s[FAIL]%s %s\n' "$C_RED" "$C_OFF" "$1"; }
step() { printf '\n%s── %s%s\n' "$C_CYN" "$1" "$C_OFF"; }

confirm() {
    # confirm "<问题>" —— 返回 0 表示用户确认继续
    local prompt="$1" ans
    if [ "$ASSUME_YES" = "1" ]; then
        printf '%s (由 --yes 自动确认)\n' "$prompt"
        return 0
    fi
    printf '\n%s%s%s\n' "$C_YEL" "$prompt" "$C_OFF"
    printf 'yes 继续 / 其它任意键取消: '
    read -r ans
    case "$ans" in
        y|Y|yes|YES) return 0 ;;
        *)           printf '已取消。\n'; return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# 0. 前置检查（全部只读）
# ---------------------------------------------------------------------------
step "0/5 前置检查"
FAIL=0

if [ -z "$DEVICE" ]; then
    bad "缺少 --device USER@HOST（例如 --device kineto@192.168.1.107）"
    usage; exit 2
fi
if ! command -v rsync >/dev/null 2>&1; then bad "本机没有 rsync"; exit 1; fi
if ! command -v ssh    >/dev/null 2>&1; then bad "本机没有 ssh"; exit 1; fi

SHA_LOCAL=""
if command -v sha256sum >/dev/null 2>&1; then SHA_LOCAL="sha256sum"
elif command -v shasum >/dev/null 2>&1; then SHA_LOCAL="shasum -a 256"
else warn "本机无 sha256sum/shasum —— 将退化为只比对文件大小"; QUICK_VERIFY=1; fi

[ -d "$LOCAL_CACHE" ]        && ok "本地缓存存在: $LOCAL_CACHE ($(du -sh "$LOCAL_CACHE" 2>/dev/null | awk '{print $1}'))" \
                             || { bad "本地缓存缺失: ${LOCAL_CACHE}（先在本机跑通一次 kineto_core.py 以下载权重）"; FAIL=1; }
[ -f "$ENGINE_DIR/yolov8n.pt" ] && ok "本地 yolov8n.pt 存在" \
                             || { bad "缺少 $ENGINE_DIR/yolov8n.pt"; FAIL=1; }

SMPL_PKL="$ENGINE_DIR/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
# [Ultra Review] SSOT 落地后 basicModel pkl 不再是“可选”：skeleton_spec 的惰性加载
# 在首次推理时触发，缺 pkl → FileNotFoundError → job 失败。warn 升级为 bad+FAIL。
[ -f "$SMPL_PKL" ]           && ok "本地 SMPL 基础模型存在 ($(du -h "$SMPL_PKL" | awk '{print $1}'))" \
                             || { bad "缺少 ${SMPL_PKL}（SSOT 惰性加载在首次推理时触发，缺 pkl → job 失败；必须提供）"; FAIL=1; }

HMR2_CKPT="$LOCAL_CACHE/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
[ -f "$HMR2_CKPT" ]          && ok "HMR2 权重存在 (2.5GB 主文件)" \
                             || { bad "缺少 HMR2 权重: $HMR2_CKPT"; FAIL=1; }
[ -f "$LOCAL_CACHE/logs/train/multiruns/hmr2/0/model_config.yaml" ] \
                             && ok "model_config.yaml 存在（load_hmr2 必需）" \
                             || { bad "缺少 model_config.yaml —— load_hmr2() 会直接抛错"; FAIL=1; }
[ -f "$LOCAL_CACHE/data/smpl/SMPL_NEUTRAL.pkl" ] \
                             && ok "SMPL_NEUTRAL.pkl 存在（check_smpl_exists 必需）" \
                             || warn "缺少 data/smpl/SMPL_NEUTRAL.pkl —— 引擎会尝试从 basicModel_*.pkl 转换（需 chumpy 可用）"

if [ -d "$LOCAL_CACHE" ] && [ -f "$LOCAL_CACHE/hmr2_data.tar.gz" ]; then
    warn "本地存在 hmr2_data.tar.gz (2.5GB) —— 已按设计**排除**，不会传输"
fi

if [ "$FAIL" -ne 0 ]; then
    bad "前置检查未通过，退出（未做任何改动）"
    exit 1
fi

step "0b/5 连通性与设备端空间（只读）"
SSH=(ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 "$DEVICE")
if ! "${SSH[@]}" 'echo ok' >/dev/null 2>&1; then
    bad "SSH 连不上 ${DEVICE}:${SSH_PORT}（BatchMode 下无法交互输密码 —— 请先配好 ssh key，或用 ssh-copy-id）"
    exit 1
fi
ok "SSH 可达: $DEVICE:$SSH_PORT"

REMOTE_USER="$("${SSH[@]}" 'id -un' 2>/dev/null)"
REMOTE_FREE_KB="$("${SSH[@]}" "df -k ${REMOTE_DIR%/*} 2>/dev/null | awk 'NR==2{print \$4}'" 2>/dev/null)"
REMOTE_FREE_GB=$(( ${REMOTE_FREE_KB:-0} / 1024 / 1024 ))
NEED_GB=4
if [ "$REMOTE_FREE_GB" -ge "$NEED_GB" ]; then
    ok "设备端可用空间 ${REMOTE_FREE_GB}GB ≥ ${NEED_GB}GB"
else
    bad "设备端可用空间仅 ${REMOTE_FREE_GB}GB（< ${NEED_GB}GB）—— 先清盘再传"
    exit 1
fi
printf '      设备用户: %s   目标目录: %s\n' "$REMOTE_USER" "$REMOTE_DIR"

# ---------------------------------------------------------------------------
# 1. 打印计划 + 确认
# ---------------------------------------------------------------------------
step "1/5 传输计划（执行前请核对）"
cat <<EOF
  本机 → 设备           : $(whoami)@$HOSTNAME → $DEVICE:$SSH_PORT
  远端根目录            : $REMOTE_DIR
  传输项 1  $LOCAL_CACHE/{logs,data}  →  $REMOTE_DIR/4DHumans/
            排除: hmr2_data.tar.gz / .DS_Store / __pycache__/ / *.log
            包含: logs/*** (HMR2 ckpt + model_config.yaml + dataset_config.yaml)
                  data/*** (smpl/SMPL_NEUTRAL.pkl, SMPL_to_J19.pkl, smpl_mean_params.npz)
  传输项 2  $ENGINE_DIR/yolov8n.pt   →  $REMOTE_DIR/engine/yolov8n.pt
  传输项 3  4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
                                       →  $REMOTE_DIR/engine/basicModel_...pkl
  rsync 参数            : -az --partial --info=progress2  (**无 --delete**)
  预计流量              : ≈ 2.6 GB（千兆局域网约 30-60s，Wi-Fi 可能 10min+）
  校验方式              : $([ "$QUICK_VERIFY" = 1 ] && echo '仅比对文件大小' || echo '逐文件 sha256 比对')
  是否布线(--wire)      : $([ "$DO_WIRE" = 1 ] && echo '是（传输后创建软链接，需设备 sudo，会二次确认）' || echo '否（只打印命令，由你手动执行）')
  试运行(--dry-run)     : $([ "$DRY_RUN" = 1 ] && echo '是（不写任何数据）' || echo '否')

  ${C_GRN}本脚本不会删除设备上的任何文件，也不会改动 systemd / nginx / MoFang。${C_OFF}
EOF

# macOS 15+ 自带的是 openrsync（功能子集，不支持 --info=progress2），
# 先探测能力再决定进度条参数；如需完整体验可 `brew install rsync`。
RSYNC_FLAGS=(-az --partial
             --exclude 'hmr2_data.tar.gz' --exclude '.DS_Store'
             --exclude '__pycache__/' --exclude '*.log')
if rsync --version 2>&1 | head -n1 | grep -qi 'openrsync'; then
    RSYNC_FLAGS+=(-v)
    warn "本机 rsync 为 openrsync（macOS 自带）—— 使用 -v 代替进度条；如需进度条请 brew install rsync"
else
    RSYNC_FLAGS+=(--human-readable --info=progress2)
fi
[ "$DRY_RUN" = "1" ] && RSYNC_FLAGS+=(--dry-run --itemize-changes)

confirm "确认按上述计划开始 rsync 传输？" || exit 0

# ---------------------------------------------------------------------------
# 2. 创建远端目录（普通用户可写；--wire 阶段再改属主给 kineto）
# ---------------------------------------------------------------------------
step "2/5 准备远端目录"
if [ "$DRY_RUN" = "1" ]; then
    warn "dry-run：跳过目录创建"
else
    "${SSH[@]}" "mkdir -p '$REMOTE_DIR/4DHumans' '$REMOTE_DIR/engine'" \
        && ok "远端目录就绪: $REMOTE_DIR/{4DHumans,engine}" \
        || { bad "无法创建远端目录（${REMOTE_DIR} 可能需要 sudo mkdir + chown ${REMOTE_USER}）"; exit 1; }
fi

# ---------------------------------------------------------------------------
# 3. 执行 rsync
# ---------------------------------------------------------------------------
RSYNC_SSH="ssh -p $SSH_PORT -o BatchMode=yes"
T_START=$(date +%s)

step "3/5 rsync ① 4DHumans 缓存 (logs/*** + data/***)"
printf '      %s\n' "rsync ${RSYNC_FLAGS[*]} --include '/logs/***' --include '/data/***' --exclude '*' $LOCAL_CACHE/ $DEVICE:$REMOTE_DIR/4DHumans/"
rsync "${RSYNC_FLAGS[@]}" \
      -e "$RSYNC_SSH" \
      --include '/logs/***' --include '/data/***' --exclude '*' \
      "$LOCAL_CACHE/" "$DEVICE:$REMOTE_DIR/4DHumans/"
RC1=$?
[ $RC1 -eq 0 ] && ok "4DHumans 缓存同步完成" || bad "4DHumans 缓存同步失败 (rc=$RC1)"

step "3b/5 rsync ② YOLOv8n 权重"
rsync "${RSYNC_FLAGS[@]}" -e "$RSYNC_SSH" \
      "$ENGINE_DIR/yolov8n.pt" "$DEVICE:$REMOTE_DIR/engine/yolov8n.pt"
RC2=$?
[ $RC2 -eq 0 ] && ok "yolov8n.pt 同步完成" || bad "yolov8n.pt 同步失败 (rc=$RC2)"

step "3c/5 rsync ③ SMPL 基础模型"
if [ -f "$SMPL_PKL" ]; then
    rsync "${RSYNC_FLAGS[@]}" -e "$RSYNC_SSH" \
          "$SMPL_PKL" "$DEVICE:$REMOTE_DIR/engine/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    RC3=$?
    [ $RC3 -eq 0 ] && ok "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 同步完成" || bad "SMPL pkl 同步失败 (rc=$RC3)"
else
    RC3=0; warn "本地无 SMPL pkl，跳过（缓存里的 SMPL_NEUTRAL.pkl 已满足 check_smpl_exists）"
fi

T_END=$(date +%s)
printf '\n      传输耗时: %s 秒\n' "$((T_END - T_START))"

if [ $RC1 -ne 0 ] || [ $RC2 -ne 0 ] || [ $RC3 -ne 0 ]; then
    bad "rsync 存在失败项 —— 修复后重跑本脚本即可（幂等，已传部分会跳过）"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. 校验（sha256 或 size）
# ---------------------------------------------------------------------------
step "4/5 完整性校验"

sha_local() {  # sha_local <path>
    if [ -n "$SHA_LOCAL" ]; then $SHA_LOCAL "$1" 2>/dev/null | awk '{print $1}'; fi
}
sha_remote() { # sha_remote <remote-abs-path>
    "${SSH[@]}" "sha256sum '$1' 2>/dev/null | awk '{print \$1}'" 2>/dev/null
}
size_local()  { stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null; }
size_remote() { "${SSH[@]}" "stat -c%s '$1' 2>/dev/null" 2>/dev/null; }

VERIFY_FAIL=0
# 待校验清单： "<本地绝对路径>|<远端绝对路径>"
PAIRS=""
add_pair() { PAIRS="$PAIRS$1|$2
"; }

add_pair "$HMR2_CKPT" "$REMOTE_DIR/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
add_pair "$LOCAL_CACHE/logs/train/multiruns/hmr2/0/model_config.yaml" "$REMOTE_DIR/4DHumans/logs/train/multiruns/hmr2/0/model_config.yaml"
add_pair "$LOCAL_CACHE/data/SMPL_to_J19.pkl" "$REMOTE_DIR/4DHumans/data/SMPL_to_J19.pkl"
add_pair "$LOCAL_CACHE/data/smpl_mean_params.npz" "$REMOTE_DIR/4DHumans/data/smpl_mean_params.npz"
add_pair "$LOCAL_CACHE/data/smpl/SMPL_NEUTRAL.pkl" "$REMOTE_DIR/4DHumans/data/smpl/SMPL_NEUTRAL.pkl"
add_pair "$ENGINE_DIR/yolov8n.pt" "$REMOTE_DIR/engine/yolov8n.pt"
if [ -f "$SMPL_PKL" ]; then
    add_pair "$SMPL_PKL" "$REMOTE_DIR/engine/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
fi

while IFS='|' read -r LP RP; do
    [ -z "$LP" ] && continue
    if [ ! -f "$LP" ]; then
        warn "本地缺失，跳过: $LP"
        continue
    fi
    NAME="$(basename "$LP")"
    LS="$(size_local "$LP")"; RS="$(size_remote "$RP")"
    if [ -z "$RS" ]; then
        bad "$NAME — 远端不存在: $RP"; VERIFY_FAIL=1
        continue
    fi
    if [ "$LS" != "$RS" ]; then
        bad "$NAME — 大小不一致 (本地 $LS / 远端 $RS)"; VERIFY_FAIL=1
        continue
    fi
    if [ "$QUICK_VERIFY" = "1" ]; then
        ok "$NAME — 大小一致 ($LS bytes)"
    else
        printf '      校验 %s (本地/远端 sha256，2.5GB 约需 30-60s)...\n' "$NAME"
        LSH="$(sha_local "$LP")"; RSH="$(sha_remote "$RP")"
        if [ -n "$LSH" ] && [ "$LSH" = "$RSH" ]; then
            ok "$NAME — sha256 一致 (${LSH:0:16}…)"
        else
            bad "$NAME — sha256 不一致 (本地 ${LSH:0:16}… / 远端 ${RSH:0:16}…) → 重跑本脚本"
            VERIFY_FAIL=1
        fi
    fi
done < <(printf '%s\n' "$PAIRS")
step "4b/5 远端清单复核"
"${SSH[@]}" "find '$REMOTE_DIR' -maxdepth 6 -type f -printf '%10s  %p\n' 2>/dev/null | sort -k2" \
    || "${SSH[@]}" "find '$REMOTE_DIR' -type f -exec ls -l {} \; 2>/dev/null | awk '{print \$5, \$9}'"
REMOTE_COUNT="$("${SSH[@]}" "find '$REMOTE_DIR' -type f 2>/dev/null | wc -l" | tr -d ' ')"
if [ "${REMOTE_COUNT:-0}" -ge 6 ]; then
    ok "远端共 $REMOTE_COUNT 个文件（期望 ≥ 6）"
else
    bad "远端仅 $REMOTE_COUNT 个文件 —— 校验段落里有 FAIL，请重跑"
    VERIFY_FAIL=1
fi

# ---------------------------------------------------------------------------
# 5. 布线（可选，需设备 sudo；默认只打印）
# ---------------------------------------------------------------------------
step "5/5 缓存解析布线"
WIRE_CMDS=$(cat <<EOF
set -euo pipefail
# 1) 目录骨架
sudo mkdir -p $REMOTE_DIR /srv/kineto/.cache /srv/kineto/jobs
# 2) systemd 路径: HOME=/srv/kineto → \$HOME/.cache/4DHumans 必须解析到规范副本
sudo ln -sfn $REMOTE_DIR/4DHumans /srv/kineto/.cache/4DHumans
# 3) 引擎相对路径引用的权重（kineto_core.py 以 /opt/kineto/kineto-engine 为 cwd）
if [ -d /opt/kineto/kineto-engine ]; then
  sudo ln -sfn $REMOTE_DIR/engine/yolov8n.pt /opt/kineto/kineto-engine/yolov8n.pt
  sudo mkdir -p /opt/kineto/kineto-engine/4D-Humans/data
  sudo ln -sfn $REMOTE_DIR/engine/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl \\
       /opt/kineto/kineto-engine/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
fi
# 4) 属主/权限：kineto 服务账号必须能读模型、能写 jobs
if id kineto >/dev/null 2>&1; then
  sudo chown -R kineto:kineto /srv/kineto
fi
sudo chmod -R a+rX $REMOTE_DIR
sudo chmod 0755 /srv/kineto/jobs
# 5) 自检：缓存路径是否真的能解析
ls -lL /srv/kineto/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/ >/dev/null && echo '[wire] cache OK'
EOF
)

printf '%s\n' "  需要在**设备上**执行的布线命令："
printf '%s\n' "$WIRE_CMDS" | sed 's/^/      /'

if [ "$DO_WIRE" = "1" ] && [ "$DRY_RUN" != "1" ]; then
    if confirm "确认通过 SSH 在设备上执行上述布线命令（含 sudo）？"; then
        printf '%s\n' "$WIRE_CMDS" | "${SSH[@]}" 'bash -s' && ok "布线完成" || bad "布线失败（可手动逐条执行上面的命令）"
    else
        warn "已跳过布线"
    fi
elif [ "$DRY_RUN" = "1" ]; then
    warn "dry-run：不执行布线"
else
    printf '\n      %s提示%s: 加 --wire 可让本脚本代为执行；或把上面的命令复制到设备终端里跑。\n' "$C_YEL" "$C_OFF"
    printf '      Docker 备选路径**无需**第 2 步软链（compose 直接把 %s 挂到 /root/.cache）。\n' "$REMOTE_DIR"
fi

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
printf '\n%s\n' "=============================================================================="
if [ "$VERIFY_FAIL" -eq 0 ]; then
    printf '%s传输与校验完成。%s 下一步：\n' "$C_GRN" "$C_OFF"
    printf '  1) 完成上面的布线（--wire 或手动）\n'
    printf '  2) sudo -u kineto HOME=/srv/kineto /opt/kineto/venv/bin/python -c \\\n'
    printf '       "from hmr2.configs import CACHE_DIR_4DHUMANS; import os; print(CACHE_DIR_4DHUMANS, os.path.isdir(CACHE_DIR_4DHUMANS))"\n'
    printf '  3) bash deploy/validate.sh   # E2E 验收\n'
    printf '\n%s⚠ weights_only 补丁提醒%s：本脚本只传模型权重，**不传 4D-Humans 代码树**。\n' "$C_YEL" "$C_OFF"
    printf '   设备端的 4D-Humans/ 必须带 torch>=2.6 的 weights_only=False 补丁，否则加载 .ckpt 会 UnpicklingError：\n'
    printf '     · 若代码树是从本仓库 rsync 过去的（DEPLOY_MOFANG.md §5.5）——已含补丁，无需再做。\n'
    printf '     · 若设备端是 `git clone` 上游 4D-Humans——必须补上：\n'
    printf '         cd <设备>/kineto-engine/4D-Humans && git apply -p1 /opt/kineto/deploy/patches/hmr2_weights_only.patch\n'
    printf '     · 校验：`grep -q weights_only <设备>/kineto-engine/4D-Humans/hmr2/models/__init__.py && echo OK`\n'
else
    printf '%s存在校验失败项，请按上面的 FAIL 提示重跑本脚本（幂等）。%s\n' "$C_RED" "$C_OFF"
fi
printf '%s\n' "=============================================================================="
exit "$VERIFY_FAIL"
