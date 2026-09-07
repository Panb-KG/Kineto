#!/usr/bin/env bash
# =============================================================================
# Kineto / deploy — strip_mofang.sh   (可逆剥离 MoFang 上层业务服务)
# -----------------------------------------------------------------------------
# 运行处 : 设备端（用户 SSH 登录后自己跑；需要 sudo 才能真正 disable）。
# 作用   : 把 DEPLOY_MOFANG.md §5.2「停用 MoFang 上层业务服务」脚本化，取代易错的
#          手工 for 循环 disable。核心是**安全**：
#            · 禁停硬拦截 —— 候选单元命中禁停清单（SSH/网络/systemd/GPU/风扇/看门狗/
#              OTA/固件/授权…）立即 exit 1，绝不停用；
#            · 只用 `systemctl disable --now`（**不 mask、不 purge、不卸载**），随时可逆；
#            · 自动生成 undo 文件；`--rollback` 读它 `systemctl enable --now` 一键恢复；
#            · `--dry-run` 只打印「将停用 / 将拒绝」的单元，不改动任何状态。
# 安全性 : 默认交互确认（除非 --yes）；不装包、不改配置、不删文件、不动 nginx。
#          唯一的状态改动是对**非禁停**候选单元执行 disable --now（可逆）。
# 幂等性 : disable --now / enable --now 均可重复执行。
#
# 候选单元来源（三选一，可组合）：
#   --units "a.service b.service"   直接给定
#   --from-file PATH                从 §5.1 存证或 discover_device.sh §10 输出里抽取 *.service
#   （都不给时）--pattern REGEX      现场 systemctl list-unit-files 发现（默认 mofang-ish）
#
# 用法:
#   bash deploy/strip_mofang.sh --dry-run                       # 先看会停哪些、会拒哪些
#   bash deploy/strip_mofang.sh --from-file ~/kineto-rollback-evidence/all_services_before.txt
#   bash deploy/strip_mofang.sh --units "mofang-assistant.service mofang-bridge.service"
#   bash deploy/strip_mofang.sh --rollback ~/kineto-rollback-evidence/strip_undo_<ts>.txt   # 回滚（直接跟路径）
#   bash deploy/strip_mofang.sh --rollback --undo ~/kineto-rollback-evidence/strip_undo_*.txt
#   # 误拦逃生阀（需 --yes，会打印并记录 [OVERRIDE] 审计行；仅供人工核对确认安全后使用）:
#   bash deploy/strip_mofang.sh --units "some-unit.service" --allow-forbidden --yes
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# 颜色 / 输出辅助
# ---------------------------------------------------------------------------
C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_CYN=$'\033[36m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_RED=""; C_GRN=""; C_YEL=""; C_CYN=""; C_OFF=""; }
ok()   { printf '%s[ OK ]%s %s\n' "$C_GRN" "$C_OFF" "$1"; }
warn() { printf '%s[WARN]%s %s\n' "$C_YEL" "$C_OFF" "$1"; }
bad()  { printf '%s[BLOCK]%s %s\n' "$C_RED" "$C_OFF" "$1"; }
info() { printf '      %s\n' "$1"; }
step() { printf '\n%s── %s%s\n' "$C_CYN" "$1" "$C_OFF"; }

# ---------------------------------------------------------------------------
# 禁停清单（硬拦截）—— 命中任意禁停词即拒绝停用，宁可误拒不可误停。
#   说明：用**前导词边界** \b 匹配（token 处于词首或 - _ . @ 等分隔符之后）。
#   这样 'fan' 只会命中 fancontrol/风扇类单元，而**不会**误伤 'mofang' 里的 fan
#   （mofang 的 fan 前面是词字符 o，无词边界）；'ota'/'power' 等短词同理不会误命中
#   storage/empower 之类。代价是极端内嵌写法（如 'openssh'）可能漏配，故显式列入 openssh。
#   ★ 'xe'（Intel Xe GPU 驱动）用 \bxe\b **精确 token** 匹配：设备是 Xeon 平台的 Intel AI Box，
#     若只用前导 \bxe 会把 'mofang-xeon-agent.service' 里的 xeon 误判命中而 REJECT；xe\b 要求
#     xe 自成 token（后接分隔符/结尾），故 xeon 被 ALLOW、而 xe/xe-drm 等仍被拦截。
#   ★ 另补拦时钟同步/自动升级/snap/权限/审计类系统单元，避免被误停导致设备失联或安全降级。
# ---------------------------------------------------------------------------
FORBIDDEN_RE='\b(ssh|sshd|openssh|network|systemd|systemd-timesyncd|getty|dbus|nginx|docker|containerd|udev|kmod|intel|i915|xe\b|gpu|drm|fwupd|thermald|thermal|chrony|ntp|ntpsec|cron|crond|rsyslog|fan|power|battery|watchdog|ota|firmware|license|unattended-upgrades|snapd|polkit|apparmor|audit)'

# 现场发现时用的候选正则（与 discover_device.sh §10 对齐；rag 用词边界 \brag\b 避免误命中 storage）
PATTERN='mofang|bridge|claw|\brag\b|assistant'

DEFAULT_EVIDENCE_DIR="$HOME/kineto-rollback-evidence"
UNDO=""
ROLLBACK=0
DRY_RUN=0
ASSUME_YES=0
ALLOW_FORBIDDEN=0
FROM_FILE=""
UNITS_ARG=""
DO_LIVE=0

usage() {
    cat <<EOF
用法: bash deploy/strip_mofang.sh [选项]

可逆地停用 MoFang 上层业务服务（DEPLOY_MOFANG.md §5.2）。只做 \`systemctl disable --now\`，
不 mask、不 purge、不卸载；命中禁停清单的单元会被**硬拦截并 exit 1**。

候选来源（可组合；都不给则现场发现）:
  --units "A.service B.service"   直接指定候选单元
  --from-file PATH                从存证/discover 输出文件里抽取 *.service（§5.1 / discover §10）
  --pattern REGEX                 现场 systemctl list-unit-files 的过滤正则（默认 '$PATTERN'）

选项:
  --dry-run               只打印「将停用 / 将拒绝」的单元，不改动任何状态
  --undo PATH             undo 文件路径（默认 $DEFAULT_EVIDENCE_DIR/strip_undo_<时间戳>.txt）
  --rollback [PATH...]    读取 undo 文件并 systemctl enable --now 恢复；可直接跟 undo 路径
                          （等价于 --rollback --undo PATH），不给路径则自动取最新的 undo 文件
  --allow-forbidden       逃生阀：允许停用命中禁停清单的单元（**必须**同时 --yes；会打印并
                          记录 [OVERRIDE] 审计行）。仅供人工核对确认「确属误拦」后使用，切勿滥用
  --yes                   跳过交互确认（CI/脚本化；仍受禁停硬拦截保护，除非再加 --allow-forbidden）
  -h, --help              显示本帮助

退出码: 0=成功/已回滚/dry-run 完成   1=触发禁停硬拦截或执行失败   2=参数错误
EOF
}

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --units)      UNITS_ARG="${2:-}"; shift 2 ;;
        --from-file)  FROM_FILE="${2:-}"; shift 2 ;;
        --pattern)    PATTERN="${2:-}"; DO_LIVE=1; shift 2 ;;
        --undo)       UNDO="${2:-}"; shift 2 ;;
        --rollback)
            ROLLBACK=1; shift
            # 吞掉紧随其后的「非选项」参数作为 undo 路径（支持 glob 展开成多个）；
            # 这样 `--rollback PATH`、`--rollback A B`、`--rollback --undo PATH` 都可接受。
            while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do
                UNDO="${UNDO:+$UNDO }$1"; shift
            done
            ;;
        --allow-forbidden) ALLOW_FORBIDDEN=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --yes)        ASSUME_YES=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) printf '未知参数: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done

# --allow-forbidden 是危险逃生阀：必须显式 --yes，避免脚本化误用
if [ "$ALLOW_FORBIDDEN" = "1" ] && [ "$ASSUME_YES" != "1" ]; then
    bad "--allow-forbidden 必须与 --yes 同用（强制停用禁停单元是高危操作，需显式确认）。"
    exit 2
fi

# ---------------------------------------------------------------------------
# sudo 能力探测（disable/enable 需要 root）
# ---------------------------------------------------------------------------
SUDO=""
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    SUDO="sudo -n"
else
    SUDO="sudo"   # 交互 sudo（真正执行时会提示输密码）；dry-run 不需要
fi

confirm() {
    local prompt="$1" ans
    if [ "$ASSUME_YES" = "1" ]; then printf '%s (由 --yes 自动确认)\n' "$prompt"; return 0; fi
    printf '\n%s%s%s\n' "$C_YEL" "$prompt" "$C_OFF"
    printf 'yes 继续 / 其它任意键取消: '
    read -r ans
    case "$ans" in y|Y|yes|YES) return 0 ;; *) printf '已取消。\n'; return 1 ;; esac
}

# 从任意文本抽取形如 xxx.service 的单元名（去重、保序）
extract_units() {  # stdin → stdout
    grep -oE '[A-Za-z0-9@_.\\-]+\.service' 2>/dev/null | awk '!seen[$0]++'
}

# ===========================================================================
# 回滚模式
# ===========================================================================
if [ "$ROLLBACK" = "1" ]; then
    step "回滚模式 (--rollback)：按 undo 文件 systemctl enable --now 恢复"
    # 展开可能的 glob（用户传 strip_undo_*.txt）
    UNDO_FILES=""
    for f in $UNDO; do [ -f "$f" ] && UNDO_FILES="$UNDO_FILES$f"$'\n'; done
    if [ -z "$UNDO" ]; then
        # 未指定 → 找 evidence 目录里最新的 undo 文件
        UNDO_FILES="$(ls -1t "$DEFAULT_EVIDENCE_DIR"/strip_undo_*.txt 2>/dev/null | head -n1)"
        [ -n "$UNDO_FILES" ] && UNDO_FILES="$UNDO_FILES"$'\n'
    fi
    if [ -z "$(printf '%s' "$UNDO_FILES" | tr -d '[:space:]')" ]; then
        bad "找不到可读的 undo 文件（用 --undo PATH 指定，或确认 $DEFAULT_EVIDENCE_DIR/strip_undo_*.txt 存在）"
        exit 1
    fi
    RC=0
    # 用 here-doc 而非 `printf | while`：管道会把 while 放进子 shell，RC 的累加将丢失（exit 恒 0）。
    while IFS= read -r uf; do
        [ -z "$uf" ] && continue
        info "undo 文件: $uf"
        while IFS= read -r u; do
            [ -z "$u" ] && continue
            if [ "$DRY_RUN" = "1" ]; then
                info "[dry-run] 将恢复: systemctl enable --now $u"
            else
                printf '      恢复 %s ... ' "$u"
                if $SUDO systemctl enable --now "$u" 2>/dev/null; then
                    printf '%sOK%s\n' "$C_GRN" "$C_OFF"
                else
                    printf '%sFAIL%s\n' "$C_RED" "$C_OFF"; RC=1
                fi
            fi
        done < "$uf"
    done <<EOF_UNDO
$UNDO_FILES
EOF_UNDO
    if [ "$RC" -ne 0 ]; then
        bad "回滚中有单元 enable --now 失败（见上）——请 systemctl status <unit> 排查后重跑（幂等）。"
    else
        ok "回滚流程完成（disable 是可逆的；undo 文件内所有单元已 enable --now 恢复）"
    fi
    exit "$RC"
fi

# ===========================================================================
# 收集候选单元
# ===========================================================================
step "收集候选单元"
CANDIDATES=""

if [ -n "$UNITS_ARG" ]; then
    add="$(printf '%s\n' $UNITS_ARG | extract_units)"
    CANDIDATES="$CANDIDATES$add"$'\n'
    info "来自 --units: $(printf '%s' "$add" | tr '\n' ' ')"
fi

if [ -n "$FROM_FILE" ]; then
    if [ ! -r "$FROM_FILE" ]; then
        bad "--from-file 不可读: $FROM_FILE"; exit 2
    fi
    add="$(extract_units < "$FROM_FILE")"
    # 从存证文件抽取时，只保留 mofang-ish（避免把整机所有 service 都当候选）
    add="$(printf '%s\n' "$add" | grep -iE "$PATTERN" 2>/dev/null)"
    CANDIDATES="$CANDIDATES$add"$'\n'
    info "来自 --from-file ($FROM_FILE)，按 pattern '$PATTERN' 过滤后: $(printf '%s' "$add" | tr '\n' ' ')"
fi

# 若既没 --units 也没 --from-file（或显式给了 --pattern），则现场发现
if [ -z "$(printf '%s' "$CANDIDATES" | tr -d '[:space:]')" ] || [ "$DO_LIVE" = "1" ]; then
    if command -v systemctl >/dev/null 2>&1; then
        add="$(systemctl list-unit-files --type=service --no-legend 2>/dev/null | awk '{print $1}' | grep -iE "$PATTERN" 2>/dev/null)"
        CANDIDATES="$CANDIDATES$add"$'\n'
        info "现场发现 (systemctl list-unit-files, pattern '$PATTERN'): $(printf '%s' "$add" | tr '\n' ' ')"
    else
        warn "本机无 systemctl（非设备端？）——只能靠 --units / --from-file 提供候选"
    fi
fi

# 去重
CANDIDATES="$(printf '%s' "$CANDIDATES" | grep -E '\.service$' | awk 'NF' | awk '!seen[$0]++')"

if [ -z "$CANDIDATES" ]; then
    warn "没有任何候选单元。用 --units / --from-file 指定，或确认设备上确有 mofang-ish 服务。"
    info "查看候选: systemctl list-unit-files --type=service | grep -iE '$PATTERN'"
    exit 0
fi

# ===========================================================================
# 禁停硬拦截分类
# ===========================================================================
step "禁停清单硬拦截 (前导词边界 \\b 匹配；默认命中即拒绝，--allow-forbidden 可强制放行并审计)"
ALLOWED=""
REJECTED=""
while IFS= read -r u; do
    [ -z "$u" ] && continue
    lname="$(printf '%s' "$u" | tr 'A-Z' 'a-z')"
    if printf '%s' "$lname" | grep -qE "$FORBIDDEN_RE"; then
        hit="$(printf '%s' "$lname" | grep -oE "$FORBIDDEN_RE" | head -n1)"
        REJECTED="$REJECTED$u	(命中禁停词: $hit)"$'\n'
    else
        ALLOWED="$ALLOWED$u"$'\n'
    fi
done <<EOF2
$CANDIDATES
EOF2

printf '\n%s将拒绝（禁停硬拦截）:%s\n' "$C_RED" "$C_OFF"
if [ -n "$(printf '%s' "$REJECTED" | tr -d '[:space:]')" ]; then
    printf '%s' "$REJECTED" | while IFS= read -r line; do [ -n "$line" ] && printf '   ✗ %s\n' "$line"; done
else
    printf '   (无)\n'
fi

printf '\n%s将停用（disable --now，可逆）:%s\n' "$C_GRN" "$C_OFF"
if [ -n "$(printf '%s' "$ALLOWED" | tr -d '[:space:]')" ]; then
    printf '%s' "$ALLOWED" | while IFS= read -r u; do [ -n "$u" ] && printf '   ✓ %s\n' "$u"; done
else
    printf '   (无)\n'
fi

# 禁停硬拦截：非 dry-run 下，候选命中禁停清单默认 exit 1（绝不停用任何东西）。
# 唯一例外：--allow-forbidden --yes（误拦逃生阀）—— 此时把命中单元转入 OVERRIDE，
# 照常停用但打印并记录 [OVERRIDE] 审计行，避免与「禁止手工 disable」形成死锁。
OVERRIDE=""
if [ -n "$(printf '%s' "$REJECTED" | tr -d '[:space:]')" ] && [ "$DRY_RUN" != "1" ]; then
    if [ "$ALLOW_FORBIDDEN" = "1" ]; then
        OVERRIDE="$REJECTED"
        warn "⚠️ --allow-forbidden：下列命中禁停清单的单元将被**强制停用**（已记录 [OVERRIDE] 审计）："
        printf '%s' "$REJECTED" | while IFS= read -r line; do [ -n "$line" ] && printf '   %s[OVERRIDE]%s %s\n' "$C_YEL" "$C_OFF" "$line"; done
        warn "请确认你已人工核对它们确属误拦、且停用不会导致设备失联/变砖/安全降级。"
    else
        bad "候选中含禁停单元（见上）——已硬拦截，未停用任何服务。"
        info "请从候选里移除这些单元后重试。若确需停用某个被**误拦**的单元，必须人工核对确认它不属于禁停类后，"
        info "用 --allow-forbidden --yes 强制停用（脚本会打印并记录 [OVERRIDE] 审计行）；该逃生阀仅供误拦时使用，切勿滥用。"
        exit 1
    fi
fi

if [ "$DRY_RUN" = "1" ]; then
    warn "--dry-run：以上仅为预览，未改动任何状态。"
    exit 0
fi

if [ -z "$(printf '%s' "$ALLOWED" | tr -d '[:space:]')" ] && [ -z "$(printf '%s' "$OVERRIDE" | tr -d '[:space:]')" ]; then
    warn "没有可停用的单元（全部被拦截或候选为空）。"
    exit 0
fi

# ===========================================================================
# 执行停用（写 undo 文件）
# ===========================================================================
step "执行停用 + 生成 undo 文件"
if [ -z "$UNDO" ]; then
    mkdir -p "$DEFAULT_EVIDENCE_DIR" 2>/dev/null
    UNDO="$DEFAULT_EVIDENCE_DIR/strip_undo_$(date +%F_%H%M%S).txt"
fi
: > "$UNDO" 2>/dev/null || { bad "无法写 undo 文件: $UNDO"; exit 1; }
info "undo 文件: $UNDO"
# [OVERRIDE] 审计日志（仅 --allow-forbidden 时产生）：记录被强制停用的禁停单元，便于事后追溯
AUDIT="$DEFAULT_EVIDENCE_DIR/strip_override_$(date +%F_%H%M%S).log"

if [ -n "$(printf '%s' "$OVERRIDE" | tr -d '[:space:]')" ]; then
    confirm "⚠️ 确认对含**禁停单元**的上列清单执行强制 disable --now（--allow-forbidden）？此操作已审计" \
        || { warn "已取消，未改动。"; exit 0; }
else
    confirm "确认对上列「将停用」单元执行 systemctl disable --now？" || { warn "已取消，未改动。"; exit 0; }
fi

FAIL=0
# ① 正常候选（未命中禁停清单）
while IFS= read -r u; do
    [ -z "$u" ] && continue
    printf '      disable --now %s ... ' "$u"
    if $SUDO systemctl disable --now "$u" 2>/dev/null; then
        printf '%sOK%s\n' "$C_GRN" "$C_OFF"
        printf '%s\n' "$u" >> "$UNDO"
    else
        printf '%sFAIL%s\n' "$C_RED" "$C_OFF"
        FAIL=1
    fi
done <<EOF3
$ALLOWED
EOF3

# ② --allow-forbidden 误拦逃生阀：强制停用命中禁停清单的单元，并写 [OVERRIDE] 审计行
if [ -n "$(printf '%s' "$OVERRIDE" | tr -d '[:space:]')" ]; then
    mkdir -p "$DEFAULT_EVIDENCE_DIR" 2>/dev/null
    : > "$AUDIT" 2>/dev/null || { bad "无法写审计文件: $AUDIT"; exit 1; }
    printf '# [OVERRIDE] %s 强制停用禁停单元（--allow-forbidden --yes）\n' "$(date '+%F %T')" >> "$AUDIT"
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        u="${line%%$'\t'*}"          # REJECTED 行格式: unit<TAB>(命中禁停词: x) → 取 TAB 前的单元名
        reason="${line#*$'\t'}"
        printf '      %s[OVERRIDE]%s disable --now %s (%s) ... ' "$C_YEL" "$C_OFF" "$u" "$reason"
        printf '[OVERRIDE] %s\t%s\t%s\n' "$(date '+%F %T')" "$u" "$reason" >> "$AUDIT"
        if $SUDO systemctl disable --now "$u" 2>/dev/null; then
            printf '%sOK%s\n' "$C_GRN" "$C_OFF"
            printf '%s\n' "$u" >> "$UNDO"
        else
            printf '%sFAIL%s\n' "$C_RED" "$C_OFF"
            FAIL=1
        fi
    done <<EOF4
$OVERRIDE
EOF4
    warn "[OVERRIDE] 审计已写入: $AUDIT（请连同 §5.1 存证一并保存，便于事后追溯/回滚）"
fi

step "复核"
info "上层 UI 应已不可达，而 nginx/SSH/GPU 必须照常："
info "  curl -sk -o /dev/null -w 'MoFang.html → HTTP %{http_code}\\n' https://<设备IP>/next/MoFang.html"
info "  systemctl is-active nginx ssh docker ; ls -l /dev/dri"
printf '\n%s回滚（可逆性证明）:%s bash deploy/strip_mofang.sh --rollback %s\n' "$C_YEL" "$C_OFF" "$UNDO"

if [ "$FAIL" -ne 0 ]; then
    bad "有单元 disable 失败（见上）——请用 systemctl status <unit> 排查后重跑（幂等）。"
    exit 1
fi
ok "剥离完成；undo 文件已写入 $UNDO"
exit 0
