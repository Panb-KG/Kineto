#!/usr/bin/env bash
# =============================================================================
# Kineto / deploy — discover_device.sh
# -----------------------------------------------------------------------------
# 目的  : 在 MoFang M01 / Intel AI Box 上做一次 **只读** 体检，收集部署 Kineto
#         Engine 所需的全部事实，并对「决策门禁」给出 PASS/FAIL 结论。
# 运行处 : 设备端（用户 SSH 登录后自己跑）。云端 Agent 无 SSH 凭据，不会代跑。
# 安全性 : 本脚本 **不修改任何状态** —— 不装包、不改配置、不启停服务、不写文件。
#         全部命令均为查询类（cat / ls / lspci / lscpu / df / ss / systemctl
#         list-units / nginx -T）。唯一可能的“写入”是你自己选择的重定向，例如：
#             bash discover_device.sh 2>&1 | tee ~/discover_$(date +%F_%H%M).log
# 幂等性 : 可重复运行任意次，输出一致（除时间戳/负载类字段）。
#
# 用法  :
#     scp deploy/discover_device.sh user@<device-ip>:/tmp/
#     ssh user@<device-ip> 'bash /tmp/discover_device.sh'
#     # 需要 sudo 的项目（lshw / dmesg / nginx -T / oom）会自动尝试 sudo -n，
#     # 若无免密 sudo，这些段落会标记 SKIPPED，不会中断脚本。
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# 参数（本脚本无副作用，只接受帮助类参数；其余一律拒绝，避免误用）
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
用法: bash deploy/discover_device.sh [-h|--help]

在 MoFang M01 / Intel AI Box 上做一次 **只读** 体检，为部署 Kineto Engine 采集事实，
并在末尾给出决策门禁的 PASS/FAIL 汇总。

采集内容（全部为查询类命令，不装包/不改配置/不启停服务/不写文件）:
  1  OS 发行版与 Secure Boot   2  GPU 身份 + /dev/dri + dmesg   3  Intel GPU 用户态包
  4  CPU / AVX / AMX            5  RAM                           6  磁盘与 inode
  7  docker + cgroup 版本       8  python3 与出网能力            9  systemd 状态
  10 监听端口（含 8000 是否空闲） 11 MoFang 服务清单              12 nginx 摘要
  13 既有 Kineto 部署痕迹（幂等性）

决策门禁: G1 RAM>=16G  G2 可用磁盘>=25G  G3 Intel GPU 存在  G4 docker 存在
退出码  : 0 = GO（可继续部署）  1 = NO-GO（先看 FAIL 项与 DEPLOY_MOFANG.md §2.2）

提示    : 需要 root 的段落（lshw / dmesg / nginx -T / oom）会自动尝试 sudo -n；
          无免密 sudo 时标记 SKIPPED 而不中断。要完整报告请用:
              sudo bash deploy/discover_device.sh
          存档:
              bash deploy/discover_device.sh 2>&1 | tee ~/kineto_discover_$(date +%F).log
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        *) printf '未知参数: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# 输出与权限辅助
# ---------------------------------------------------------------------------
C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_CYN=$'\033[36m'; C_OFF=$'\033[0m'
if [ ! -t 1 ]; then C_RED=""; C_GRN=""; C_YEL=""; C_CYN=""; C_OFF=""; fi

SUDO=""
SUDO_NOTE=""
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    SUDO="sudo -n"
else
    SUDO_NOTE="(无免密 sudo，标记为 SKIPPED；如需完整报告请用 sudo bash $0)"
fi

section() { printf '\n%s══ %s %s\n' "$C_CYN" "$1" "$C_OFF"; }
note()    { printf '   %s\n' "$1"; }
# run "<说明>" <命令...> —— 回显命令本身再执行，输出缩进，失败不中断
run() {
    local desc="$1"; shift
    printf '   %s$%s %s\n' "$C_YEL" "$C_OFF" "$*"
    if ! command -v "$1" >/dev/null 2>&1 && [ "${1#sudo*}" = "$1" ]; then
        note "SKIPPED — 未找到命令: $1"
        return 0
    fi
    "$@" 2>&1 | sed 's/^/     /'
    return 0
}
run_sudo() {
    local desc="$1"; shift
    if [ -z "$SUDO" ] && [ "$(id -u)" -ne 0 ]; then
        printf '   %s$%s %s\n' "$C_YEL" "$C_OFF" "$*"
        note "SKIPPED — 需要 root/sudo ${SUDO_NOTE}"
        return 0
    fi
    printf '   %s$%s %s\n' "$C_YEL" "$C_OFF" "$*"
    $SUDO "$@" 2>&1 | sed 's/^/     /'
    return 0
}
ok()   { printf '   %s[PASS]%s %s\n' "$C_GRN" "$C_OFF" "$1"; }
warn() { printf '   %s[WARN]%s %s\n' "$C_YEL" "$C_OFF" "$1"; }
bad()  { printf '   %s[FAIL]%s %s\n' "$C_RED" "$C_OFF" "$1"; }

printf '%s\n' "=============================================================================="
printf ' Kineto Engine 设备体检 (READ-ONLY)  —  %s\n' "$(date '+%F %T %Z')"
printf ' 主机: %s   执行者: %s   %s\n' "$(hostname 2>/dev/null || echo unknown)" "$(id -un)" "$SUDO_NOTE"
printf "=============================================================================="
printf '\n%s本脚本不会修改任何系统状态。%s\n' "$C_GRN" "$C_OFF"

# ---------------------------------------------------------------------------
# 1. 操作系统
# ---------------------------------------------------------------------------
section "1. 操作系统发行版"
run "lsb_release" lsb_release -a
run "os-release" cat /etc/os-release
OS_ID=$(. /etc/os-release 2>/dev/null; echo "${ID:-unknown}")
OS_VER=$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID:-unknown}")
OS_PRETTY=$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")
note "解析结果: ID=$OS_ID  VERSION_ID=$OS_VER  ($OS_PRETTY)"

section "1b. 内核 / 架构 / Secure Boot"
run "kernel" uname -r
run "kernel full" uname -a
ARCH="$(dpkg --print-architecture 2>/dev/null || uname -m)"
run "arch" dpkg --print-architecture
# 内核版本：仅对 Linux 有意义（在 macOS 上 uname -r 返回 Darwin 版本号，会误判）
KERNEL_REL="$(uname -r 2>/dev/null | cut -d. -f1)"
KERNEL_MAJ="$(uname -r 2>/dev/null | cut -d. -f2)"
case "$KERNEL_REL" in ''|*[!0-9]*) KERNEL_REL=0 ;; esac
case "$KERNEL_MAJ" in ''|*[!0-9]*) KERNEL_MAJ=0 ;; esac
if command -v mokutil >/dev/null 2>&1; then
    run_sudo "secure boot state" mokutil --sb-state
else
    note "mokutil 未安装 —— 无法判断 Secure Boot（Intel 驱动包安装可能受其影响）"
fi

# ---------------------------------------------------------------------------
# 2. GPU 身份（最关键）
# ---------------------------------------------------------------------------
section "2. GPU 身份 (Intel Arc Pro B60 期望值)"
run "lspci vga/display/3d" bash -c "lspci -nn 2>/dev/null | grep -iE 'vga|display|3d'"
LSPCI_GPU="$(lspci -nn 2>/dev/null | grep -iE 'vga|display|3d')"
LSPCI_INTEL_GPU="$(printf '%s' "$LSPCI_GPU" | grep -i intel)"
run "lspci -k (驱动绑定)" bash -c "lspci -nnk 2>/dev/null | grep -iE -A3 'vga|display|3d'"
run "/dev/dri" ls -l /dev/dri
run "/sys/class/drm" ls -l /sys/class/drm
run "内核模块 i915/xe" bash -c "lsmod 2>/dev/null | grep -iE '^i915|^xe|drm'"
run_sudo "lshw -C display" lshw -C display
run_sudo "dmesg i915/xe/drm/arc (tail)" bash -c "dmesg 2>/dev/null | grep -iE 'i915|xe|drm|arc' | tail -n 30"

RENDER_NODES="$(ls /dev/dri/renderD* 2>/dev/null | wc -l | tr -d ' ')"
GPU_VENDOR_OK="no"
if [ -n "$LSPCI_INTEL_GPU" ]; then GPU_VENDOR_OK="yes"; fi

# 显式找 Arc Pro B60 / Battlemage 关键字（8086:xxxx）
if printf '%s' "$LSPCI_GPU" | grep -qiE 'arc|battlemage|b60|\[8086:'; then
    GPU_MODEL_HINT="$(printf '%s' "$LSPCI_GPU" | grep -iE 'arc|battlemage|b60|\[8086:' | head -n1)"
else
    GPU_MODEL_HINT="(未从 lspci 名称中识别出 Arc/Battlemage，请人工比对上面的 PCI ID)"
fi
note "GPU 识别提示: $GPU_MODEL_HINT"
note "render 节点数: $RENDER_NODES  (/dev/dri/renderD128 起为 GPU 计算节点)"

section "2b. Intel GPU 用户态运行时是否已安装"
run "dpkg intel gpu pkgs" bash -c "dpkg -l 2>/dev/null | grep -iE 'intel-opencl-icd|intel-level-zero-gpu|level-zero|libze1|libze-intel-gpu1|intel-media-va-driver|intel-omix|compute-runtime' | awk '{print \$1, \$2, \$3}'"
run "apt repo (intel gpu)" bash -c "grep -rhiE 'repositories.intel.com|intel-graphics' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null"
run "clinfo/ze 工具" bash -c "command -v clinfo ze_info sycl-ls intel_gpu_top 2>/dev/null"
run "apt-cache policy intel-opencl-icd" bash -c "apt-cache policy intel-opencl-icd 2>/dev/null | head -n 6"

# ---------------------------------------------------------------------------
# 3. CPU / RAM / 磁盘
# ---------------------------------------------------------------------------
section "3. CPU"
run "lscpu (head)" bash -c "lscpu 2>/dev/null | head -n 25"
run "nproc" nproc
CPU_FLAGS="$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null)"
for f in avx avx2 avx512f avx512_bf16 amx_bf16 sse4_2; do
    if printf '%s' "$CPU_FLAGS" | grep -qw "$f"; then ok "CPU flag: $f"; else warn "CPU flag 缺失: $f"; fi
done
run "model name" bash -c "grep -m1 'model name' /proc/cpuinfo"

section "4. 内存"
run "free -h" free -h
MEM_TOTAL_KB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null)"
MEM_TOTAL_GB=$(( ${MEM_TOTAL_KB:-0} / 1024 / 1024 ))
MEM_AVAIL_KB="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)"
MEM_AVAIL_GB=$(( ${MEM_AVAIL_KB:-0} / 1024 / 1024 ))
note "MemTotal ≈ ${MEM_TOTAL_GB} GB   MemAvailable ≈ ${MEM_AVAIL_GB} GB"
run "swap" bash -c "swapon --show 2>/dev/null; free -h | grep -i swap"

section "5. 磁盘"
run "df -hT" df -hT
run "lsblk" lsblk
# 门禁用：根分区可用空间（GB）。若 /srv 或 /opt 是独立分区，也一并报告。
avail_kb_root="$(df -k / 2>/dev/null | awk 'NR==2{print $4}')"
DISK_AVAIL_GB=$(( ${avail_kb_root:-0} / 1024 / 1024 ))
for m in /srv /opt /opt/kineto /srv/kineto; do
    if [ -d "$m" ]; then
        a="$(df -k "$m" 2>/dev/null | awk 'NR==2{print $4}')"
        note "$m 可用 ≈ $(( ${a:-0} / 1024 / 1024 )) GB  (挂载点: $(df --output=target "$m" 2>/dev/null | tail -n1))"
    fi
done
run "inode 使用" bash -c "df -i / | tail -n 2"

# ---------------------------------------------------------------------------
# 6. 运行时环境：docker / python / systemd
# ---------------------------------------------------------------------------
section "6. Docker"
if command -v docker >/dev/null 2>&1; then
    DOCKER_PRESENT="yes"
    run "docker version" docker --version
    run "docker info (摘要)" bash -c "docker info --format 'Server={{.ServerVersion}} Cgroup={{.CgroupVersion}} Runtimes={{.Runtimes}}' 2>&1 | head -n 5"
    run "docker compose" bash -c "docker compose version 2>&1 | head -n 2"
    run "当前用户能否访问 docker.sock" bash -c "docker ps >/dev/null 2>&1 && echo 'yes (可免 sudo)' || echo 'no (需 sudo 或加入 docker 组)'"
    run "docker 镜像" bash -c "docker images 2>/dev/null | head -n 10"
else
    DOCKER_PRESENT="no"
    warn "未安装 docker —— 容器路径不可用（systemd 路径不受影响）"
fi
note "提示: Intel GPU 容器直通使用 '--device /dev/dri'，**不是** '--gpus'（那是 NVIDIA 运行时）"

section "7. Python"
run "python3 -V" python3 -V
run "python 候选版本" bash -c "ls -1 /usr/bin/python3.* 2>/dev/null"
run "venv 模块可用?" bash -c "python3 -c 'import venv,ensurepip; print(\"venv OK\")' 2>&1"
run "pip3" bash -c "pip3 --version 2>&1 | head -n 1"
run "conda/mamba" bash -c "command -v conda mamba 2>/dev/null || echo '(无 conda/mamba — 用 venv 即可)'"

section "8. systemd 状态"
run "is-system-running" bash -c "systemctl is-system-running 2>&1"
run "failed units" bash -c "systemctl --failed --no-legend 2>&1 | head -n 20"
run "cgroup 版本" bash -c "stat -fc %T /sys/fs/cgroup 2>/dev/null"
note "cgroup v2 (输出 cgroup2fs) 才能让 systemd 的 MemoryMax= 生效"

# ---------------------------------------------------------------------------
# 7. 网络 / 端口 / nginx / MoFang 上层应用清单
# ---------------------------------------------------------------------------
section "9. 监听端口 (Kineto Engine 需要 127.0.0.1:8000 空闲)"
run "ss -tlnp" bash -c "ss -tlnp 2>/dev/null | head -n 30"
run "8000 端口占用?" bash -c "ss -tlnp 2>/dev/null | grep -E ':8000\\b' || echo 'FREE (8000 未被占用)'"
run "80/443 占用" bash -c "ss -tlnp 2>/dev/null | grep -E ':(80|443)\\b' || echo '(无)'"
run "本机 IP" bash -c "ip -4 -o addr show 2>/dev/null | awk '{print \$2, \$4}'"
run "出网连通性" bash -c "curl -sS -m 8 -o /dev/null -w 'pypi=%{http_code}\\n' https://pypi.org/simple/ 2>&1; curl -sS -m 8 -o /dev/null -w 'pytorch-xpu-index=%{http_code}\\n' https://download.pytorch.org/whl/xpu 2>&1"

section "10. MoFang 上层应用服务清单 (待停用对象)"
run "running services (mofang|bridge|ai)" bash -c "systemctl list-units --type=service --state=running --no-legend 2>/dev/null | grep -iE 'mofang|bridge|ai' | head -n 40"
run "all mofang-ish units" bash -c "systemctl list-unit-files --type=service --no-legend 2>/dev/null | grep -iE 'mofang|bridge|claw|\brag\b|assistant' | head -n 40"
run "docker 中的 MoFang 容器" bash -c "docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null | head -n 20"
run "MoFang Web 根目录线索" bash -c "ls -ld /opt/MoFang /opt/mofang /usr/share/nginx/html/next /var/www/html/next /next 2>/dev/null"
note "把上面 'list-unit-files' 的输出抄进 DEPLOY_MOFANG.md 的 OPEN QUESTIONS 里 —— 停用清单必须以实测为准"
note "⚠ 'rag' 用词边界 \\brag\\b 匹配，避免误命中 storage（sto-rag-e）等必须保留的厂商/系统单元"
note "实际停用请用 deploy/strip_mofang.sh（内置禁停硬拦截 + --dry-run + undo），**不要**手工 for 循环 disable"

section "11. nginx 现状摘要"
run_sudo "nginx -T (关键行)" bash -c "nginx -T 2>/dev/null | grep -iE 'server_name|location|proxy_pass|root|listen' | head -n 40"
run "nginx 版本" bash -c "nginx -v 2>&1"
run "nginx 服务状态" bash -c "systemctl is-active nginx 2>&1; systemctl is-enabled nginx 2>&1"
run "sites 清单" bash -c "ls -l /etc/nginx/sites-enabled/ /etc/nginx/conf.d/ 2>/dev/null"
note "Kineto Engine 只监听 127.0.0.1:8000，不与 nginx 争端口；公网出口由 Cloudflare Tunnel 承担"

section "12. 现有 Kineto 部署痕迹 (幂等性检查)"
run "/opt/kineto" bash -c "ls -l /opt/kineto 2>/dev/null || echo '(不存在 — 全新部署)'"
run "/srv/kineto" bash -c "ls -l /srv/kineto 2>/dev/null || echo '(不存在 — 全新部署)'"
run "kineto 用户" bash -c "id kineto 2>/dev/null || echo '(kineto 用户不存在)'"
run "kineto systemd unit" bash -c "systemctl status kineto-engine --no-pager 2>&1 | head -n 6"
run "cloudflared" bash -c "command -v cloudflared >/dev/null 2>&1 && cloudflared --version 2>&1 || echo '(未安装 cloudflared)'"

# ---------------------------------------------------------------------------
# 决策门禁汇总
# ---------------------------------------------------------------------------
GATE_FAIL=0
section "13. 决策门禁 (DECISION GATE)"

# G1 RAM >= 16G
if [ "$MEM_TOTAL_GB" -ge 16 ]; then ok  "G1 内存 ${MEM_TOTAL_GB}GB ≥ 16GB"; else bad "G1 内存 ${MEM_TOTAL_GB}GB < 16GB —— 不足以承载 HMR2 ViT-H + 队列"; GATE_FAIL=1; fi

# G2 free disk >= 25G
if [ "$DISK_AVAIL_GB" -ge 25 ]; then ok  "G2 根分区可用 ${DISK_AVAIL_GB}GB ≥ 25GB"; else bad "G2 根分区可用 ${DISK_AVAIL_GB}GB < 25GB —— venv(~8G)+模型(~2.6G)+job 产物不足"; GATE_FAIL=1; fi

# G3 Intel GPU present
if [ "$GPU_VENDOR_OK" = "yes" ] && [ "$RENDER_NODES" -ge 1 ]; then
    ok "G3 Intel GPU 已识别且存在 /dev/dri/renderD* 节点 ($RENDER_NODES 个)"
elif [ "$GPU_VENDOR_OK" = "yes" ]; then
    bad "G3 lspci 看到 Intel 显卡，但 /dev/dri/renderD* 缺失 —— GPU 驱动未就绪，必须先装 compute-runtime + Level Zero"; GATE_FAIL=1
else
    bad "G3 未识别到 Intel GPU —— 核对 lspci 输出；若确无独显，本机不满足生产条件"; GATE_FAIL=1
fi

# G4 docker present
if [ "$DOCKER_PRESENT" = "yes" ]; then ok "G4 docker 已安装（容器路径可选）"; else warn "G4 docker 未安装 —— 容器路径不可用，走 systemd 主路径即可（不阻断）"; fi

# 附加非阻断建议
case "$OS_VER" in
    24.04|26.04) ok  "A1 Ubuntu $OS_VER —— XPU wheel 官方验证版本" ;;
    22.04)       warn "A1 Ubuntu 22.04 —— XPU wheel 未在该版本验证；建议升级到 24.04，或走 IPEX 路径（intel-extension-for-pytorch==2.7.10+xpu）" ;;
    *)           warn "A1 Ubuntu $OS_VER 非预期版本，请人工确认 XPU wheel 兼容性" ;;
esac
if [ "$(uname -s 2>/dev/null)" != "Linux" ]; then
    warn "A2 非 Linux 内核（$(uname -s 2>/dev/null)）—— 本项不适用；请在设备上（Ubuntu）重跑本脚本"
elif [ "$KERNEL_REL" -gt 6 ] || { [ "$KERNEL_REL" -eq 6 ] && [ "$KERNEL_MAJ" -ge 8 ]; }; then
    ok "A2 内核 $(uname -r) ≥ 6.8 —— Battlemage(Arc Pro B60) xe/i915 支持较完整"
else
    warn "A2 内核 $(uname -r) 偏旧（< 6.8）—— Battlemage 需要较新内核/HWE 栈，建议：sudo apt-get install linux-generic-hwe-24.04（升级前先备好恢复镜像，见 DEPLOY_MOFANG.md §6）"
fi
# amd64 (dpkg 说法) == x86_64 (uname 说法)；两者都算通过
case "$ARCH" in
    amd64|x86_64) ok "A3 架构 $ARCH —— XPU wheel 可用" ;;
    *)            bad "A3 架构 $ARCH —— 无 XPU wheel，方案不成立"; GATE_FAIL=1 ;;
esac

printf '\n'
if [ "$GATE_FAIL" -eq 0 ]; then
    printf '%s==> 门禁结论: GO —— 可以进入 DEPLOY_MOFANG.md 的 Phase 1 (共存验证)%s\n' "$C_GRN" "$C_OFF"
else
    printf '%s==> 门禁结论: NO-GO —— 先解决上面的 FAIL 项，再进入部署阶段%s\n' "$C_RED" "$C_OFF"
fi
printf '%s\n' "=============================================================================="
printf ' 请把本报告完整回传给云端（含 lspci / df / free / MoFang 单元清单）。\n'
printf ' 本脚本未做任何修改；重复运行安全。\n'
printf '%s\n' "=============================================================================="
exit "$GATE_FAIL"
