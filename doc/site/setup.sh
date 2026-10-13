#!/usr/bin/env bash
#
# 深度学习框架学习手册 —— 构建依赖准备脚本（Ubuntu / Debian）
#
#   只负责「检查 + 安装」依赖，不执行构建本身。
#   可反复执行：已满足的项自动跳过，不会重复安装。
#
#   ./setup.sh           检查并安装缺失的依赖
#   ./setup.sh --check   只检查环境，不做任何安装
#
# 装完后执行构建：
#   .venv/bin/python tools/build.py     （若脚本创建了虚拟环境）
#   python3 tools/build.py              （若系统 Python 已具备依赖）
#
set -euo pipefail
cd "$(dirname "$0")"

MODE="install"
case "${1:-}" in
  --check|-c) MODE="check" ;;
  --help|-h) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "未知参数: $1（可用 --check）" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------ 输出辅助
if [ -t 1 ]; then
  C_OK=$'\033[32m'; C_DIM=$'\033[90m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'
  C_B=$'\033[1m'; C_0=$'\033[0m'
else
  C_OK=; C_DIM=; C_WARN=; C_ERR=; C_B=; C_0=
fi
ok()    { printf '  %s✓%s %s\n' "$C_OK" "$C_0" "$1"; }
skip()  { printf '  %s·%s %s\n' "$C_DIM" "$C_0" "$1"; }
warn()  { printf '  %s!%s %s\n' "$C_WARN" "$C_0" "$1"; }
die()   { printf '  %s✗%s %s\n' "$C_ERR" "$C_0" "$1" >&2; exit 1; }
head2() { printf '\n%s%s%s\n' "$C_B" "$1" "$C_0"; }

MISSING=0

# ------------------------------------------------------------------ apt 封装
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
APT_UPDATED=0

apt_install() {
  if [ "$MODE" = "check" ]; then
    MISSING=$((MISSING + 1))
    warn "需要安装: $* （--check 模式，未执行）"
    return 1
  fi
  if [ "$APT_UPDATED" -eq 0 ]; then
    printf '  %s执行 apt-get update…%s\n' "$C_DIM" "$C_0"
    $SUDO apt-get update -qq
    APT_UPDATED=1
  fi
  printf '  %s执行 apt-get install %s…%s\n' "$C_DIM" "$*" "$C_0"
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "$@"
}

printf '%s深度学习框架学习手册 · 依赖检查%s\n' "$C_B" "$C_0"
[ "$MODE" = "check" ] && printf '%s（--check 模式：只检查，不安装）%s\n' "$C_DIM" "$C_0"

if ! command -v apt-get >/dev/null 2>&1; then
  warn "未检测到 apt-get：本脚本面向 Ubuntu / Debian，其他发行版请手动安装下列依赖"
fi

# ------------------------------------------------------------------ 1. Python 3
head2 "[1/4] Python 3"
if command -v python3 >/dev/null 2>&1; then
  skip "已安装：$(python3 -V 2>&1)"
else
  warn "未检测到 python3"
  apt_install python3
  command -v python3 >/dev/null 2>&1 || die "python3 仍不可用，请手动安装后重试"
  ok "python3 安装完成：$(python3 -V 2>&1)"
fi

# ------------------------------------------------------------------ 2. venv 支持
head2 "[2/4] Python venv 模块"
VENV_OK=0
if [ "$MODE" = "check" ]; then
  # 只检查不实测（实测会真的创建一次临时环境）
  if python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
    VENV_OK=1
    skip "venv 可用"
  else
    warn "venv 模块或 ensurepip 缺失"
  fi
elif python3 -c "import venv" >/dev/null 2>&1; then
  # 光有模块不够，ensurepip 缺失时创建环境会失败，这里实测一次
  TMPV="$(mktemp -d)"
  if python3 -m venv "$TMPV/probe" >/dev/null 2>&1; then
    VENV_OK=1
    skip "venv 可用"
  else
    warn "venv 模块存在但无法创建环境（缺少 ensurepip）"
  fi
  rm -rf "$TMPV"
else
  warn "缺少 python3-venv"
fi
if [ "$VENV_OK" -eq 0 ]; then
  apt_install python3-venv python3-pip || true
  TMPV="$(mktemp -d)"
  if python3 -m venv "$TMPV/probe" >/dev/null 2>&1; then
    VENV_OK=1
    ok "python3-venv 安装完成"
  fi
  rm -rf "$TMPV"
  [ "$VENV_OK" -eq 1 ] || warn "venv 仍不可用，将尝试直接使用系统 Python"
fi

# ------------------------------------------------------------------ 3. Python 依赖
head2 "[3/4] Python 依赖：markdown、pygments"
PYTHON_CMD="python3"

if python3 -c "import markdown, pygments" >/dev/null 2>&1; then
  skip "系统 python3 已具备 markdown + pygments，无需虚拟环境"
else
  VENV_DIR=".venv"
  if [ -x "$VENV_DIR/bin/python" ]; then
    skip "复用已有虚拟环境 $VENV_DIR"
  else
    if [ "$MODE" = "check" ]; then
      MISSING=$((MISSING + 1))
      warn "需要创建虚拟环境 $VENV_DIR（--check 模式，未执行）"
    elif [ "$VENV_OK" -eq 1 ]; then
      python3 -m venv "$VENV_DIR"
      ok "已创建虚拟环境 $VENV_DIR"
    else
      die "无法创建虚拟环境；请先安装 python3-venv，或用 pip install --user markdown pygments"
    fi
  fi

  if [ -x "$VENV_DIR/bin/python" ]; then
    if "$VENV_DIR/bin/python" -c "import markdown, pygments" >/dev/null 2>&1; then
      skip "虚拟环境内 markdown + pygments 已就绪"
    else
      if [ "$MODE" = "check" ]; then
        MISSING=$((MISSING + 1))
        warn "需要在 $VENV_DIR 内安装 markdown + pygments（--check 模式，未执行）"
      else
        "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
        "$VENV_DIR/bin/python" -m pip install --quiet markdown pygments
        "$VENV_DIR/bin/python" -c "import markdown, pygments" >/dev/null 2>&1 \
          || die "依赖安装失败，请检查网络或 pip 源"
        ok "已在 $VENV_DIR 内安装 markdown + pygments"
      fi
    fi
    PYTHON_CMD="$VENV_DIR/bin/python"
  fi
fi

# ------------------------------------------------------------------ 4. Node.js
head2 "[4/4] Node.js（构建期渲染 KaTeX 公式需要）"
NODE_MIN=16
if command -v node >/dev/null 2>&1; then
  NODE_VER="$(node -v | sed 's/^v//')"
  NODE_MAJOR="${NODE_VER%%.*}"
  if [ "$NODE_MAJOR" -ge "$NODE_MIN" ] 2>/dev/null; then
    skip "已安装：node v$NODE_VER"
  else
    warn "node v$NODE_VER 版本偏低（建议 ≥ $NODE_MIN）"
    printf '  %s  升级方式：curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs%s\n' "$C_DIM" "$C_0"
  fi
else
  warn "未检测到 node"
  if apt_install nodejs npm; then
    if command -v node >/dev/null 2>&1; then
      ok "node 安装完成：$(node -v)"
      NODE_MAJOR="$(node -v | sed 's/^v//; s/\..*//')"
      [ "$NODE_MAJOR" -ge "$NODE_MIN" ] 2>/dev/null || \
        warn "仓库版本偏低，如构建报错请按上面的 NodeSource 方式升级"
    else
      warn "node 仍不可用；没有 node 时 tools/build.py 无法渲染 LaTeX 公式"
    fi
  fi
fi

# ------------------------------------------------------------------ 汇总
head2 "结果"
if [ "$MODE" = "check" ]; then
  if [ "$MISSING" -eq 0 ]; then
    ok "环境已满足构建要求，无需安装任何东西"
  else
    warn "有 $MISSING 项待安装，去掉 --check 重新执行即可"
  fi
  exit 0
fi

printf '  %s构建命令%s\n' "$C_B" "$C_0"
printf '    %s tools/build.py\n' "$PYTHON_CMD"
printf '\n  %s说明%s\n' "$C_B" "$C_0"
printf '    · 本脚本只准备依赖，不会执行构建\n'
printf '    · 内容更新后重跑上面的构建命令即可\n'
printf '    · 可选：想在本地跑自检 tools/smoke_test.cjs，需先 %snpm install jsdom%s\n' "$C_B" "$C_0"
