#!/usr/bin/env bash
# setup.sh — one-shot setup for gemma-cli
#
# What this script does (in order):
#   1. Detects the OS (macOS or Linux) and checks prerequisites
#   2. Checks Python >=3.10 and Ollama (installs missing deps automatically)
#   3. Creates a Python virtual environment at .venv (skipped if it already exists)
#   4. Installs gemma-cli with memory + dev extras into the venv
#   5. Ensures Redis is running — installs it natively (brew/apt/dnf) if needed;
#      falls back to Docker only if Docker is already available
#   6. Pulls gemma4:e4b-it-q5_K_M and nomic-embed-text into Ollama (skipped per model if already present)
#   7. Runs the test suite to confirm everything wired up correctly
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # full setup
#   ./setup.sh --skip-models  # skip Ollama model pulls (useful on slow connections)
#   ./setup.sh --skip-tests   # skip the test suite at the end
#
# Re-running is safe: each step is idempotent and skips work already done.
# Docker is NOT required — Redis is installed natively when possible.

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
DIM="\033[2m"
RESET="\033[0m"

step()    { echo -e "\n${BOLD}${CYAN}==> $*${RESET}"; }
ok()      { echo -e "  ${GREEN}✓${RESET}  $*"; }
skip()    { echo -e "  ${YELLOW}–${RESET}  ${DIM}$* (skipped)${RESET}"; }
warn()    { echo -e "  ${YELLOW}!${RESET}  $*"; }
die()     { echo -e "\n${RED}${BOLD}ERROR: $*${RESET}\n" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SKIP_MODELS=false
SKIP_TESTS=false

for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=true ;;
    --skip-tests)  SKIP_TESTS=true  ;;
    --help|-h)
      echo "Usage: ./setup.sh [--skip-models] [--skip-tests]"
      exit 0
      ;;
    *)
      die "Unknown argument: $arg (use --help to see options)"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Resolve the repo root (the directory this script lives in)
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

OS_TYPE="unknown"
case "$(uname -s)" in
  Darwin) OS_TYPE="macos" ;;
  Linux)  OS_TYPE="linux" ;;
esac

echo -e "\n${BOLD}gemma-cli setup${RESET}"
echo -e "${DIM}repo: $REPO_ROOT  |  OS: $OS_TYPE${RESET}"

# ---------------------------------------------------------------------------
# Step 1 — Prerequisite checks
# ---------------------------------------------------------------------------

step "Checking prerequisites"

# --- Python ---
# Find a python3 that is >= 3.10
PYTHON=""
for candidate in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
  if command -v "$candidate" &>/dev/null; then
    version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    major=${version%%.*}
    minor=${version##*.}
    if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  step "Installing Python >=3.10"
  if [ "$OS_TYPE" = "macos" ]; then
    if ! command -v brew &>/dev/null; then
      die "Homebrew not found and Python >=3.10 is missing. Install Homebrew first: https://brew.sh"
    fi
    brew install python@3.12
    PYTHON="python3.12"
  elif [ "$OS_TYPE" = "linux" ]; then
    if command -v apt-get &>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y python3.12 python3.12-venv
      PYTHON="python3.12"
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y python3.12
      PYTHON="python3.12"
    else
      die "Cannot auto-install Python on this Linux distro. Install Python >=3.10 manually."
    fi
  else
    die "Python >=3.10 not found. Install it from https://python.org"
  fi
fi
ok "Python $("$PYTHON" --version 2>&1 | awk '{print $2}') found at $(command -v "$PYTHON")"

# --- Docker (optional — only used as Redis fallback if native install isn't possible) ---
DOCKER_AVAILABLE=false
COMPOSE=""
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
  DOCKER_AVAILABLE=true
  if docker compose version &>/dev/null 2>&1; then
    COMPOSE="docker compose"
  elif command -v docker-compose &>/dev/null; then
    COMPOSE="docker-compose"
  fi
  ok "Docker available ($(docker --version | awk '{print $3}' | tr -d ',')) — will use as Redis fallback if needed"
else
  skip "Docker not available or not running — Redis will be installed natively"
fi

# --- Ollama ---
if ! command -v ollama &>/dev/null; then
  step "Installing Ollama"
  if [ "$OS_TYPE" = "macos" ]; then
    if command -v brew &>/dev/null; then
      brew install ollama
    else
      die "Homebrew not found. Install Ollama manually from https://ollama.com"
    fi
  elif [ "$OS_TYPE" = "linux" ]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    die "Cannot auto-install Ollama on this OS. Install it from https://ollama.com"
  fi
fi
# Check the daemon is actually reachable (ollama serve must be running)
if ! ollama list &>/dev/null 2>&1; then
  die "Ollama is installed but the daemon isn't running. Start it with: ollama serve  (or open the Ollama menu-bar app)"
fi
ok "Ollama $(ollama --version 2>&1 | head -1)"

# ---------------------------------------------------------------------------
# Step 2 — Virtual environment
# ---------------------------------------------------------------------------

step "Setting up Python virtual environment"

VENV_DIR="$REPO_ROOT/.venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
  skip ".venv already exists"
else
  "$PYTHON" -m venv "$VENV_DIR"
  ok "Created .venv"
fi

# Activate for the rest of this script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Activated .venv (Python: $(python --version 2>&1 | awk '{print $2}'))"

# Upgrade pip quietly
python -m pip install --upgrade pip --quiet
ok "pip up to date ($(pip --version | awk '{print $2}'))"

# ---------------------------------------------------------------------------
# Step 3 — Install gemma-cli
# ---------------------------------------------------------------------------

step "Installing gemma-cli with memory + dev extras"

# Check if already installed at the right path
if pip show gemma-cli &>/dev/null 2>&1; then
  INSTALLED_LOC=$(pip show gemma-cli | grep "Location" | awk '{print $2}')
  # If it's an editable install pointing to this repo, just upgrade deps
  pip install -e ".[memory,dev]" --quiet
  ok "gemma-cli already installed — dependencies refreshed"
else
  pip install -e ".[memory,dev]" --quiet
  ok "gemma-cli installed in editable mode"
fi

# Confirm the CLI entry point is on PATH
if ! command -v gemma &>/dev/null; then
  warn "'gemma' command not found on PATH after install."
  warn "Either activate the venv first: source .venv/bin/activate"
  warn "Or run directly:               .venv/bin/gemma --help"
else
  ok "CLI entry point: $(command -v gemma)"
fi

# ---------------------------------------------------------------------------
# Step 4 — Redis
# ---------------------------------------------------------------------------

step "Ensuring Redis is running"

# Helper: check if a native Redis server is already listening on 6379
redis_native_running() {
  if command -v redis-cli &>/dev/null && redis-cli ping &>/dev/null 2>&1; then
    return 0
  fi
  return 1
}

# Helper: install Redis natively for the current OS
install_redis_native() {
  if [ "$OS_TYPE" = "macos" ]; then
    if ! command -v brew &>/dev/null; then
      die "Homebrew not found. Install it from https://brew.sh then re-run this script."
    fi
    brew install redis
    brew services start redis
  elif [ "$OS_TYPE" = "linux" ]; then
    if command -v apt-get &>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y redis-server
      sudo systemctl enable --now redis-server 2>/dev/null || sudo service redis-server start
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y redis
      sudo systemctl enable --now redis
    else
      die "Cannot auto-install Redis on this Linux distro. Install redis-server manually."
    fi
  else
    die "Unsupported OS for auto-install. Install Redis manually from https://redis.io"
  fi
}

if redis_native_running; then
  skip "Redis already running natively (localhost:6379)"
else
  # Try native install first
  if command -v redis-server &>/dev/null; then
    # redis-server binary exists but isn't running — start it
    if [ "$OS_TYPE" = "macos" ]; then
      brew services start redis 2>/dev/null || redis-server --daemonize yes --logfile /tmp/redis.log
    else
      sudo systemctl start redis-server 2>/dev/null || sudo systemctl start redis 2>/dev/null || redis-server --daemonize yes
    fi
    ok "Started existing native Redis installation"
  elif [ "$DOCKER_AVAILABLE" = true ] && [ -n "$COMPOSE" ] && [ -f "$REPO_ROOT/docker-compose.yml" ]; then
    # Fall back to Docker only if it's available and a compose file exists
    warn "Redis not found natively — falling back to Docker"
    $COMPOSE up -d --wait 2>&1 | sed 's/^/    /'
    # Give Redis up to 15 seconds to report healthy
    REDIS_HEALTHY=false
    for i in $(seq 1 15); do
      if $COMPOSE ps redis 2>/dev/null | grep -q "healthy"; then
        REDIS_HEALTHY=true
        break
      fi
      sleep 1
    done
    if ! $REDIS_HEALTHY; then
      warn "Redis container started but health check hasn't passed yet. Run '$COMPOSE ps' to check."
    else
      ok "Redis is up via Docker (localhost:6379)"
    fi
  else
    # Install natively
    echo -e "  ${DIM}Redis not found — installing natively...${RESET}"
    install_redis_native
    ok "Redis installed and started natively (localhost:6379)"
  fi
fi

# ---------------------------------------------------------------------------
# Step 5 — Ollama models
# ---------------------------------------------------------------------------

step "Pulling Ollama models"

CHAT_MODEL="gemma4:e4b-it-q5_K_M"
EMBED_MODEL="nomic-embed-text"

pull_model_if_missing() {
  local model="$1"
  local size_hint="$2"
  if ollama list 2>/dev/null | grep -q "^${model}"; then
    skip "$model already present"
  else
    if [ "$SKIP_MODELS" = true ]; then
      warn "$model not found — skipped (pass without --skip-models to pull)"
    else
      echo -e "  ${DIM}Pulling $model ($size_hint) — this may take a while...${RESET}"
      ollama pull "$model"
      ok "$model pulled"
    fi
  fi
}

pull_model_if_missing "$CHAT_MODEL"  "~3–4 GB"
pull_model_if_missing "$EMBED_MODEL" "~274 MB"

# ---------------------------------------------------------------------------
# Step 6 — Test suite
# ---------------------------------------------------------------------------

step "Running test suite"

if [ "$SKIP_TESTS" = true ]; then
  skip "Tests skipped (--skip-tests)"
else
  echo ""
  # Run pytest; show output directly so failures are readable
  if python -m pytest tests/ -v --tb=short 2>&1; then
    ok "All tests passed"
  else
    echo ""
    die "Tests failed. Fix the errors above before using the CLI."
  fi
fi

# ---------------------------------------------------------------------------
# Done — print a usage summary
# ---------------------------------------------------------------------------

echo -e "\n${BOLD}${GREEN}Setup complete.${RESET}\n"

echo -e "${BOLD}Activate the virtual environment:${RESET}"
echo -e "  ${DIM}source .venv/bin/activate${RESET}"
echo ""
echo -e "${BOLD}Quick-start commands:${RESET}"
echo -e "  ${CYAN}gemma ask \"Hello\"${RESET}           — single-shot query"
echo -e "  ${CYAN}gemma chat${RESET}                   — interactive REPL with memory"
echo -e "  ${CYAN}gemma chat --no-memory${RESET}       — stateless chat (no Redis needed)"
echo -e "  ${CYAN}cat file.txt | gemma pipe${RESET}    — analyse piped content"
echo -e "  ${CYAN}gemma history stats${RESET}          — view memory statistics"
echo ""
echo -e "${BOLD}To stop Redis when you're done:${RESET}"
if [ "$OS_TYPE" = "macos" ]; then
  echo -e "  ${DIM}brew services stop redis${RESET}     — stop native Redis"
elif [ "$OS_TYPE" = "linux" ]; then
  echo -e "  ${DIM}sudo systemctl stop redis-server${RESET}  — stop native Redis"
fi
if [ "$DOCKER_AVAILABLE" = true ] && [ -n "$COMPOSE" ]; then
  echo -e "  ${DIM}docker compose down${RESET}           — stop Docker Redis (if used)"
  echo -e "  ${DIM}docker compose down -v${RESET}        — stop Docker Redis and wipe data"
fi
echo ""
echo -e "${DIM}See README.md for full documentation.${RESET}"
