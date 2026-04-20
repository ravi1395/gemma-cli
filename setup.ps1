# setup.ps1 — one-shot setup for gemma-cli on Windows
#
# What this script does (in order):
#   1. Checks/installs prerequisites: Python >=3.10, Ollama
#   2. Creates a Python virtual environment at .venv (skipped if it already exists)
#   3. Installs gemma-cli with memory + dev extras into the venv
#   4. Ensures Redis is running — installs via winget/chocolatey or falls back to
#      Docker if already available
#   5. Pulls gemma3:4b-it-q4_K_M and nomic-embed-text into Ollama (skipped per model if present)
#   6. Runs the test suite to confirm everything wired up correctly
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup.ps1                # full setup
#   .\setup.ps1 -SkipModels    # skip Ollama model pulls
#   .\setup.ps1 -SkipTests     # skip the test suite
#
# Re-running is safe: each step is idempotent and skips work already done.
# Docker is NOT required — Redis is installed natively when possible.

[CmdletBinding()]
param(
    [switch]$SkipModels,
    [switch]$SkipTests,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

function Step   { param([string]$msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok     { param([string]$msg) Write-Host "  ✓  $msg" -ForegroundColor Green }
function Skip-  { param([string]$msg) Write-Host "  –  $msg (skipped)" -ForegroundColor DarkGray }
function Warn-  { param([string]$msg) Write-Host "  !  $msg" -ForegroundColor Yellow }
function Die    { param([string]$msg) Write-Host "`nERROR: $msg`n" -ForegroundColor Red; exit 1 }

if ($Help) {
    Write-Host "Usage: .\setup.ps1 [-SkipModels] [-SkipTests]"
    exit 0
}

# ---------------------------------------------------------------------------
# Resolve the repo root
# ---------------------------------------------------------------------------

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "`ngemma-cli setup" -ForegroundColor White
Write-Host "repo: $RepoRoot  |  OS: Windows" -ForegroundColor DarkGray

# ---------------------------------------------------------------------------
# Helper: check if a command exists
# ---------------------------------------------------------------------------

function Test-Command { param([string]$Name) return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# ---------------------------------------------------------------------------
# Helper: check if winget is available
# ---------------------------------------------------------------------------

function Test-Winget { return (Test-Command "winget") }

# ---------------------------------------------------------------------------
# Helper: check if chocolatey is available
# ---------------------------------------------------------------------------

function Test-Choco { return (Test-Command "choco") }

# ---------------------------------------------------------------------------
# Step 1 — Prerequisite checks
# ---------------------------------------------------------------------------

Step "Checking prerequisites"

# --- Python ---
$Python = $null
foreach ($candidate in @("python3", "python")) {
    if (Test-Command $candidate) {
        try {
            $ver = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver) {
                $parts = $ver.Split(".")
                $major = [int]$parts[0]
                $minor = [int]$parts[1]
                if ($major -ge 3 -and $minor -ge 10) {
                    $Python = $candidate
                    break
                }
            }
        } catch {
            # candidate not usable, try next
        }
    }
}

if (-not $Python) {
    Step "Installing Python"
    if (Test-Winget) {
        winget install --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        # Refresh PATH so python is found
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        $Python = "python"
    } elseif (Test-Choco) {
        choco install python312 -y
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        $Python = "python"
    } else {
        Die "Python >=3.10 not found. Install from https://python.org or run: winget install Python.Python.3.12"
    }
}
$pyVersion = & $Python --version 2>&1
Ok "Python $pyVersion found"

# --- Docker (optional) ---
$DockerAvailable = $false
$Compose = ""
if ((Test-Command "docker") -and (docker info 2>$null)) {
    $DockerAvailable = $true
    if (docker compose version 2>$null) {
        $Compose = "docker compose"
    } elseif (Test-Command "docker-compose") {
        $Compose = "docker-compose"
    }
    Ok "Docker available — will use as Redis fallback if needed"
} else {
    Skip- "Docker not available or not running — Redis will be installed natively"
}

# --- Ollama ---
if (-not (Test-Command "ollama")) {
    Step "Installing Ollama"
    if (Test-Winget) {
        winget install --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    } elseif (Test-Choco) {
        choco install ollama -y
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Die "Cannot auto-install Ollama. Install from https://ollama.com or run: winget install Ollama.Ollama"
    }
}
# Check daemon is reachable
try {
    $null = ollama list 2>$null
    if ($LASTEXITCODE -ne 0) { throw "not running" }
    Ok "Ollama is available"
} catch {
    Die "Ollama is installed but the daemon isn't running. Start it from the system tray or run: ollama serve"
}

# ---------------------------------------------------------------------------
# Step 2 — Virtual environment
# ---------------------------------------------------------------------------

Step "Setting up Python virtual environment"

$VenvDir = Join-Path $RepoRoot ".venv"
$VenvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"

if ((Test-Path $VenvDir) -and (Test-Path $VenvActivate)) {
    Skip- ".venv already exists"
} else {
    & $Python -m venv $VenvDir
    Ok "Created .venv"
}

# Activate for the rest of this script
& $VenvActivate
Ok "Activated .venv"

# Upgrade pip
python -m pip install --upgrade pip --quiet
Ok "pip up to date"

# ---------------------------------------------------------------------------
# Step 3 — Install gemma-cli
# ---------------------------------------------------------------------------

Step "Installing gemma-cli with memory + dev extras"

pip install -e ".[memory,dev]" --quiet
Ok "gemma-cli installed in editable mode"

# Confirm the CLI entry point is on PATH
if (Test-Command "gemma") {
    Ok "CLI entry point: gemma"
} else {
    Warn- "'gemma' command not found on PATH after install."
    Warn- "Activate the venv first: .venv\Scripts\Activate.ps1"
}

# ---------------------------------------------------------------------------
# Step 4 — Redis
# ---------------------------------------------------------------------------

Step "Ensuring Redis is running"

# Helper: check if Redis is reachable
function Test-RedisRunning {
    if (Test-Command "redis-cli") {
        try {
            $reply = redis-cli ping 2>$null
            return ($reply -eq "PONG")
        } catch {
            return $false
        }
    }
    return $false
}

# Helper: install Redis on Windows
function Install-RedisNative {
    # Memurai is a Redis-compatible server for Windows, or use winget Redis
    if (Test-Winget) {
        Write-Host "    Installing Redis via winget..." -ForegroundColor DarkGray
        winget install --id Redis.Redis --accept-source-agreements --accept-package-agreements 2>$null
        if ($LASTEXITCODE -ne 0) {
            # Try Memurai as fallback (Redis-compatible, native Windows)
            winget install --id Memurai.MemuraiDeveloper --accept-source-agreements --accept-package-agreements 2>$null
        }
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    } elseif (Test-Choco) {
        Write-Host "    Installing Redis via Chocolatey..." -ForegroundColor DarkGray
        choco install redis-64 -y 2>$null
        if ($LASTEXITCODE -ne 0) {
            choco install memurai-developer -y 2>$null
        }
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Die "Cannot auto-install Redis. Install from https://github.com/tporadowski/redis/releases or use Docker."
    }
}

if (Test-RedisRunning) {
    Skip- "Redis already running (localhost:6379)"
} elseif ((Test-Command "redis-server")) {
    # redis-server exists but not running — start it
    Start-Process -FilePath "redis-server" -WindowStyle Hidden
    Start-Sleep -Seconds 2
    if (Test-RedisRunning) {
        Ok "Started existing Redis installation (localhost:6379)"
    } else {
        Warn- "redis-server started but not responding yet"
    }
} elseif ($DockerAvailable -and $Compose -and (Test-Path (Join-Path $RepoRoot "docker-compose.yml"))) {
    Warn- "Redis not found natively — falling back to Docker"
    if ($Compose -eq "docker compose") {
        docker compose up -d --wait
    } else {
        docker-compose up -d
    }
    $healthy = $false
    for ($i = 0; $i -lt 15; $i++) {
        if ($Compose -eq "docker compose") {
            $status = docker compose ps redis 2>$null
        } else {
            $status = docker-compose ps redis 2>$null
        }
        if ($status -match "healthy") {
            $healthy = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    if ($healthy) {
        Ok "Redis is up via Docker (localhost:6379)"
    } else {
        Warn- "Redis container started but health check hasn't passed yet."
    }
} else {
    Write-Host "    Redis not found — installing..." -ForegroundColor DarkGray
    Install-RedisNative
    # Try starting it
    if (Test-Command "redis-server") {
        Start-Process -FilePath "redis-server" -WindowStyle Hidden
        Start-Sleep -Seconds 2
    }
    if (Test-RedisRunning) {
        Ok "Redis installed and started (localhost:6379)"
    } else {
        Warn- "Redis installed but may need a manual start. Run: redis-server"
    }
}

# ---------------------------------------------------------------------------
# Step 5 — Ollama models
# ---------------------------------------------------------------------------

Step "Pulling Ollama models"

$ChatModel = "gemma3:4b-it-q4_K_M"
$EmbedModel = "nomic-embed-text"

function Pull-ModelIfMissing {
    param([string]$Model, [string]$SizeHint)
    $list = ollama list 2>$null
    if ($list -match "^$([regex]::Escape($Model))") {
        Skip- "$Model already present"
    } elseif ($SkipModels) {
        Warn- "$Model not found — skipped (run without -SkipModels to pull)"
    } else {
        Write-Host "    Pulling $Model ($SizeHint) — this may take a while..." -ForegroundColor DarkGray
        ollama pull $Model
        if ($LASTEXITCODE -eq 0) {
            Ok "$Model pulled"
        } else {
            Warn- "Failed to pull $Model"
        }
    }
}

Pull-ModelIfMissing $ChatModel  "~3-4 GB"
Pull-ModelIfMissing $EmbedModel "~274 MB"

# ---------------------------------------------------------------------------
# Step 6 — Test suite
# ---------------------------------------------------------------------------

Step "Running test suite"

if ($SkipTests) {
    Skip- "Tests skipped (-SkipTests)"
} else {
    Write-Host ""
    python -m pytest tests/ -v --tb=short
    if ($LASTEXITCODE -eq 0) {
        Ok "All tests passed"
    } else {
        Die "Tests failed. Fix the errors above before using the CLI."
    }
}

# ---------------------------------------------------------------------------
# Done — usage summary
# ---------------------------------------------------------------------------

Write-Host "`nSetup complete.`n" -ForegroundColor Green

Write-Host "Activate the virtual environment:" -ForegroundColor White
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Quick-start commands:" -ForegroundColor White
Write-Host '  gemma ask "Hello"' -ForegroundColor Cyan
Write-Host "           — single-shot query"
Write-Host "  gemma chat" -ForegroundColor Cyan
Write-Host "                   — interactive REPL with memory"
Write-Host "  gemma chat --no-memory" -ForegroundColor Cyan
Write-Host "       — stateless chat (no Redis needed)"
Write-Host '  Get-Content file.txt | gemma pipe' -ForegroundColor Cyan
Write-Host "    — analyse piped content"
Write-Host "  gemma history stats" -ForegroundColor Cyan
Write-Host "          — view memory statistics"
Write-Host ""
Write-Host "To stop Redis when you're done:" -ForegroundColor White
Write-Host "  Stop-Service redis          — stop native Redis service" -ForegroundColor DarkGray
if ($DockerAvailable -and $Compose) {
    Write-Host "  docker compose down         — stop Docker Redis (if used)" -ForegroundColor DarkGray
    Write-Host "  docker compose down -v      — stop Docker Redis and wipe data" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "See README.md for full documentation." -ForegroundColor DarkGray
