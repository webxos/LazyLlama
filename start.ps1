# ============================================================
# LAZY LLAMA v3.6 - Windows PowerShell Startup Script
# ============================================================
# Requires -ExecutionPolicy Bypass
# To run: powershell -ExecutionPolicy Bypass -File start.ps1

# Get script directory and set paths
$APP_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR = Join-Path $APP_DIR "venv"
$LOG_FILE = Join-Path $APP_DIR "startup.log"
$LOCK_FILE = Join-Path $APP_DIR ".startup.lock"
$env:PYTHONPATH = "$APP_DIR;$env:PYTHONPATH"

# Platform (can be overridden by LAZY_PLATFORM env var)
$PLATFORM = if ($env:LAZY_PLATFORM) { $env:LAZY_PLATFORM } else { "windows" }

# Write initial log entry
$startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$startTime] Starting Lazy Llama v3.6 on Windows..." | Out-File -FilePath $LOG_FILE -Encoding utf8

# Function to log messages
function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] $Message"
    $logLine | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
    Write-Host $Message -ForegroundColor $Color
}

# Function to check command existence
function Test-CommandExists {
    param($Command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Stop'
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Retry wrapper for pip commands
function Invoke-PipRetry {
    param(
        [string[]]$Arguments,
        [int]$MaxAttempts = 5
    )
    $attempt = 1
    $delay = 5
    while ($attempt -le $MaxAttempts) {
        Write-Log "Running: pip $($Arguments -join ' ') (attempt $attempt/$MaxAttempts)" -Color Yellow
        & python -m pip $Arguments 2>&1 | Out-File -FilePath $LOG_FILE -Append
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Command succeeded." -Color Green
            return $true
        } else {
            Write-Log "Command failed (exit $LASTEXITCODE). Retrying in ${delay}s..." -Color Red
            Start-Sleep -Seconds $delay
            $delay *= 2
            $attempt++
        }
    }
    Write-Log "Command failed after $MaxAttempts attempts." -Color Red
    return $false
}

# ---- Cleanup function ----
function Cleanup {
    if (Test-Path $LOCK_FILE) {
        Remove-Item -Force $LOCK_FILE
        Write-Log "Removed lock file." -Color Yellow
    }
}

# ---- Trap for errors ----
$ErrorActionPreference = "Stop"
trap {
    Write-Log "An error occurred. Check $LOG_FILE" -Color Red
    Cleanup
    Read-Host "Press Enter to exit"
    exit 1
}

# ---- Lock file to prevent concurrent runs ----
if (Test-Path $LOCK_FILE) {
    Write-Log "Another instance is already running (lock file exists). Exiting." -Color Red
    Read-Host "Press Enter to exit"
    exit 1
}
New-Item -Path $LOCK_FILE -ItemType File -Force | Out-Null

# =============================================================================
# 1. Check Python Installation (fallback to 'py' if needed)
# =============================================================================
Write-Log "Checking Python installation..." -Color Cyan
$pythonCmd = $null
if (Test-CommandExists "python") {
    $pythonCmd = "python"
} elseif (Test-CommandExists "py") {
    $pythonCmd = "py"
} else {
    Write-Log "Python not found. Please install Python 3.10 or later and ensure it is in your PATH." -Color Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Log "Using Python command: $pythonCmd" -Color Cyan

$pythonVersion = (& $pythonCmd --version 2>&1).ToString()
if ($pythonVersion -match "Python (\d+)\.(\d+)") {
    $pyMajor = [int]$matches[1]
    $pyMinor = [int]$matches[2]
    if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 10)) {
        Write-Log "Python 3.10+ required. Found $pythonVersion" -Color Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Log "Python version: $pythonVersion" -Color Green
} else {
    Write-Log "Could not parse Python version. Found: $pythonVersion" -Color Yellow
}

# Check venv module
Write-Log "Checking Python venv module..." -Color Cyan
& $pythonCmd -m venv --help 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Log "Python venv module is not available. Please install python3-venv or ensure it is included." -Color Red
    Read-Host "Press Enter to exit"
    exit 1
}

# =============================================================================
# 2. Memory check (optional)
# =============================================================================
if ($env:LAZY_NO_MEMCHECK -ne "1") {
    Write-Log "Checking system memory..." -Color Cyan
    try {
        $mem = Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction Stop
        $totalRamGB = [math]::Round($mem.TotalPhysicalMemory / 1GB, 1)
        $freeRamGB = [math]::Round((Get-CimInstance -ClassName Win32_OperatingSystem).FreePhysicalMemory / 1MB / 1024, 1)
        Write-Log "RAM: ${freeRamGB}GB available / ${totalRamGB}GB total." -Color Green
    } catch {
        Write-Log "Memory check skipped (requires admin or WMI)." -Color Yellow
    }
}

# =============================================================================
# 3. Virtual Environment (with optional force recreate)
# =============================================================================
$forceVenv = $env:LAZY_FORCE_VENV -eq "1"
if ($forceVenv -and (Test-Path $VENV_DIR)) {
    Write-Log "LAZY_FORCE_VENV=1, removing existing venv..." -Color Yellow
    Remove-Item -Recurse -Force $VENV_DIR
}
if (-not (Test-Path $VENV_DIR)) {
    Write-Log "Creating virtual environment..." -Color Cyan
    & $pythonCmd -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Failed to create virtual environment." -Color Red
        Write-Log "Ensure Python 3.10+ and venv module are installed." -Color Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Log "Virtual environment created." -Color Green
} else {
    Write-Log "Using existing virtual environment." -Color Green
}

# =============================================================================
# 4. Activate Virtual Environment
# =============================================================================
Write-Log "Activating virtual environment..." -Color Cyan
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Log "Activation script not found: $activateScript" -Color Red
    Read-Host "Press Enter to exit"
    exit 1
}
& $activateScript
if ($LASTEXITCODE -ne 0) {
    Write-Log "Failed to activate virtual environment." -Color Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Log "Virtual environment activated." -Color Green

# =============================================================================
# 5. Install/Upgrade Dependencies (with retry)
# =============================================================================
Write-Log "Upgrading pip..." -Color Cyan
Invoke-PipRetry -Arguments @("install", "--upgrade", "pip", "setuptools", "wheel", "--no-cache-dir")

# ---- Remove existing PyTorch (ignore errors if not installed) ----
Write-Log "Removing existing PyTorch (if any)..." -Color Cyan
& python -m pip uninstall torch torchvision torchaudio -y 2>&1 | Out-Null

Write-Log "Installing numpy 2.0+ for Python 3.13 compatibility..." -Color Cyan
if (-not (Invoke-PipRetry -Arguments @("install", "numpy>=2.0.0,<3.0.0", "--only-binary", ":all:", "--no-cache-dir"))) {
    Write-Log "Binary numpy not available, trying to build..." -Color Yellow
    $env:CFLAGS = "-O2 -pipe"
    $env:MAKEFLAGS = "-j2"
    Invoke-PipRetry -Arguments @("install", "numpy>=2.0.0,<3.0.0", "--no-cache-dir")
}

$requirementsFile = Join-Path $APP_DIR "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Log "Installing dependencies from requirements.txt..." -Color Cyan
    if (-not (Invoke-PipRetry -Arguments @("install", "-r", $requirementsFile, "--no-warn-script-location", "--no-cache-dir"))) {
        Write-Log "Failed to install dependencies. Check $LOG_FILE" -Color Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Log "requirements.txt not found. Please ensure it exists in $APP_DIR" -Color Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Log "Upgrading packaging..." -Color Cyan
Invoke-PipRetry -Arguments @("install", "--upgrade", "packaging", "--no-warn-script-location", "--no-cache-dir")

# ---- Install CPU-only PyTorch with force reinstall ----
Write-Log "Installing CPU-only PyTorch..." -Color Cyan
Invoke-PipRetry -Arguments @("install", "--force-reinstall", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu", "--no-warn-script-location", "--no-cache-dir")

# Install local package in editable mode
if (Test-Path (Join-Path $APP_DIR "setup.py")) {
    Write-Log "Installing local package (setup.py)..." -Color Cyan
    Invoke-PipRetry -Arguments @("install", "-e", $APP_DIR, "--no-warn-script-location", "--no-cache-dir")
} elseif (Test-Path (Join-Path $APP_DIR "pyproject.toml")) {
    Write-Log "Installing local package (pyproject.toml)..." -Color Cyan
    Invoke-PipRetry -Arguments @("install", "-e", $APP_DIR, "--no-warn-script-location", "--no-cache-dir")
} else {
    Write-Log "No setup.py or pyproject.toml found. Skipping local install." -Color Yellow
}

# =============================================================================
# 6. Health Check for Critical Packages
# =============================================================================
Write-Log "Verifying critical packages..." -Color Cyan
& $pythonCmd -c "import rich, torch, transformers" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Log "Critical package missing. Retrying installation..." -Color Yellow
    Invoke-PipRetry -Arguments @("install", "--force-reinstall", "rich", "torch", "transformers", "--no-warn-script-location", "--no-cache-dir")
}

# =============================================================================
# 7. Create Necessary Directories
# =============================================================================
Write-Log "Creating required directories..." -Color Cyan
$lazyDir = Join-Path $APP_DIR ".lazy_llama"
$subDirs = @("models", "checkpoints", "logs", "cache", "lazytorch_cache", "exports")
foreach ($sub in $subDirs) {
    $dir = Join-Path $lazyDir $sub
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Log "Created directory: $dir" -Color Green
    }
}

# =============================================================================
# 8. Launch Application with Platform Override (using @args for safe quoting)
# =============================================================================
Write-Log "Platform: $PLATFORM" -Color Green
Write-Log "Launching Lazy Llama..." -Color Cyan

# Use & with @args to preserve quoting
& $pythonCmd -m lazy_llama.bootstrap --platform $PLATFORM @args
$exitCode = $LASTEXITCODE

# Clean up lock file
Cleanup

if ($exitCode -ne 0) {
    Write-Log "Bootstrap.py exited with code $exitCode. Check $LOG_FILE" -Color Red
    Read-Host "Press Enter to exit"
    exit $exitCode
}

Write-Log "Lazy Llama exited normally." -Color Green
exit 0
