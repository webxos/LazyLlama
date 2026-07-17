@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem LAZY LLAMA v3.6 - Windows Command Prompt Startup Script
rem ============================================================

set APP_DIR=%~dp0
set VENV_DIR=%APP_DIR%venv
set LOG_FILE=%APP_DIR%startup.log
set LOCK_FILE=%APP_DIR%.startup.lock
set PYTHONPATH=%APP_DIR%;%PYTHONPATH%

rem Platform (can be overridden by LAZY_PLATFORM env var)
if defined LAZY_PLATFORM (
    set PLATFORM=%LAZY_PLATFORM%
) else (
    set PLATFORM=windows
)

rem Write initial log entry
echo [%date% %time%] Starting Lazy Llama v3.6 on Windows... > "%LOG_FILE%" 2>&1

rem ------------------------------------------------------------------
rem Helper: log messages (prepend timestamp, write to log and console)
rem ------------------------------------------------------------------
call :log "Starting Lazy Llama v3.6..." Cyan

rem ------------------------------------------------------------------
rem Lock file to prevent concurrent runs
rem ------------------------------------------------------------------
if exist "%LOCK_FILE%" (
    call :log "Another instance is already running (lock file exists). Exiting." Red
    pause
    exit /b 1
)
echo %date% %time% > "%LOCK_FILE%"

rem ------------------------------------------------------------------
rem 1. Check Python installation (try 'python' then 'py')
rem ------------------------------------------------------------------
call :log "Checking Python installation..." Cyan
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
) else (
    py --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py
    ) else (
        call :log "Python not found. Please install Python 3.10 or later and ensure it is in your PATH." Red
        del "%LOCK_FILE%" 2>nul
        pause
        exit /b 1
    )
)
call :log "Using Python command: %PYTHON_CMD%" Cyan

for /f "tokens=2" %%I in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VER=%%I
call :log "Python version: %PYTHON_VER%" Green

rem Parse version (e.g., "3.13.0" or "3.10.4")
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    call :log "Python 3.10+ required. Found %PYTHON_VER%" Red
    del "%LOCK_FILE%" 2>nul
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    call :log "Python 3.10+ required. Found %PYTHON_VER%" Red
    del "%LOCK_FILE%" 2>nul
    pause
    exit /b 1
)

rem Check venv module
call :log "Checking Python venv module..." Cyan
%PYTHON_CMD% -m venv --help >nul 2>&1
if errorlevel 1 (
    call :log "Python venv module is not available. Please install python3-venv." Red
    del "%LOCK_FILE%" 2>nul
    pause
    exit /b 1
)

rem ------------------------------------------------------------------
rem 2. Memory check (optional)
rem ------------------------------------------------------------------
if "%LAZY_NO_MEMCHECK%"=="1" goto :skip_memcheck
call :log "Checking system memory..." Cyan
wmic os get FreePhysicalMemory,TotalVisibleMemorySize 2>nul | findstr /r "[0-9]" > mem.txt
if errorlevel 1 (
    call :log "Memory check skipped (wmic not available)." Yellow
) else (
    for /f "tokens=1,2" %%a in (mem.txt) do (
        set FREE_KB=%%a
        set TOTAL_KB=%%b
    )
    del mem.txt
    set /a FREE_GB=!FREE_KB! / 1024 / 1024
    set /a TOTAL_GB=!TOTAL_KB! / 1024 / 1024
    call :log "RAM: !FREE_GB!GB available / !TOTAL_GB!GB total." Green
)
:skip_memcheck

rem ------------------------------------------------------------------
rem 3. Virtual Environment (with optional force recreate)
rem ------------------------------------------------------------------
if "%LAZY_FORCE_VENV%"=="1" (
    if exist "%VENV_DIR%" (
        call :log "LAZY_FORCE_VENV=1, removing existing venv..." Yellow
        rmdir /s /q "%VENV_DIR%"
    )
)
if not exist "%VENV_DIR%" (
    call :log "Creating virtual environment..." Cyan
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        call :log "Failed to create virtual environment." Red
        call :log "Ensure Python 3.10+ and venv module are installed." Yellow
        del "%LOCK_FILE%" 2>nul
        pause
        exit /b 1
    )
    call :log "Virtual environment created." Green
) else (
    call :log "Using existing virtual environment." Green
)

rem ------------------------------------------------------------------
rem 4. Activate virtual environment
rem ------------------------------------------------------------------
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    call :log "Failed to activate virtual environment." Red
    del "%LOCK_FILE%" 2>nul
    pause
    exit /b 1
)
call :log "Virtual environment activated." Green

rem ------------------------------------------------------------------
rem 5. Install/Upgrade Dependencies (with retry)
rem ------------------------------------------------------------------
set MAX_ATTEMPTS=5
set ATTEMPT=1
set DELAY=5

call :log "Upgrading pip..." Cyan
call :pip_retry install --upgrade pip setuptools wheel --no-cache-dir
if errorlevel 1 goto :install_failed

rem ---- Remove existing PyTorch (ignore errors if not installed) ----
call :log "Removing existing PyTorch (if any)..." Cyan
pip uninstall torch torchvision torchaudio -y 2>nul
set errorlevel=0

call :log "Installing numpy 2.0+ for Python 3.13 compatibility..." Cyan
call :pip_retry install "numpy>=2.0.0,<3.0.0" --only-binary :all: --no-cache-dir
if errorlevel 1 (
    call :log "Binary numpy not available, trying to build..." Yellow
    set CFLAGS=-O2 -pipe
    set MAKEFLAGS=-j2
    call :pip_retry install "numpy>=2.0.0,<3.0.0" --no-cache-dir
    if errorlevel 1 goto :install_failed
)

if exist "%APP_DIR%requirements.txt" (
    call :log "Installing dependencies from requirements.txt..." Cyan
    call :pip_retry install -r "%APP_DIR%requirements.txt" --no-warn-script-location --no-cache-dir
    if errorlevel 1 goto :install_failed
) else (
    call :log "requirements.txt not found. Please ensure it exists in %APP_DIR%" Red
    del "%LOCK_FILE%" 2>nul
    pause
    exit /b 1
)

call :log "Upgrading packaging..." Cyan
call :pip_retry install --upgrade packaging --no-warn-script-location --no-cache-dir
if errorlevel 1 goto :install_failed

rem ---- Install CPU-only PyTorch with force reinstall ----
call :log "Installing CPU-only PyTorch..." Cyan
call :pip_retry install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location --no-cache-dir
if errorlevel 1 goto :install_failed

rem Install local package in editable mode
if exist "%APP_DIR%setup.py" (
    call :log "Installing local package (setup.py)..." Cyan
    call :pip_retry install -e "%APP_DIR%" --no-warn-script-location --no-cache-dir
    if errorlevel 1 goto :install_failed
) else (
    if exist "%APP_DIR%pyproject.toml" (
        call :log "Installing local package (pyproject.toml)..." Cyan
        call :pip_retry install -e "%APP_DIR%" --no-warn-script-location --no-cache-dir
        if errorlevel 1 goto :install_failed
    ) else (
        call :log "No setup.py or pyproject.toml found. Skipping local install." Yellow
    )
)

goto :install_done
:install_failed
call :log "Failed to install dependencies. Check %LOG_FILE%" Red
del "%LOCK_FILE%" 2>nul
pause
exit /b 1
:install_done

rem ------------------------------------------------------------------
rem 6. Health Check for Critical Packages
rem ------------------------------------------------------------------
call :log "Verifying critical packages..." Cyan
%PYTHON_CMD% -c "import rich, torch, transformers" >nul 2>&1
if errorlevel 1 (
    call :log "Critical package missing. Retrying installation..." Yellow
    call :pip_retry install --force-reinstall rich torch transformers --no-warn-script-location --no-cache-dir
)

rem ------------------------------------------------------------------
rem 7. Create Necessary Directories
rem ------------------------------------------------------------------
call :log "Creating required directories..." Cyan
if not exist "%APP_DIR%.lazy_llama" mkdir "%APP_DIR%.lazy_llama"
if not exist "%APP_DIR%.lazy_llama\models" mkdir "%APP_DIR%.lazy_llama\models"
if not exist "%APP_DIR%.lazy_llama\checkpoints" mkdir "%APP_DIR%.lazy_llama\checkpoints"
if not exist "%APP_DIR%.lazy_llama\logs" mkdir "%APP_DIR%.lazy_llama\logs"
if not exist "%APP_DIR%.lazy_llama\cache" mkdir "%APP_DIR%.lazy_llama\cache"
if not exist "%APP_DIR%.lazy_llama\lazytorch_cache" mkdir "%APP_DIR%.lazy_llama\lazytorch_cache"
if not exist "%APP_DIR%.lazy_llama\exports" mkdir "%APP_DIR%.lazy_llama\exports"

rem ------------------------------------------------------------------
rem 8. Launch Application with Platform Override (using module)
rem ------------------------------------------------------------------
call :log "Platform: %PLATFORM%" Green
call :log "Launching Lazy Llama..." Cyan
%PYTHON_CMD% -m lazy_llama.bootstrap --platform %PLATFORM% %*
set EXIT_CODE=%ERRORLEVEL%

rem Remove lock file
del "%LOCK_FILE%" 2>nul

if %EXIT_CODE% neq 0 (
    call :log "Lazy Llama exited with code %EXIT_CODE%. Check %LOG_FILE%" Red
    pause
    exit /b %EXIT_CODE%
)

call :log "Lazy Llama exited normally." Green
exit /b 0

rem =========================================================================
rem Subroutines
rem =========================================================================

:log
echo [%date% %time%] %~1 >> "%LOG_FILE%" 2>&1
if "%~2"=="Red" (
    echo %~1
) else if "%~2"=="Green" (
    echo %~1
) else if "%~2"=="Yellow" (
    echo %~1
) else if "%~2"=="Cyan" (
    echo %~1
) else (
    echo %~1
)
exit /b

:pip_retry
set /a ATTEMPT=1
set /a DELAY=5
:retry_loop
if !ATTEMPT! leq %MAX_ATTEMPTS% (
    call :log "Running: pip %* (attempt !ATTEMPT!/%MAX_ATTEMPTS%)" Yellow
    pip %* >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log "Command failed (exit !ERRORLEVEL!). Retrying in !DELAY!s..." Red
        timeout /t !DELAY! /nobreak >nul
        set /a DELAY*=2
        set /a ATTEMPT+=1
        goto retry_loop
    ) else (
        call :log "Command succeeded." Green
        exit /b 0
    )
) else (
    call :log "Command failed after %MAX_ATTEMPTS% attempts." Red
    exit /b 1
)
