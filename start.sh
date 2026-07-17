#!/bin/bash
# =============================================
# LAZY LLAMA v3.6 - Seamless Startup
# Auto-creates venv, installs deps, runs app
# =============================================

set -e

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║             LAZY LLAMA v3.6 + LazyTorch                      ║${NC}"
echo -e "${BLUE}║          Seamless Startup (Kali / Ubuntu)                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# ============================================================
# 0. Install system build dependencies (if missing)
# ============================================================
echo -e "${YELLOW}→ Checking system build tools...${NC}"
if ! command -v gcc &> /dev/null; then
    echo -e "${YELLOW}→ Installing build tools...${NC}"
    sudo apt update && sudo apt install -y python3-dev gcc g++ build-essential libatlas-base-dev gfortran
else
    echo -e "${GREEN}✓ Build tools found${NC}"
fi

# ============================================================
# 1. Check Python
# ============================================================
echo -e "${YELLOW}→ Checking Python...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 not found. Please install: sudo apt install python3${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# ============================================================
# 2. Setup Virtual Environment
# ============================================================
VENV_DIR="$APP_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}→ Creating virtual environment...${NC}"
    
    # Ensure python3-venv is installed
    if ! python3 -m venv --help &> /dev/null; then
        echo -e "${YELLOW}→ Installing python3-venv...${NC}"
        sudo apt update && sudo apt install python3-venv python3-full -y
    fi
    
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Verify we're in the venv
echo -e "${YELLOW}→ Virtual environment: $VIRTUAL_ENV${NC}"

# ============================================================
# 3. Upgrade pip & install dependencies (verbose for debugging)
# ============================================================
echo -e "${YELLOW}→ Upgrading pip...${NC}"
python -m pip install --upgrade pip setuptools wheel

# ---- Remove any existing PyTorch to avoid conflicts ----
echo -e "${YELLOW}→ Removing existing PyTorch (if any)...${NC}"
python -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# ---- Install CPU-only PyTorch with force reinstall ----
echo -e "${YELLOW}→ Installing PyTorch (CPU version)...${NC}"
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ---- Install NumPy with fallback ----
echo -e "${YELLOW}→ Installing NumPy (prefer binary wheel)...${NC}"
if ! python -m pip install "numpy>=2.0.0,<3.0.0" --only-binary :all: --no-cache-dir; then
    echo -e "${YELLOW}  NumPy 2.0+ binary not available, falling back to 1.26.x...${NC}"
    python -m pip install "numpy>=1.26.0,<2.0.0" --no-cache-dir
fi

if [ -f "$APP_DIR/requirements.txt" ]; then
    echo -e "${YELLOW}→ Installing remaining dependencies from requirements.txt...${NC}"
    python -m pip install -r "$APP_DIR/requirements.txt"
else
    echo -e "${RED}✗ requirements.txt not found!${NC}"
    exit 1
fi

# ============================================================
# 4. Install the package in editable mode (with verbose output)
# ============================================================
echo -e "${YELLOW}→ Installing Lazy Llama package...${NC}"

# ---- FIXED: Check for the correct package source directory ----
if [ ! -d "$APP_DIR/lazy_llama" ]; then
    echo -e "${RED}✗ Error: lazy_llama/ folder not found!${NC}"
    echo -e "${YELLOW}  Please ensure all Python files are in lazy_llama/ directory${NC}"
    exit 1
fi

# Install with verbose output to see what's failing
python -m pip install -e "$APP_DIR" --verbose

# ============================================================
# 5. Verify installation with detailed checks
# ============================================================
echo -e "${YELLOW}→ Verifying installation...${NC}"

# Check if the package is installed
if python -c "import lazy_llama; print('✅ Package imported successfully')" 2>/dev/null; then
    echo -e "${GREEN}✓ Package installed successfully${NC}"
else
    echo -e "${RED}✗ Package installation failed${NC}"
    echo -e "${YELLOW}  Diagnostic information:${NC}"
    
    # Show what's in site-packages
    echo -e "${YELLOW}  Contents of site-packages:${NC}"
    python -c "import site; print(site.getsitepackages())"
    ls -la "$(python -c 'import site; print(site.getsitepackages()[0])')" | grep -E "lazy|llama" || echo "  No lazy_llama found in site-packages"
    
    # Check if the egg-link exists
    if [ -f "$(python -c 'import site; print(site.getsitepackages()[0])')/lazy_llama.egg-link" ]; then
        echo -e "${YELLOW}  egg-link found:${NC}"
        cat "$(python -c 'import site; print(site.getsitepackages()[0])')/lazy_llama.egg-link"
    fi
    
    # Try to import with explicit path
    echo -e "${YELLOW}  Trying import with explicit sys.path:${NC}"
    python -c "import sys; sys.path.insert(0, '.'); import lazy_llama; print('✅ Import with sys.path worked')" 2>&1 || echo "  Import with sys.path failed"
    
    echo -e "${RED}  Exiting. Please fix the installation manually.${NC}"
    exit 1
fi

# ============================================================
# 6. Setup required directories
# ============================================================
echo -e "${YELLOW}→ Creating required directories...${NC}"
mkdir -p "$HOME/.lazy_llama"/{models,checkpoints,logs,cache,lazytorch_cache,exports}

# ============================================================
# 7. Launch the application
# ============================================================
echo -e "${GREEN}✓ Starting Lazy Llama...${NC}"
echo ""

python -m lazy_llama.bootstrap "$@"

# ============================================================
# 8. Cleanup
# ============================================================
deactivate 2>/dev/null
echo -e "${GREEN}✓ Done${NC}"