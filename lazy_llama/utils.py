"""
Utility functions: memory, checkpoints, retry, logging, Ollama export, download retry, SHA256 verification, and LazyTorch helpers.
Fixed: export_model_to_lazytorch now correctly converts string dtype to torch.dtype.
Fixed: Ollama export for LazyTorch models now checks both 'source_path' and 'original_path' in manifest.
Added: convert_to_safetensors() for vLLM compatibility.
Added: validate_model_zip() to verify model zip archives.
Added: get_global_student_teacher() to retrieve global selections from dashboard state.
FIXED: export_to_ollama() now validates model path validity (HF directory or GGUF file) before export.
FIXED: Increased timeout in export_to_ollama from 120 to 600 seconds, with configurable parameter.
FIXED: Added docstring note about `estimate_memory_need` being a rough estimate.

================================================================================
IMPROVED: Deep tokenizer validation with fallback to use_fast=False and better error messages.
ADDED: copy_tokenizer_files() helper to copy all tokenizer-related files with logging.
FIXED: export_to_ollama now always validates tokenizer before creating Modelfile.
FIXED: validate_model_zip now runs deep tokenizer validation after unpacking.

FURTHER FIX (v3.2):
- copy_tokenizer_files now also copies config.json to ensure the model configuration
  is preserved along with tokenizer files, which is required for some Hugging Face models.

FIXED: export_to_ollama now checks model architecture (model_type) against a list of
       supported architectures for Ollama (Llama, Mistral, Phi, Qwen, GPT-NeoX, etc.)
       and fails early with a clear error if unsupported.

NEW (v3.3.6):
- Added `check_ollama_model()` helper to verify Ollama reachability and model existence.
  Used by dashboard and bootstrap for pre‑distillation checks.
- Enhanced `_validate_tokenizer_deep` with additional logging and a note about
  potential false negatives; added a `strict` parameter (default True) to allow
  permissive validation if needed (but kept strict by default).

================================================================================
PLATFORM SUPPORT (v3.5):
- Added `detect_platform()` to detect OS ('linux', 'darwin', 'windows').
- Added `is_wsl2()` to detect Windows Subsystem for Linux.
- Added `get_platform_defaults()` to return platform‑specific defaults (Ollama binary, etc.).
- Added `expand_windows_path()` convenience function to expand %VAR% on Windows.
- Updated `export_to_ollama()` to use the platform‑aware Ollama binary from config.

================================================================================
FIXES (2026-07-06):
- In export_to_ollama, added explicit subprocess CalledProcessError handling and improved error messages.
- Enhanced _validate_tokenizer_deep with a third fallback using trust_remote_code=True
  for custom tokenizers (with appropriate warnings).
- Verified copy_tokenizer_files already includes config.json; no change needed.

================================================================================
FURTHER FIX (2026-07-10):
- Removed restrictive architecture whitelist in export_to_ollama; now only warns for unknown types.
- Added retry mechanism with timeout escalation for ollama create.
- Improved error messages in validate_model_zip and other functions.

FIX (2026-07-10) - Additional:
- Fixed `_validate_tokenizer_deep` to properly use the `strict` parameter: if `strict=False`,
  the function returns True even after all attempts fail (with a warning), allowing permissive validation.
- Fixed `detect_platform()` to detect WSL2 and return 'windows'.
- Fixed `is_wsl2()` to use explicit imports and clean logic.

================================================================================
IMPROVED TOKENIZER VALIDATION (2026-07-11):
- Enhanced `_validate_tokenizer_deep` with more verbose logging at each fallback stage.
- Added explicit comments explaining when to use `strict=False` (e.g., when a model is known
  to have a slightly non‑standard tokenizer but is still usable).
- Ensured that `copy_tokenizer_files` copies all essential tokenizer files including
  `config.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `special_tokens_map.json`, `added_tokens.json`, `chat_template.json`,
  `generation_config.json`, and `tokenizer.model`.

================================================================================
FIX (2026-07-13) - Stricter architecture check in `export_to_ollama`:
- Unknown model architectures now raise a ValueError instead of just a warning,
  unless the environment variable `LAZY_ALLOW_UNSUPPORTED_OLLAMA` is set to "1".
- This prevents exporting models that are known to cause issues with Ollama.
- The check now uses a curated list of architectures known to work well with Ollama.
- A clear error message is printed with instructions on how to override if needed.

================================================================================
REAP PIPELINE HELPERS (v3.6):
- Added `get_model_checklist_path()`: returns the path to the checklist JSON for a model.
- Added `read_checklist()`: loads the checklist JSON or returns a default empty structure.
- Added `write_checklist()`: writes the checklist JSON to the model directory.
- Added `update_stage_status()`: updates a single stage in the checklist.
- Added `log_stage_summary()`: logs a concise summary of a pipeline stage.
- Added `get_student_model_dir()`: returns the directory where a student model is stored.
- Added `get_lazytorch_path()`: returns the path to the .lazytorch marker/directory.

ENHANCEMENTS (2026-07-15):
- Added tokenizer validation caching (`validate_tokenizer_cached`) to avoid repeated loads.
- Added `atomic_move()` for safe cross‑device file moves with retry and fallback.
- Improved `estimate_memory_need()` by reading `config.json` for parameter count when available.
- Added `get_model_size()` to compute total disk usage of a model.
- Added `safe_rmtree()` with retry for Windows permission issues.
- Added `_tokenizer_cache` dictionary with path+stats as key.
- All new functions are fully documented.

================================================================================
NEW (2026-07-16): Centralised Logging & Reporting Helpers.
- Added `format_error_report()` to produce a detailed error report including traceback.
- Added `setup_lazy_llama_logging()` to configure rotating file logging for all modules.

================================================================================
NEW (2026-07-16): Operation History Logging.
- Added `log_operation_result()` to append an operation history entry to a model's
  registry metadata. Used by distillation, pruning, finetuning, and endless loops.

================================================================================
FIX (2026-07-16): Added missing import for `logging.handlers` in `setup_lazy_llama_logging()`.

REMOVED (2026-07-17): Removed all HydraHead helper functions. These have been removed
from the project.
"""

import psutil
import torch
import logging
import logging.handlers  # <-- FIX: required for RotatingFileHandler
import gc
import time
import json
import subprocess
import hashlib
import re
import shutil
import tempfile
import requests
import platform as _platform
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Union, Tuple
from datetime import datetime
import traceback

# ---- Do NOT import from config at module level (circular import) ----
# Instead, we import locally inside functions that need them.

logger = logging.getLogger(__name__)

# =============================================================================
# Tokenizer validation cache (keyed by path + mtime + size)
# =============================================================================
_TOKENIZER_CACHE: Dict[Tuple[Path, float, int], bool] = {}
_TOKENIZER_CACHE_MAXSIZE = 128  # limit to prevent unbounded growth


def _tokenizer_cache_key(path: Path) -> Tuple[Path, float, int]:
    """Generate a cache key from path, modification time, and size."""
    try:
        stat = path.stat()
        return (path, stat.st_mtime, stat.st_size)
    except OSError:
        # If stat fails, use path only (fallback)
        return (path, 0.0, 0)


def validate_tokenizer_cached(path: Path, strict: bool = True) -> bool:
    """
    Validate tokenizer with caching based on file modification time and size.
    This avoids repeatedly loading the tokenizer for the same path.

    Args:
        path: Directory containing tokenizer files.
        strict: Passed to _validate_tokenizer_deep.

    Returns:
        True if tokenizer is valid, False otherwise.
    """
    key = _tokenizer_cache_key(path)
    if key in _TOKENIZER_CACHE:
        logger.debug(f"Using cached tokenizer validation for {path}")
        return _TOKENIZER_CACHE[key]

    result = _validate_tokenizer_deep(path, strict=strict)

    # Update cache, respecting max size
    if len(_TOKENIZER_CACHE) >= _TOKENIZER_CACHE_MAXSIZE:
        # Remove oldest entry (ordered insertion in Python 3.7+)
        oldest = next(iter(_TOKENIZER_CACHE))
        del _TOKENIZER_CACHE[oldest]
    _TOKENIZER_CACHE[key] = result
    logger.debug(f"Cached tokenizer validation for {path}: {result}")
    return result


def clear_tokenizer_cache() -> None:
    """Clear the tokenizer validation cache."""
    _TOKENIZER_CACHE.clear()
    logger.debug("Tokenizer validation cache cleared.")


# ----------------------------------------------------------------------
# Platform detection helpers (v3.5) - FIXED
# ----------------------------------------------------------------------
def detect_platform() -> str:
    """
    Detect the current operating system.
    Returns: 'linux', 'darwin', or 'windows'.
    """
    sys_plat = _platform.system().lower()
    if sys_plat == "windows":
        return "windows"
    elif sys_plat == "darwin":
        return "darwin"
    else:
        # Check for WSL2 inside Linux
        if "microsoft" in _platform.uname().release.lower():
            return "windows"
        return "linux"


def is_wsl2() -> bool:
    """
    Return True if running inside WSL2 (Windows Subsystem for Linux).
    Checks both the kernel release and the WSL_DISTRO_NAME environment variable.
    """
    import platform as _platform
    import os
    # Check kernel release for 'microsoft'
    if "microsoft" in _platform.uname().release.lower():
        return True
    # Also check for WSL environment variable
    return os.environ.get("WSL_DISTRO_NAME") is not None


def get_platform_defaults() -> dict:
    """
    Return a dict with platform‑specific defaults for various components.
    Keys: 'ollama_binary', 'llama_cpp_package', 'pip_index'.
    """
    plat = detect_platform()
    defaults = {
        "ollama_binary": "ollama",
        "llama_cpp_package": "llama-cpp-python",
        "pip_index": "https://pypi.org/simple"
    }
    if plat == "windows":
        # Typical Windows install location; expand environment variable
        defaults["ollama_binary"] = r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    elif is_wsl2():
        # In WSL2, Ollama might be installed on the Windows side; we can use the Windows binary
        # if it's in PATH, otherwise fallback to 'ollama' (if installed inside WSL).
        defaults["ollama_binary"] = "ollama.exe"  # assumes in PATH
    # For macOS/Linux, defaults are fine.
    return defaults


def expand_windows_path(path: str) -> str:
    """
    Expand Windows environment variables in a path using os.path.expandvars.
    For example, '%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe' becomes the actual path.
    On non-Windows, this function simply returns the original path.
    """
    return os.path.expandvars(path)


# ----------------------------------------------------------------------
# Helper: Deep tokenizer validation (actually loads the tokenizer) with fallback
# ----------------------------------------------------------------------
def _validate_tokenizer_deep(path: Path, strict: bool = True) -> bool:
    """
    Attempt to load the tokenizer from the given path using AutoTokenizer.
    Returns True if successful, False otherwise.
    This catches the "untagged enum ModelWrapper" error and other parsing issues.
    If default loading fails, it retries with `use_fast=False` as a fallback.
    If that also fails, it tries with `trust_remote_code=True` as a last resort
    (for custom tokenizers that require remote code execution).

    Logs a detailed error message on failure, including a suggestion to delete the model.

    The `strict` parameter controls the behavior when all attempts fail:
      - strict=True (default): returns False, meaning the tokenizer is considered invalid.
      - strict=False: returns True with a warning, allowing the caller to proceed
        even if tokenizer validation fails. This is useful for cases where the
        tokenizer is known to be slightly non‑standard but still functional
        (e.g., some older GPT‑2 models). Use with caution.

    Args:
        path: Directory containing tokenizer files.
        strict: If True (default), return False on any failure.
                If False, return True even if validation fails (use with caution).
    """
    try:
        from transformers import AutoTokenizer
        # First attempt: default settings
        logger.debug(f"Attempting to load tokenizer from {path} with default settings...")
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        # Sanity check: vocab size > 0 and can encode
        if tokenizer.vocab_size == 0:
            logger.error(f"Tokenizer at {path} has vocab size 0")
            return False
        tokenizer.encode("test")
        logger.debug(f"Tokenizer loaded successfully with default settings.")
        return True
    except Exception as e:
        logger.debug(f"Default tokenizer loading failed for {path}: {e}")
        # Fallback 1: try with use_fast=False (some tokenizers are not fast-compatible)
        try:
            logger.warning(f"Fallback: Trying tokenizer with use_fast=False for {path}.")
            tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=False)
            if tokenizer.vocab_size == 0:
                logger.error(f"Tokenizer at {path} (use_fast=False) has vocab size 0")
                return False
            tokenizer.encode("test")
            logger.debug(f"Tokenizer loaded with use_fast=False.")
            return True
        except Exception as e2:
            logger.debug(f"Tokenizer with use_fast=False failed: {e2}")
            # Fallback 2: try with trust_remote_code=True for custom tokenizers
            try:
                logger.warning(f"Fallback: Trying tokenizer with trust_remote_code=True for {path}.")
                tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
                if tokenizer.vocab_size == 0:
                    logger.error(f"Tokenizer at {path} (trust_remote_code=True) has vocab size 0")
                    return False
                tokenizer.encode("test")
                logger.debug(f"Tokenizer loaded with trust_remote_code=True.")
                return True
            except Exception as e3:
                # All attempts failed; log detailed error with actionable advice
                logger.error(
                    f"Deep tokenizer validation failed for {path}.\n"
                    f"First error: {e}\nSecond error: {e2}\nThird error: {e3}\n"
                    "This usually means the tokenizer files are corrupt or incompatible.\n"
                    "Please delete the model and re-download it, or repair the tokenizer files.\n"
                    f"You can delete it using: python bootstrap.py remove --model {path.stem}"
                )
                # If not strict, return True as a fallback (permissive mode)
                if not strict:
                    logger.warning(
                        f"Tokenizer validation failed but strict=False; returning True as fallback. "
                        "This may cause issues if the tokenizer is actually corrupt."
                    )
                    return True
                return False


# ----------------------------------------------------------------------
# Helper: Check Ollama model availability
# ----------------------------------------------------------------------
def check_ollama_model(model_name: str) -> bool:
    """
    Return True if Ollama is reachable and the specified model exists.
    Used for pre‑distillation validation.
    """
    try:
        requests.get("http://localhost:11434", timeout=2)
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return any(m.get("name") == model_name for m in models)
        return False
    except Exception:
        return False


# ----------------------------------------------------------------------
# Helper: Copy tokenizer files from src to dst with logging
# ----------------------------------------------------------------------
def copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy all tokenizer-related files from src_dir to dst_dir.
    Logs each file copied.
    Includes config.json for safety, as some models rely on it for tokenizer configuration.
    Also copies generation_config.json and chat_template.json if present, as they are
    often required for correct tokenizer behavior.
    """
    # Extended list of tokenizer-related files; includes config.json and others
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.json",
        "generation_config.json",
        "tokenizer.model",
        "config.json",   # added for safety; sometimes needed for tokenizer setup
    ]
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in tokenizer_files:
        src_file = src_dir / fname
        dst_file = dst_dir / fname
        if src_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
                logger.debug(f"Copied tokenizer file: {fname}")
            except Exception as e:
                logger.warning(f"Failed to copy {fname}: {e}")


# ----------------------------------------------------------------------
# System information (no circular imports)
# ----------------------------------------------------------------------
def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024**3)


def get_total_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def get_cpu_percent() -> float:
    return psutil.cpu_percent(interval=0.5)


def get_memory_usage_gb() -> float:
    """Cross‑platform memory usage."""
    try:
        return psutil.Process().memory_info().rss / (1024**3)
    except AttributeError:
        return psutil.virtual_memory().used / (1024**3)


def clear_cuda_memory() -> None:
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def get_gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_system_profile() -> Dict[str, Any]:
    """
    Return system RAM, GPU, CPU info for auto-optimization.
    Implemented inline to avoid circular imports.
    """
    ram_total_gb = psutil.virtual_memory().total / (1024**3)
    ram_available_gb = psutil.virtual_memory().available / (1024**3)
    cpu_cores = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 0

    gpu_available = torch.cuda.is_available()
    gpu_memory_gb = 0
    if gpu_available:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return {
        "total_ram_gb": ram_total_gb,
        "available_ram_gb": ram_available_gb,
        "cpu_cores": cpu_cores,
        "cpu_freq_mhz": cpu_freq,
        "gpu_available": gpu_available,
        "gpu_memory_gb": gpu_memory_gb,
    }


def recommend_model_size(ram_gb: float) -> str:
    if ram_gb < 4:
        return "none"
    elif ram_gb < 6:
        return "2B -> distill to 0.5B"
    elif ram_gb < 8:
        return "3B"
    else:
        return "7B (with aggressive pruning)"


def is_ram_sufficient(required_gb: float, margin: float = 0.5) -> bool:
    return get_available_ram_gb() >= (required_gb + margin)


def check_low_ram(warning_threshold_gb: float = 1.0) -> bool:
    free = get_available_ram_gb()
    if free < warning_threshold_gb:
        logger.warning(f"Low RAM: {free:.1f} GB free")
        return True
    return False


def get_device(device_str: str = "cpu") -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_bytes(num_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


# ----------------------------------------------------------------------
# Checkpoint management – using local imports for CHECKPOINTS_DIR
# ----------------------------------------------------------------------
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    step: int,
    path: Path,
    metadata: Optional[Dict] = None
) -> None:
    from .config import CHECKPOINTS_DIR  # local import avoids circularity
    
    # Capture architecture information for compatibility checks
    arch_info = {}
    if hasattr(model, 'config'):
        arch_info = {
            "num_layers": getattr(model.config, "num_hidden_layers", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "vocab_size": getattr(model.config, "vocab_size", None),
            "model_type": getattr(model.config, "model_type", None),
        }
    # Fallback for models without config
    if not arch_info or all(v is None for v in arch_info.values()):
        # Try to infer from model structure
        num_layers = sum(1 for n, _ in model.named_modules() if 'layer' in n or 'h.' in n)
        arch_info = {"inferred_num_layers": num_layers}
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'step': step,
        'metadata': metadata or {},
        'architecture': arch_info,
    }, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
    map_location: str = 'cpu',
    strict: bool = True
) -> tuple:
    from .config import CHECKPOINTS_DIR  # local import
    ckpt = torch.load(path, map_location=map_location)
    
    # ---- NEW: Architecture compatibility check ----
    arch_info = ckpt.get('architecture', {})
    if arch_info:
        current_arch = {}
        if hasattr(model, 'config'):
            current_arch = {
                "num_layers": getattr(model.config, "num_hidden_layers", None),
                "hidden_size": getattr(model.config, "hidden_size", None),
                "vocab_size": getattr(model.config, "vocab_size", None),
                "model_type": getattr(model.config, "model_type", None),
            }
        # Compare key fields (skip None values)
        mismatch = False
        for key, value in arch_info.items():
            if key in current_arch and current_arch[key] is not None and value is not None:
                if current_arch[key] != value:
                    mismatch = True
                    break
        
        if mismatch:
            logger.warning(
                f"Checkpoint architecture mismatch!\n"
                f"Checkpoint: {arch_info}\n"
                f"Current model: {current_arch}\n"
                "This checkpoint was created for a different model architecture.\n"
                "Loading with strict=False to attempt partial loading. "
                "This may cause missing key errors."
            )
            strict = False
    
    # Load with optional strict=False
    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=strict)
    except RuntimeError as e:
        # Re-raise with clearer message
        raise RuntimeError(
            f"Failed to load checkpoint {path}:\n"
            f"{e}\n\n"
            "This usually means the checkpoint was created for a different model.\n"
            f"To fix: delete the checkpoint and restart without resume:\n"
            f"  rm -f {path}\n"
            "Or use the model that matches this checkpoint."
        ) from e
    
    if optimizer and ckpt.get('optimizer_state_dict'):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, ckpt['epoch'], ckpt.get('step', 0), ckpt.get('metadata', {})


def find_latest_checkpoint(model_name: str) -> Optional[Path]:
    from .config import CHECKPOINTS_DIR  # local import
    checkpoints = list(CHECKPOINTS_DIR.glob(f"{model_name}_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def retry(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """Exponential backoff retry decorator."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_attempts} after error: {e}")
                    time.sleep(_delay)
                    _delay *= backoff
            return None
        return wrapper
    return decorator


# ----------------------------------------------------------------------
# Download utilities (SHA256 verification - empty hashes = skip)
# ----------------------------------------------------------------------
KNOWN_BINARY_HASHES = {
    "https://github.com/ggerganov/llama.cpp/releases/download/b4488/llama-b4488-bin-ubuntu-x64.zip": "",
    "https://github.com/ggerganov/llama.cpp/releases/download/b4488/llama-b4488-bin-macos-arm64.zip": "",
    "https://github.com/ggerganov/llama.cpp/releases/download/b4488/llama-b4488-bin-macos-x64.zip": "",
    "https://github.com/ggerganov/llama.cpp/releases/download/b4488/llama-b4488-bin-win-avx2-x64.zip": "",
}

def download_with_retry(url: str, dest: Path, max_retries: int = 3) -> bool:
    """Download a file with exponential backoff retry."""
    import urllib.request
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            logger.warning(f"Download failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Verify SHA256 checksum of a file. If expected_hash is empty, skip verification."""
    if not expected_hash:
        logger.warning("No expected hash provided, skipping SHA256 verification.")
        return True
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    computed = sha256.hexdigest()
    if computed != expected_hash:
        logger.error(f"SHA256 mismatch: expected {expected_hash}, got {computed}")
        return False
    return True


# ----------------------------------------------------------------------
# Memory estimation (improved)
# ----------------------------------------------------------------------
def estimate_memory_need(model_path: Path) -> float:
    """
    Estimate memory needed to load a model (rough upper bound).
    For LazyTorch models, returns a small value because weights are not loaded into RAM.
    If config.json is present and contains `num_hidden_layers` or `num_parameters`,
    compute a more accurate estimate.

    NOTE: This is only an estimate; actual memory usage during inference may be higher
    due to activations, cache, and framework overhead.
    """
    # LazyTorch models use almost no RAM
    if is_lazytorch_model(model_path):
        return 0.1  # Extremely low memory (manifest + overhead)

    # Try to read config.json for parameter estimation
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                cfg = json.load(f)
            # Estimate parameters from model architecture
            hidden_size = cfg.get('hidden_size') or cfg.get('d_model')
            num_hidden_layers = cfg.get('num_hidden_layers') or cfg.get('num_layers')
            vocab_size = cfg.get('vocab_size')
            # Rough estimate: for transformer, params ~ 12 * hidden_size^2 * num_layers + vocab_size * hidden_size
            if hidden_size and num_hidden_layers and vocab_size:
                # 12 includes attention (4*W_q, W_k, W_v, W_o) + FFN (2*W1, W2) = 4 + 2 = 6, times 2 for QKV? Actually typical: 12 * d_model^2 per layer.
                num_params = 12 * (hidden_size ** 2) * num_hidden_layers + vocab_size * hidden_size
                # Assume 4 bytes per parameter (FP32), plus 20% overhead
                memory_gb = (num_params * 4) / 1e9 * 1.2
                return memory_gb
        except Exception as e:
            logger.debug(f"Could not read config.json for memory estimation: {e}")

    # Fallback estimation
    try:
        # Try to import from e8_quantize if available (relative import)
        from .e8_quantize import estimate_memory_need as _e8_estimate
        return _e8_estimate(model_path)
    except (ImportError, AttributeError):
        # Fallback estimation
        if model_path.is_dir():
            size_gb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1e9
            return size_gb * 1.5
        elif model_path.suffix == ".gguf":
            return model_path.stat().st_size / 1e9 * 1.2
        return 2.0


# ----------------------------------------------------------------------
# Safe atomic move with retry and fallback
# ----------------------------------------------------------------------
def atomic_move(src: Path, dst: Path, max_retries: int = 5, delay: float = 0.5) -> None:
    """
    Atomically move a file or directory from src to dst.
    Uses shutil.move (rename) which is atomic on most Unix systems.
    If rename fails (e.g., cross-device), falls back to shutil.copytree + shutil.rmtree
    with retries on Windows for permission issues.

    Args:
        src: Source path (file or directory).
        dst: Destination path.
        max_retries: Number of retries on failure.
        delay: Initial delay between retries (exponential backoff).

    Raises:
        shutil.Error: If the move fails after all retries.
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    # If dst already exists, remove it if it's a directory or file
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
        else:
            dst.unlink()

    # First attempt: shutil.move (rename)
    try:
        shutil.move(str(src), str(dst))
        logger.debug(f"Moved {src} -> {dst} via rename")
        return
    except (OSError, shutil.Error) as e:
        logger.warning(f"Rename failed (cross-device?): {e}. Falling back to copy+delete.")

    # Fallback: copy + delete with retries
    for attempt in range(max_retries):
        try:
            if src.is_dir():
                shutil.copytree(src, dst, symlinks=True, ignore_dangling_symlinks=True)
            else:
                shutil.copy2(src, dst)
            # Remove source after successful copy
            if src.is_dir():
                shutil.rmtree(src, ignore_errors=True)
            else:
                src.unlink()
            logger.debug(f"Moved {src} -> {dst} via copy+delete (attempt {attempt+1})")
            return
        except Exception as e:
            logger.warning(f"Copy+delete attempt {attempt+1}/{max_retries} failed: {e}")
            # Clean up partial destination
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                else:
                    dst.unlink()
            time.sleep(delay * (2 ** attempt))

    raise shutil.Error(f"Failed to move {src} to {dst} after {max_retries} attempts.")


# ----------------------------------------------------------------------
# Safe rmtree with retry (Windows permission issues)
# ----------------------------------------------------------------------
def safe_rmtree(path: Path, max_retries: int = 5, delay: float = 0.5) -> None:
    """
    Safely remove a directory tree with retries, useful for Windows permission issues.
    """
    if not path.exists():
        return
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Permission error removing {path} (attempt {attempt+1}): {e}. Retrying...")
            time.sleep(delay * (2 ** attempt))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Error removing {path} (attempt {attempt+1}): {e}. Retrying...")
            time.sleep(delay * (2 ** attempt))


# ----------------------------------------------------------------------
# Model size helper
# ----------------------------------------------------------------------
def get_model_size(model_path: Path) -> int:
    """
    Return the total disk size of a model directory or file in bytes.
    For directories, it recursively sums all file sizes.
    For files, returns the file size.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return 0
    if model_path.is_file():
        return model_path.stat().st_size
    total = 0
    for f in model_path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


# ----------------------------------------------------------------------
# Ollama export with improved error handling and optional registry update
# (Uses platform‑aware binary from config)
# ----------------------------------------------------------------------
def export_to_ollama(
    model_path: str,
    model_name: str,
    quantize: str = "q4_0",
    update_registry: bool = True,
    timeout: int = 600
) -> bool:
    """
    Export a local model (GGUF, Hugging Face directory, or LazyTorch) to Ollama.
    For LazyTorch models, the original Hugging Face source is used if available,
    otherwise a warning is issued.

    This function uses the platform‑aware Ollama binary path from the configuration,
    so it works correctly on Windows, macOS, Linux, and WSL2.

    Args:
        model_path: Path to the model file or directory.
        model_name: Name to give the model in Ollama.
        quantize: (currently unused) intended for future quantization options.
        update_registry: If True, update the ModelManager registry with the exported model.
        timeout: Timeout in seconds for the `ollama create` subprocess (default 600).

    Returns:
        True if successful, False otherwise.

    Raises:
        ValueError: If the model architecture is not supported by Ollama and the
                    environment variable LAZY_ALLOW_UNSUPPORTED_OLLAMA is not set to "1".
    """
    # Load config locally to avoid circular import
    from .config import load_config
    config = load_config()
    ollama_bin = config.get_ollama_binary()
    if not ollama_bin:
        ollama_bin = "ollama"  # fallback

    # Expand Windows environment variables if any
    ollama_bin = expand_windows_path(ollama_bin)

    # Check if ollama binary is available
    try:
        subprocess.run([ollama_bin, "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Ollama binary check failed: {e.stderr.decode() if e.stderr else e}")
        return False
    except FileNotFoundError:
        logger.error(f"Ollama not found at: {ollama_bin}")
        logger.error("Please ensure Ollama is installed and the binary is accessible.")
        return False

    # Convert to absolute path and normalize slashes (Windows → forward slashes)
    abs_path = Path(model_path).resolve()
    if not abs_path.exists():
        logger.error(f"Model path does not exist: {abs_path}")
        return False

    # Validate model path is a valid HF directory or GGUF file
    original_abs_path = abs_path  # keep for later use
    if abs_path.is_dir():
        # Check for HF model: config.json + weights
        config_exists = (abs_path / "config.json").exists()
        weights_exist = (abs_path / "pytorch_model.bin").exists() or (abs_path / "model.safetensors").exists()
        is_lazytorch = is_lazytorch_model(abs_path)
        if not (config_exists and weights_exist) and not is_lazytorch:
            logger.error(f"Directory {abs_path} is not a valid Hugging Face model (missing config.json or weight files)")
            return False

        # ---- If it's a LazyTorch model, try to find the original HF path from manifest ----
        if is_lazytorch:
            logger.info("LazyTorch model detected. Exporting to Ollama may require the original Hugging Face model.")
            manifest_path = abs_path / "manifest.json" if abs_path.is_dir() else abs_path.with_suffix('') / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    # Try both possible keys: 'source_path' (old) or 'original_path' (new)
                    original_path = manifest.get("source_path") or manifest.get("original_path")
                    if original_path and Path(original_path).exists():
                        logger.info(f"Using original Hugging Face model at {original_path} for export.")
                        # Validate tokenizer of the original model
                        if not validate_tokenizer_cached(Path(original_path)):
                            logger.error(f"Tokenizer in original model {original_path} is corrupt. Cannot export.")
                            return False
                        abs_path = Path(original_path)
                    else:
                        logger.warning("Original model path not found in manifest; attempting export with LazyTorch path may fail.")
                        # Still allow export, but we'll validate tokenizer of LazyTorch directory later
                except Exception as e:
                    logger.warning(f"Failed to read manifest: {e}")
            else:
                logger.warning("Cannot find original model source; export may fail.")

        # ---- Validate tokenizer for HF directory (if not LazyTorch or we found original) ----
        if not is_lazytorch_model(abs_path) and not validate_tokenizer_cached(abs_path):
            logger.error(f"Tokenizer in {abs_path} is corrupt or incompatible. Cannot export to Ollama.")
            return False

    elif abs_path.is_file():
        if not abs_path.suffix == ".gguf":
            logger.error(f"File {abs_path} is not a .gguf file")
            return False
        # GGUF models don't have tokenizer files; skip validation.
    else:
        logger.error(f"Path {abs_path} is neither a directory nor a file")
        return False

    # ---- Check model architecture compatibility with Ollama ----
    # We now block unknown architectures unless the user explicitly overrides.
    if abs_path.is_dir():
        config_file = abs_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    cfg = json.load(f)
                model_type = cfg.get("model_type", "").lower()
                # Curated list of architectures known to work well with Ollama
                supported_types = [
                    "llama", "mistral", "phi", "qwen", "gptneox",
                    "gptj", "falcon", "mpt", "stablelm", "starcoder",
                    "codegen", "bloom", "opt", "gpt2", "gemma", "gemma2"
                ]
                if model_type and not any(supported in model_type for supported in supported_types):
                    # Check for override environment variable
                    allow_override = os.environ.get("LAZY_ALLOW_UNSUPPORTED_OLLAMA") == "1"
                    if not allow_override:
                        raise ValueError(
                            f"Model architecture '{model_type}' is not in the list of architectures "
                            "known to work with Ollama. Export may fail or produce unexpected results.\n"
                            "To force export anyway, set the environment variable:\n"
                            "  LAZY_ALLOW_UNSUPPORTED_OLLAMA=1\n"
                            "and try again. If you are sure the model is compatible, this will allow "
                            "export, but be aware that Ollama may not support it fully."
                        )
                    else:
                        logger.warning(
                            f"Model architecture '{model_type}' is not in the supported list, "
                            "but LAZY_ALLOW_UNSUPPORTED_OLLAMA=1 is set. Proceeding with export."
                        )
            except Exception as e:
                logger.warning(f"Could not read config.json for architecture check: {e}")
        else:
            # No config.json – likely not a HF model, skip architecture check.
            pass

    # Use the resolved path (may be original HF path or LazyTorch path)
    model_path_fixed = abs_path.as_posix()

    # Clean model name: remove .gguf extension and sanitize
    clean_name = model_name
    if clean_name.lower().endswith('.gguf'):
        clean_name = clean_name[:-5]
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)

    # Create Modelfile
    modelfile_content = (
        f"FROM {model_path_fixed}\n"
        f"TEMPLATE \"\"\"{{ .Prompt }}\"\"\"\n"
        f"PARAMETER temperature 0.7\n"
        f"PARAMETER top_p 0.9\n"
        f"SYSTEM \"\"\"You are a helpful assistant.\"\"\"\n"
    )

    modelfile_path = Path.home() / f".ollama/modelfiles/{clean_name}.Modelfile"
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(modelfile_content)

    # Create the model in Ollama using the platform‑aware binary with retry
    cmd = [ollama_bin, "create", clean_name, "-f", str(modelfile_path)]
    logger.info(f"Running: {' '.join(cmd)} (timeout {timeout}s)")

    max_retries = 2
    retry_count = 0
    current_timeout = timeout
    while retry_count < max_retries:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=current_timeout, check=True)
            # If we get here, success
            break
        except subprocess.TimeoutExpired:
            retry_count += 1
            if retry_count < max_retries:
                # Increase timeout and retry
                current_timeout = int(current_timeout * 1.5)
                logger.warning(
                    f"Ollama create timed out after {timeout} seconds. "
                    f"Retrying with timeout {current_timeout}s (attempt {retry_count+1}/{max_retries})..."
                )
            else:
                logger.error(
                    f"Ollama create timed out after {current_timeout} seconds (after {max_retries} attempts). "
                    "This may indicate the model is too large or Ollama is slow. "
                    "Consider increasing the timeout in config (ollama_timeout) and retry."
                )
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama create failed (return code {e.returncode}): {e.stderr if e.stderr else e.stdout}")
            return False

    logger.info(f"Model {clean_name} exported to Ollama successfully")

    # Optionally update registry if we have a ModelManager instance
    if update_registry:
        try:
            from .lazy_model_manager import ModelManager
            mm = ModelManager()
            existing = mm.get_model(clean_name)
            if not existing:
                # Use the original path (if it was a LazyTorch model, we may have resolved to HF)
                size_mb = sum(f.stat().st_size for f in abs_path.rglob("*") if f.is_file()) / (1024*1024)
                mm.registry[clean_name] = mm._create_model_info(
                    name=clean_name,
                    path=str(abs_path),
                    size_mb=size_mb
                )
                mm._save_registry()
                logger.debug(f"Added {clean_name} to registry")
        except Exception as e:
            logger.warning(f"Could not update registry: {e}")

    return True


# ----------------------------------------------------------------------
# LazyTorch helper functions
# ----------------------------------------------------------------------
def is_lazytorch_model(path: Path) -> bool:
    """Check if a given path points to a LazyTorch model directory or marker."""
    path = Path(path)
    if path.is_dir():
        return (path / "manifest.json").exists()
    elif path.suffix == ".lazytorch":
        return True
    return False


def get_lazytorch_model_size(path: Path) -> int:
    """Return total disk size of a LazyTorch model in bytes."""
    path = Path(path)
    if is_lazytorch_model(path):
        if path.suffix == ".lazytorch":
            lazy_dir = path.with_suffix('')
        else:
            lazy_dir = path
        if lazy_dir.exists():
            total = 0
            for f in lazy_dir.glob("*"):
                if f.is_file() and f.name != "manifest.json":
                    total += f.stat().st_size
            return total
    return 0


def export_model_to_lazytorch(
    model: Path,
    output_path: Optional[Path] = None,
    dtype: str = "float32",
    progress_callback: Optional[Callable[[str], None]] = None
) -> Path:
    """
    Convert a Hugging Face model (directory or loaded model) to LazyTorch format.

    Args:
        model: Path to Hugging Face model directory
        output_path: Destination for .lazytorch folder (default: model_path.with_suffix('.lazytorch'))
        dtype: "float32" or "float16"
        progress_callback: Optional function called with status messages

    Returns:
        Path to the created .lazytorch directory
    """
    from .lazytorch_core import export_to_lazytorch as _export

    # Convert string dtype to torch.dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    return _export(
        model,
        output_path=output_path,
        dtype=torch_dtype,
        progress_callback=progress_callback
    )


def convert_hf_to_lazytorch(
    hf_path: Path,
    output_path: Optional[Path] = None,
    dtype: str = "float32",
    progress_callback: Optional[Callable[[str], None]] = None
) -> Path:
    """
    Convenience alias for export_model_to_lazytorch.
    """
    return export_model_to_lazytorch(hf_path, output_path, dtype, progress_callback)


# =============================================================================
# REAP PIPELINE HELPERS (v3.6)
# =============================================================================

def get_model_checklist_path(model_name: str, models_dir: Optional[Path] = None) -> Path:
    """
    Return the path to the checklist JSON for a model.
    The checklist tracks which REAP stages (distillation, pruning, finetuning, evaluation)
    have been completed and when.

    Args:
        model_name: Name of the model.
        models_dir: Optional base models directory; if None, uses the global MODELS_DIR.

    Returns:
        Path to the checklist JSON file (e.g., MODELS_DIR/model_name/checklist.json).
    """
    from .config import MODELS_DIR as _MODELS_DIR
    if models_dir is None:
        models_dir = _MODELS_DIR
    return models_dir / model_name / "checklist.json"


def read_checklist(model_name: str, models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the checklist JSON for a model, or return a default empty structure.

    The default structure:
        {
            "model_name": model_name,
            "stages": {
                "distillation": {"completed": False, "timestamp": None},
                "pruning": {"completed": False, "timestamp": None},
                "finetuning": {"completed": False, "timestamp": None},
                "evaluation": {"completed": False, "timestamp": None, "score": None}
            },
            "metadata": {}
        }

    Args:
        model_name: Name of the model.
        models_dir: Optional base models directory.

    Returns:
        Dictionary with the checklist data.
    """
    path = get_model_checklist_path(model_name, models_dir)
    default = {
        "model_name": model_name,
        "stages": {
            "distillation": {"completed": False, "timestamp": None},
            "pruning": {"completed": False, "timestamp": None},
            "finetuning": {"completed": False, "timestamp": None},
            "evaluation": {"completed": False, "timestamp": None, "score": None},
        },
        "metadata": {},
    }
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # Ensure all default keys exist
            if "stages" not in data:
                data["stages"] = default["stages"]
            for stage in default["stages"].keys():
                if stage not in data["stages"]:
                    data["stages"][stage] = default["stages"][stage]
                else:
                    # Ensure sub-keys exist
                    for key in default["stages"][stage]:
                        if key not in data["stages"][stage]:
                            data["stages"][stage][key] = default["stages"][stage][key]
            if "metadata" not in data:
                data["metadata"] = {}
            return data
        except Exception as e:
            logger.warning(f"Failed to read checklist for {model_name}: {e}; using default.")
            return default
    return default


def write_checklist(model_name: str, checklist: Dict[str, Any], models_dir: Optional[Path] = None) -> None:
    """
    Write the checklist JSON to the model directory.

    Args:
        model_name: Name of the model.
        checklist: The checklist dictionary (must contain at least "stages").
        models_dir: Optional base models directory.
    """
    path = get_model_checklist_path(model_name, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure the checklist has the required structure
    if "model_name" not in checklist:
        checklist["model_name"] = model_name
    if "stages" not in checklist:
        checklist["stages"] = {}
    with open(path, "w") as f:
        json.dump(checklist, f, indent=2)


def update_stage_status(
    model_name: str,
    stage: str,
    completed: bool,
    metadata: Optional[Dict] = None,
    models_dir: Optional[Path] = None
) -> None:
    """
    Update a single stage in the checklist.

    Args:
        model_name: Name of the model.
        stage: The stage name (e.g., "distillation", "pruning", "finetuning", "evaluation").
        completed: Whether the stage has been completed.
        metadata: Optional dictionary with additional data to store for this stage
                  (e.g., teacher name, pruning strategy, eval score).
        models_dir: Optional base models directory.
    """
    checklist = read_checklist(model_name, models_dir)
    if "stages" not in checklist:
        checklist["stages"] = {}
    if stage not in checklist["stages"]:
        checklist["stages"][stage] = {}
    checklist["stages"][stage]["completed"] = completed
    checklist["stages"][stage]["timestamp"] = datetime.now().isoformat()
    if metadata:
        for k, v in metadata.items():
            checklist["stages"][stage][k] = v
    write_checklist(model_name, checklist, models_dir)


def log_stage_summary(stage: str, model_name: str, success: bool, extra: str = "") -> None:
    """
    Log a concise summary of a pipeline stage.

    Args:
        stage: The stage name (e.g., "Distillation", "Pruning", "Finetuning", "Evaluation").
        model_name: The model name.
        success: Whether the stage succeeded.
        extra: Optional extra message (e.g., "teacher=llama2").
    """
    status = "✅" if success else "❌"
    logger.info(f"{status} {stage} for {model_name} {extra}".strip())


def get_student_model_dir(model_name: str, models_dir: Optional[Path] = None) -> Path:
    """
    Return the directory where a student model should be stored.

    Args:
        model_name: Name of the student model.
        models_dir: Optional base models directory; if None, uses the global MODELS_DIR.

    Returns:
        Path to the model directory.
    """
    from .config import MODELS_DIR as _MODELS_DIR
    if models_dir is None:
        models_dir = _MODELS_DIR
    return models_dir / model_name


def get_lazytorch_path(model_name: str, models_dir: Optional[Path] = None) -> Path:
    """
    Return the path to the .lazytorch marker/directory for a model.

    Args:
        model_name: Name of the model.
        models_dir: Optional base models directory.

    Returns:
        Path ending with .lazytorch (e.g., MODELS_DIR/model_name.lazytorch).
    """
    return get_student_model_dir(model_name, models_dir).with_suffix(".lazytorch")


# =============================================================================
# SafeTensor conversion for vLLM compatibility
# =============================================================================
def convert_to_safetensors(model_path: Path) -> Path:
    """
    Convert PyTorch .bin files to safetensors format for vLLM compatibility.

    Args:
        model_path: Path to the Hugging Face model directory containing .bin files.

    Returns:
        Path to the directory where safetensors files are saved.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading model from {model_path} for safetensors conversion...")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    output_path = model_path / "safetensors"
    output_path.mkdir(exist_ok=True)

    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Model saved in safetensors format to {output_path}")
    return output_path


# =============================================================================
# Model zip validation (with tokenizer check)
# =============================================================================
def validate_model_zip(zip_path: Path) -> bool:
    """
    Validate that a zip file contains a valid model structure.
    Checks for config.json, at least one of model.safetensors or pytorch_model.bin,
    and attempts to load the tokenizer to ensure it is not corrupt.

    Args:
        zip_path: Path to the zip file.

    Returns:
        True if valid, False otherwise.
    """
    if not zip_path.exists():
        logger.error(f"Zip file not found: {zip_path}")
        return False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(str(zip_path), temp_dir)
            temp_path = Path(temp_dir)

            # Check for required files
            required = ["config.json"]
            # Check if safetensors exists; if not, require pytorch_model.bin
            has_safetensors = (temp_path / "model.safetensors").exists()
            required.append("model.safetensors" if has_safetensors else "pytorch_model.bin")

            for req in required:
                if not (temp_path / req).exists():
                    logger.error(f"Missing required file: {req}")
                    return False

            # Try loading config to ensure it's valid JSON
            import json
            try:
                with open(temp_path / "config.json") as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid config.json: {e}")
                return False

            # ---- Deep tokenizer validation (cached) ----
            if not validate_tokenizer_cached(temp_path):
                logger.error(f"Tokenizer in zip is corrupt or incompatible: {zip_path}")
                return False

            logger.info("Zip validation passed (including tokenizer)")
            return True
    except Exception as e:
        logger.error(f"Zip validation failed: {e}")
        return False


# =============================================================================
# Global state helpers (dashboard integration)
# =============================================================================
def get_global_student_teacher():
    """
    Retrieve the global student and teacher model names as set in the dashboard.
    Returns a tuple (teacher, student) or (None, None) if not available.
    """
    try:
        from .dashboard_server import _global_student, _global_teacher
        return _global_teacher, _global_student
    except (ImportError, AttributeError):
        # Fallback: try reading the persisted state file directly
        try:
            from .config import LAZY_DIR
            state_file = LAZY_DIR / "global_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    data = json.load(f)
                    return data.get("teacher"), data.get("student")
        except Exception:
            pass
    return None, None


# =============================================================================
# Centralised Logging & Reporting Helpers
# =============================================================================
def format_error_report(exception: Exception, context: str = "") -> str:
    """
    Format a detailed error report including exception type, message, and full traceback.

    Args:
        exception: The exception instance.
        context: A string describing the operation or context where the error occurred.

    Returns:
        A multi-line string with the error report.
    """
    lines = [
        f"Error in {context or 'unknown context'}:",
        f"Type: {type(exception).__name__}",
        f"Message: {str(exception)}",
        "Traceback:",
        traceback.format_exc()
    ]
    return "\n".join(lines)


def setup_lazy_llama_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 10
) -> None:
    """
    Set up a rotating file logger for the Lazy Llama application and its modules.
    This configures both a console handler and a rotating file handler.

    Args:
        log_dir: Directory where log files will be stored. If None, uses `LOGS_DIR` from config.
        log_level: Logging level (default: logging.INFO).
        max_bytes: Maximum size of a log file before rotation (default 10 MB).
        backup_count: Number of backup files to keep (default 10).
    """
    if log_dir is None:
        from .config import LOGS_DIR
        log_dir = LOGS_DIR
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "lazy_llama.log"

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation (using logging.handlers.RotatingFileHandler)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Console handler (optional, but included for convenience)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Get the root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers that might duplicate (optional)
    for handler in root_logger.handlers[:]:
        # Avoid removing handlers that are not ours if they exist
        if isinstance(handler, (logging.StreamHandler, logging.handlers.RotatingFileHandler)):
            root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Set the logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: file={log_file}, level={logging.getLevelName(log_level)}")


# =============================================================================
# Operation History Logging
# =============================================================================
def log_operation_result(
    model_name: str,
    operation: str,
    success: bool,
    details: Dict[str, Any],
    manager: Optional[Any] = None,
    timestamp: Optional[str] = None
) -> None:
    """
    Append an operation history entry to a model's registry metadata.

    This function is used to track distillation, pruning, finetuning, and other
    operations performed on a model. The history is stored in the model's
    metadata under the key 'operation_history'.

    Args:
        model_name: The model name.
        operation: The operation name (e.g., 'distill', 'prune', 'finetune', 'auto_cycle').
        success: Whether the operation succeeded.
        details: Additional parameters (teacher, passes, error, strategy, etc.).
        manager: ModelManager instance. If None, a new one is created.
        timestamp: ISO timestamp (default: current time).
    """
    # Import ModelManager inside to avoid circular import
    from .lazy_model_manager import ModelManager

    if manager is None:
        manager = ModelManager()

    info = manager.get_model(model_name)
    if not info:
        logger.warning(f"Model '{model_name}' not found in registry; cannot log operation.")
        return

    # Ensure metadata exists
    if not hasattr(info, 'metadata') or info.metadata is None:
        info.metadata = {}

    # Initialize operation history list if not present
    if 'operation_history' not in info.metadata:
        info.metadata['operation_history'] = []

    # Build entry
    entry = {
        'timestamp': timestamp or datetime.now().isoformat(),
        'operation': operation,
        'success': success,
        'details': details
    }

    # Append to history
    info.metadata['operation_history'].append(entry)

    # Save registry
    try:
        manager._save_registry()
        logger.debug(f"Logged operation '{operation}' for model '{model_name}' (success={success})")
    except Exception as e:
        logger.error(f"Failed to save operation history for '{model_name}': {e}")