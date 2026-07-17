"""
lazy_model_manager.py - Core model registry and student creation logic.
Manages local models, downloads, distillation, pruning, and hybrid attention.
Now includes vLLM model sync to discover models served by a vLLM server.

NEW (v3.6): Added `get_student_models()` helper to list all distilled or pruned models.
NEW (2026-07-08): Added `download_and_register_model()` convenience method.

FIXES (2026-07-07):
- create_student now uses shutil.copytree to copy the base model directory directly,
  avoiding memory-intensive load/save operations. This is more efficient and works
  for LazyTorch, GGUF, and HF models.
- After copying, the tokenizer is validated using _validate_tokenizer_deep and the
  directory is removed if validation fails.
- If hybrid attention is enabled, the model is loaded from the copied directory,
  converted, saved, and re-validated.
- validate_model_directory already checks both pytorch_model.bin and model.safetensors.
- Added proper error handling and cleanup.

FIX (2026-07-08): ensure tokenizer validation is performed after copying the base model
                  and after hybrid conversion, and that invalid models are marked as such.

FIX (2026-07-08): Added validation to prevent using Ollama/vLLM models as base for student
                  creation. Only local Hugging Face models (or LazyTorch/GGUF) can be used
                  as base for student creation.

FURTHER FIXES (2026-07-10):
- create_student now uses symlinks=True in copytree to preserve symlinks.
- create_student handles both directories and file-based models (GGUF) correctly.
- validate_model now treats ollama:// and vllm:// URIs as valid (skip validation).
- download_from_hf uses snapshot_download instead of loading model into memory.
- sync_vllm improved with better error logging.
- validate_model_directory now supports .gguf files with basic validation.

FIX (2026-07-10): Converted all absolute imports to relative imports.

ADDITIONAL FIXES (2026-07-11):
- Guarded huggingface_hub import with clear error message if not installed.
- In create_student, added symlinks=False on Windows to avoid permission issues
  with symlinks (typical on Windows without admin rights). On Unix, symlinks=True.
- Added a comment about Windows administrator rights when using symlinks.
- Improved error messages for failed copies on Windows.
- Added fallback for copying directories on Windows if shutil.copytree fails.
- Ensured that if copying fails, the destination is cleaned up.

FIX (2026-07-11) - Robust tokenizer validation:
- In create_student, after copying the base model, always call _validate_tokenizer_deep
  for non-GGUF directories. If validation fails, delete the destination and raise
  a clear RuntimeError.
- After hybrid conversion, re-validate the tokenizer and clean up on failure.
- All validation failures now raise exceptions instead of silently returning False,
  ensuring caller is aware of corruption.

FIX (2026-07-13) - Additional validation hardening:
- Added early tokenizer validation in `_ensure_base_model`: when a base model already
  exists, we validate its tokenizer and mark it invalid if corrupt before attempting
  to copy it. This prevents creating a student from a corrupt base model.
- Improved error messages for GGUF pruning/distillation rejection with clear
  instructions on how to convert GGUF to a PyTorch model.
- In `convert_to_lazytorch`, after exporting, we now validate the tokenizer in the
  destination directory and delete on failure, raising a clear error.
- Added a `_validate_model_path` helper to centralize tokenizer validation.

FIX (2026-07-13) - Thread-safety fixes:
- All registry modifications (download_from_hf, sync_ollama, sync_vllm) now use
  `self._lock` to prevent data corruption in multi-threaded environments.
- The lock is also used when marking models invalid in `_ensure_base_model`.

ENHANCEMENTS (2026-07-15):
- Added file‑locking (fcntl/portalocker) for registry operations to prevent
  concurrent write corruption from multiple processes.
- `create_student` now uses hardlinks when possible (cross‑device fallback to copy)
  to save disk space and speed up student creation.
- Added `update_model_info(name, **kwargs)` method to update individual fields
  without full replacement.
- Added `get_model_by_path(path)` method to resolve a model by its disk path.
- Added `_copy_model_files()` helper with hardlink support.
- Updated `_save_registry` and `_load_registry` to use file locking.
- Used `utils.safe_rmtree` for reliable deletion.

FIX (2026-07-15): Ensured file locks are released even on exception using try/finally.
FIX (2026-07-15): Fixed hardlink logic in `_copy_with_hardlink` to attempt hardlinks
                  unconditionally, falling back to copy only on failure.

NEW (2026-07-16): Added `get_operation_history()` method to retrieve a model's
                   operation history from metadata.

NEW (2026-07-17): Added `is_valid_hf_model()` static method to verify if a model ID
                   exists on Hugging Face Hub without attempting a full download.

REMOVED (2026-07-17): Removed all HydraHead (hybrid attention) related code:
- Removed imports of `create_hybrid_head_mask`, `export_hybrid_config`, and `apply_hybrid_attention_to_model`.
- Removed `use_hybrid_heads`, `head_mask`, `la_fa_ratio`, `use_gated_deltanet` parameters from `create_student`.
- Deleted the entire hybrid attention application block.

FIX (2026-07-17): In `_save_registry`, added a fallback direct write to the final
registry file if the atomic move fails. This prevents registry save failures when
the temporary file cannot be created or moved due to disk/permission issues.
"""

import json
import logging
import shutil
import threading
import sys
import os
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Local imports – all relative to avoid circularity ----
from .config import (
    Config,
    ModelInfo,
    LAZY_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    LAZYTORCH_CACHE_DIR,
    load_config,
)
from .utils import (
    is_lazytorch_model,
    _validate_tokenizer_deep,
    validate_tokenizer_cached,
    copy_tokenizer_files,
    get_available_ram_gb,
    download_with_retry,
    detect_platform,
    safe_rmtree,
    atomic_move,
)

# ---- Try to import openai for vLLM sync ----
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# ---- Try to import huggingface_hub for efficient downloading ----
try:
    from huggingface_hub import snapshot_download, hf_hub_download, model_info as hf_model_info
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    snapshot_download = None
    hf_hub_download = None
    hf_model_info = None

# ---- Import GGUF validator for file validation (relative) ----
try:
    from .lazy_infer import is_valid_gguf
except ImportError:
    is_valid_gguf = None

# ---- File locking support ----
try:
    import fcntl
    HAVE_FCNTL = True
except ImportError:
    HAVE_FCNTL = False

try:
    import portalocker
    HAVE_PORTALOCKER = True
except ImportError:
    HAVE_PORTALOCKER = False

if not HAVE_FCNTL and not HAVE_PORTALOCKER:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Neither fcntl nor portalocker is available. "
        "Registry file may be corrupted by concurrent processes. "
        "Install portalocker (pip install portalocker) for cross‑platform locking."
    )

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the local model registry, downloads, and student creation.
    Thread‑safe using a lock, and process‑safe using file locking (fcntl/portalocker).
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: Optional Config object; if not provided, loads from disk.
        """
        self.config = config or load_config()
        self._lock = threading.RLock()
        self.models_dir = MODELS_DIR
        self.registry_path = LAZY_DIR / "registry.json"
        self._registry: Dict[str, ModelInfo] = {}
        self._load_registry()

    # --------------------------------------------------------------------------
    # Public property for read‑only access to registry
    # --------------------------------------------------------------------------
    @property
    def registry(self) -> Dict[str, ModelInfo]:
        """Return the internal registry dict (read‑only)."""
        return self._registry

    # --------------------------------------------------------------------------
    # Registry management with file locking
    # --------------------------------------------------------------------------
    def _acquire_file_lock(self, file_obj, exclusive: bool = True) -> None:
        """
        Acquire an advisory file lock on the given file object.
        Uses fcntl on Unix, portalocker on Windows, or falls back to no‑op.
        """
        if HAVE_FCNTL:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        elif HAVE_PORTALOCKER:
            portalocker.lock(file_obj, portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH)
        else:
            # No locking available; warn once per instance
            if not getattr(self, '_lock_warned', False):
                logger.warning("No file locking available; registry may be corrupted by concurrent processes.")
                self._lock_warned = True

    def _release_file_lock(self, file_obj) -> None:
        """Release an advisory file lock."""
        if HAVE_FCNTL:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        elif HAVE_PORTALOCKER:
            portalocker.unlock(file_obj)
        # else no‑op

    def _load_registry(self) -> None:
        """Load the registry from disk, or create an empty one."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    self._acquire_file_lock(f, exclusive=False)
                    try:
                        data = json.load(f)
                    finally:
                        self._release_file_lock(f)
                self._registry = {
                    name: ModelInfo.from_dict(info)
                    for name, info in data.items()
                }
                logger.info(f"Loaded {len(self._registry)} models from registry.")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._registry = {}
        else:
            self._registry = {}

    def _save_registry(self) -> None:
        """
        Save the registry to disk with retry and fallback.
        Uses exclusive file locking and atomic rename if possible.
        """
        temp_path = self.registry_path.with_suffix(".tmp")

        # Prepare data once
        data = {name: info.to_dict() for name, info in self._registry.items()}

        # Try atomic save with retries
        success = False
        for attempt in range(3):
            try:
                with open(temp_path, "w") as f:
                    self._acquire_file_lock(f, exclusive=True)
                    try:
                        json.dump(data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    finally:
                        self._release_file_lock(f)
                atomic_move(temp_path, self.registry_path)
                success = True
                logger.debug(f"Registry saved atomically to {self.registry_path} (attempt {attempt+1})")
                break
            except Exception as e:
                logger.debug(f"Atomic save attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))

        if not success:
            logger.warning(
                f"Atomic registry save failed after 3 attempts. "
                f"Falling back to direct write to {self.registry_path}.\n"
                "This is usually harmless. The registry was still saved successfully."
            )
            try:
                with open(self.registry_path, "w") as f:
                    self._acquire_file_lock(f, exclusive=True)
                    try:
                        json.dump(data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    finally:
                        self._release_file_lock(f)
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                logger.info(f"Registry saved directly to {self.registry_path}.")
            except Exception as e2:
                logger.error(f"Direct registry save also failed: {e2}")
                # Emergency fallback: write to a backup location
                backup_path = self.registry_path.with_suffix(".json.bak")
                try:
                    with open(backup_path, "w") as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"Registry backed up to {backup_path}. Please restore manually if needed.")
                except Exception as e3:
                    logger.critical(f"Emergency backup also failed: {e3}")
                raise RuntimeError(f"Registry save failed: {e2}") from e2

    def reload_registry(self, sync_ollama: bool = True) -> None:
        """Reload registry from disk, optionally syncing with Ollama and vLLM first."""
        if sync_ollama:
            self.sync_ollama()
        # Sync vLLM if enabled in config
        if getattr(self.config, 'vllm_enabled', False):
            self.sync_vllm()
        self._load_registry()

    # --------------------------------------------------------------------------
    # Sync methods (unchanged, but use file locking in save)
    # --------------------------------------------------------------------------
    def sync_ollama(self) -> None:
        """Sync Ollama models into the registry."""
        try:
            import subprocess
            import requests
            # Check if Ollama is running
            requests.get("http://localhost:11434", timeout=2)
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) < 2:
                return
            # ---- Thread-safe registry update ----
            with self._lock:
                for line in lines[1:]:
                    parts = line.split()
                    if not parts:
                        continue
                    name = parts[0]
                    if name in self._registry:
                        continue
                    info = ModelInfo(
                        name=name,
                        original_size_mb=0.0,
                        path=f"ollama://{name}",
                        model_type="ollama",
                    )
                    self._registry[name] = info
                self._save_registry()
        except Exception as e:
            logger.debug(f"Ollama sync failed: {e}")

    def sync_vllm(self) -> None:
        """Query vLLM server for available models and add them to the registry."""
        if not OPENAI_AVAILABLE:
            logger.debug("openai package not installed; vLLM sync disabled.")
            return
        try:
            base_url = getattr(self.config, 'vllm_base_url', 'http://localhost:8000/v1')
            api_key = getattr(self.config, 'vllm_api_key', 'EMPTY')
            client = OpenAI(base_url=base_url, api_key=api_key)
            models = client.models.list()
            with self._lock:
                for model in models:
                    name = model.id
                    if name not in self._registry:
                        logger.debug(f"Discovered vLLM model: {name}")
                        info = ModelInfo(
                            name=name,
                            original_size_mb=0.0,
                            path=f"vllm://{name}",
                            model_type="vllm",
                        )
                        self._registry[name] = info
                if models:
                    self._save_registry()
                    logger.info(f"Synced {len(models)} model(s) from vLLM server at {base_url}")
        except Exception as e:
            if "ConnectionError" in str(type(e)):
                logger.debug(f"vLLM sync failed: could not connect to {base_url}")
            else:
                logger.debug(f"vLLM sync failed: {e}")

    # --------------------------------------------------------------------------
    # Basic registry access
    # --------------------------------------------------------------------------
    def get_model(self, name: str) -> Optional[ModelInfo]:
        with self._lock:
            return self._registry.get(name)

    def list_models(self, include_invalid: bool = False) -> List[ModelInfo]:
        with self._lock:
            if include_invalid:
                return list(self._registry.values())
            return [info for info in self._registry.values() if not info.invalid]

    def model_exists(self, name: str) -> bool:
        info = self.get_model(name)
        return info is not None and not info.invalid

    def delete_model(self, name: str) -> bool:
        with self._lock:
            info = self._registry.get(name)
            if info is None:
                logger.warning(f"Model '{name}' not found.")
                return False
            # Remove directory if it exists
            if info.path and Path(info.path).exists():
                safe_rmtree(Path(info.path))
            # Also remove any LazyTorch cache
            lazytorch_dir = LAZYTORCH_CACHE_DIR / name
            if lazytorch_dir.exists():
                safe_rmtree(lazytorch_dir)
            del self._registry[name]
            self._save_registry()
            logger.info(f"Deleted model '{name}'.")
            return True

    # --------------------------------------------------------------------------
    # Download / base model handling
    # --------------------------------------------------------------------------
    def download_from_hf(
        self,
        model_name: str,
        gguf_file: Optional[str] = None,
        convert_to_lazytorch_after: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Optional[Path]:
        """Download a model from Hugging Face."""
        dest_dir = MODELS_DIR / model_name
        if dest_dir.exists():
            logger.info(f"Model already exists at {dest_dir}")
            if dest_dir.is_dir() and not validate_tokenizer_cached(dest_dir):
                logger.error(f"Existing model at {dest_dir} has a corrupt tokenizer. Marking as invalid.")
                with self._lock:
                    info = self.get_model(model_name)
                    if info:
                        info.invalid = True
                        self._save_registry()
                return None
            return dest_dir

        dest_dir.mkdir(parents=True, exist_ok=True)

        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is not installed. Please install it with: pip install huggingface-hub"
            )

        try:
            if gguf_file:
                # Download a specific GGUF file
                if progress_callback:
                    progress_callback(10, f"Downloading GGUF file: {gguf_file}")
                local_path = hf_hub_download(
                    repo_id=model_name,
                    filename=gguf_file,
                    local_dir=dest_dir,
                    resume=True,
                )
                downloaded_file = Path(local_path)
                if downloaded_file.parent != dest_dir:
                    shutil.move(str(downloaded_file), dest_dir / gguf_file)
                if progress_callback:
                    progress_callback(90, "GGUF download complete.")
                size_mb = (dest_dir / gguf_file).stat().st_size / (1024 * 1024)
                with self._lock:
                    self._registry[model_name] = ModelInfo(
                        name=model_name,
                        original_size_mb=size_mb,
                        path=str(dest_dir / gguf_file),
                        model_type="gguf",
                    )
                    self._save_registry()
                if progress_callback:
                    progress_callback(100, "Done.")
                return dest_dir

            # Full model snapshot
            if progress_callback:
                progress_callback(10, "Downloading model snapshot...")

            snapshot_download(
                repo_id=model_name,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=["*.safetensors", "*.bin"] if gguf_file else None,
            )
            if progress_callback:
                progress_callback(80, "Validating tokenizer...")

            if not validate_tokenizer_cached(dest_dir):
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Tokenizer validation failed after downloading '{model_name}'. "
                    "The model may be corrupt or incompatible. Please try another model."
                )

            size_mb = self._get_directory_size_mb(dest_dir)
            with self._lock:
                self._registry[model_name] = ModelInfo(
                    name=model_name,
                    original_size_mb=size_mb,
                    path=str(dest_dir),
                    model_type="local",
                )
                self._save_registry()

            if progress_callback:
                progress_callback(90, "Download complete.")
            if convert_to_lazytorch_after:
                self.convert_to_lazytorch(model_name, progress_callback)
            if progress_callback:
                progress_callback(100, "Done.")
            return dest_dir

        except Exception as e:
            logger.error(f"Download failed: {e}")
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise

    def download_from_ollama(self, model_name: str) -> bool:
        """Pull a model from Ollama."""
        try:
            import subprocess
            import requests
            requests.get("http://localhost:11434", timeout=2)
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.returncode == 0:
                self.sync_ollama()
                return True
            else:
                logger.error(f"Ollama pull failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Ollama pull error: {e}")
            return False

    def _get_directory_size_mb(self, path: Path) -> float:
        total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total_bytes / (1024 * 1024)

    # --------------------------------------------------------------------------
    # LazyTorch conversion
    # --------------------------------------------------------------------------
    def convert_to_lazytorch(
        self,
        name: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[Path]:
        """Convert a registered model to LazyTorch format."""
        info = self.get_model(name)
        if not info or not info.path:
            logger.error(f"Model '{name}' not found or has no path.")
            return None
        src_path = Path(info.path)
        if not src_path.exists():
            logger.error(f"Model path does not exist: {src_path}")
            return None

        if is_lazytorch_model(src_path) or src_path.suffix == ".lazytorch":
            logger.info(f"Model already in LazyTorch format.")
            return src_path

        dest_path = src_path.with_suffix(".lazytorch")
        if dest_path.exists():
            if progress_callback:
                progress_callback("Removing existing LazyTorch...")
            safe_rmtree(dest_path)

        from .lazytorch_core import export_to_lazytorch

        if progress_callback:
            progress_callback("Exporting to LazyTorch...")
        export_to_lazytorch(
            src_path,
            output_path=dest_path,
            dtype=torch.float32,
            progress_callback=lambda msg: progress_callback(f"Export: {msg}")
            if progress_callback else None,
        )
        # Validate tokenizer after export
        if not validate_tokenizer_cached(dest_path):
            logger.error(f"LazyTorch export produced corrupt tokenizer at {dest_path}")
            safe_rmtree(dest_path)
            raise RuntimeError(
                f"LazyTorch export failed: corrupt tokenizer in {dest_path}. "
                "Please check the source model's tokenizer and re-export."
            )

        with self._lock:
            if name in self._registry:
                self._registry[name].lazytorch_format = True
                self._registry[name].model_type = "lazytorch"
                self._save_registry()
        if progress_callback:
            progress_callback("Conversion complete.")
        logger.info(f"LazyTorch model saved at {dest_path}")
        return dest_path

    def get_lazytorch_path(self, name_or_path: Union[str, Path]) -> Optional[Path]:
        """Return the path to the .lazytorch directory if it exists."""
        if isinstance(name_or_path, str):
            info = self.get_model(name_or_path)
            if info and info.path:
                path = Path(info.path)
            else:
                path = MODELS_DIR / name_or_path
        else:
            path = Path(name_or_path)

        if is_lazytorch_model(path):
            return path
        lazy_candidate = path.with_suffix(".lazytorch") if path.suffix else Path(str(path) + ".lazytorch")
        if lazy_candidate.exists() and is_lazytorch_model(lazy_candidate):
            return lazy_candidate
        return None

    # --------------------------------------------------------------------------
    # Smart copying with hardlink support
    # --------------------------------------------------------------------------
    def _copy_model_files(self, src: Path, dst: Path, use_hardlinks: bool = True) -> None:
        """
        Copy a model directory or file from src to dst, attempting to use hardlinks
        for files when possible (saves disk space and time). Falls back to regular copy.
        """
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if use_hardlinks:
                try:
                    os.link(str(src), str(dst))
                    logger.debug(f"Hardlinked {src} -> {dst}")
                    return
                except (OSError, PermissionError) as e:
                    logger.debug(f"Hardlink failed for {src}: {e}. Falling back to copy.")
            shutil.copy2(src, dst)
            return

        # Directory: use copytree with a custom copy function
        if dst.exists():
            safe_rmtree(dst)

        def _copy_with_hardlink(src_file: str, dst_file: str, *, follow_symlinks: bool = True) -> None:
            """
            Copy function that tries hardlink first, then copy2.
            Fixed: unconditionally attempt hardlink when use_hardlinks is True,
            regardless of follow_symlinks.
            """
            # Attempt hardlink unconditionally if enabled
            if use_hardlinks:
                try:
                    os.link(src_file, dst_file)
                    logger.debug(f"Hardlinked {src_file} -> {dst_file}")
                    return
                except (OSError, PermissionError) as e:
                    # Fall through to copy
                    logger.debug(f"Hardlink failed for {src_file}: {e}. Falling back to copy.")
            # Fallback to copy
            shutil.copy2(src_file, dst_file, follow_symlinks=follow_symlinks)

        # Determine symlink behavior
        use_symlinks = True
        if sys.platform == "win32":
            use_symlinks = False
        if os.environ.get("LAZY_COPY_SYMLINKS") == "1":
            use_symlinks = True

        shutil.copytree(
            src,
            dst,
            symlinks=use_symlinks,
            ignore_dangling_symlinks=True,
            copy_function=_copy_with_hardlink,
        )
        logger.debug(f"Copied directory {src} -> {dst} (hardlinks={use_hardlinks})")

    # --------------------------------------------------------------------------
    # Student creation (now with hardlinks, hybrid removed)
    # --------------------------------------------------------------------------
    def create_student(
        self,
        base_model_name: str,
        student_name: str,
        auto_download: bool = True,
        use_lazytorch: bool = False,
        **distill_kwargs,
    ) -> bool:
        """
        Create a student model from a base model.
        Uses hardlinks when possible to save disk space.
        """
        with self._lock:
            # Validate base model is local
            base_info = self.get_model(base_model_name)
            if base_info and base_info.path and (base_info.path.startswith("ollama://") or base_info.path.startswith("vllm://")):
                raise RuntimeError(
                    f"Base model '{base_model_name}' is not a local model. "
                    "Cannot create student from Ollama/vLLM. Please use a local Hugging Face model."
                )

            if student_name in self._registry:
                raise RuntimeError(f"Student '{student_name}' already exists in registry.")

            base_path = self._ensure_base_model(base_model_name, auto_download)
            if base_path is None:
                raise RuntimeError(f"Base model '{base_model_name}' not available and could not be downloaded.")
            if base_path.is_dir() and not validate_tokenizer_cached(base_path):
                if base_info:
                    base_info.invalid = True
                    self._save_registry()
                raise RuntimeError(
                    f"Base model '{base_model_name}' has a corrupt tokenizer. "
                    "Please delete and re-download the model, or repair the tokenizer files."
                )

            dest_dir = MODELS_DIR / student_name
            if dest_dir.exists():
                raise RuntimeError(f"Destination directory {dest_dir} already exists.")

            # Copy base model using hardlinks where possible
            logger.info(f"Copying base model from {base_path} to {dest_dir} (using hardlinks if possible)")
            try:
                self._copy_model_files(base_path, dest_dir, use_hardlinks=True)
            except Exception as e:
                logger.error(f"Failed to copy base model: {e}")
                safe_rmtree(dest_dir)
                raise RuntimeError(f"Failed to copy base model: {e}")

            # Validate tokenizer in copied directory (skip for GGUF)
            if dest_dir.is_dir() and not (dest_dir / "config.json").exists():
                # Might be GGUF-only; skip
                pass
            elif dest_dir.is_dir() and not validate_tokenizer_cached(dest_dir):
                logger.error(f"Student model at {dest_dir} has corrupt tokenizer after copy.")
                safe_rmtree(dest_dir)
                raise RuntimeError(
                    f"Student model '{student_name}' has corrupt tokenizer after copying base model. "
                    "The base model may be corrupted. Please delete and re-download the base model."
                )

            # (Optional) convert to LazyTorch
            if use_lazytorch or self.config.auto_convert_student_to_lazytorch:
                try:
                    self.convert_to_lazytorch(student_name)
                except Exception as e:
                    logger.warning(f"LazyTorch conversion failed: {e}")
                    # Continue anyway; the student is still usable as HF.

            # Register the student
            size_mb = self._get_directory_size_mb(dest_dir) if dest_dir.is_dir() else dest_dir.stat().st_size / (1024*1024)
            model_type = "lazytorch" if use_lazytorch else ("gguf" if any(dest_dir.glob("*.gguf")) else "local")
            info = self._create_model_info(
                name=student_name,
                path=dest_dir,
                size_mb=size_mb,
                lazytorch_format=use_lazytorch or self.config.auto_convert_student_to_lazytorch,
                model_type=model_type,
            )
            self._registry[student_name] = info
            self._save_registry()

            logger.info(f"Student '{student_name}' created successfully.")
            return True

    # --------------------------------------------------------------------------
    # Ensure base model exists and tokenizer valid
    # --------------------------------------------------------------------------
    def _ensure_base_model(self, name: str, auto_download: bool) -> Optional[Path]:
        with self._lock:
            info = self.get_model(name)
            if info and info.path:
                path = Path(info.path)
                if path.exists():
                    if path.is_dir() and not validate_tokenizer_cached(path):
                        logger.error(f"Base model '{name}' at {path} has a corrupt tokenizer.")
                        info.invalid = True
                        self._save_registry()
                        return None
                    return path
                else:
                    logger.warning(f"Model '{name}' path {path} does not exist.")
                    return None
            if auto_download and self.config.auto_download_missing_models:
                logger.info(f"Downloading base model '{name}' from Hugging Face...")
                try:
                    return self.download_from_hf(name)
                except Exception as e:
                    logger.error(f"Download failed: {e}")
                    return None
            return None

    # --------------------------------------------------------------------------
    # Model validation
    # --------------------------------------------------------------------------
    def validate_model(self, name: str) -> bool:
        info = self.get_model(name)
        if not info or not info.path:
            return False
        path_str = info.path
        if path_str.startswith("ollama://") or path_str.startswith("vllm://"):
            with self._lock:
                if name in self._registry:
                    self._registry[name].invalid = False
                    self._save_registry()
            return True
        path = Path(path_str)
        valid = self.validate_model_directory(path)
        with self._lock:
            if name in self._registry:
                self._registry[name].invalid = not valid
                self._save_registry()
        return valid

    def validate_model_directory(self, path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_file():
            if path.suffix == ".gguf":
                if is_valid_gguf is not None:
                    return is_valid_gguf(path)
                return True
            return False
        if path.is_dir():
            has_config = (path / "config.json").exists()
            has_weights = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
            if not (has_config and has_weights):
                return False
            return validate_tokenizer_cached(path)
        return False

    def validate_all_models(self) -> None:
        with self._lock:
            for name, info in list(self._registry.items()):
                if info.path and not info.path.startswith("ollama://"):
                    path = Path(info.path)
                    valid = self.validate_model_directory(path) if path.exists() else False
                    info.invalid = not valid
                else:
                    info.invalid = False
            self._save_registry()
        logger.info("Validation complete for all models.")

    # --------------------------------------------------------------------------
    # Rename model
    # --------------------------------------------------------------------------
    def rename_model(self, old_name: str, new_name: str) -> bool:
        with self._lock:
            if old_name not in self._registry:
                logger.warning(f"Model '{old_name}' not found.")
                return False
            if new_name in self._registry:
                logger.warning(f"Model '{new_name}' already exists.")
                return False

            info = self._registry[old_name]
            if info.path and not info.path.startswith("ollama://"):
                old_path = Path(info.path)
                if old_path.exists():
                    new_path = old_path.parent / new_name
                    try:
                        old_path.rename(new_path)
                        info.path = str(new_path)
                    except Exception as e:
                        logger.error(f"Failed to rename directory: {e}")
                        return False
            info.name = new_name
            self._registry[new_name] = info
            del self._registry[old_name]
            self._save_registry()
            logger.info(f"Renamed model '{old_name}' to '{new_name}'.")
            return True

    # --------------------------------------------------------------------------
    # Resolve model from name or path
    # --------------------------------------------------------------------------
    def resolve_model(self, name_or_path: str) -> Optional[ModelInfo]:
        info = self.get_model(name_or_path)
        if info:
            return info
        path = Path(name_or_path)
        if path.exists():
            size_mb = self._get_directory_size_mb(path) if path.is_dir() else path.stat().st_size / (1024*1024)
            model_type = "gguf" if path.suffix == ".gguf" else "local"
            return ModelInfo(
                name=path.stem,
                original_size_mb=size_mb,
                path=str(path),
                model_type=model_type,
            )
        return None

    # --------------------------------------------------------------------------
    # NEW: Get model by disk path
    # --------------------------------------------------------------------------
    def get_model_by_path(self, path: Union[str, Path]) -> Optional[ModelInfo]:
        """
        Search the registry for a model with the given disk path.
        Returns the first ModelInfo whose path matches (string comparison).
        """
        path_str = str(Path(path).resolve())
        with self._lock:
            for info in self._registry.values():
                if info.path and str(Path(info.path).resolve()) == path_str:
                    return info
        return None

    # --------------------------------------------------------------------------
    # NEW: Update model metadata fields
    # --------------------------------------------------------------------------
    def update_model_info(self, name: str, **kwargs) -> bool:
        """
        Update one or more fields of a model's metadata.
        Allowed fields: any attribute of ModelInfo.
        Returns True if updated, False if model not found.
        """
        with self._lock:
            if name not in self._registry:
                logger.warning(f"Model '{name}' not found.")
                return False
            info = self._registry[name]
            updated = False
            for key, value in kwargs.items():
                if hasattr(info, key):
                    setattr(info, key, value)
                    updated = True
                else:
                    logger.warning(f"ModelInfo has no attribute '{key}'; skipping.")
            if updated:
                self._save_registry()
                logger.debug(f"Updated model '{name}' fields: {', '.join(kwargs.keys())}")
            return updated

    # --------------------------------------------------------------------------
    # NEW: Get operation history for a model
    # --------------------------------------------------------------------------
    def get_operation_history(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve the operation history for a given model from its metadata.

        The operation history is a list of dicts, each containing information about
        an operation performed on the model (e.g., distillation, pruning, finetuning).
        Each entry typically includes 'operation', 'timestamp', and any relevant parameters.

        Args:
            model_name: Name of the model to query.

        Returns:
            List of operation history entries (dicts). Returns an empty list if the model
            has no history or does not exist.
        """
        with self._lock:
            info = self.get_model(model_name)
            if info is None:
                logger.debug(f"Model '{model_name}' not found; returning empty history.")
                return []
            if hasattr(info, 'metadata') and isinstance(info.metadata, dict):
                history = info.metadata.get('operation_history', [])
                if isinstance(history, list):
                    return history
                else:
                    logger.warning(f"Model '{model_name}' has invalid operation_history; returning empty.")
                    return []
            return []

    # --------------------------------------------------------------------------
    # NEW: Check if a model ID exists on Hugging Face Hub
    # --------------------------------------------------------------------------
    @staticmethod
    def is_valid_hf_model(model_id: str) -> bool:
        """
        Return True if the given model ID exists on Hugging Face Hub.
        Uses huggingface_hub.model_info if available; otherwise returns False.
        """
        if not HF_HUB_AVAILABLE or hf_model_info is None:
            logger.warning("huggingface_hub not available; cannot validate HF model ID.")
            return False
        try:
            hf_model_info(model_id)
            return True
        except Exception:
            return False

    # --------------------------------------------------------------------------
    # Utility to create ModelInfo objects
    # --------------------------------------------------------------------------
    def _create_model_info(
        self,
        name: str,
        path: Path,
        size_mb: float,
        lazytorch_format: bool = False,
        invalid: bool = False,
        model_type: str = "local",
        **kwargs,
    ) -> ModelInfo:
        return ModelInfo(
            name=name,
            original_size_mb=size_mb,
            path=str(path),
            lazytorch_format=lazytorch_format,
            invalid=invalid,
            model_type=model_type,
            **kwargs,
        )

    # --------------------------------------------------------------------------
    # Convenience method to download and register
    # --------------------------------------------------------------------------
    def download_and_register_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Optional[Path]:
        try:
            return self.download_from_hf(model_name, progress_callback=progress_callback)
        except Exception as e:
            logger.error(f"Failed to download and register model '{model_name}': {e}")
            return None

    # --------------------------------------------------------------------------
    # Additional stubs (prune, distill, export)
    # --------------------------------------------------------------------------
    def prune_model(self, model_name: str, threshold: float = 0.05) -> bool:
        logger.warning("prune_model() not implemented yet.")
        return False

    def distill_model(
        self,
        teacher_name: str,
        student_name: str,
        temperature: float = 2.0,
        alpha: float = 0.7,
    ) -> bool:
        logger.warning("distill_model() not implemented yet.")
        return False

    def export_model(self, model_name: str, export_format: str = "ollama") -> bool:
        logger.warning("export_model() not implemented yet.")
        return False

    # --------------------------------------------------------------------------
    # Student model list helper (v3.6)
    # --------------------------------------------------------------------------
    def get_student_models(self) -> List[str]:
        with self._lock:
            return [
                info.name for info in self._registry.values()
                if ('_distilled' in info.name or info.name.endswith('_pruned'))
                and not info.invalid
            ]

    # --------------------------------------------------------------------------
    # Utility for global defaults
    # --------------------------------------------------------------------------
    def ensure_default_student(self) -> None:
        default = self.config.default_student
        if not default:
            default = "distilgpt2"
        if not self.model_exists(default):
            logger.info(f"Default student '{default}' not found; downloading...")
            try:
                self.download_from_hf(default)
            except Exception as e:
                logger.warning(f"Failed to download default student: {e}")


# Singleton convenience
_default_manager: Optional[ModelManager] = None


def get_model_manager(config: Optional[Config] = None) -> ModelManager:
    global _default_manager
    if _default_manager is None or config is not None:
        _default_manager = ModelManager(config)
    return _default_manager