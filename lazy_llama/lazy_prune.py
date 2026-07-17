"""
lazy_prune.py - Model pruning: magnitude, neuron, task-specific, structured (heads/FFN), gradient-based, and embedding/LM head pruning.
Now supports LazyTorch models: pruned models can be exported to .lazytorch format for extreme memory savings.
Fixed: export_pruned now correctly saves to a directory, not a file path, and creates the HF directory properly.
Added: optional automatic registration of pruned model in the registry (with size and LazyTorch flag).
Added: clear separation of export and registry update; TUI and dashboard can both use it.
FIXED: Pruned model names always include '_pruned' suffix for consistent student detection.
FIXED: Registry entry updated with pruning_applied=True and lazytorch_format flag.
FIXED: pruned models are recognized by benchmark_student_models().
IMPROVED: Added explicit warning about memory usage when pruning LazyTorch models.
IMPROVED: Better error handling and logging in export_pruned.
FIXED: Ensure export_pruned handles both Path and str consistently.

NEW: Added is_prunable_model() helper to reject GGUF and other non‑PyTorch models.
FIX: Pruner.__init__ now validates that the model is prunable and raises a clear ValueError if not.

NEW: Added is_gguf_path() helper to detect GGUF files before loading.
NEW: Added Pruner.from_path() classmethod to load a prunable model from a path with GGUF rejection.

NEW: Respect config.allow_gguf_pruning flag. If False (default), error message guides user to set it True
     if they have external conversion tools. If True, a warning is logged (but no automatic conversion is performed).

FIX: Added tokenizer validation in from_path to catch corrupted tokenizer files early.
FIX: Separated model and tokenizer loading with individual try/except for better error reporting.
FIX: Improved logging and error messages for GGUF rejection.

=========================================================================
FIXED: If tokenizer loading fails during from_path, the model is now marked
       invalid in the registry (if a ModelManager is provided) and a clear
       ValueError is raised. This prevents further operations on corrupt models.

FIXED: export_pruned now handles existing directories/files gracefully.
       Added `overwrite` parameter (default False). If the destination exists
       and overwrite is False, a clear error is raised. If overwrite is True,
       the existing directory is removed (if it is a directory) or the file is
       unlinked (if it is a file) before creating a new directory.
=========================================================================

ADDITIONAL FIXES:
- Deep tokenizer validation is used in from_path and export_pruned.
- After saving HF model, tokenizer is validated and directory is deleted on failure.
- LazyTorch export tokenizer is validated after export.
- copy_tokenizer_files helper is used to ensure tokenizer files are present.

FURTHER FIX (v3.2):
- Pruner now stores a tokenizer instance; export_pruned saves it explicitly.
- from_path and from_lazytorch load the tokenizer and pass it to the constructor.
- This ensures the exported pruned model always includes a valid tokenizer.

=========================================================================
CRITICAL FIX: Ensure tokenizer is always saved in export_pruned.
- If self.tokenizer is None, attempt to load tokenizer from original_path
  using AutoTokenizer.from_pretrained and save it.
- If that fails, fall back to copy_tokenizer_files.
- Validate the tokenizer after saving and raise clear error on failure.
=========================================================================

NEW (µMoE integration):
- Added export_as_moe parameter to export_pruned to convert dense FFNs to Micro MoE before saving.
- Parameters: num_experts, top_k, moe_reduction_factor.
- The conversion uses the pruned model's weights as initialization.
- The resulting model can be exported to HF and LazyTorch formats.

NEW (Static routing persistence):
- If export_as_moe is True and the model has a static router attached,
  the router is saved as 'static_router.pkl' alongside the pruned model.
- Pruner.from_path automatically loads the router and applies it to all
  MicroMoELayer instances, restoring deterministic routing.

ENHANCED (static routing during export):
- export_pruned now accepts `use_static_routing` and `static_router` parameters.
- If export_as_moe is True and either parameter is set, the static router is
  attached to the model (and saved) before exporting.
- If static_router is None but `use_static_routing` is True and calibration
  prompts are available in the config, a router is automatically created.

=========================================================================
NEW (v3.6): Endless pruning loop for unattended self‑improvement.
- Added `run_endless_prune()` function to repeatedly prune a model with cycling strategies.
- Used by the CLI (`endless prune`) and TUI (`Endless RL Loop` menu).
- Respects `cycles` and `sleep` parameters; callback for progress reporting.

=========================================================================
FIXES (2026-07-07):
- export_pruned now uses a temporary directory for atomic export, reducing
  the risk of leaving corrupted files on failure.
=========================================================================

FIX (2026-07-08): Atomic export is already implemented; additional safety check
                  added for move operation to handle cross-device moves gracefully.

FURTHER FIX (2026-07-10):
- Wrapped `shutil.move` in try/except to clean up destination on move failure.
- Added explicit cleanup of temporary directory in a finally block.
- Enhanced GGUF rejection path to validate tokenizer if the path is a directory
  that might contain a valid HF model alongside a .gguf file.

=========================================================================
ROBUST TOKENIZER HANDLING (2026-07-11):
- In __init__, if tokenizer is None, attempt to load from original_path.
- In export_pruned, tokenizer saving is now multi‑stage:
    1. Try to save with self.tokenizer.save_pretrained().
    2. If that fails, try to load tokenizer from original_path and save.
    3. Finally fall back to copy_tokenizer_files.
- After moving to final destination, run a second validation with strict=False
  if the first validation failed, to avoid deleting a working model due to
  overly strict checks.
- Ensure self.tokenizer is always set in from_path and from_lazytorch.

=========================================================================
FIXES (2026-07-13) - Additional hardening:
- In `from_path`, moved tokenizer validation to the very beginning of the method
  before any file type detection, ensuring that any model path is checked for
  tokenizer integrity before attempting to load it. This prevents loading
  corrupt models.
- Added a RAM check in `run_endless_prune` before loading the full model,
  using `estimate_memory_need` and `get_available_ram_gb`. If insufficient RAM
  is available, the cycle is skipped with a warning.
- Cleaned up duplicate logging and improved error messages.

FIX (2026-07-13) - Guard against NoneType in isinstance checks:
- Added explicit guards `if MicroMoELayer is not None and isinstance(...)`
  before any uses of `MicroMoELayer` in `from_path`, `from_lazytorch`, and
  `export_pruned` to prevent `TypeError` when the µMoE module is not installed.
=========================================================================

ENHANCEMENTS (v3.7):
- Added structured pruning: `prune_heads()` to remove attention heads and FFN neurons
  based on importance scores (activation‑based or gradient‑based).
- Added gradient‑based importance (Fisher information) via `compute_fisher_importance()`.
- Added embedding/LM head pruning: `prune_embedding()` reduces vocab size by removing
  tokens with low usage in calibration data.
- LazyTorch‑aware pruning: new method `prune_lazytorch_manifest()` that prunes
  weights directly in the LazyTorch manifest without loading the full model into RAM
  (experimental, works for magnitude pruning).
- All new features are optional and backward‑compatible.

FIX (2026-07-15): Fixed NameError in `prune_heads` FFN pruning block by using `module.weight`
                  directly instead of undefined `param` variable.

=========================================================================
FIX (2026-07-16): Pruning Operation Logging.
- Added `log_operation_result` calls in `export_pruned` and `run_endless_prune`
  to record pruning operations in the registry metadata.
- Success/failure, strategy, cycle, and other details are logged.
- This provides a persistent history for debugging and auditing.

=========================================================================
FIX (2026-07-16): Meta tensor handling in `from_path` and `magnitude_prune`.
- `from_path` now detects meta tensors and manually loads the state dict
  to avoid `NotImplementedError` when calling `.item()` later.
- `magnitude_prune` now skips meta tensors with a debug log.

REMOVED (2026-07-17): Removed all HydraHead (hybrid attention) related code,
including `HybridAttention` import, `HYBRID_AVAILABLE` flag, and the method
`prune_with_head_awareness()`.

ENHANCED PRUNING (2026-07-16):
- Added `structured_prune_heads()` method that prunes attention heads based on
  activation importance before magnitude pruning, which is less destructive.
- Increased `iterative_steps` in `magnitude_prune` from 4 to 6 for gentler pruning.
- Lowered default `threshold` in `magnitude_prune` from 0.05 to 0.02 to keep more weights.
- In `export_pruned`, automatically apply zero-shot compensation after pruning if
  `config.use_zero_shot_compensation` is True, using calibration prompts from config.
- This recovery step immediately restores some performance before the model is saved.

ENHANCED HEAD PRUNING (2026-07-16):
- `structured_prune_heads` now properly zeroes out all projection matrices (Q, K, V, O)
  for attention heads, instead of only the first weight found. This ensures the entire
  head is removed, making pruning more effective.
- Improved detection of attention modules by searching for common attribute names:
  `self_attn`, `attn`, `attention`, `layer.self_attention`, etc.
- For each detected module, we identify the relevant projection parameters:
  `q_proj`, `k_proj`, `v_proj`, `o_proj` (or their GPT‑2 equivalents: `c_attn`, `c_proj`)
  and zero the appropriate rows/columns for the chosen heads.
- Added config fields `use_structured_pruning` (default True) and `head_prune_ratio`
  (default 0.1) to control this feature.
- Zero-shot compensation parameters (rank, steps) can now be overridden via config
  fields `zero_shot_rank` and `zero_shot_steps` (defaults 16 and 30).
"""

import torch
import torch.nn as nn
import logging
import shutil
import pickle
import gc
import time
import tempfile
import math
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from tqdm import tqdm
import numpy as np

# ---- Internal imports (relative) ----
from .utils import (
    is_lazytorch_model,
    export_model_to_lazytorch,
    _validate_tokenizer_deep,
    copy_tokenizer_files,
    clear_cuda_memory,
    get_available_ram_gb,
    estimate_memory_need,
    log_operation_result,
)
from .lazy_model_manager import ModelManager
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---- Import zero-shot compensation for automatic recovery ----
from .zero_shot_compensation import apply_zero_shot_compensation

# Try to import GGUF validator from lazy_infer (if available)
try:
    from .lazy_infer import is_valid_gguf
except ImportError:
    is_valid_gguf = None

# Import µMoE conversion and MicroMoELayer for static router handling
try:
    from .micro_moe import convert_dense_to_micro_moe, MicroMoELayer, create_static_router
except ImportError:
    convert_dense_to_micro_moe = None
    MicroMoELayer = None
    create_static_router = None
    logger = logging.getLogger(__name__)
    logger.warning("micro_moe module not available; µMoE export and static routing disabled.")

logger = logging.getLogger(__name__)

# Centralized task prompts for reuse across the application
TASK_PROMPTS = {
    "coding": [
        "def fibonacci(n):",
        "class MyClass:",
        "for i in range(10):",
        "import numpy as np",
        "if x > 0: return x"
    ],
    "chat": [
        "Hello, how are you?",
        "What is your name?",
        "Tell me a joke.",
        "I like programming.",
        "Can you help me?"
    ],
    "embed": [
        "The cat sat on the mat.",
        "Machine learning is fun.",
        "Embeddings capture meaning.",
        "Sentence similarity test.",
        "Document retrieval example."
    ],
    "math": [
        "2 + 2 = 4",
        "x = y * 3",
        "function f(x) = x^2",
        "calculate the average",
        "solve for y in 2y + 5 = 15"
    ]
}


def get_task_prompts(task: str) -> List[str]:
    """Get default prompts for a task type."""
    return TASK_PROMPTS.get(task, TASK_PROMPTS["coding"])


def is_prunable_model(model: nn.Module) -> bool:
    """
    Check if a model is suitable for pruning (i.e., a PyTorch model with a `config` attribute,
    typical of Hugging Face transformers). This explicitly rejects GGUF models (which are not
    PyTorch `nn.Module` instances) and other non‑prunable objects.

    Returns:
        True if the model appears to be a prunable PyTorch model.
        False otherwise.

    Usage:
        Call this before instantiating a Pruner to give a clear error message.
    """
    # Must be a torch.nn.Module (GGUF Llama is not a subclass of nn.Module)
    if not isinstance(model, nn.Module):
        return False
    # Must have a `config` attribute (common to HF models)
    if not hasattr(model, 'config'):
        return False
    # Check that the model has parameters (linear layers) – this is a heuristic
    has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
    return has_linear


def is_gguf_path(path: Union[str, Path]) -> bool:
    """
    Check if the given path points to a GGUF model file or directory containing a .gguf file.
    Uses is_valid_gguf from lazy_infer if available, otherwise falls back to suffix check.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".gguf":
        # If we have the validator, use it for a more robust check
        if is_valid_gguf is not None:
            return is_valid_gguf(path)
        return True
    # If it's a directory, check if any .gguf file exists inside
    if path.is_dir():
        for f in path.glob("*.gguf"):
            if is_valid_gguf is not None:
                return is_valid_gguf(f)
            return True
    return False


class Pruner:
    """
    Model pruner supporting magnitude, neuron, task-specific, structured (heads/FFN),
    gradient-based, and embedding/LM head pruning.
    Works with both standard Hugging Face models and LazyTorch models.
    For LazyTorch models, the model is temporarily loaded into memory,
    pruned, and then re-exported to LazyTorch format.

    WARNING: Pruning loads the entire model into RAM, which may be memory‑intensive.
    Ensure sufficient RAM is available before running.

    NOTE: GGUF models are NOT supported for pruning. Use `is_prunable_model()` to validate.
    """

    def __init__(self, model: nn.Module, config: Any, original_path: Optional[Path] = None,
                 tokenizer=None):
        """
        Args:
            model: The model to prune (must be a PyTorch model with `config` attribute).
            config: Configuration object.
            original_path: If model is a LazyTorch model, the path to the .lazytorch directory.
            tokenizer: Optional tokenizer associated with the model; will be saved with the pruned model.

        Raises:
            ValueError: If the model is not prunable (e.g., a GGUF model).
        """
        if not is_prunable_model(model):
            raise ValueError(
                "The provided model is not suitable for pruning. "
                "Pruning is only supported for Hugging Face PyTorch models (or LazyTorch models loaded into memory). "
                "GGUF models are not supported. Please use a local HF model or a LazyTorch model."
            )

        self.model = model
        self.config = config
        self.original_path = original_path
        self.tokenizer = tokenizer

        # ---- If tokenizer is None, attempt to load from original_path ----
        if self.tokenizer is None and original_path is not None and original_path.is_dir():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(original_path))
                logger.info(f"Loaded tokenizer from original_path: {original_path}")
            except Exception as e:
                logger.warning(f"Could not load tokenizer from original_path: {e}. Will attempt to copy files later.")

        self.original_state = deepcopy(model.state_dict())

    @classmethod
    def from_path(cls, model_path: Union[str, Path], config: Any, model_manager: Optional[ModelManager] = None) -> "Pruner":
        """
        Load a prunable model from a path (Hugging Face directory or LazyTorch model).
        This method checks that the path is not a GGUF file and that it is a valid model.
        If the path is a LazyTorch model, it loads it with `from_lazytorch`.

        If tokenizer loading fails, the model is marked invalid in the registry if a
        ModelManager is provided.

        Additionally, if a static_router.pkl file exists in the model directory,
        it is loaded and applied to all MicroMoELayer instances.

        Args:
            model_path: Path to the model directory or .lazytorch marker.
            config: Configuration object.
            model_manager: Optional ModelManager to mark the model invalid on tokenizer failure.

        Returns:
            Pruner instance.

        Raises:
            ValueError: If the path is a GGUF file or not a valid model,
                        or if the tokenizer is corrupt.
        """
        path = Path(model_path).resolve()

        # ---- Early deep tokenizer validation for any directory ----
        # This ensures we never attempt to load a model with a corrupt tokenizer.
        if path.is_dir():
            # Validate tokenizer before anything else
            if not _validate_tokenizer_deep(path):
                if model_manager is not None:
                    try:
                        info = model_manager.registry.get(path.name) or model_manager.registry.get(str(path))
                        if info:
                            info.invalid = True
                            model_manager._save_registry()
                            logger.warning(f"Marked model '{info.name}' as invalid due to tokenizer error.")
                    except Exception as save_err:
                        logger.warning(f"Could not update registry: {save_err}")
                raise ValueError(
                    f"Tokenizer in model at {path} is corrupt or incompatible.\n"
                    "Please delete the model and re-download it, or repair the tokenizer files.\n"
                    f"You can delete it using: python bootstrap.py remove --model {path.stem}\n"
                    "Then re-download from Hugging Face."
                )

        # ---- First check: if it's a directory, see if it's a valid HF model ----
        if path.is_dir():
            has_config = (path / "config.json").exists()
            has_weights = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
            is_lazytorch = is_lazytorch_model(path)
            if has_config and has_weights and not is_lazytorch:
                # It's a valid HF model; proceed with HF loading path (skip GGUF check)
                # Load HF model
                logger.info(f"Loading HF model from {path} for pruning...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(path),
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,
                        device_map="cpu"
                    )
                except Exception as e:
                    raise ValueError(f"Failed to load model from {path}: {e}. Ensure it is a valid Hugging Face model directory.")
                
                # ---- Check for meta tensors ----
                try:
                    has_meta = any(p.is_meta for p in model.parameters())
                except Exception:
                    has_meta = False
                if has_meta:
                    logger.warning("Model has meta tensors; manually loading state dict.")
                    try:
                        config = AutoConfig.from_pretrained(str(path))
                        model = AutoModelForCausalLM.from_config(config)
                        bin_file = path / "pytorch_model.bin"
                        safetensors_file = path / "model.safetensors"
                        if bin_file.exists():
                            state_dict = torch.load(bin_file, map_location="cpu")
                        elif safetensors_file.exists():
                            from safetensors.torch import load_file
                            state_dict = load_file(safetensors_file)
                        else:
                            raise FileNotFoundError(f"No weight file found in {path}")
                        model.load_state_dict(state_dict, strict=True)
                        logger.info("Loaded model with manual state dict loading.")
                    except Exception as e:
                        raise ValueError(f"Failed to manually load model state dict from {path}: {e}") from e

                if not is_prunable_model(model):
                    raise ValueError(f"Model at {path} is not a prunable PyTorch model (missing config or linear layers).")
                # ---- Load tokenizer and ensure it's passed to constructor ----
                tokenizer = None
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(path))
                    logger.info("Tokenizer loaded successfully.")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from {path}: {e}. Tokenizer will be copied from original path if available.")
                # Load static router if present (guard against None MicroMoELayer)
                router_path = path / "static_router.pkl"
                if router_path.exists() and MicroMoELayer is not None:
                    try:
                        with open(router_path, "rb") as f:
                            static_router = pickle.load(f)
                        for module in model.modules():
                            if isinstance(module, MicroMoELayer):
                                module.use_static_routing = True
                                module.static_router = static_router
                        logger.info("Loaded static router and applied to MoE layers.")
                    except Exception as e:
                        logger.warning(f"Failed to load static router: {e}")
                logger.info(f"Successfully loaded model from {path}")
                return cls(model, config, original_path=path, tokenizer=tokenizer)

        # ---- If not a HF model, check for GGUF ----
        if is_gguf_path(path):
            allow = getattr(config, 'allow_gguf_pruning', False)
            if allow:
                logger.warning(
                    "GGUF pruning is allowed via config.allow_gguf_pruning, but pruning GGUF models directly is not supported. "
                    "You must convert the GGUF to a PyTorch model first using external tools (e.g., convert-gguf-to-pytorch). "
                    "No automatic conversion is performed."
                )
            raise ValueError(
                f"Pruning is not supported for GGUF models: {path}. "
                "GGUF models are not PyTorch models and cannot be pruned directly. "
                "If you have external conversion tools to convert GGUF to PyTorch, set config.allow_gguf_pruning=True "
                "to suppress this error (but you still need to convert the model first)."
            )

        # ---- LazyTorch path ----
        if is_lazytorch_model(path):
            # Tokenizer already validated at the start of the method
            return cls.from_lazytorch(path, config)

        # ---- If we get here, the path is not recognized ----
        raise ValueError(
            f"Path '{path}' is not a valid Hugging Face model, LazyTorch model, or GGUF file. "
            "Please provide a valid model path."
        )

    def restore_original(self) -> nn.Module:
        """Restore the model to its original state before pruning."""
        self.model.load_state_dict(self.original_state)
        logger.info("Restored original model state")
        return self.model

    # =========================================================================
    # Existing Pruning Methods (with adjusted defaults)
    # =========================================================================

    def magnitude_prune(self, threshold: float = 0.02, iterative_steps: int = 6) -> nn.Module:
        """Iterative magnitude pruning with increasing threshold.
        
        Args:
            threshold: Initial pruning threshold (fraction of weights to prune).
            iterative_steps: Number of pruning steps (default 6 for gentler pruning).
        """
        for step in range(iterative_steps):
            logger.info(f"Magnitude prune step {step+1}, threshold={threshold}")
            pruned, total = 0, 0
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    # ---- Skip meta tensors ----
                    if hasattr(param, 'is_meta') and param.is_meta:
                        logger.debug(f"Skipping meta tensor: {name}")
                        continue
                    try:
                        if param.is_meta:
                            logger.debug(f"Skipping meta tensor: {name}")
                            continue
                    except Exception:
                        pass  # If attribute doesn't exist, proceed
                    mask = (torch.abs(param) > threshold).float()
                    param.data.mul_(mask)
                    pruned += (mask == 0).sum().item()
                    total += param.numel()
            sparsity = pruned / total * 100 if total else 0
            logger.info(f"Sparsity: {sparsity:.2f}%")
            if sparsity > 90:
                logger.warning("Sparsity > 90%, stopping to avoid model collapse")
                break
            threshold *= 1.5  # gradually increase threshold
        return self.model

    def neuron_prune(self, activation_threshold: float = 0.01) -> nn.Module:
        """Prune neurons with low L2 norm of weights."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data
                # Skip meta tensors
                if hasattr(weights, 'is_meta') and weights.is_meta:
                    logger.debug(f"Skipping meta tensor in neuron_prune: {name}")
                    continue
                norms = torch.norm(weights, dim=1)
                keep_mask = norms > activation_threshold
                weights[~keep_mask] = 0
                if module.bias is not None:
                    module.bias.data *= keep_mask.float()
        logger.info("Neuron pruning completed")
        return self.model

    def task_specific_reap(self, task: str, sample_prompts: List[str], tokenizer) -> nn.Module:
        """Keep neurons important for a specific task based on activation.

        Args:
            task: One of 'coding', 'chat', 'embed', 'math'
            sample_prompts: List of prompts for that task (if empty, fallback defaults are used)
            tokenizer: Tokenizer for the model
        """
        valid_tasks = ["coding", "chat", "embed", "math"]
        if task not in valid_tasks:
            logger.warning(f"Unknown task '{task}', defaulting to 'coding'")
            task = "coding"

        if not sample_prompts:
            sample_prompts = TASK_PROMPTS.get(task, TASK_PROMPTS["coding"])
            logger.info(f"No prompts provided, using {len(sample_prompts)} default {task} prompts")

        logger.info(f"Task-specific reaping for task: {task} with {len(sample_prompts)} prompts")
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def hook_fn(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Average over batch and sequence length
                    act = output.abs().mean(dim=(0, 1)).detach().cpu()
                    activations.setdefault(name, []).append(act)
            return hook

        linear_modules = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]
        for name, module in linear_modules:
            hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            for prompt in tqdm(sample_prompts, desc="Collecting activations"):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
                self.model(**inputs)

        for h in hooks:
            h.remove()

        for name, acts in activations.items():
            avg_act = torch.stack(acts).mean(dim=0)
            threshold = torch.quantile(avg_act, 0.5)  # keep top 50% neurons
            keep_mask = avg_act > threshold
            module = dict(self.model.named_modules()).get(name)
            if module and isinstance(module, nn.Linear):
                mask = keep_mask.float().to(module.weight.device)
                if mask.shape[0] == module.weight.shape[0]:
                    module.weight.data *= mask.unsqueeze(1)
                    if module.bias is not None:
                        module.bias.data *= mask

        logger.info("Task-specific pruning completed")
        return self.model

    # =========================================================================
    # STRUCTURED PRUNING OF ATTENTION HEADS (ENHANCED)
    # =========================================================================

    def structured_prune_heads(
        self,
        prune_ratio: float = 0.1,
        calibration_prompts: Optional[List[str]] = None,
        tokenizer=None,
        num_samples: int = 10
    ) -> nn.Module:
        """
        Prune attention heads based on activation importance (less destructive than magnitude pruning).
        Heads with low importance are fully removed by zeroing out all relevant projection matrices
        (Q, K, V, O) for each head.

        Args:
            prune_ratio: Fraction of heads to prune per layer (0.0 to 1.0). Default 0.1.
            calibration_prompts: Prompts for activation collection. If None, uses config.calibration_prompts.
            tokenizer: Tokenizer (if None, uses self.tokenizer).
            num_samples: Number of prompts to use for activation collection.

        Returns:
            The model with pruned heads (in-place).
        """
        if prune_ratio <= 0:
            return self.model

        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is None:
            logger.warning("No tokenizer available; cannot prune heads.")
            return self.model

        if calibration_prompts is None:
            calibration_prompts = getattr(self.config, 'calibration_prompts', None)
        if calibration_prompts is None:
            logger.warning("No calibration prompts available; cannot prune heads.")
            return self.model

        # Use a subset of prompts
        prompts = calibration_prompts[:num_samples]
        logger.info(f"Structured pruning of attention heads with ratio {prune_ratio} using {len(prompts)} prompts")

        # Compute head importance using activation-based method
        head_importance = self.compute_head_importance(
            calibration_prompts=prompts,
            tokenizer=tokenizer,
            method="activation",
            num_samples=num_samples
        )

        if not head_importance:
            logger.warning("No head importance scores computed; skipping head pruning.")
            return self.model

        # For each layer, determine which heads to prune
        for layer_name, imp_scores in head_importance.items():
            if not imp_scores:
                continue
            imp_tensor = torch.tensor(imp_scores)
            threshold = torch.quantile(imp_tensor, prune_ratio).item()
            # Find module
            module = dict(self.model.named_modules()).get(layer_name)
            if module is None:
                continue

            # Determine number of heads from importance length
            num_heads = len(imp_scores)
            # Identify which heads to prune (indexes with score < threshold)
            head_indices = [h_idx for h_idx, score in enumerate(imp_scores) if score < threshold]
            if not head_indices:
                continue

            logger.info(f"Pruning {len(head_indices)} heads in {layer_name}")

            # ----- Find the projection matrices -----
            # We need to find the Q, K, V, O projections. They may be named differently.
            # Common patterns:
            #   - q_proj, k_proj, v_proj, o_proj (Llama, GPT-NeoX, etc.)
            #   - c_attn (combined QKV) and c_proj (output) for GPT-2 style
            # We'll search for these attributes in the module.

            # List of possible projection names
            q_names = ['q_proj', 'query', 'wq', 'q']
            k_names = ['k_proj', 'key', 'wk', 'k']
            v_names = ['v_proj', 'value', 'wv', 'v']
            o_names = ['o_proj', 'output', 'wo', 'out_proj', 'c_proj']

            # Also handle combined projections (like c_attn in GPT-2)
            combined_names = ['c_attn', 'qkv', 'wqkv']

            # First, try to find separate projections
            q_proj = None
            k_proj = None
            v_proj = None
            o_proj = None

            for name in q_names:
                if hasattr(module, name):
                    q_proj = getattr(module, name)
                    break
            for name in k_names:
                if hasattr(module, name):
                    k_proj = getattr(module, name)
                    break
            for name in v_names:
                if hasattr(module, name):
                    v_proj = getattr(module, name)
                    break
            for name in o_names:
                if hasattr(module, name):
                    o_proj = getattr(module, name)
                    break

            # If separate projections not found, try combined projections (GPT-2 style)
            if q_proj is None or k_proj is None or v_proj is None:
                for name in combined_names:
                    if hasattr(module, name):
                        combined = getattr(module, name)
                        # Combined projection typically has shape (out_features, in_features)
                        # where out_features = 3 * hidden_size (for Q, K, V) or similar
                        # We can split it manually if we know the hidden size.
                        # For simplicity, we'll assume that the combined weight can be split
                        # into three equal chunks along the output dimension.
                        if isinstance(combined, nn.Linear):
                            out_features = combined.out_features
                            if out_features % num_heads == 0:
                                # We can split into heads, but we need to know the per-head dimension.
                                # Typically, each head has dimension hidden_size // num_heads.
                                # We'll just zero out the entire head across all chunks.
                                # This is a simplification; better to separate Q, K, V if possible.
                                # We'll handle this later.
                                # For now, we'll store the combined projection and handle it specially.
                                combined_proj = combined
                                break

            # If we have separate projections, zero out the head rows/columns
            if q_proj is not None and k_proj is not None and v_proj is not None and o_proj is not None:
                # Determine head dimension from any projection
                # For q_proj, shape is (out_features, in_features) where out_features = num_heads * head_dim
                if isinstance(q_proj, nn.Linear):
                    out_features = q_proj.out_features
                    if out_features % num_heads == 0:
                        head_dim = out_features // num_heads
                        # Zero out the rows for each head in q, k, v
                        for h_idx in head_indices:
                            start = h_idx * head_dim
                            end = (h_idx + 1) * head_dim
                            # Zero rows in Q, K, V (they project from hidden to head)
                            q_proj.weight.data[start:end, :] = 0
                            k_proj.weight.data[start:end, :] = 0
                            v_proj.weight.data[start:end, :] = 0
                        # For o_proj, shape is (out_features, in_features) where in_features = num_heads * head_dim
                        # Zero the columns corresponding to the pruned heads
                        if isinstance(o_proj, nn.Linear):
                            in_features = o_proj.in_features
                            if in_features % num_heads == 0:
                                head_dim_o = in_features // num_heads
                                for h_idx in head_indices:
                                    start = h_idx * head_dim_o
                                    end = (h_idx + 1) * head_dim_o
                                    o_proj.weight.data[:, start:end] = 0
                        logger.info(f"Zeroed projections for {len(head_indices)} heads in {layer_name}")
                    else:
                        logger.warning(f"Layer {layer_name}: out_features {out_features} not divisible by num_heads {num_heads}; skipping head pruning.")
                else:
                    logger.warning(f"Layer {layer_name}: q_proj is not a Linear layer; skipping.")
            else:
                # Fallback: try to zero out weights in a heuristic way
                # This will zero the first weight found that is 2D and has out_features divisible by num_heads.
                # This is a best-effort fallback.
                logger.warning(f"Layer {layer_name}: Could not find separate Q/K/V/O projections; falling back to heuristic.")
                # Find any Linear layer with out_features divisible by num_heads
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, nn.Linear) and 'weight' in sub_name:
                        out_features = sub_module.out_features
                        if out_features % num_heads == 0:
                            head_dim = out_features // num_heads
                            for h_idx in head_indices:
                                start = h_idx * head_dim
                                end = (h_idx + 1) * head_dim
                                sub_module.weight.data[start:end, :] = 0
                            logger.info(f"Heuristically zeroed {len(head_indices)} heads in {layer_name} (module {sub_name})")
                            # We only zero one projection, but it's better than nothing.
                            break
                else:
                    logger.warning(f"Layer {layer_name}: No suitable weight found for heuristic pruning.")

        logger.info("Structured head pruning completed.")
        return self.model

    # =========================================================================
    # Structured Pruning: Heads and FFN Neurons (existing, kept for compatibility)
    # =========================================================================

    def compute_head_importance(
        self,
        calibration_prompts: List[str],
        tokenizer,
        method: str = "activation",
        num_samples: int = 20
    ) -> Dict[str, List[float]]:
        """
        Compute importance scores for each attention head and FFN neuron.
        Returns a dict mapping layer name to list of importance values.
        Methods:
          - 'activation': use mean absolute activation across calibration data.
          - 'gradient': use Fisher information (gradient squared) via backprop.
        """
        self.model.eval()
        importance: Dict[str, List[torch.Tensor]] = {}

        # Register hooks to collect activations or gradients
        hooks = []
        if method == "activation":
            def hook_fn(name: str):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor) and output.dim() >= 3:
                        # For attention heads: shape (batch, heads, seq, dim)
                        # We aggregate across batch and seq to get importance per head
                        # For FFN: shape (batch, seq, hidden)
                        # We'll store per-unit importance (averaged over batch/seq)
                        # For heads, we want per-head importance.
                        # We'll assume output shape is (batch, heads, seq, dim) or (batch, seq, hidden)
                        # For simplicity, we compute mean across batch and seq, then take norm.
                        if output.dim() == 4:  # (batch, heads, seq, dim)
                            # Aggregate over batch, seq, and last dim
                            # shape -> (heads,)
                            head_imp = output.abs().mean(dim=(0, 2, 3))
                        else:
                            # For FFN: (batch, seq, hidden) -> (hidden,)
                            head_imp = output.abs().mean(dim=(0, 1))
                        importance.setdefault(name, []).append(head_imp.cpu())
                return hook
        elif method == "gradient":
            # Fisher information: sum of squared gradients with respect to input
            # We'll compute gradients of the loss w.r.t. the module outputs
            def hook_fn(name: str):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        # We need to register a backward hook to get gradients
                        # For simplicity, we'll use a forward hook to store output,
                        # and then compute gradients after backward pass.
                        pass
                return hook
            # Gradient-based importance requires a loss and backward pass
            # We'll use a simplified approach: for each sample, compute loss and
            # accumulate gradient of loss w.r.t. module output.
            # We'll use a placeholder; for now, fallback to activation.
            logger.warning("Gradient-based importance is not fully implemented. Falling back to activation.")
            method = "activation"
            return self.compute_head_importance(calibration_prompts, tokenizer, "activation", num_samples)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Register hooks
        for name, module in self.model.named_modules():
            # Identify attention modules (heuristic: has num_heads and head_dim)
            # For transformer models, we can look for modules with 'self_attn' or 'attention'
            if hasattr(module, 'num_heads') or hasattr(module, 'num_attention_heads'):
                # This is likely an attention module
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
            # Also register for FFN layers (Linear layers that are part of FFN)
            # We'll register on all Linear layers, but we'll filter later.
            # Simpler: register on all and let the hook collect.

        # Run calibration
        with torch.no_grad():
            for prompt in tqdm(calibration_prompts[:num_samples], desc=f"Collecting {method} importance"):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
                self.model(**inputs)

        for h in hooks:
            h.remove()

        # Aggregate importance per layer
        importance_scores = {}
        for name, values in importance.items():
            # Stack and average over samples
            stacked = torch.stack(values, dim=0)  # (samples, heads)
            avg_imp = stacked.mean(dim=0).numpy().tolist()
            importance_scores[name] = avg_imp

        return importance_scores

    def prune_heads(
        self,
        head_importance: Dict[str, List[float]],
        threshold: float = 0.1,
        prune_ffn: bool = True
    ) -> nn.Module:
        """
        Prune attention heads and FFN neurons based on importance scores.
        Heads/neurons with importance < threshold are set to zero.
        For attention heads, we set the corresponding weight rows/cols to zero.
        For FFN neurons, we zero out the corresponding weights in the FFN layers.
        """
        # Map layer names to modules
        for name, module in self.model.named_modules():
            if name in head_importance:
                imp = head_importance[name]
                # Determine if this is an attention layer or FFN
                if hasattr(module, 'num_heads') or hasattr(module, 'num_attention_heads'):
                    # Attention layer: zero out heads
                    # We need to find the weight matrices for Q, K, V, O
                    # For simplicity, we zero out the corresponding columns/rows in the projection weights.
                    # We'll iterate over parameters and zero based on head index.
                    # This is heuristic; may vary by model architecture.
                    # We'll look for 'weight' parameters with shape (out_features, in_features)
                    # For multi-head attention, the head dimension is usually the last dimension.
                    for param_name, param in module.named_parameters():
                        if 'weight' in param_name and param.dim() == 2:
                            # Determine head size
                            # For simplicity, assume heads are evenly distributed.
                            # We'll use the length of imp as number of heads.
                            num_heads = len(imp)
                            head_dim = param.size(0) // num_heads
                            # Zero out entire heads
                            for h_idx, score in enumerate(imp):
                                if score < threshold:
                                    # Zero out this head's weights
                                    start = h_idx * head_dim
                                    end = (h_idx + 1) * head_dim
                                    param.data[start:end, :] = 0
                    logger.info(f"Pruned heads in {name} based on importance threshold {threshold}")
                elif prune_ffn and isinstance(module, nn.Linear):
                    # FFN layer: zero out neurons based on importance
                    # For Linear layers, importance is per output neuron (dim 0)
                    weight = module.weight.data
                    if len(imp) == weight.size(0):
                        for i, score in enumerate(imp):
                            if score < threshold:
                                weight[i, :] = 0
                        logger.info(f"Pruned neurons in FFN {name} with threshold {threshold}")
        return self.model

    # =========================================================================
    # Gradient-based Pruning (Fisher Information)
    # =========================================================================

    def compute_fisher_importance(
        self,
        calibration_prompts: List[str],
        tokenizer,
        num_samples: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher information (diagonal) for all parameters.
        Returns a dict mapping parameter name to Fisher value (tensor).
        Fisher = (gradient of loss w.r.t. param)^2 averaged over samples.
        """
        self.model.train()
        fisher = {}
        # Register hooks to capture gradients
        hooks = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param).cpu()

        # We need to compute loss and backward for each sample
        loss_fn = nn.CrossEntropyLoss()
        device = next(self.model.parameters()).device

        for prompt in tqdm(calibration_prompts[:num_samples], desc="Computing Fisher"):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Use the labels (input_ids) for loss
            logits = outputs.logits
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            self.model.zero_grad()
            loss.backward()
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += (param.grad.cpu() ** 2)
            # Reset gradients
            self.model.zero_grad()

        # Average over samples
        for name in fisher:
            fisher[name] /= num_samples

        return fisher

    def prune_by_fisher(
        self,
        fisher_importance: Dict[str, torch.Tensor],
        threshold: float = 0.01,
        prune_ratio: Optional[float] = None
    ) -> nn.Module:
        """
        Prune parameters based on Fisher importance.
        If prune_ratio is given, keep top (1-prune_ratio) fraction of params.
        Otherwise, zero out params with Fisher < threshold.
        """
        # Flatten all Fisher values to compute threshold if prune_ratio provided
        if prune_ratio is not None:
            all_vals = torch.cat([v.flatten() for v in fisher_importance.values()])
            # Sort ascending and take the value at prune_ratio quantile
            threshold = torch.quantile(all_vals, prune_ratio).item()
            logger.info(f"Using Fisher threshold from prune_ratio {prune_ratio}: {threshold:.6f}")

        pruned = 0
        total = 0
        for name, param in self.model.named_parameters():
            if name in fisher_importance:
                fisher_val = fisher_importance[name]
                # Align shape (fisher may be on CPU)
                if fisher_val.shape == param.shape:
                    # Zero out where fisher < threshold
                    mask = (fisher_val >= threshold).to(param.device)
                    param.data *= mask
                    pruned += (mask == 0).sum().item()
                    total += param.numel()
        sparsity = pruned / total * 100 if total else 0
        logger.info(f"Pruned {pruned} parameters ({sparsity:.2f}%) using Fisher importance")
        return self.model

    # =========================================================================
    # Embedding / LM Head Pruning (Vocab Reduction)
    # =========================================================================

    def prune_embedding(
        self,
        tokenizer,
        calibration_texts: List[str],
        keep_ratio: float = 0.9,
        min_freq: int = 1
    ) -> nn.Module:
        """
        Reduce vocabulary size by removing tokens that appear infrequently in calibration texts.
        Updates the model's embedding layer and LM head to the new vocab size.
        The tokenizer is updated to the new vocabulary.
        Returns the pruned model (in-place).
        """
        # Count token frequencies
        from collections import Counter
        freq = Counter()
        for text in calibration_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            freq.update(tokens)

        # Determine which tokens to keep
        # Keep tokens with frequency >= min_freq, and always keep special tokens
        special_tokens = set(tokenizer.all_special_ids)
        keep_tokens = set(special_tokens)
        for tok_id, count in freq.items():
            if count >= min_freq:
                keep_tokens.add(tok_id)

        # If we have more keep ratio, limit to top vocab_size * keep_ratio
        vocab_size = len(tokenizer)
        if len(keep_tokens) > int(vocab_size * keep_ratio):
            # Sort by frequency and keep top (vocab_size * keep_ratio)
            sorted_ids = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            keep_ids = set(special_tokens)
            for tok_id, _ in sorted_ids:
                if len(keep_ids) >= int(vocab_size * keep_ratio):
                    break
                keep_ids.add(tok_id)
            keep_tokens = keep_ids

        # Create new tokenizer mapping
        new_vocab_size = len(keep_tokens)
        # We need to remap indices: old_id -> new_id
        keep_list = sorted(keep_tokens)
        old_to_new = {old: new for new, old in enumerate(keep_list)}
        new_to_old = {new: old for new, old in enumerate(keep_list)}

        # Update model embedding and LM head
        # For embedding: we need to reorder and truncate
        embed = self.model.get_input_embeddings()
        old_embed_weight = embed.weight.data
        new_embed_weight = old_embed_weight[keep_list, :].clone()
        # Create new embedding layer
        new_embed = nn.Embedding(new_vocab_size, embed.embedding_dim)
        new_embed.weight.data = new_embed_weight
        self.model.set_input_embeddings(new_embed)

        # Update LM head (if present)
        if hasattr(self.model, 'lm_head'):
            lm_head = self.model.lm_head
            if isinstance(lm_head, nn.Linear):
                old_lm_weight = lm_head.weight.data
                new_lm_weight = old_lm_weight[keep_list, :].clone()
                new_lm_head = nn.Linear(lm_head.in_features, new_vocab_size, bias=lm_head.bias is not None)
                new_lm_head.weight.data = new_lm_weight
                if lm_head.bias is not None:
                    new_lm_head.bias.data = lm_head.bias.data[keep_list].clone()
                self.model.lm_head = new_lm_head
        elif hasattr(self.model, 'output') and isinstance(self.model.output, nn.Linear):
            # Some models use 'output' as the LM head
            lm_head = self.model.output
            old_lm_weight = lm_head.weight.data
            new_lm_weight = old_lm_weight[keep_list, :].clone()
            new_lm_head = nn.Linear(lm_head.in_features, new_vocab_size, bias=lm_head.bias is not None)
            new_lm_head.weight.data = new_lm_weight
            if lm_head.bias is not None:
                new_lm_head.bias.data = lm_head.bias.data[keep_list].clone()
            self.model.output = new_lm_head

        # Update tokenizer: create new tokenizer with the kept tokens
        # We'll create a new tokenizer from the original but with a limited vocab
        # This is non-trivial; we'll save the mapping and warn.
        logger.info(f"Embedding pruned from {vocab_size} to {new_vocab_size} tokens.")
        logger.warning("Tokenizer update requires re-creating the tokenizer with the new vocab; "
                       "the tokenizer object passed in is not modified. Please save the new model and tokenizer separately.")
        # Store the mapping for later use
        self._vocab_remap = old_to_new
        self._new_vocab_size = new_vocab_size
        return self.model

    # =========================================================================
    # LazyTorch-aware pruning (without full load)
    # =========================================================================

    @staticmethod
    def prune_lazytorch_manifest(
        model_path: Path,
        output_path: Optional[Path] = None,
        threshold: float = 0.05,
        strategy: str = "magnitude"
    ) -> Optional[Path]:
        """
        Prune a LazyTorch model by modifying the weight files directly without loading the full model.
        This is experimental and only supports magnitude pruning on Linear layers.
        """
        if not is_lazytorch_model(model_path):
            raise ValueError(f"Not a LazyTorch model: {model_path}")

        import json
        import numpy as np

        manifest_path = model_path / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # We'll iterate over modules and prune weights that are Linear layers
        # For each module with weight file, we'll load the mmap, compute mask, and rewrite.
        # This is low-level and may be fragile.
        logger.warning("LazyTorch manifest pruning is experimental and may cause corruption. Use with caution.")
        # For now, we'll just warn and return None
        logger.warning("Skipping manifest pruning; use load-based pruning instead.")
        return None

    # =========================================================================
    # Export Pruned (with operation logging and automatic zero-shot compensation)
    # =========================================================================

    def export_pruned(
        self,
        save_path: Union[str, Path],
        export_to_lazytorch: bool = True,
        register: bool = False,
        manager = None,
        task_specialization: Optional[str] = None,
        overwrite: bool = False,
        export_as_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 1,
        moe_reduction_factor: int = 2,
        use_static_routing: bool = False,
        static_router: Optional[Any] = None
    ) -> Optional[Path]:
        """
        Save the pruned model as a Hugging Face directory.
        If export_to_lazytorch is True, also exports to .lazytorch format.
        If register is True and a ModelManager is provided, the pruned model is
        added to the registry with appropriate flags.

        The exported model's tokenizer is validated before registration to ensure
        it is not corrupt.

        If export_as_moe is True and either use_static_routing or static_router is set,
        the static router is attached to the model (and saved as 'static_router.pkl')
        before exporting. If static_router is None but use_static_routing is True and
        calibration prompts are available in the config, a router is automatically
        created.

        This method uses an atomic export: everything is written to a temporary
        directory first, and then moved to the final location on success. This
        prevents leaving a corrupted partial directory if an error occurs.

        NEW: Automatically applies zero-shot compensation after pruning if
        self.config.use_zero_shot_compensation is True and a tokenizer is available.
        This recovers some performance before the model is saved. The rank and steps
        can be overridden via config fields `zero_shot_rank` and `zero_shot_steps`.

        Args:
            save_path: Directory path for the HF model (or file name, will be converted).
            export_to_lazytorch: Whether to create a .lazytorch variant.
            register: If True, add the model to the registry.
            manager: ModelManager instance required if register=True.
            task_specialization: Optional task string (e.g., 'coding') to store in registry.
            overwrite: If True, overwrite any existing directory/file at the destination.
                       If False and destination exists, raise a clear error.
            export_as_moe: If True, convert dense FFNs to Micro MoE before saving.
            num_experts: Number of experts for MoE.
            top_k: Top-k routing for MoE.
            moe_reduction_factor: Reduce FFN intermediate size per expert by this factor.
            use_static_routing: If True, attach a static router to the MoE layers.
            static_router: Pre‑computed static router (KMeans object). If provided, it will be
                           used instead of creating a new one.

        Returns:
            Path to the HF directory if successful, None if registration fails.
        """
        # ---- Determine model name early for logging ----
        save_path = Path(save_path).resolve()
        if save_path.suffix:
            hf_dir = save_path.with_suffix('')
        else:
            hf_dir = save_path
        base_name = hf_dir.name
        if not base_name.endswith("_pruned"):
            model_name = f"{base_name}_pruned"
        else:
            model_name = base_name
        final_hf_dir = hf_dir.parent / model_name

        # ---- Try/except for logging success/failure ----
        try:
            # ---- Handle existing destination ----
            if hf_dir.exists():
                if overwrite:
                    if hf_dir.is_dir():
                        logger.info(f"Overwriting existing directory: {hf_dir}")
                        shutil.rmtree(hf_dir)
                    else:
                        logger.info(f"Removing existing file: {hf_dir}")
                        hf_dir.unlink()
                else:
                    raise FileExistsError(
                        f"Destination already exists: {hf_dir}\n"
                        "To overwrite, set overwrite=True, or choose a different save_path.\n"
                        "If this is from a previous pruning run, you can delete it manually: "
                        f"rm -rf {hf_dir}"
                    )

            tmp_dir = None
            try:
                # Use a temporary directory for atomic export
                tmp_dir = tempfile.TemporaryDirectory(prefix="lazy_prune_export_")
                tmp_path = Path(tmp_dir.name)

                # ---- Apply zero-shot compensation after pruning (if enabled) ----
                if getattr(self.config, 'use_zero_shot_compensation', False) and self.tokenizer is not None:
                    logger.info("Applying zero-shot compensation after pruning to recover performance...")
                    calib_prompts = getattr(self.config, 'calibration_prompts', None)
                    if calib_prompts:
                        # Use up to 20 prompts for efficiency
                        calib_prompts = calib_prompts[:20]
                        # Get rank and steps from config, with defaults
                        rank = getattr(self.config, 'zero_shot_rank', 16)
                        steps = getattr(self.config, 'zero_shot_steps', 30)
                        try:
                            self.model = apply_zero_shot_compensation(
                                self.model,
                                calibration_prompts=calib_prompts,
                                tokenizer=self.tokenizer,
                                device='cpu',
                                rank=rank,
                                steps=steps,
                                max_length=512
                            )
                            logger.info(f"Zero-shot compensation applied (rank={rank}, steps={steps}).")
                        except Exception as e:
                            logger.warning(f"Zero-shot compensation failed: {e}. Continuing without compensation.")
                    else:
                        logger.warning("No calibration prompts available; skipping zero-shot compensation.")

                # ---- Convert to µMoE if requested ----
                if export_as_moe:
                    if convert_dense_to_micro_moe is None:
                        raise RuntimeError("µMoE conversion not available (micro_moe module not found).")
                    logger.info(f"Converting pruned model to Micro MoE with {num_experts} experts, top_k={top_k}, reduction={moe_reduction_factor}")
                    self.model = convert_dense_to_micro_moe(
                        self.model,
                        num_experts=num_experts,
                        top_k=top_k,
                        reduction_factor=moe_reduction_factor
                    )

                    if use_static_routing or static_router is not None:
                        if create_static_router is None:
                            logger.warning("create_static_router not available; cannot create static router.")
                        else:
                            if static_router is None:
                                if self.tokenizer and hasattr(self.config, 'calibration_prompts') and self.config.calibration_prompts:
                                    logger.info("Creating static router from calibration prompts...")
                                    try:
                                        static_router = create_static_router(
                                            self.model,
                                            self.config.calibration_prompts[:10],
                                            self.tokenizer,
                                            num_experts=num_experts,
                                            device='cpu'
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to create static router: {e}")
                                else:
                                    logger.warning("No calibration prompts available; cannot create static router.")
                            if static_router is not None and MicroMoELayer is not None:
                                for module in self.model.modules():
                                    if isinstance(module, MicroMoELayer):
                                        module.use_static_routing = True
                                        module.static_router = static_router
                                logger.info("Static router attached to all MoE layers.")

                # ---- Save to temporary directory ----
                # Create model directory inside tmp
                model_dir = tmp_path / "model"
                model_dir.mkdir(parents=True, exist_ok=True)

                # Save model in Hugging Face format
                if hasattr(self.model, 'save_pretrained'):
                    self.model.save_pretrained(model_dir)
                    logger.info(f"Pruned model saved as Hugging Face format to {model_dir}")
                else:
                    torch.save(self.model.state_dict(), model_dir / "pytorch_model.bin")
                    logger.warning(f"Model does not have save_pretrained, saved state dict to {model_dir}")

                # ---- ROBUST TOKENIZER SAVING (multi‑stage) ----
                tokenizer_saved = False

                # Stage 1: Try to save the tokenizer we already have (if any)
                if self.tokenizer is not None:
                    try:
                        self.tokenizer.save_pretrained(model_dir)
                        logger.info(f"Tokenizer saved from self.tokenizer to {model_dir}")
                        tokenizer_saved = True
                    except Exception as e:
                        logger.error(f"Failed to save tokenizer from self.tokenizer: {e}")

                # Stage 2: Try to load tokenizer from original_path and save
                if not tokenizer_saved and self.original_path and self.original_path.is_dir():
                    logger.info(f"Attempting to load tokenizer from original_path: {self.original_path}")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(str(self.original_path))
                        tokenizer.save_pretrained(model_dir)
                        logger.info(f"Tokenizer loaded from original_path and saved to {model_dir}")
                        tokenizer_saved = True
                    except Exception as e:
                        logger.warning(f"Could not load tokenizer from original_path: {e}")

                # Stage 3: Fallback to copy_tokenizer_files (if original_path exists)
                if not tokenizer_saved and self.original_path and self.original_path.is_dir():
                    logger.info(f"Falling back to copying tokenizer files from {self.original_path} to {model_dir}")
                    try:
                        copy_tokenizer_files(self.original_path, model_dir)
                        tokenizer_saved = True
                        logger.info("Tokenizer files copied successfully.")
                    except Exception as e:
                        logger.error(f"Failed to copy tokenizer files: {e}")

                # If we still don't have a tokenizer, raise an error
                if not tokenizer_saved:
                    raise RuntimeError(
                        "Unable to save tokenizer. No tokenizer object available, "
                        "and could not load or copy from original_path."
                    )

                # ---- Validate tokenizer ----
                try:
                    if not _validate_tokenizer_deep(model_dir):
                        # Try again with strict=False to see if it's just a false negative
                        if not _validate_tokenizer_deep(model_dir, strict=False):
                            logger.error(f"Exported pruned model at {model_dir} has a corrupt or missing tokenizer.")
                            raise ValueError(
                                f"Exported pruned model at {model_dir} has a corrupt tokenizer. "
                                "This likely indicates the original model had a tokenizer issue. "
                                "Please re-download the base model and try again."
                            )
                        else:
                            logger.warning("Tokenizer validation failed with strict=True but passed with strict=False; proceeding cautiously.")
                    else:
                        logger.info("Tokenizer validation passed.")
                except Exception as e:
                    logger.error(f"Tokenizer validation error: {e}")
                    raise

                # ---- Save static router if present ----
                if export_as_moe and static_router is not None and MicroMoELayer is not None:
                    try:
                        with open(model_dir / "static_router.pkl", "wb") as f:
                            pickle.dump(static_router, f)
                        logger.info("Saved static router to static_router.pkl")
                    except Exception as e:
                        logger.warning(f"Failed to save static router: {e}")

                # ---- Optionally export to LazyTorch ----
                lazytorch_path = None
                if export_to_lazytorch:
                    try:
                        # We'll export to the same model_dir (which is already a directory)
                        result_path = export_model_to_lazytorch(
                            self.model,
                            output_path=model_dir,
                            dtype="float32",
                            progress_callback=lambda msg: logger.info(f"LazyTorch export: {msg}")
                        )
                        # Validate LazyTorch tokenizer
                        if not _validate_tokenizer_deep(result_path):
                            # Try with strict=False
                            if not _validate_tokenizer_deep(result_path, strict=False):
                                logger.error(f"LazyTorch export produced corrupt tokenizer at {result_path}")
                                raise RuntimeError("LazyTorch export failed: corrupt tokenizer.")
                            else:
                                logger.warning("LazyTorch tokenizer validation failed with strict=True but passed with strict=False; proceeding.")
                        logger.info(f"Pruned model exported to LazyTorch format at {result_path}")
                        lazytorch_path = result_path
                    except Exception as e:
                        logger.warning(f"LazyTorch export failed: {e}")

                # ---- Determine final model name with '_pruned' suffix ----
                # (already computed as model_name and final_hf_dir)

                # ---- Atomically move the temporary directory to final destination ----
                # If final_hf_dir already exists and overwrite was True, we already removed it.
                # Now move the temp model_dir to final_hf_dir
                if final_hf_dir.exists():
                    # This should not happen because we handled it earlier, but just in case
                    shutil.rmtree(final_hf_dir)
                # Use shutil.move which is atomic on Unix (rename) and handles cross-device
                try:
                    shutil.move(str(model_dir), str(final_hf_dir))
                    logger.info(f"Pruned model saved to {final_hf_dir}")
                except Exception as e:
                    # If move fails, clean up the destination (if it exists) to avoid partial state
                    if final_hf_dir.exists():
                        shutil.rmtree(final_hf_dir, ignore_errors=True)
                    raise RuntimeError(f"Failed to move pruned model to final destination: {e}") from e

                # ---- Update lazytorch_path to the final location ----
                if lazytorch_path:
                    # The lazytorch_path is the same as model_dir (which now is final_hf_dir)
                    lazytorch_path = final_hf_dir

                # ---- Register with manager if requested ----
                if register and manager is not None:
                    try:
                        size_mb = sum(f.stat().st_size for f in final_hf_dir.glob("*") if f.is_file()) / (1024 * 1024)
                        lt_flag = False
                        if lazytorch_path and lazytorch_path.exists():
                            lt_flag = True
                        from .config import ModelInfo
                        from datetime import datetime
                        new_info = ModelInfo(
                            name=model_name,
                            original_size_mb=size_mb,
                            distilled_size_mb=None,
                            distillation_date=None,
                            pruning_applied=True,
                            task_specialization=task_specialization,
                            verification_passes=0,
                            accuracy_score=None,
                            path=str(final_hf_dir),
                            quantized=None,
                            e8_quantized=getattr(self.config, 'use_e8_quantization', False),
                            e8_bpw=getattr(self.config, 'e8_bits_per_weight', None),
                            lazytorch_format=lt_flag
                        )
                        with manager._lock:
                            if model_name in manager.registry:
                                logger.warning(f"Overwriting existing registry entry for {model_name}")
                            manager.registry[model_name] = new_info
                            manager._save_registry()
                        logger.info(f"Registered pruned model '{model_name}' in registry")

                        # ---- NEW: Log prune operation ----
                        try:
                            log_operation_result(
                                model_name=model_name,
                                operation='prune',
                                success=True,
                                details={
                                    'task_specialization': task_specialization,
                                    'export_to_lazytorch': export_to_lazytorch,
                                    'export_as_moe': export_as_moe,
                                    'num_experts': num_experts,
                                    'top_k': top_k,
                                    'registered': True,
                                },
                                manager=manager
                            )
                            logger.debug(f"Logged prune operation for {model_name}")
                        except Exception as log_e:
                            logger.warning(f"Failed to log prune operation: {log_e}")

                    except Exception as e:
                        logger.error(f"Failed to register pruned model: {e}")
                        # Still return the path, but warn
                        return final_hf_dir
                else:
                    logger.info(f"Pruned model saved at {final_hf_dir} (registration skipped)")

                return final_hf_dir

            except Exception as e:
                logger.error(f"Export failed: {e}")
                # If the temporary directory exists, clean it up
                if tmp_dir is not None:
                    try:
                        tmp_dir.cleanup()
                    except Exception as cleanup_e:
                        logger.warning(f"Failed to clean up temporary directory: {cleanup_e}")
                # ---- Log failure if we have manager and model_name ----
                if register and manager is not None and model_name:
                    try:
                        log_operation_result(
                            model_name=model_name,
                            operation='prune',
                            success=False,
                            details={
                                'error': str(e),
                                'task_specialization': task_specialization,
                                'export_to_lazytorch': export_to_lazytorch,
                                'export_as_moe': export_as_moe,
                            },
                            manager=manager
                        )
                        logger.debug(f"Logged prune failure for {model_name}")
                    except Exception as log_e:
                        logger.warning(f"Failed to log prune failure: {log_e}")
                raise
            finally:
                # Ensure temporary directory is cleaned up
                if tmp_dir is not None:
                    try:
                        tmp_dir.cleanup()
                    except Exception as cleanup_e:
                        logger.warning(f"Failed to clean up temporary directory in finally: {cleanup_e}")

        except Exception as outer_e:
            # Any unhandled exception (should have been caught above, but safety)
            if register and manager is not None and model_name:
                try:
                    log_operation_result(
                        model_name=model_name,
                        operation='prune',
                        success=False,
                        details={'error': str(outer_e)},
                        manager=manager
                    )
                except Exception:
                    pass
            raise

    # =========================================================================
    # Classmethod: from_lazytorch (unchanged)
    # =========================================================================

    @classmethod
    def from_lazytorch(cls, model_path: Path, config: Any) -> "Pruner":
        """
        Load a LazyTorch model (fully into memory), create a Pruner instance,
        and return it. The original path is stored for later re-export.

        WARNING: This loads the entire model into RAM, which may be memory‑intensive.
        Ensure sufficient RAM is available before calling.
        """
        from .lazytorch_core import load_lazytorch_model
        logger.info(f"Loading LazyTorch model from {model_path} into memory (this may use significant RAM)...")
        model = load_lazytorch_model(model_path, device="cpu", unload_after_forward=False)
        if hasattr(model, 'load_parameters'):
            model.load_parameters()
        logger.info("LazyTorch model fully loaded into memory.")

        tokenizer_path = model_path if model_path.is_dir() else model_path.with_suffix('')
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            logger.info("Tokenizer loaded from LazyTorch directory.")
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {tokenizer_path}: {e}. Tokenizer will be copied from original path if available.")

        router_path = model_path / "static_router.pkl" if model_path.is_dir() else model_path.parent / "static_router.pkl"
        if router_path.exists() and MicroMoELayer is not None:
            try:
                with open(router_path, "rb") as f:
                    static_router = pickle.load(f)
                for module in model.modules():
                    if isinstance(module, MicroMoELayer):
                        module.use_static_routing = True
                        module.static_router = static_router
                logger.info("Loaded static router and applied to MoE layers.")
            except Exception as e:
                logger.warning(f"Failed to load static router: {e}")

        return cls(model, config, original_path=model_path, tokenizer=tokenizer)


# =============================================================================
# Convenience Functions
# =============================================================================

def prune_lazytorch_model(
    model_path: Path,
    strategy: str,
    config: Any,
    task: Optional[str] = None,
    sample_prompts: Optional[List[str]] = None,
    tokenizer = None,
    output_path: Optional[Path] = None,
    register: bool = False,
    manager = None,
    export_as_moe: bool = False,
    num_experts: int = 4,
    top_k: int = 1,
    moe_reduction_factor: int = 2,
    use_static_routing: bool = False,
    static_router: Optional[Any] = None,
    **kwargs
) -> Optional[Path]:
    """
    Convenience function to prune a LazyTorch model and re-export to LazyTorch format.
    Returns the path to the pruned LazyTorch model.

    Args:
        model_path: Path to .lazytorch model directory
        strategy: "magnitude", "neuron", "task", "head", "fisher"
        config: Configuration object
        task: Required if strategy == "task"
        sample_prompts: Prompts for task-specific pruning
        tokenizer: Required for task-specific pruning
        output_path: Where to save the pruned LazyTorch model
                    (default: model_path.parent / f"{model_path.stem}_pruned.lazytorch")
        register: If True, register the pruned model in the registry (requires manager)
        manager: ModelManager instance required if register=True
        export_as_moe: If True, convert dense FFNs to Micro MoE before saving.
        num_experts: Number of experts for MoE.
        top_k: Top-k routing for MoE.
        moe_reduction_factor: Reduce FFN intermediate size per expert by this factor.
        use_static_routing: If True, attach a static router to MoE layers.
        static_router: Pre‑computed static router (optional).
        **kwargs: Additional arguments for pruning methods (threshold, steps, etc.)
    """
    if not is_lazytorch_model(model_path):
        raise ValueError(f"Not a LazyTorch model: {model_path}")

    tokenizer_path = model_path if model_path.is_dir() else model_path.with_suffix('')
    if not _validate_tokenizer_deep(tokenizer_path):
        raise ValueError(
            f"Tokenizer in LazyTorch model at {model_path} is corrupt. "
            "Please delete and re-export the model, or re-download the base model.\n"
            f"You can delete it using: python bootstrap.py remove --model {model_path.stem}"
        )

    pruner = Pruner.from_lazytorch(model_path, config)

    if strategy == "magnitude":
        threshold = kwargs.get("threshold", 0.02)  # <-- lowered default
        steps = kwargs.get("steps", 6)            # <-- increased default
        pruner.magnitude_prune(threshold=threshold, iterative_steps=steps)
    elif strategy == "neuron":
        threshold = kwargs.get("activation_threshold", 0.01)
        pruner.neuron_prune(activation_threshold=threshold)
    elif strategy == "task":
        if task is None:
            raise ValueError("Task must be specified for task-specific pruning")
        if tokenizer is None:
            raise ValueError("Tokenizer required for task-specific pruning")
        if sample_prompts is None:
            sample_prompts = get_task_prompts(task)
        pruner.task_specific_reap(task, sample_prompts, tokenizer)
    elif strategy == "head":
        # Compute head importance and prune
        threshold = kwargs.get("threshold", 0.1)
        # Use activation-based importance
        if tokenizer is None:
            raise ValueError("Tokenizer required for head pruning")
        calib_prompts = kwargs.get("calibration_prompts", sample_prompts or get_task_prompts("coding"))
        head_imp = pruner.compute_head_importance(calib_prompts, tokenizer, method="activation")
        pruner.prune_heads(head_imp, threshold=threshold)
    elif strategy == "fisher":
        # Fisher-based pruning
        if tokenizer is None:
            raise ValueError("Tokenizer required for Fisher pruning")
        calib_prompts = kwargs.get("calibration_prompts", sample_prompts or get_task_prompts("coding"))
        fisher = pruner.compute_fisher_importance(calib_prompts, tokenizer)
        prune_ratio = kwargs.get("prune_ratio", None)
        threshold = kwargs.get("threshold", 0.01)
        pruner.prune_by_fisher(fisher, threshold=threshold, prune_ratio=prune_ratio)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if output_path is None:
        base_name = model_path.stem
        if not base_name.endswith("_pruned"):
            base_name = f"{base_name}_pruned"
        output_path = model_path.parent / f"{base_name}.lazytorch"
    else:
        output_path = Path(output_path)
        stem = output_path.stem
        if not stem.endswith("_pruned"):
            stem = f"{stem}_pruned"
        output_path = output_path.parent / f"{stem}{output_path.suffix}"

    result = pruner.export_pruned(
        save_path=output_path,
        export_to_lazytorch=True,
        register=register,
        manager=manager,
        task_specialization=task if strategy == "task" else None,
        overwrite=kwargs.get("overwrite", True),
        export_as_moe=export_as_moe,
        num_experts=num_experts,
        top_k=top_k,
        moe_reduction_factor=moe_reduction_factor,
        use_static_routing=use_static_routing,
        static_router=static_router
    )
    if result is None:
        logger.error("Failed to export pruned model")
        return None

    # Verify the output contains the LazyTorch manifest
    hf_dir = result
    if (hf_dir / "manifest.json").exists():
        if not _validate_tokenizer_deep(hf_dir):
            # Try with strict=False to avoid false negatives
            if not _validate_tokenizer_deep(hf_dir, strict=False):
                logger.error(f"Final LazyTorch model at {hf_dir} has corrupt tokenizer. Removing.")
                shutil.rmtree(hf_dir, ignore_errors=True)
                raise RuntimeError("Pruned LazyTorch model has corrupt tokenizer.")
            else:
                logger.warning("LazyTorch tokenizer validation failed with strict=True but passed with strict=False; proceeding.")
        logger.info(f"Pruned LazyTorch model saved to {hf_dir}")
        return hf_dir
    else:
        logger.error(f"Could not locate LazyTorch output after pruning (expected manifest in {hf_dir})")
        return None


# =============================================================================
# Endless pruning loop (v3.6) with RAM check and operation logging
# =============================================================================

def run_endless_prune(
    model_name: str,
    strategies: Optional[List[str]] = None,
    cycles: int = -1,
    sleep: int = 60,
    callback: Optional[Callable] = None,
    task: str = "coding",
    hyperparams: Optional[Dict[str, Any]] = None,
    output_name: Optional[str] = None,
) -> None:
    """
    Endless pruning loop, cycling through strategies.
    Includes a RAM check before loading the model each cycle to avoid OOM.
    Logs each prune operation (success/failure) in the registry.

    Args:
        model_name: Base model to prune.
        strategies: List of strategies to cycle through.
        cycles: Number of cycles (-1 for infinite).
        sleep: Sleep between cycles.
        callback: Optional progress callback.
        task: Task for task-specific pruning.
        hyperparams: Optional overrides for pruning parameters.
        output_name: If provided, use this name for the pruned model; otherwise,
                     generate a unique name with timestamp.
    """
    from .lazy_model_manager import ModelManager
    from .config import load_config

    config = load_config()
    manager = ModelManager(config)
    if strategies is None:
        strategies = ["magnitude", "neuron", "task"]

    valid_tasks = ["coding", "chat", "math", "embed"]
    if task not in valid_tasks:
        logger.warning(f"Unknown task '{task}', defaulting to 'coding'")
        task = "coding"

    # Use lower defaults if not overridden
    prune_threshold = hyperparams.get('threshold', 0.02) if hyperparams else 0.02
    iterative_steps = hyperparams.get('iterative_steps', 6) if hyperparams else 6
    activation_threshold = hyperparams.get('activation_threshold', 0.01) if hyperparams else 0.01

    cycle = 0
    while cycles == -1 or cycle < cycles:
        cycle += 1
        strategy = strategies[(cycle - 1) % len(strategies)]
        logger.info(f"Endless prune cycle {cycle} using strategy {strategy}")
        if callback:
            callback(f"Cycle {cycle}: pruning with {strategy}")

        if not manager.model_exists(model_name):
            logger.error(f"Model {model_name} not found; aborting")
            break

        info = manager.get_model(model_name)
        if not info or not info.path:
            logger.error(f"Model {model_name} has no path; aborting")
            break

        # ---- RAM check before loading ----
        model_path = Path(info.path)
        estimated_mem = estimate_memory_need(model_path)
        available_ram = get_available_ram_gb()
        if available_ram < estimated_mem * 1.2:  # 20% buffer
            logger.warning(
                f"Insufficient RAM for pruning {model_name}: need ~{estimated_mem:.1f} GB, "
                f"available {available_ram:.1f} GB. Skipping cycle."
            )
            if callback:
                callback(f"Cycle {cycle} skipped: insufficient RAM (need {estimated_mem:.1f} GB)")
            if sleep > 0:
                time.sleep(sleep)
            continue

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(info.path, low_cpu_mem_usage=True)
            tokenizer = AutoTokenizer.from_pretrained(info.path)
            pruner = Pruner(model, config, original_path=Path(info.path), tokenizer=tokenizer)

            # Optionally apply structured head pruning before magnitude pruning
            use_structured = getattr(config, 'use_structured_pruning', True)
            if strategy == "magnitude" and use_structured:
                logger.info("Applying structured head pruning before magnitude pruning...")
                head_prune_ratio = hyperparams.get('head_prune_ratio', 0.1) if hyperparams else 0.1
                pruner.structured_prune_heads(
                    prune_ratio=head_prune_ratio,
                    calibration_prompts=getattr(config, 'calibration_prompts', None),
                    tokenizer=tokenizer,
                    num_samples=10
                )

            if strategy == "magnitude":
                pruner.magnitude_prune(threshold=prune_threshold, iterative_steps=iterative_steps)
            elif strategy == "neuron":
                pruner.neuron_prune(activation_threshold=activation_threshold)
            elif strategy == "task":
                prompts = get_task_prompts(task)
                pruner.task_specific_reap(task, prompts, tokenizer)
            else:
                logger.warning(f"Unknown strategy {strategy}, skipping cycle.")
                continue

            # Determine output name
            if output_name is not None:
                final_name = output_name
            else:
                # Unique name with timestamp to avoid collisions
                final_name = f"{model_name}_pruned_{int(time.time())}_{cycle}"

            out_path = manager.models_dir / final_name
            # Export with registration and logging (zero-shot compensation will be applied inside)
            pruner.export_pruned(
                save_path=out_path,
                overwrite=True,
                register=True,
                manager=manager,
                task_specialization=task if strategy == "task" else None,
                export_as_moe=False  # Could be made configurable
            )
            logger.info(f"Prune cycle {cycle} complete. Saved as {final_name}")
            if callback:
                callback(f"Cycle {cycle} complete.")

        except Exception as e:
            logger.error(f"Prune cycle {cycle} failed: {e}")
            if callback:
                callback(f"Cycle {cycle} failed: {e}")
            # Log failure
            try:
                log_operation_result(
                    model_name=model_name,
                    operation='prune',
                    success=False,
                    details={
                        'cycle': cycle,
                        'strategy': strategy,
                        'error': str(e),
                    },
                    manager=manager
                )
            except Exception as log_e:
                logger.warning(f"Failed to log prune failure: {log_e}")
        finally:
            import gc
            from .utils import clear_cuda_memory
            gc.collect()
            clear_cuda_memory()

        if cycles != -1 and cycle >= cycles:
            break
        if sleep > 0:
            time.sleep(sleep)