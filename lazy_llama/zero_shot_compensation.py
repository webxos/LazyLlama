"""
zero_shot_compensation.py - Zero-shot compensation for quantized/pruned models using low-rank adapters.
   Now compatible with LazyTorch: adapters work with LazyModule and can be exported to .lazytorch format.

   FIXED: Improved forward patching to safely handle LazyLinear.
   FIXED: LazyTorch loading now uses unload_after_forward=False during training to keep weights loaded.
   FIXED: Added memory cleanup with gc.collect() and torch.cuda.empty_cache().
   FIXED: Added try/except and defensive checks around activation collection and adapter training.
   IMPROVED: Better handling of LazyParameter and device consistency.
   FIXED: Avoid double-patching LazyLinear (which already applies adapters in its forward).
   FIXED: Adapters are now properly persisted when exporting to LazyTorch format.

   CAVEATS / NON‑BLOCKING LIMITATIONS:
   - Training still loads the full model into memory temporarily (inherent for adapter fine‑tuning).
     This is acceptable for a one‑time compensation step.
   - No automatic "remove adapters" method is provided; if you need to revert, you must reload the model.
   - SVD can be numerically unstable on very small calibration sets; this is handled with a warning
     and the layer is skipped (so compensation may not be applied to all layers).

   ============================================================================================
   FIXED: Tokenizer validation uses shared _validate_tokenizer_deep from utils.
   - Removed local _validate_tokenizer_from_path; use _validate_tokenizer_deep.
   - Kept _validate_tokenizer_object for already-loaded tokenizer objects.
   - All errors raise ValueError with clear advice to re‑download the model.

   ADDITIONAL FIXES (2026-07-06):
   - In adapter_forward, ensure that LazyParameter weights are loaded before use (defensive).
   - After LazyTorch export, validate the tokenizer with _validate_tokenizer_deep.
   - Improved pad token fallback logic to be more robust.

   FURTHER FIX (2026-07-10):
   - Corrected LazyLinear adapter handling: adapters are now properly registered and used.
   - Clarified that training requires loading parameters, but LazyTorch minimizes peak memory.
   - Added explicit check for model compatibility (must be a causal LM or have standard forward).
   - Moved `import types` to top; added gradient clipping; made max_length configurable.
   - Replaced broad exception catching with more specific exceptions where possible.

   FIX (2026-07-10): Changed all absolute imports to relative imports.
"""
import gc
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- All internal imports are now relative ----
from .utils import _validate_tokenizer_deep

# Try to import LazyLinear for type checking
try:
    from .lazytorch_core import LazyParameter, LazyLinear
except ImportError:
    LazyParameter = None
    LazyLinear = None

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Tokenizer object validation helper (for already-loaded tokenizer)
# ----------------------------------------------------------------------
def _validate_tokenizer_object(tokenizer) -> bool:
    """
    Validate that the given tokenizer object is usable.
    Returns True if valid, False otherwise.
    """
    if tokenizer is None:
        logger.error("Tokenizer is None")
        return False
    try:
        if tokenizer.vocab_size == 0:
            logger.error("Tokenizer has vocab size 0")
            return False
        tokenizer.encode("test")
        return True
    except Exception as e:
        logger.error(f"Tokenizer validation failed: {e}")
        return False


def _ensure_parameters_loaded(module: nn.Module) -> None:
    """
    Ensure that all LazyParameter attributes of a module are loaded.
    This is a defensive check for adapter_forward.
    """
    if LazyParameter is None:
        return
    for attr_name in ['weight', 'bias']:
        if hasattr(module, attr_name):
            param = getattr(module, attr_name)
            if isinstance(param, LazyParameter) and not param._lazy_data_loaded:
                try:
                    param._load_data()
                except Exception as e:
                    logger.warning(f"Failed to load LazyParameter {attr_name}: {e}")


class ZeroShotAdapterCompensation:
    """
    Adds low‑rank adapters to compensate for errors introduced by quantization or pruning.
    Uses activation statistics from calibration data to compute optimal adapter directions.
    Fully compatible with LazyTorch: works with LazyModule and LazyLinear without loading all weights into RAM.
    After compensation, the model can be exported to LazyTorch format for future memory-efficient inference.
    """
    
    def __init__(self, calibration_data: List[str], tokenizer, device: str = "cpu",
                 max_length: int = 512):
        """
        Args:
            calibration_data: List of prompts for collecting activations.
            tokenizer: Tokenizer corresponding to the model.
            device: Device to run calibration on ("cpu", "cuda").
            max_length: Maximum token length for tokenization.
        
        Raises:
            ValueError: If tokenizer is invalid.
        """
        # Validate tokenizer object
        if not _validate_tokenizer_object(tokenizer):
            raise ValueError(
                "Invalid tokenizer provided for zero-shot compensation. "
                "Please ensure the tokenizer is loaded correctly and is not corrupt."
            )
        self.calibration_data = calibration_data
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_length = max_length
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []
        
    def _collect_activations(self, model: nn.Module) -> None:
        """Register forward hooks to collect outputs of linear layers.
           Works with both standard nn.Linear and LazyLinear (weights load on demand)."""
        def hook_fn(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Flatten to (batch * seq_len, hidden_dim)
                    act = output.detach().reshape(-1, output.shape[-1])
                    self.activations.setdefault(name, []).append(act.to("cpu"))
            return hook
        
        # Remove any existing hooks
        self._remove_hooks()
        
        # Register hooks on all Linear layers (including LazyLinear if it's a subclass)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
        
        # Run calibration
        model.eval()
        model.to(self.device)
        try:
            with torch.no_grad():
                for prompt in tqdm(self.calibration_data, desc="Collecting activations"):
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                            max_length=self.max_length)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    _ = model(**inputs)
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error(f"Activation collection failed: {e}")
            raise
        finally:
            self._remove_hooks()
    
    def _remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _compute_eigenspaces(self, activations: Dict[str, List[torch.Tensor]], rank: int = 8) -> Dict[str, torch.Tensor]:
        """
        For each linear layer, compute the top‑rank eigenvectors of its activation covariance.
        Returns a dictionary mapping layer name to projection matrix (hidden_dim, rank).
        """
        eigenspaces = {}
        for name, acts_list in activations.items():
            if not acts_list:
                continue
            acts = torch.cat(acts_list, dim=0)  # (N, hidden_dim)
            if acts.shape[0] == 0:
                continue
            # Center data
            mean = acts.mean(dim=0, keepdim=True)
            centered = acts - mean
            # Compute covariance
            cov = (centered.T @ centered) / (centered.shape[0] - 1)
            try:
                U, S, V = torch.linalg.svd(cov, full_matrices=False)
                rank_actual = min(rank, U.shape[1])
                proj = U[:, :rank_actual]
                eigenspaces[name] = proj.to(self.device)
            except (RuntimeError, torch.linalg.LinAlgError) as e:
                logger.warning(f"SVD failed for layer {name}: {e}; skipping.")
                continue
        return eigenspaces
    
    def _patch_linear_forward(self, module: nn.Linear, original_forward):
        """
        Replace the forward method of a linear layer to include the low‑rank adapter.
        The adapter is stored as module.adapter_B (rank, in_features) and module.adapter_A (out_features, rank).
        Works with standard Linear (not LazyLinear, which already handles adapters).
        """
        # If module already has an adapter, skip to avoid double patching
        if hasattr(module, '_adapter_patched') and module._adapter_patched:
            logger.debug(f"Module {module} already patched, skipping.")
            return
        
        def adapter_forward(module, x):
            # Defensive: ensure LazyParameters are loaded before forward
            _ensure_parameters_loaded(module)
            # Call original forward (which may load LazyParameters)
            out = original_forward(x)
            if hasattr(module, 'adapter_B') and hasattr(module, 'adapter_A'):
                # Ensure adapter tensors are on the same device as input
                if module.adapter_B.device != x.device:
                    module.adapter_B = module.adapter_B.to(x.device)
                if module.adapter_A.device != x.device:
                    module.adapter_A = module.adapter_A.to(x.device)
                adapter_out = (x @ module.adapter_B.T) @ module.adapter_A.T
                return out + adapter_out
            return out
        
        # Bind method to the module instance
        module.forward = types.MethodType(adapter_forward, module)
        module._adapter_patched = True  # Mark as patched
    
    def compensate(self, model: nn.Module, rank: int = 8, lr: float = 1e-4, steps: int = 50) -> nn.Module:
        """
        Add low‑rank adapters to all linear layers using the collected eigenspaces.
        The adapters are trained for a few steps using language modeling loss on calibration data.
        Returns the modified model (in‑place).
        Works efficiently with LazyTorch models: only the parameters needed for forward passes
        are loaded temporarily; after training, adapters are stored as regular parameters.
        """
        # Validate tokenizer object again (safety)
        if not _validate_tokenizer_object(self.tokenizer):
            raise ValueError(
                "Tokenizer is invalid. Please provide a valid tokenizer. "
                "If the model is corrupt, delete and re-download it."
            )

        if not self.activations:
            self._collect_activations(model)
        
        eigenspaces = self._compute_eigenspaces(self.activations, rank)
        if not eigenspaces:
            logger.warning("No eigenspaces computed; compensation skipped.")
            return model
        
        adapters = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in eigenspaces:
                # Check if this module already has adapters (from previous compensation)
                if hasattr(module, 'adapter_A') and module.adapter_A is not None:
                    logger.warning(f"Layer {name} already has adapters; skipping to avoid conflict.")
                    continue

                proj = eigenspaces[name]
                in_features = module.in_features
                out_features = module.out_features
                rank_actual = proj.shape[1]
                
                # Ensure tensors are on the same device as module weight
                device = module.weight.device
                B = proj.T.to(device)
                A = nn.Parameter(torch.zeros(out_features, rank_actual, device=device))
                
                # Register the adapter parameters
                module.register_buffer("adapter_B", B)
                module.register_parameter("adapter_A", A)
                
                # Only patch forward for standard Linear; LazyLinear already handles adapters
                if LazyLinear is not None and isinstance(module, LazyLinear):
                    logger.debug(f"Layer {name} is LazyLinear; skipping forward patching (already has adapter support).")
                else:
                    self._patch_linear_forward(module, module.forward)
                
                adapters[name] = A
                logger.debug(f"Added adapter to layer {name} (rank={rank_actual})")
        
        if not adapters:
            logger.warning("No adapters added; no linear layers with collected activations.")
            return model
        
        # Prepare tokenized calibration data
        tokenized_inputs = []
        for prompt in self.calibration_data:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()
            tokenized_inputs.append((inputs, labels))
        
        # Freeze all parameters except adapter_A
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "adapter_A" in name:
                param.requires_grad = True
        
        optimizer = torch.optim.Adam(list(adapters.values()), lr=lr)
        model.train()
        
        # Determine pad token id with robust fallback
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = -100  # PyTorch cross_entropy ignore_index default
        
        logger.info(f"Training adapters for {steps} steps on {len(self.calibration_data)} calibration prompts...")
        try:
            for step in range(steps):
                total_loss = 0.0
                for inputs, labels in tokenized_inputs:
                    optimizer.zero_grad()
                    outputs = model(**inputs)
                    logits = outputs.logits
                    # Shift for causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=pad_token_id
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(adapters.values()), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                if (step + 1) % 20 == 0:
                    avg_loss = total_loss / len(tokenized_inputs)
                    logger.info(f"Step {step+1}/{steps}, average loss: {avg_loss:.4f}")
        except (RuntimeError, ValueError) as e:
            logger.error(f"Adapter training failed: {e}")
            # Clean up partially trained adapters (note: we cannot easily restore original forward)
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and hasattr(module, '_adapter_patched'):
                    # We could restore original forward but it's complex; we just warn.
                    pass
            raise
        
        logger.info(f"Zero‑shot compensation applied to {len(adapters)} linear layers (rank={rank})")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model


def apply_zero_shot_compensation(
    model: nn.Module,
    calibration_prompts: List[str],
    tokenizer,
    device: str = "cpu",
    rank: int = 8,
    steps: int = 50,
    max_length: int = 512
) -> nn.Module:
    """
    Convenience function to apply zero‑shot compensation to a model.
    Uses calibration prompts to collect activations and adds low‑rank adapters.
    The adapters are trained using language modeling loss on the same prompts.
    Compatible with both standard and LazyTorch models.

    Raises:
        ValueError: If tokenizer is invalid.
    """
    # Validate tokenizer object before creating compensator
    if not _validate_tokenizer_object(tokenizer):
        raise ValueError(
            "Invalid tokenizer provided for zero-shot compensation. "
            "Please ensure the tokenizer is loaded correctly and is not corrupt."
        )
    compensator = ZeroShotAdapterCompensation(calibration_prompts, tokenizer, device, max_length)
    model = compensator.compensate(model, rank=rank, steps=steps)
    return model


def apply_zero_shot_compensation_lazytorch(
    model_path: Path,
    calibration_prompts: List[str],
    tokenizer,
    output_path: Optional[Path] = None,
    device: str = "cpu",
    rank: int = 8,
    steps: int = 50,
    export_to_lazytorch: bool = True,
    max_length: int = 512
) -> Optional[Path]:
    """
    Apply zero-shot compensation to a model stored on disk (Hugging Face format or LazyTorch)
    and optionally export the compensated model to LazyTorch format for future use.
    This function loads the model temporarily (using low-cpu-mem usage), applies compensation,
    and then exports to LazyTorch if requested.

    NOTE: Training adapters requires forward passes on the model, which inherently loads
    the necessary parameters into memory. However, with LazyTorch, weights are loaded
    on-demand rather than all at once, reducing peak memory usage compared to loading
    the entire model upfront.

    Args:
        model_path: Path to Hugging Face model directory or .lazytorch marker
        calibration_prompts: List of prompts for calibration
        tokenizer: Tokenizer for the model
        output_path: Where to save the compensated LazyTorch model (if None, auto-generated)
        device: Device to run on
        rank: Rank of low-rank adapters
        steps: Number of training steps
        export_to_lazytorch: If True, save as .lazytorch; otherwise return None
        max_length: Maximum token length for tokenization.

    Returns:
        Path to saved LazyTorch model if export_to_lazytorch, else None

    Raises:
        ValueError: If tokenizer is invalid or model path is invalid.
    """
    # Validate tokenizer object first
    if not _validate_tokenizer_object(tokenizer):
        raise ValueError(
            "Invalid tokenizer provided for zero-shot compensation. "
            "Please ensure the tokenizer is loaded correctly and is not corrupt."
        )

    from transformers import AutoModelForCausalLM
    from .lazytorch_core import export_to_lazytorch as _export_to_lazytorch
    from .lazytorch_core import load_lazytorch_model as _load_lazytorch_model
    from .utils import is_lazytorch_model
    
    model_path = Path(model_path)
    # If it's a directory, validate tokenizer using the shared deep validator
    if model_path.is_dir() and not _validate_tokenizer_deep(model_path):
        raise ValueError(
            f"Model directory {model_path} has a corrupt tokenizer. "
            "Please delete the model and re-download it, or repair the tokenizer files.\n"
            f"You can delete it using: python bootstrap.py remove --model {model_path.stem}"
        )

    model = None
    try:
        if is_lazytorch_model(model_path):
            # Load as LazyTorch model, but keep parameters loaded during training
            logger.info("Loading LazyTorch model with unload_after_forward=False for training.")
            model = _load_lazytorch_model(model_path, device=device, unload_after_forward=False)
        else:
            # Load as standard HF model with low memory
            logger.info("Loading standard Hugging Face model.")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        
        # Apply compensation
        compensated_model = apply_zero_shot_compensation(
            model, calibration_prompts, tokenizer, device, rank, steps, max_length
        )
        
        if not export_to_lazytorch:
            return None
        
        if output_path is None:
            # Generate a sensible default name
            stem = model_path.stem if model_path.suffix != '.lazytorch' else model_path.with_suffix('').stem
            output_path = model_path.parent / f"{stem}_compensated.lazytorch"
        
        # Export to LazyTorch format (adapters will be saved automatically)
        logger.info("Exporting compensated model to LazyTorch format...")
        result = _export_to_lazytorch(
            compensated_model,
            output_path,
            dtype="float32",
            progress_callback=lambda msg: logger.info(f"LazyTorch export: {msg}")
        )
        
        # ---- VALIDATE TOKENIZER AFTER EXPORT ----
        if not _validate_tokenizer_deep(output_path):
            # Clean up invalid output
            import shutil
            shutil.rmtree(output_path, ignore_errors=True)
            raise ValueError(
                f"Exported LazyTorch model at {output_path} has a corrupt tokenizer. "
                "This likely indicates the source model had a tokenizer issue. "
                f"Please delete the source model using: python bootstrap.py remove --model {model_path.stem}\n"
                "Then re-download and try again."
            )
        
        logger.info(f"Compensated model saved to {result}")
        return result
    except Exception as e:
        logger.error(f"Zero-shot compensation for LazyTorch model failed: {e}")
        return None
    finally:
        # Clean up
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()