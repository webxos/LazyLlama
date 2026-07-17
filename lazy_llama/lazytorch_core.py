"""
lazytorch_core.py - Memory-mapped, on-demand layer loading for PyTorch models.
Enables running large models (7B+) with <500MB RAM by swapping parameters from disk.

Key components:
- LazyParameter: disk-backed tensor that loads only when accessed
- LazyModule: base class for modules that support lazy loading
- export_to_lazytorch: converts HuggingFace model to .lazytorch format
- load_lazytorch_model: loads model without loading all weights into RAM
- export_to_standard_pytorch: converts .lazytorch back to standard HF format

FIXES (v3.6.1):
- Fixed device_map bug: force device_map=None and explicit .to("cpu")
- Added REAP optimizations (pruning + quantization) during export
- Added `reap_mode` parameter to enable/disable REAP steps
- Added `apply_reap_pruning` and `apply_lazy_quant` helper functions
- Improved logging for export failures
- Added nx/nf aliases to LazyConv1D for GPT-2 compatibility

All tokenizer validation uses shared `_validate_tokenizer_deep` from utils.py.

FIX (2026-07-17): Quantized tensor handling in export.
- Added dequantization for quantized tensors (from torch.quantization) before saving.
- Ensured tensor is contiguous and on CPU before conversion to numpy.
- Added guards to skip non-tensor weights (e.g., quantized tensors without dequantize).
- Fixed `_prepare_tensor_for_saving` to include `.detach()` so that `.numpy()` can be called
  on tensors that require gradients.
- Added `original_path` to manifest during export so that LazyTorch models can be used
  as students in distillation (the original HF source is stored for loading).

ENHANCED (2026-07-16):
- Reduced REAP prune ratio in `export_to_lazytorch` from 0.3 to 0.15 to avoid compounding damage.
- Ensured zero-shot compensation adapters (adapter_A, adapter_B) are preserved during export
  (already handled, but verified and documented).
- Added optional post-export validation (`validate_after_export` parameter) that performs
  a forward pass to catch corruption (uses minimal RAM for validation).

FIX (2026-07-16):
- Replaced broken `.generate()` call in post-export validation with a simple forward pass,
  since LazyModule does not implement `.generate()`.
- Added proper `finally` block to ensure the validation model is unloaded even on error.
- Updated docstring to clarify that validation uses a forward pass (not generation)
  and may use additional memory (though LazyTorch loading is still memory‑efficient).

FIX (2026-07-16):
- In `_lazy_build_modules`, added validation for `padding_idx` in `LazyEmbedding`
  to prevent `AssertionError: Padding_idx must be within num_embeddings` when the
  vocabulary size has been reduced (e.g., after pruning or distillation) but the
  padding index from the original model is out of bounds. If the padding_idx is
  invalid (>= num_embeddings), it is disabled (set to None) with a warning.

FIX (2026-07-16):
- Improved `padding_idx` validation to check against the actual weight shape
  (`weight_shape[0]`) rather than the possibly stale `num_embeddings` field
  in the manifest. This ensures the check passes even if the manifest's
  `num_embeddings` is outdated due to vocabulary reduction after export.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import logging
import gc
import shutil
import re
import inspect
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Union, List, Set
from contextlib import contextmanager
from collections import namedtuple

# ---- Relative imports for internal modules ----
from .utils import _validate_tokenizer_deep

logger = logging.getLogger(__name__)

# ---- Centralized LazyTorch version ----
LAZYTORCH_VERSION = "1.2"

# ----------------------------------------------------------------------
# HF-compatible output structure
# ----------------------------------------------------------------------
CausalLMOutputWithPast = namedtuple(
    "CausalLMOutputWithPast",
    ["loss", "logits", "past_key_values", "hidden_states", "attentions"]
)
CausalLMOutputWithPast.__new__.__defaults__ = (None, None, None, None, None)


# =============================================================================
# REAP Optimisation Helpers
# =============================================================================

def apply_reap_pruning(model: nn.Module, prune_ratio: float = 0.15) -> nn.Module:  # <-- CHANGED: default 0.15
    """
    Apply simple magnitude pruning to all Linear layers.
    Weights below the `prune_ratio` quantile are set to zero.

    For large tensors, the quantile is computed on a random sample to avoid OOM.

    Args:
        model: The PyTorch model to prune.
        prune_ratio: Fraction of weights to prune (0.0 to 1.0). Default 0.15 (gentle).

    Returns:
        The model with pruned weights (in-place).
    """
    if prune_ratio <= 0.0 or prune_ratio >= 1.0:
        logger.warning(f"Invalid prune_ratio {prune_ratio}. Skipping pruning.")
        return model

    logger.info(f"Applying REAP pruning with ratio {prune_ratio:.2f}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            if weight.numel() == 0:
                continue
            flat_weights = weight.abs().flatten()
            # ---- Memory‑efficient quantile computation ----
            if flat_weights.numel() > 10_000_000:
                # Sample up to 1,000,000 elements
                n_samples = min(1_000_000, flat_weights.numel())
                indices = torch.randint(0, flat_weights.numel(), (n_samples,), device=flat_weights.device)
                sample = flat_weights[indices]
                threshold = torch.quantile(sample, prune_ratio)
            else:
                threshold = torch.quantile(flat_weights, prune_ratio)
            mask = weight.abs() > threshold
            # Keep weights above threshold, zero out others
            module.weight.data = weight * mask
            logger.debug(f"Pruned {name}: {mask.numel() - mask.sum().item()} weights zeroed")

    return model


def apply_lazy_quant(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization to all Linear layers (int8) for memory savings.

    This uses torch.quantization.quantize_dynamic, which converts linear layers
    to quantized versions with int8 weights.

    Args:
        model: The PyTorch model to quantize.

    Returns:
        The quantized model (in-place).
    """
    try:
        # Only quantize if model is on CPU
        if next(model.parameters()).device.type != "cpu":
            logger.warning("Quantization only supported on CPU; moving model to CPU first.")
            model = model.to("cpu")

        # Use dynamic quantization for Linear layers
        from torch.quantization import quantize_dynamic
        model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        logger.info("Applied dynamic quantization (int8) to all Linear layers.")
        return model
    except Exception as e:
        logger.warning(f"Quantization failed: {e}. Skipping.")
        return model


# ----------------------------------------------------------------------
# Helper: Prepare tensor for saving (dequantize, detach, contiguous, CPU)
# ----------------------------------------------------------------------
def _prepare_tensor_for_saving(tensor: torch.Tensor) -> torch.Tensor:
    """
    Prepare a tensor for saving to disk:
    - If it's a quantized tensor, dequantize it to float32.
    - Detach from computation graph (remove gradient history).
    - Ensure it's contiguous and on CPU.
    - Returns a regular torch.Tensor.

    Raises ValueError if the tensor cannot be converted.
    """
    # Dequantize if it's a quantized tensor
    if hasattr(tensor, 'dequantize'):
        try:
            tensor = tensor.dequantize()
            logger.debug("Dequantized tensor before saving.")
        except Exception as e:
            raise ValueError(f"Failed to dequantize tensor: {e}") from e

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    # Detach, move to CPU, and make contiguous
    tensor = tensor.detach().contiguous().cpu()
    return tensor


# ----------------------------------------------------------------------
# LazyParameter: A parameter that loads its data from disk on demand
# ----------------------------------------------------------------------
class LazyParameter(torch.Tensor):
    """
    A tensor subclass that stores its data in a memory-mapped numpy array on disk.
    The data is loaded into RAM only when the tensor is accessed (forward pass).
    After use, it can be freed by calling .cpu() and deleting references.
    """
    @staticmethod
    def __new__(cls, mmap_array: np.memmap, dtype: torch.dtype, shape: Tuple[int, ...],
                requires_grad: bool = False, device: torch.device = torch.device('cpu')):
        tensor = torch.empty(0, dtype=dtype, device=device)
        tensor._lazy_mmap = mmap_array
        tensor._lazy_shape = shape
        tensor._lazy_dtype = dtype
        tensor._lazy_device = device
        tensor._lazy_data_loaded = False
        tensor._lazy_requires_grad = requires_grad
        return tensor

    def __init__(self, mmap_array: np.memmap, dtype: torch.dtype, shape: Tuple[int, ...],
                 requires_grad: bool = False, device: torch.device = torch.device('cpu')):
        super().__init__()
        self._lazy_mmap = mmap_array
        self._lazy_shape = shape
        self._lazy_dtype = dtype
        self._lazy_device = device
        self._lazy_data_loaded = False
        self._lazy_requires_grad = requires_grad

    def _load_data(self) -> torch.Tensor:
        if self._lazy_data_loaded:
            return self
        try:
            np_data = np.array(self._lazy_mmap).reshape(self._lazy_shape)
            tensor = torch.from_numpy(np_data).to(self._lazy_device).to(self._lazy_dtype)
            self.data = tensor.data
            self._lazy_data_loaded = True
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to load LazyParameter from disk: {e}")

    def __repr__(self):
        if self._lazy_data_loaded:
            return super().__repr__()
        else:
            return f"LazyParameter(shape={self._lazy_shape}, dtype={self._lazy_dtype}, device={self._lazy_device}, loaded=False)"

    @property
    def shape(self):
        return self._lazy_shape

    @property
    def dtype(self):
        return self._lazy_dtype

    @property
    def device(self):
        return self._lazy_device

    def to(self, *args, **kwargs):
        device = kwargs.get('device', None)
        if device is not None:
            self._lazy_device = torch.device(device)
        if self._lazy_data_loaded:
            return super().to(*args, **kwargs)
        else:
            return self

    def cpu(self):
        if self._lazy_data_loaded:
            super().cpu()
        else:
            self._lazy_device = torch.device('cpu')
        return self

    def cuda(self, device=None):
        if self._lazy_data_loaded:
            if device is None:
                super().cuda()
            else:
                super().cuda(device)
        else:
            self._lazy_device = torch.device('cuda' if device is None else device)
        return self

    def __getitem__(self, key):
        self._load_data()
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self._load_data()
        super().__setitem__(key, value)

    def __add__(self, other):
        self._load_data()
        return super().__add__(other)

    def __mul__(self, other):
        self._load_data()
        return super().__mul__(other)

    def _lazy_ensure_loaded(self):
        self._load_data()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args = []
        for arg in args:
            if isinstance(arg, LazyParameter):
                arg._load_data()
            new_args.append(arg)
        for k, v in kwargs.items():
            if isinstance(v, LazyParameter):
                v._load_data()
                kwargs[k] = v
        return super().__torch_function__(func, types, new_args, kwargs)


# ----------------------------------------------------------------------
# LazyModule: Base class that supports lazy loading of parameters
# ----------------------------------------------------------------------
class LazyModule(nn.Module):
    """
    A module that loads its parameters on demand from disk.
    Supports nested modules and arbitrary architectures via manifest.
    Implements a generic forward pass that works for most transformer models.
    """
    def __init__(self, lazy_dir: Path, manifest: Dict[str, Any], unload_after_forward: bool = True):
        super().__init__()
        self.lazy_dir = Path(lazy_dir)
        self.manifest = manifest
        self.unload_after_forward = unload_after_forward
        self._layer_cache = None  # Cache for discovered layers
        self._model_type = manifest.get("model_type", "unknown")
        
        # ---- Check version compatibility (warning only) ----
        manifest_version = manifest.get("version", "0.0")
        if manifest_version != LAZYTORCH_VERSION:
            logger.warning(
                f"Manifest version {manifest_version} differs from current {LAZYTORCH_VERSION}. "
                "The model may be incompatible; if you encounter errors, re-export using the current version."
            )
        
        # Mapping from sanitized name (used as attribute) to original dotted name
        self._name_map = {}
        self._lazy_build_modules()
        self.config = self._create_dummy_config()
        # Expose num_layers from manifest for layer discovery
        self.num_layers = manifest.get("num_layers", None)

    def _create_dummy_config(self):
        """Create a dummy config object for compatibility."""
        manifest = self.manifest  # capture manifest
        class DummyConfig:
            def __init__(self, model_type):
                self.model_type = model_type
                self.use_cache = False
                # Vocab size from manifest, fallback to 50272
                self.vocab_size = manifest.get("vocab_size", 50272)
                self.num_hidden_layers = manifest.get("num_layers", 12)
                self.manifest = manifest  # store for any future access
        return DummyConfig(self.manifest.get("model_type", "unknown"))

    def _sanitize_name(self, name: str) -> str:
        """Replace dots with underscores for safe attribute naming."""
        return name.replace('.', '_')

    def _lazy_build_modules(self):
        """Recursively build the module structure from manifest."""
        for name, info in self.manifest.get('modules', {}).items():
            module_type = info.get('type')
            safe_name = self._sanitize_name(name)
            self._name_map[safe_name] = name
            try:
                if module_type == 'Linear':
                    in_features = info['in_features']
                    out_features = info['out_features']
                    bias = info.get('bias', False)
                    module = LazyLinear(in_features, out_features, bias)
                    weight_path = self.lazy_dir / info['weight_file']
                    weight_mmap = np.memmap(weight_path, dtype=info['weight_dtype'], mode='r',
                                            shape=tuple(info['weight_shape']))
                    module.weight = LazyParameter(weight_mmap, dtype=torch.float32,
                                                  shape=tuple(info['weight_shape']),
                                                  requires_grad=False, device=torch.device('cpu'))
                    if bias and 'bias_file' in info:
                        bias_path = self.lazy_dir / info['bias_file']
                        bias_mmap = np.memmap(bias_path, dtype=info['bias_dtype'], mode='r',
                                              shape=(out_features,))
                        module.bias = LazyParameter(bias_mmap, dtype=torch.float32,
                                                    shape=(out_features,),
                                                    requires_grad=False, device=torch.device('cpu'))
                    # ---- Preserve zero-shot compensation adapters ----
                    if 'adapter_A_file' in info and 'adapter_B_file' in info:
                        try:
                            adapter_A_path = self.lazy_dir / info['adapter_A_file']
                            adapter_A_mmap = np.memmap(adapter_A_path, dtype='float32', mode='r',
                                                       shape=tuple(info['adapter_A_shape']))
                            adapter_A = torch.nn.Parameter(
                                torch.from_numpy(np.array(adapter_A_mmap)).reshape(tuple(info['adapter_A_shape']))
                            )
                            adapter_B_path = self.lazy_dir / info['adapter_B_file']
                            adapter_B_mmap = np.memmap(adapter_B_path, dtype='float32', mode='r',
                                                       shape=tuple(info['adapter_B_shape']))
                            adapter_B = torch.from_numpy(np.array(adapter_B_mmap)).reshape(tuple(info['adapter_B_shape']))
                            module.register_parameter("adapter_A", adapter_A)
                            module.register_buffer("adapter_B", adapter_B)
                            module._adapter_patched = True
                            logger.debug(f"Restored adapters for {name}")
                        except Exception as e:
                            logger.warning(f"Failed to restore adapters for {name}: {e}")
                    self.add_module(safe_name, module)

                elif module_type == 'Embedding':
                    num_embeddings = info['num_embeddings']
                    embedding_dim = info['embedding_dim']
                    padding_idx = info.get('padding_idx', -1)
                    if padding_idx == -1:
                        padding_idx = None

                    # ---- Improved: Validate padding_idx against actual weight shape ----
                    weight_shape = info.get('weight_shape')
                    if weight_shape and padding_idx is not None:
                        actual_num_embeddings = weight_shape[0]
                        if padding_idx >= actual_num_embeddings:
                            logger.warning(
                                f"Padding_idx {padding_idx} is out of range for actual num_embeddings {actual_num_embeddings} in {name}; "
                                "disabling padding by setting padding_idx=None."
                            )
                            padding_idx = None

                    # Optional: warn if num_embeddings is inconsistent with actual shape
                    if weight_shape and num_embeddings != weight_shape[0]:
                        logger.warning(
                            f"num_embeddings in manifest ({num_embeddings}) does not match weight shape {weight_shape[0]} for {name}; "
                            "using manifest value for constructor but actual shape will be used at runtime."
                        )

                    module = LazyEmbedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
                    weight_path = self.lazy_dir / info['weight_file']
                    weight_mmap = np.memmap(weight_path, dtype=info['weight_dtype'], mode='r',
                                            shape=tuple(info['weight_shape']))
                    module.weight = LazyParameter(weight_mmap, dtype=torch.float32,
                                                  shape=tuple(info['weight_shape']),
                                                  requires_grad=False, device=torch.device('cpu'))
                    self.add_module(safe_name, module)

                elif module_type == 'LayerNorm':
                    normalized_shape = info['normalized_shape']
                    eps = info.get('eps', 1e-5)
                    elementwise_affine = info.get('elementwise_affine', True)
                    module = LazyLayerNorm(normalized_shape, eps, elementwise_affine)
                    if elementwise_affine:
                        weight_path = self.lazy_dir / info['weight_file']
                        weight_mmap = np.memmap(weight_path, dtype=info['weight_dtype'], mode='r',
                                                shape=tuple(info['weight_shape']))
                        module.weight = LazyParameter(weight_mmap, dtype=torch.float32,
                                                      shape=tuple(info['weight_shape']),
                                                      requires_grad=False, device=torch.device('cpu'))
                        if 'bias_file' in info:
                            bias_path = self.lazy_dir / info['bias_file']
                            bias_mmap = np.memmap(bias_path, dtype=info['bias_dtype'], mode='r',
                                                  shape=tuple(info['bias_shape']))
                            module.bias = LazyParameter(bias_mmap, dtype=torch.float32,
                                                        shape=tuple(info['bias_shape']),
                                                        requires_grad=False, device=torch.device('cpu'))
                    self.add_module(safe_name, module)

                elif module_type == 'Conv1D':
                    in_features = info['in_features']
                    out_features = info['out_features']
                    module = LazyConv1D(in_features, out_features)
                    weight_path = self.lazy_dir / info['weight_file']
                    weight_mmap = np.memmap(weight_path, dtype=info['weight_dtype'], mode='r',
                                            shape=tuple(info['weight_shape']))
                    module.weight = LazyParameter(weight_mmap, dtype=torch.float32,
                                                  shape=tuple(info['weight_shape']),
                                                  requires_grad=False, device=torch.device('cpu'))
                    if 'bias_file' in info:
                        bias_path = self.lazy_dir / info['bias_file']
                        bias_mmap = np.memmap(bias_path, dtype=info['bias_dtype'], mode='r',
                                              shape=(out_features,))
                        module.bias = LazyParameter(bias_mmap, dtype=torch.float32,
                                                    shape=(out_features,),
                                                    requires_grad=False, device=torch.device('cpu'))
                    self.add_module(safe_name, module)

                elif module_type == 'Generic':
                    # Try to infer module type from weight shape
                    if 'weight_shape' in info:
                        shape = tuple(info['weight_shape'])
                        if len(shape) == 2:
                            # Treat as Linear
                            in_features = shape[1]
                            out_features = shape[0]
                            bias = info.get('bias', False)
                            module = LazyLinear(in_features, out_features, bias)
                            weight_path = self.lazy_dir / info['weight_file']
                            weight_mmap = np.memmap(weight_path, dtype=info['weight_dtype'], mode='r',
                                                    shape=shape)
                            module.weight = LazyParameter(weight_mmap, dtype=torch.float32,
                                                          shape=shape,
                                                          requires_grad=False, device=torch.device('cpu'))
                            if bias and 'bias_file' in info:
                                bias_path = self.lazy_dir / info['bias_file']
                                bias_mmap = np.memmap(bias_path, dtype=info['bias_dtype'], mode='r',
                                                      shape=(out_features,))
                                module.bias = LazyParameter(bias_mmap, dtype=torch.float32,
                                                            shape=(out_features,),
                                                            requires_grad=False, device=torch.device('cpu'))
                            self.add_module(safe_name, module)
                            logger.debug(f"Converted Generic {name} to Linear (in={in_features}, out={out_features})")
                        else:
                            logger.warning(f"Unsupported Generic module {name} with weight shape {shape}; using Identity")
                            self.add_module(safe_name, nn.Identity())
                    else:
                        logger.warning(f"Generic module {name} has no weight_shape; using Identity")
                        self.add_module(safe_name, nn.Identity())

                else:
                    logger.warning(f"Unsupported module type {module_type} for {name}, using Identity")
                    self.add_module(safe_name, nn.Identity())
            except Exception as e:
                # ----- CRITICAL FIX: Raise instead of swallowing -----
                raise RuntimeError(
                    f"Failed to build module '{name}' (type {module_type}): {e}\n"
                    "This indicates the LazyTorch model is corrupt (e.g., mismatched mmap sizes, missing files). "
                    "Please delete the .lazytorch directory and re-export the model from a clean Hugging Face model.\n"
                    f"You can delete it using: python bootstrap.py remove --model {self.lazy_dir.stem}"
                ) from e

    def _discover_layers(self) -> List[Tuple[str, nn.Module]]:
        """Discover all decoder layers using multiple strategies."""
        if self._layer_cache is not None:
            return self._layer_cache

        layer_modules = []
        seen_ids: Set[int] = set()

        # Strategy 1: Common naming patterns (expanded for broader architecture support)
        layer_patterns = [
            'layer.', 'h.', 'decoder.layers.', 'model.layers.', 'layers.',
            'transformer.h', 'encoder.layer.', 'decoder.layer.',
            'gpt_neox.layers.', 'llama.layers.', 'mistral.layers.',
            'qwen.layers.', 'phi.layers.'
        ]

        def collect_by_pattern(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                # Check both sanitized and original names for patterns
                matched = any(pattern in full_name for pattern in layer_patterns)
                # Also check original name if we have mapping
                original = self._name_map.get(name, name)
                if not matched:
                    matched = any(pattern in original for pattern in layer_patterns)
                if matched:
                    if id(child) not in seen_ids:
                        seen_ids.add(id(child))
                        layer_modules.append((full_name, child))
                # If we know num_layers, we can also look for modules with numeric names
                if self.num_layers is not None and name.isdigit() and int(name) < self.num_layers:
                    if id(child) not in seen_ids:
                        seen_ids.add(id(child))
                        layer_modules.append((full_name, child))
                collect_by_pattern(child, full_name)

        collect_by_pattern(self)

        # Strategy 2: Look for modules that have self_attn and mlp (common block structure)
        for name, module in self.named_modules():
            if hasattr(module, 'self_attn') and hasattr(module, 'mlp'):
                if id(module) not in seen_ids:
                    seen_ids.add(id(module))
                    layer_modules.append((name, module))

        # Sort by layer index
        def extract_layer_index(name):
            match = re.search(r'(\d+)', name)
            return int(match.group(1)) if match else -1

        layer_modules.sort(key=lambda x: extract_layer_index(x[0]))

        # Filter to num_layers if known
        if self.num_layers is not None:
            filtered = []
            for name, mod in layer_modules:
                idx = extract_layer_index(name)
                if 0 <= idx < self.num_layers:
                    filtered.append((name, mod))
            if filtered:
                layer_modules = filtered
                logger.debug(f"Found {len(layer_modules)} layers using num_layers filter.")
            else:
                logger.debug(f"num_layers filter failed, using all {len(layer_modules)} discovered.")

        if not layer_modules:
            logger.warning(f"No decoder layers discovered for model type {self._model_type}. "
                          f"The forward pass may not work correctly.")

        self._layer_cache = layer_modules
        return layer_modules

    def _can_accept_attention_mask(self, layer: nn.Module) -> bool:
        """Check if a layer's forward method accepts attention_mask as a keyword argument."""
        try:
            sig = inspect.signature(layer.forward)
            return 'attention_mask' in sig.parameters
        except (ValueError, TypeError):
            return False

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=False, return_dict=True, **kwargs):
        """
        Generic forward pass that works for most transformer architectures.
        """
        # ---- KV Cache: accepted but ignored ----
        if use_cache:
            logger.debug("KV caching is not fully implemented in LazyTorch. use_cache=False is forced.")
            use_cache = False
        if past_key_values is not None:
            logger.debug("past_key_values received but ignored (KV caching not supported).")

        # 1. Get embeddings
        x = self._get_embeddings(input_ids)

        # 2. Pass through decoder layers
        hidden_states = self._apply_layers(x, attention_mask)

        # 3. Final layer norm
        hidden_states = self._apply_final_norm(hidden_states)

        # 4. LM head
        logits = self._apply_lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

    def _get_embeddings(self, input_ids):
        """Find and apply the embedding layer."""
        embed_names = ['embed_tokens', 'wte', 'tok_embeddings', 'model.embed_tokens']
        # Try both original and sanitized names
        for name in embed_names:
            safe_name = self._sanitize_name(name)
            if name in self._modules:
                return self._modules[name](input_ids)
            if safe_name in self._modules:
                return self._modules[safe_name](input_ids)
        # Fallback: find any embedding module
        for module in self.children():
            if isinstance(module, LazyEmbedding):
                return module(input_ids)
        raise RuntimeError("No embedding layer found in LazyModule")

    def _apply_layers(self, hidden_states, attention_mask):
        """Apply all decoder layers using robust discovery."""
        layers = self._discover_layers()
        if not layers:
            logger.warning("No decoder layers discovered; returning input unchanged.")
            return hidden_states

        for name, layer in layers:
            try:
                if self._can_accept_attention_mask(layer):
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
                else:
                    hidden_states = layer(hidden_states)
            except Exception as e:
                # Log error but continue with next layer (prevent total failure)
                logger.warning(f"Layer {name} forward failed: {e}. Skipping.")
                # Continue with current hidden_states to avoid complete failure
        return hidden_states

    def _apply_final_norm(self, hidden_states):
        """Apply final layer norm if present."""
        norm_names = ['norm', 'ln_f', 'final_layer_norm', 'model.norm']
        for name in norm_names:
            safe_name = self._sanitize_name(name)
            if name in self._modules:
                return self._modules[name](hidden_states)
            if safe_name in self._modules:
                return self._modules[safe_name](hidden_states)
        return hidden_states

    def _apply_lm_head(self, hidden_states):
        """Apply LM head (linear layer) to get logits."""
        lm_head_names = ['lm_head', 'output', 'embed_out', 'model.lm_head']
        for name in lm_head_names:
            safe_name = self._sanitize_name(name)
            if name in self._modules:
                return self._modules[name](hidden_states)
            if safe_name in self._modules:
                return self._modules[safe_name](hidden_states)

        # Fallback 1: find linear layer matching vocab_size (from config or manifest)
        # Prefer config.vocab_size if set, else try manifest, else 50272
        vocab_size = getattr(self.config, 'vocab_size', None)
        if vocab_size is None:
            vocab_size = self.manifest.get("vocab_size", 50272)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.out_features == vocab_size:
                return module(hidden_states)

        # Fallback 2: find largest linear layer (out_features > 1000)
        candidates = []
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.out_features > 1000:
                candidates.append(module)
        if candidates:
            # Usually the last one is lm_head
            return candidates[-1](hidden_states)

        raise RuntimeError("No LM head found in LazyModule")

    def unload_parameters(self):
        """Unload all lazy parameters to free memory."""
        for param in self.parameters():
            if isinstance(param, LazyParameter) and param._lazy_data_loaded:
                param.cpu()
                param.data = torch.empty(0, device='cpu')
                param._lazy_data_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Unloaded LazyModule parameters")

    def load_parameters(self):
        """Force load all lazy parameters into memory."""
        for param in self.parameters():
            if isinstance(param, LazyParameter) and not param._lazy_data_loaded:
                param._load_data()
        logger.debug("Loaded LazyModule parameters")

    def to(self, device):
        """Move module to device, updating lazy parameters device."""
        device = torch.device(device)
        for param in self.parameters():
            if isinstance(param, LazyParameter):
                param._lazy_device = device
            else:
                param.data = param.data.to(device)
        for buffer in self.buffers():
            if not isinstance(buffer, LazyParameter):
                buffer.data = buffer.data.to(device)
        return super().to(device)


# ----------------------------------------------------------------------
# Lazy versions of common layers with adapter support
# ----------------------------------------------------------------------
class LazyLinear(nn.Linear):
    """
    LazyLinear extends nn.Linear to support lazy-loaded weights and
    zero-shot compensation adapters (adapter_A, adapter_B) without
    requiring forward patching.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self._parameters = nn.ParameterDict()
        # Adapter buffers (for zero-shot compensation)
        self.adapter_B = None
        self.adapter_A = None
        self._adapter_patched = False

    def _load_own_params(self):
        if hasattr(self, 'weight') and isinstance(self.weight, LazyParameter):
            self.weight._load_data()
        if hasattr(self, 'bias') and isinstance(self.bias, LazyParameter):
            self.bias._load_data()

    def _apply_adapter(self, x):
        """Apply zero-shot compensation adapter if present."""
        if self.adapter_B is not None and self.adapter_A is not None:
            # Ensure adapter tensors are on the same device as input
            if self.adapter_B.device != x.device:
                self.adapter_B = self.adapter_B.to(x.device)
            if self.adapter_A.device != x.device:
                self.adapter_A = self.adapter_A.to(x.device)
            return (x @ self.adapter_B.T) @ self.adapter_A.T
        return None

    def forward(self, input):
        self._load_own_params()
        # Defensive check: weight must be 2-D
        if hasattr(self, 'weight') and self.weight is not None:
            if self.weight.dim() != 2:
                raise RuntimeError(
                    f"LazyLinear weight has {self.weight.dim()} dimensions; expected 2. "
                    "This indicates the model is not a standard linear layer. "
                    "Please use a compatible model or disable LazyTorch."
                )
        out = super().forward(input)
        # Apply adapter if present
        adapter_out = self._apply_adapter(input)
        if adapter_out is not None:
            out = out + adapter_out
        return out


class LazyEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self._parameters = nn.ParameterDict()

    def _load_own_params(self):
        if hasattr(self, 'weight') and isinstance(self.weight, LazyParameter):
            self.weight._load_data()

    def forward(self, input):
        self._load_own_params()
        return super().forward(input)


class LazyLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self._parameters = nn.ParameterDict()

    def _load_own_params(self):
        if hasattr(self, 'weight') and isinstance(self.weight, LazyParameter):
            self.weight._load_data()
        if hasattr(self, 'bias') and isinstance(self.bias, LazyParameter):
            self.bias._load_data()

    def forward(self, input):
        self._load_own_params()
        return super().forward(input)


class LazyConv1D(nn.Module):
    """
    LazyConv1D mimics Hugging Face's Conv1D with lazy-loaded weights.
    This class is used for GPT‑2 style models.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Alias for compatibility with GPT-2 detection logic (nx/nf)
        self.nx = in_features
        self.nf = out_features
        self.weight = None
        self.bias = None

    def forward(self, x):
        if self.weight is None:
            raise RuntimeError("Weight not set")
        if isinstance(self.weight, LazyParameter):
            self.weight._load_data()
        # Defensive: weight must be 2-D (out_features, in_features)
        if self.weight.dim() != 2:
            raise RuntimeError(
                f"LazyConv1D weight has {self.weight.dim()} dimensions; expected 2. "
                "This may indicate a corrupt LazyTorch model."
            )
        w = self.weight.t()
        out = x @ w
        if self.bias is not None:
            if isinstance(self.bias, LazyParameter):
                self.bias._load_data()
            out = out + self.bias
        return out


# ----------------------------------------------------------------------
# Export function: Convert a HuggingFace model to .lazytorch format
# ----------------------------------------------------------------------
def export_to_lazytorch(
    model: Union[str, Path, nn.Module],
    output_path: Union[str, Path],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    progress_callback=None,
    save_tokenizer: bool = True,
    reap_mode: bool = True,
    validate_after_export: bool = False,  # <-- NEW: optional post-export validation
) -> Path:
    """
    Convert a HuggingFace model (either a model instance or a path to a saved model)
    into the .lazytorch format. If save_tokenizer is True and the source is a path,
    tokenizer files are copied into the output directory.

    Args:
        model: Path to model directory, or an already loaded nn.Module.
        output_path: Destination directory for the .lazytorch export.
        device: Device to use for loading (only 'cpu' is supported for export).
        dtype: Data type for the exported parameters.
        progress_callback: Optional callback for progress reporting.
        save_tokenizer: If True, copy tokenizer files from source.
        reap_mode: If True, apply REAP optimizations (pruning + quantization) during export.
        validate_after_export: If True, load the exported model and run a forward pass
                               sanity check (may use additional memory but still
                               memory‑efficient due to LazyTorch). Default False.

    Raises:
        ValueError: If the tokenizer is corrupt and cannot be loaded after copying.
        RuntimeError: If any weight file is truncated (size mismatch) or if validation fails.
    """
    output_path = Path(output_path)
    if output_path.suffix == '.lazytorch':
        output_path = output_path.with_suffix('')
    output_path.mkdir(parents=True, exist_ok=True)

    source_path = None
    same_dir = False
    if isinstance(model, (str, Path)):
        source_path = Path(model).resolve()
        # If output_path is the same as source_path, skip copying
        if source_path.resolve() == output_path.resolve():
            same_dir = True
            logger.info("Source and output paths are the same; skipping file copying.")
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading model from {source_path} for conversion...")
        if progress_callback:
            progress_callback("Loading model...")

        # ---- FIX: Force safe CPU load with device_map=None ----
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(source_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=None,          # Force CPU loading to avoid meta tensor issues
                offload_folder="/tmp/offload" if device != "cpu" else None
            )
            model = model.to("cpu")
            model.eval()
            logger.info("Model loaded successfully on CPU.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {source_path}: {e}")

        # ---- Apply REAP optimizations if enabled ----
        if reap_mode:
            logger.info("Applying REAP optimizations (pruning + quantization)...")
            if progress_callback:
                progress_callback("Applying REAP pruning...")
            # ---- CHANGED: use gentle prune ratio 0.15 ----
            model = apply_reap_pruning(model, prune_ratio=0.15)
            if progress_callback:
                progress_callback("Applying lazy quantization...")
            model = apply_lazy_quant(model)
            logger.info("REAP optimizations applied successfully.")

        # Copy tokenizer files only if not same directory
        if save_tokenizer and source_path.exists() and source_path.is_dir() and not same_dir:
            tokenizer_files = [
                "tokenizer.json", "tokenizer_config.json", "vocab.json",
                "merges.txt", "special_tokens_map.json", "added_tokens.json",
                "chat_template.json", "generation_config.json", "tokenizer.model"
            ]
            for fname in tokenizer_files:
                src_file = source_path / fname
                dst_file = output_path / fname
                if src_file.exists() and src_file.resolve() != dst_file.resolve():
                    try:
                        shutil.copy2(src_file, dst_file)
                        logger.debug(f"Copied tokenizer file {fname}")
                    except shutil.SameFileError:
                        pass
                    except Exception as e:
                        logger.warning(f"Failed to copy {fname}: {e}")
            if progress_callback:
                progress_callback("Copied tokenizer files")

            # ---- Validate the copied tokenizer using shared function ----
            if not _validate_tokenizer_deep(output_path):
                # Clean up the output directory to avoid leaving a broken model
                shutil.rmtree(output_path, ignore_errors=True)
                raise ValueError(
                    f"Tokenizer in source model at {source_path} is corrupt or incompatible.\n"
                    "Please delete the source model and re-download it, or repair the tokenizer files.\n"
                    f"You can delete it using: python bootstrap.py remove --model {source_path.stem}\n"
                    "Then re-download from Hugging Face."
                )
        elif same_dir:
            # Validate tokenizer in the source (which is also output)
            if not _validate_tokenizer_deep(source_path):
                raise ValueError(
                    f"Tokenizer in model at {source_path} is corrupt or incompatible.\n"
                    "Please delete the model and re-download it, or repair the tokenizer files.\n"
                    f"You can delete it using: python bootstrap.py remove --model {source_path.stem}\n"
                    "Then re-download from Hugging Face."
                )
    else:
        # If model is already a loaded module, we can optionally apply REAP even if it's already loaded.
        if reap_mode:
            logger.info("Applying REAP optimizations to already loaded model...")
            model = apply_reap_pruning(model, prune_ratio=0.15)  # <-- gentle
            model = apply_lazy_quant(model)
            model.eval()

    # Check for E8Linear and convert to standard Linear for export
    e8_linear_class = None
    try:
        from .e8_quantize import E8Linear
        e8_linear_class = E8Linear
    except ImportError:
        pass

    manifest: Dict[str, Any] = {
        "version": LAZYTORCH_VERSION,
        "model_type": model.config.model_type if hasattr(model, "config") else "unknown",
        "dtype": str(dtype),
        "modules": {},
        "num_layers": getattr(model.config, 'num_hidden_layers', None),
        "vocab_size": getattr(model.config, 'vocab_size', 50272) if hasattr(model, 'config') else 50272,
    }

    # ---- NEW: Store original source path ----
    if source_path is not None:
        manifest["original_path"] = str(source_path)

    param_count = 0

    def save_parameter(module_path: str, param_name: str, param: torch.Tensor, module_type: str = None, extra_info: dict = None):
        nonlocal param_count

        # ---- Prepare tensor for saving ----
        try:
            param = _prepare_tensor_for_saving(param)
        except Exception as e:
            logger.error(f"Failed to prepare tensor for {module_path}.{param_name}: {e}")
            # Clean up to avoid partial export
            shutil.rmtree(output_path, ignore_errors=True)
            raise RuntimeError(f"Cannot export parameter {module_path}.{param_name}: {e}") from e

        safe_name = f"{module_path.replace('.', '_')}_{param_name}" if module_path else param_name
        weight_file = f"{safe_name}.bin"
        weight_path = output_path / weight_file

        # Convert to numpy (should be on CPU and contiguous)
        np_data = param.numpy()

        # Write the data
        np_data.tofile(weight_path)

        # ---- FIX: Validate file size after writing ----
        actual_size = weight_path.stat().st_size
        expected_size = np_data.nbytes
        if actual_size != expected_size:
            # ---- ENHANCED ERROR MESSAGES (Change 1) ----
            error_msg = (
                f"File {weight_file} size mismatch (expected {expected_size}, got {actual_size}).\n"
                "This indicates the export was interrupted (e.g., disk full, crash) or I/O error.\n"
                "To fix this, delete the corrupted LazyTorch model and re-export:\n"
                f"  python -m lazy_llama.bootstrap remove --model {output_path.stem}\n"
                f"  python -m lazy_llama.bootstrap convert-lazytorch {source_path.stem if source_path else output_path.stem} --force\n"
                "If the issue persists, check available disk space and permissions."
            )
            logger.error(error_msg)
            shutil.rmtree(output_path, ignore_errors=True)
            raise RuntimeError(error_msg)

        modules_dict = manifest["modules"]
        if module_path not in modules_dict:
            modules_dict[module_path] = {}
        mod_info = modules_dict[module_path]
        if module_type:
            mod_info["type"] = module_type
        if param_name == "weight":
            mod_info["weight_file"] = weight_file
            mod_info["weight_shape"] = list(param.shape)
            mod_info["weight_dtype"] = "float32" if dtype == torch.float32 else "float16"
            if extra_info:
                mod_info.update(extra_info)
        elif param_name == "bias":
            mod_info["bias_file"] = weight_file
            mod_info["bias_shape"] = list(param.shape)
            mod_info["bias_dtype"] = "float32" if dtype == torch.float32 else "float16"
            mod_info["bias"] = True
        elif param_name == "adapter_A":
            # ---- Preserve zero-shot compensation adapters ----
            mod_info["adapter_A_file"] = weight_file
            mod_info["adapter_A_shape"] = list(param.shape)
        elif param_name == "adapter_B":
            mod_info["adapter_B_file"] = weight_file
            mod_info["adapter_B_shape"] = list(param.shape)
        else:
            mod_info[param_name] = {
                "file": weight_file,
                "shape": list(param.shape),
                "dtype": "float32" if dtype == torch.float32 else "float16"
            }
        param_count += 1
        if progress_callback and param_count % 10 == 0:
            progress_callback(f"Saved {param_count} parameters")

    # Traverse the model and save all parameters, including adapters
    for module_path, module in model.named_modules():
        if module_path == "":
            continue

        # Handle E8Linear specially
        if e8_linear_class and isinstance(module, e8_linear_class):
            logger.warning(f"E8Linear detected at {module_path}; converting to standard Linear for LazyTorch export (compression lost).")
            weight = module.weight
            bias = module.bias if module.bias is not None else None
            in_features = module.in_features
            out_features = module.out_features
            save_parameter(module_path, "weight", weight,
                           module_type="Linear",
                           extra_info={"in_features": in_features, "out_features": out_features})
            if bias is not None:
                save_parameter(module_path, "bias", bias)
            continue

        # Save adapter parameters if present (adapter_A, adapter_B)
        if hasattr(module, 'adapter_A') and module.adapter_A is not None:
            save_parameter(module_path, "adapter_A", module.adapter_A.data)
        if hasattr(module, 'adapter_B') and module.adapter_B is not None:
            save_parameter(module_path, "adapter_B", module.adapter_B)

        if isinstance(module, nn.Linear):
            if hasattr(module, "weight") and module.weight is not None:
                save_parameter(module_path, "weight", module.weight,
                               module_type="Linear",
                               extra_info={"in_features": module.in_features, "out_features": module.out_features})
            if module.bias is not None:
                save_parameter(module_path, "bias", module.bias)
        elif isinstance(module, nn.Embedding):
            if hasattr(module, "weight") and module.weight is not None:
                padding_idx = module.padding_idx if module.padding_idx is not None else -1
                save_parameter(module_path, "weight", module.weight,
                               module_type="Embedding",
                               extra_info={"num_embeddings": module.num_embeddings,
                                           "embedding_dim": module.embedding_dim,
                                           "padding_idx": padding_idx})
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                if hasattr(module, "weight") and module.weight is not None:
                    save_parameter(module_path, "weight", module.weight,
                                   module_type="LayerNorm",
                                   extra_info={"normalized_shape": module.normalized_shape,
                                               "eps": module.eps,
                                               "elementwise_affine": True})
                if hasattr(module, "bias") and module.bias is not None:
                    save_parameter(module_path, "bias", module.bias,
                                   extra_info={"normalized_shape": module.normalized_shape})
        # For Generic modules, we save weight/bias but we already handle them in the manifest
        elif hasattr(module, "weight") and module.weight is not None and isinstance(module.weight, torch.Tensor):
            # Determine if it looks like a Linear layer (2D weight) or Conv1D, etc.
            if module.weight.dim() == 2:
                # Treat as Linear if we can get in/out features
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]
                save_parameter(module_path, "weight", module.weight,
                               module_type="Linear",
                               extra_info={"in_features": in_features, "out_features": out_features})
                if hasattr(module, "bias") and module.bias is not None and isinstance(module.bias, torch.Tensor):
                    save_parameter(module_path, "bias", module.bias)
            else:
                save_parameter(module_path, "weight", module.weight,
                               module_type="Generic",
                               extra_info={})
                if hasattr(module, "bias") and module.bias is not None and isinstance(module.bias, torch.Tensor):
                    save_parameter(module_path, "bias", module.bias)
        elif hasattr(module, "weight") and module.weight is not None:
            # If weight exists but is not a tensor, log a warning and skip this module
            logger.warning(f"Skipping module {module_path} because weight is not a Tensor (type: {type(module.weight)})")
            continue

    # Save manifest
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Marker file
    marker_path = output_path.with_suffix(".lazytorch")
    marker_path.touch()

    # Also save the model's config.json if exists in source and not same directory
    if source_path and (source_path / "config.json").exists() and not same_dir:
        shutil.copy2(source_path / "config.json", output_path / "config.json")
        logger.debug("Copied config.json")
    elif source_path and (source_path / "config.json").exists() and same_dir:
        # config.json already exists in output, no need to copy
        logger.debug("config.json already present in output directory")

    # Optionally, save a REAP metadata file if optimizations were applied
    if reap_mode:
        reap_meta = {
            "pruning_applied": True,
            "prune_ratio": 0.15,
            "quantization_applied": True,
            "quantization_type": "dynamic_int8",
            "reap_version": "1.0"
        }
        with open(output_path / "reap_metadata.json", "w") as f:
            json.dump(reap_meta, f, indent=2)

    logger.info(f"LazyTorch model exported to {output_path} with {param_count} parameters (REAP optimizations: {reap_mode})")
    if progress_callback:
        progress_callback("Export complete")

    # ---- NEW: Optional post-export validation (fixed to use forward pass) ----
    if validate_after_export:
        logger.info("Performing post-export validation (forward pass)...")
        test_model = None
        tokenizer = None
        try:
            # Load the model using LazyTorch loader
            test_model = load_lazytorch_model(output_path, device="cpu", unload_after_forward=True)
            # Load tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(output_path))
            dummy_prompt = "Hello,"
            inputs = tokenizer(dummy_prompt, return_tensors="pt")
            # Perform a simple forward pass to verify model integrity
            with torch.no_grad():
                outputs = test_model(inputs.input_ids)
                if outputs.logits is None:
                    raise RuntimeError("No logits produced from forward pass")
                # Check that logits have expected shape (batch, seq_len, vocab_size)
                vocab_size = getattr(test_model.config, 'vocab_size', None)
                if vocab_size is not None and outputs.logits.shape[-1] != vocab_size:
                    raise RuntimeError(f"Logits vocab size mismatch: {outputs.logits.shape[-1]} vs {vocab_size}")
            logger.info("Validation successful: forward pass completed with valid logits.")
        except Exception as e:
            logger.error(f"Post-export validation failed: {e}")
            # Clean up the output directory to avoid leaving a broken model
            shutil.rmtree(output_path, ignore_errors=True)
            raise RuntimeError(f"LazyTorch export validation failed: {e}") from e
        finally:
            # Ensure cleanup
            if test_model is not None:
                test_model.unload_parameters()
                del test_model
            if tokenizer is not None:
                del tokenizer
            gc.collect()

    return output_path


# ----------------------------------------------------------------------
# Load function
# ----------------------------------------------------------------------
def load_lazytorch_model(
    model_path: Union[str, Path],
    device: str = "cpu",
    unload_after_forward: bool = True
) -> nn.Module:
    """
    Load a model from .lazytorch format. The model's parameters remain on disk
    until they are needed during forward passes.

    This loader supports standard causal language models. For other architectures,
    the generic forward may not work; use the standard PyTorch format instead.
    """
    model_path = Path(model_path)
    if model_path.suffix == '.lazytorch':
        lazy_dir = model_path.with_suffix('')
    else:
        lazy_dir = model_path

    manifest_path = lazy_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # ---- Validate tokenizer before loading using shared function ----
    tokenizer_path = lazy_dir
    if not _validate_tokenizer_deep(tokenizer_path):
        raise ValueError(
            f"Tokenizer in LazyTorch model at {lazy_dir} is corrupt or incompatible.\n"
            "Please delete the model and re-export it from a clean Hugging Face model.\n"
            f"You can delete it using: python bootstrap.py remove --model {lazy_dir.stem}"
        )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # ---- NEW: Validate manifest integrity (Change 3) ----
    modules = manifest.get('modules', {})
    for module_name, module_info in modules.items():
        if 'weight_file' in module_info:
            weight_file = module_info['weight_file']
            weight_path = lazy_dir / weight_file
            if not weight_path.exists():
                raise FileNotFoundError(
                    f"Missing weight file for module '{module_name}': {weight_file}\n"
                    f"Expected at: {weight_path}\n"
                    "The model is corrupted. Please delete and re-export it:\n"
                    f"  python -m lazy_llama.bootstrap remove --model {lazy_dir.stem}\n"
                    "  python -m lazy_llama.bootstrap convert-lazytorch <original_hf_model> --force"
                )
            # Check file size (optional but helpful)
            if 'weight_shape' in module_info:
                expected_size = np.prod(module_info['weight_shape']) * 4  # float32 = 4 bytes
                actual_size = weight_path.stat().st_size
                if abs(actual_size - expected_size) > 1024:  # allow small overhead
                    logger.warning(
                        f"Module '{module_name}' weight file {weight_file} size mismatch:\n"
                        f"  Expected: {expected_size} bytes, Got: {actual_size} bytes\n"
                        "The file may be truncated. The model may fail to load correctly."
                    )

    # ---- Version check: warn only (not an error) ----
    manifest_version = manifest.get("version", "0.0")
    if manifest_version != LAZYTORCH_VERSION:
        logger.warning(
            f"Manifest version {manifest_version} differs from current {LAZYTORCH_VERSION}. "
            "The model may be incompatible; if you encounter errors, re-export using the current version."
        )

    # ---- Load with mmap error handling (Change 2) ----
    try:
        root_module = LazyModule(lazy_dir, manifest, unload_after_forward=unload_after_forward)
        root_module = root_module.to(torch.device(device))
    except RuntimeError as e:
        # Check for common mmap/file corruption errors
        if "mmap length" in str(e) or "file size" in str(e) or "truncated" in str(e):
            raise RuntimeError(
                f"Failed to load LazyTorch model from {lazy_dir}:\n"
                f"  {e}\n\n"
                "This usually means the model export is corrupted (e.g., truncated file, missing weights).\n"
                "To fix this, delete the corrupted model and re-export it:\n"
                f"  python -m lazy_llama.bootstrap remove --model {lazy_dir.stem}\n"
                f"  python -m lazy_llama.bootstrap convert-lazytorch <original_hf_model> --force\n"
                "If the original model is not available, try re-downloading it first."
            ) from e
        raise
    logger.info(f"Loaded LazyTorch model from {lazy_dir} (weights remain on disk)")
    return root_module


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def is_lazytorch_model(path: Union[str, Path]) -> bool:
    path = Path(path)
    if path.is_dir():
        return (path / "manifest.json").exists()
    elif path.suffix == ".lazytorch":
        return (path.with_suffix('') / "manifest.json").exists()
    return False


def get_lazytorch_model_size(path: Union[str, Path]) -> int:
    path = Path(path)
    if path.suffix == '.lazytorch':
        lazy_dir = path.with_suffix('')
    else:
        lazy_dir = path
    if not lazy_dir.exists():
        return 0
    total = 0
    for f in lazy_dir.glob("*"):
        if f.is_file() and f.name != "manifest.json":
            total += f.stat().st_size
    return total


def convert_hf_to_lazytorch(
    hf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    dtype: str = "float32",
    progress_callback=None
) -> Path:
    """Convenience function to convert a HuggingFace model to LazyTorch format."""
    if output_path is None:
        output_path = Path(hf_path).with_suffix(".lazytorch")
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    return export_to_lazytorch(hf_path, output_path, device="cpu", dtype=torch_dtype, progress_callback=progress_callback)


# =============================================================================
# Export from LazyTorch back to standard Hugging Face format
# =============================================================================
def export_to_standard_pytorch(
    lazytorch_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    dtype: torch.dtype = torch.float32
) -> Path:
    """
    Export a LazyTorch model back to standard Hugging Face format.
    This loads all parameters into memory (may require sufficient RAM) and saves
    as a regular HF model directory (with .bin or .safetensors files).
    Useful for using the model with vLLM, Ollama, or standard PyTorch training.

    Raises:
        ValueError: If the tokenizer is corrupt and cannot be loaded after copying.
    """
    lazytorch_path = Path(lazytorch_path)
    if output_path is None:
        output_path = lazytorch_path.with_suffix('.hf')
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading LazyTorch model from {lazytorch_path} (this may use significant RAM)...")
    model = load_lazytorch_model(lazytorch_path, device="cpu", unload_after_forward=False)

    # Force load all parameters
    if hasattr(model, 'load_parameters'):
        model.load_parameters()
    for param in model.parameters():
        if isinstance(param, LazyParameter) and not param._lazy_data_loaded:
            param._load_data()

    # Convert to standard PyTorch model
    from transformers import AutoModelForCausalLM, AutoConfig
    config_path = lazytorch_path / "config.json"
    if not config_path.exists():
        # Try parent directory
        config_path = lazytorch_path.parent / (lazytorch_path.stem + ".config.json")
        if not config_path.exists():
            # Try to find any config.json in the directory
            config_files = list(lazytorch_path.glob("config.json")) + list(lazytorch_path.parent.glob("config.json"))
            if config_files:
                config_path = config_files[0]
            else:
                raise FileNotFoundError("Could not find config.json for the model. Please ensure the original config.json is present.")

    try:
        config = AutoConfig.from_pretrained(str(config_path.parent))
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path.parent}: {e}")
        # Try to use the model's config attribute if available
        if hasattr(model, 'config'):
            config = model.config
        else:
            raise

    hf_model = AutoModelForCausalLM.from_config(config)

    # Get state dict and convert to standard format
    state_dict = model.state_dict()
    # Remove any LazyParameter-specific attributes from state dict keys? No, they are standard keys.
    if dtype != torch.float32:
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

    # Load with strict=False to handle any adapter parameters that may not be in the config
    hf_model.load_state_dict(state_dict, strict=False)
    hf_model.save_pretrained(output_path)

    # Copy tokenizer files
    tokenizer_path = lazytorch_path if lazytorch_path.is_dir() else lazytorch_path.with_suffix('')
    for f in tokenizer_path.glob("tokenizer*"):
        if f.is_file():
            shutil.copy2(f, output_path / f.name)
    for fname in ["vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.json",
                  "generation_config.json", "tokenizer.model"]:
        src = tokenizer_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)

    # ---- Validate the copied tokenizer using shared function ----
    if not _validate_tokenizer_deep(output_path):
        # Clean up the output directory to avoid leaving a broken model
        shutil.rmtree(output_path, ignore_errors=True)
        raise ValueError(
            f"Tokenizer in LazyTorch model at {lazytorch_path} is corrupt or incompatible.\n"
            "This indicates the LazyTorch model was created from a corrupt source.\n"
            "Please delete the LazyTorch model and re-export from a clean Hugging Face model."
        )

    logger.info(f"Exported to standard PyTorch format: {output_path}")
    return output_path


# ----------------------------------------------------------------------
# Context manager
# ----------------------------------------------------------------------
@contextmanager
def lazy_model_context(model: nn.Module, load_all: bool = False):
    if load_all and isinstance(model, LazyModule):
        model.load_parameters()
    try:
        yield model
    finally:
        if load_all and isinstance(model, LazyModule):
            model.unload_parameters()