"""
model_merging.py - Model merging utilities for Lazy Llama.

Provides various merging strategies:
- SLERP (Spherical Linear Interpolation) for two models.
- TIES (TrIm, Elect Sign) merging for multiple models.
- DARE (Drop And REscale) merging for multiple models.
- Helper functions to load models and state dicts.

These are used in the endless RL loop to combine top‑performing models.
"""

import logging
import copy
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelInfo, Config
from .lazy_model_manager import ModelManager

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: Load model state dicts (memory‑efficient)
# =============================================================================
def load_models(
    model_names: List[str],
    manager: ModelManager,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor]]:
    """
    Load state dicts for a list of models.

    Args:
        model_names: Names of models in the registry.
        manager: ModelManager instance.
        device: Device to load tensors onto.

    Returns:
        List of state dicts (order matches model_names).

    Raises:
        ValueError: If any model is not found or lacks a path.
    """
    state_dicts = []
    for name in model_names:
        info = manager.get_model(name)
        if not info or not info.path:
            raise ValueError(f"Model '{name}' not found or missing path.")
        path = Path(info.path)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")

        logger.info(f"Loading {name} for merging...")
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map=device,
        )
        state_dicts.append(model.state_dict())
        # Free model to save memory
        del model
        torch.cuda.empty_cache()

    return state_dicts


# =============================================================================
# SLERP (Spherical Linear Interpolation)
# =============================================================================
def slerp(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    t: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Spherical linear interpolation between two state dicts.

    Args:
        state_dict_a: First state dict.
        state_dict_b: Second state dict.
        t: Interpolation factor (0 = fully a, 1 = fully b).

    Returns:
        Interpolated state dict.
    """
    if t == 0:
        return copy.deepcopy(state_dict_a)
    if t == 1:
        return copy.deepcopy(state_dict_b)

    # Ensure keys match
    keys = set(state_dict_a.keys())
    if set(state_dict_b.keys()) != keys:
        raise ValueError("State dicts have different keys; cannot SLERP.")

    merged = {}
    for key in keys:
        a = state_dict_a[key].float()
        b = state_dict_b[key].float()
        # Normalize vectors for SLERP
        norm_a = torch.norm(a)
        norm_b = torch.norm(b)
        if norm_a == 0 or norm_b == 0:
            # Fallback to linear interpolation
            merged[key] = (1 - t) * a + t * b
            continue

        a_norm = a / norm_a
        b_norm = b / norm_b
        # Compute angle
        dot = torch.sum(a_norm * b_norm)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        if sin_theta == 0:
            # If vectors are parallel, linear interpolation
            merged[key] = (1 - t) * a + t * b
        else:
            # SLERP formula
            w_a = torch.sin((1 - t) * theta) / sin_theta
            w_b = torch.sin(t * theta) / sin_theta
            merged[key] = w_a * a + w_b * b

    return merged


# =============================================================================
# TIES Merging (TrIm, Elect Sign)
# =============================================================================
def ties_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.2,
    elect_sign: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    TIES (TrIm, Elect Sign) merging for multiple models.

    Steps:
    1. Trim: for each parameter, zero out the smallest `trim_ratio` fraction of values
       (by magnitude) across models.
    2. Elect Sign: for each parameter, compute the sign of the sum of trimmed values.
    3. Average: average the trimmed values that agree with the elected sign.

    Args:
        state_dicts: List of state dicts (all must have same keys).
        trim_ratio: Fraction of smallest (in magnitude) values to trim (0.0 to 1.0).
        elect_sign: Whether to elect sign (if False, simply average the trimmed values).

    Returns:
        Merged state dict.
    """
    if not state_dicts:
        raise ValueError("No state dicts provided.")
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if set(sd.keys()) != keys:
            raise ValueError("State dicts have different keys.")

    merged = {}
    for key in keys:
        # Stack all tensors along a new dimension: (num_models, *shape)
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)  # shape: (num_models, ...)
        num_models = stacked.size(0)

        # Flatten to (num_models, num_params)
        flat = stacked.view(num_models, -1)

        # 1. Trim: zero out the smallest trim_ratio fraction of values
        if trim_ratio > 0:
            # Compute absolute values
            abs_flat = torch.abs(flat)
            # Sort along model dimension (ascending)
            sorted_abs, _ = torch.sort(abs_flat, dim=0)  # (num_models, num_params)
            # Determine threshold: value at index floor(trim_ratio * num_models) - 1
            # We want to zero out the smallest `ceil(trim_ratio * num_models)` values.
            # If trim_ratio = 0.2 and num_models=4, we zero out the smallest 0.8 -> 1 value? Actually ceil(0.8)=1, so keep 3.
            # But the TIES paper trims the smallest 20% of values across models, i.e., if 4 models, we zero out 0.8 values per parameter? That's not integer.
            # Usually they set trim_ratio as a fraction of the number of models, e.g., 0.2 means 20% of models' weights are trimmed.
            # With 4 models, trim_ratio=0.2 means we keep 80% of models per parameter? No, the trimming is per-parameter: we keep the largest 80% of values (by magnitude) among the models, and zero out the smallest 20%.
            # So we need to find the value at the trim_ratio quantile (e.g., 20% quantile) and zero out values below it.
            # Using sorted_abs, we take the value at index int(trim_ratio * num_models) as the lower bound.
            idx = int(trim_ratio * num_models)
            if idx == 0:
                threshold = sorted_abs[0, :]  # keep all
            else:
                # The threshold is the value at index idx-1? Actually we want to keep values >= threshold.
                # If we take idx = floor(trim_ratio * num_models), then the idx-th smallest value is the cutoff.
                # Values < cutoff are zeroed.
                threshold = sorted_abs[idx, :]  # but if idx == 0, we keep all.
            # Create mask: keep values with abs >= threshold
            # But threshold is per parameter, so we need to broadcast.
            mask = abs_flat >= threshold.unsqueeze(0)  # (num_models, num_params)
            flat = flat * mask.float()

        # 2. Elect Sign: compute the sign of the sum of the remaining values
        if elect_sign:
            # Sum along model dimension
            sum_flat = torch.sum(flat, dim=0)  # (num_params,)
            # Sign of the sum
            sign = torch.sign(sum_flat)  # (num_params,)
            # Average only values that agree with sign
            # Create mask for values that have the same sign as the elected sign
            # For each parameter, we need to check sign(value) == sign (or value * sign > 0)
            agree_mask = (flat * sign.unsqueeze(0)) > 0  # (num_models, num_params)
            # Sum of agreeing values
            sum_agree = torch.sum(flat * agree_mask.float(), dim=0)  # (num_params,)
            count_agree = torch.sum(agree_mask.float(), dim=0)  # (num_params,)
            # Avoid division by zero
            avg = torch.where(count_agree > 0, sum_agree / count_agree, torch.zeros_like(sum_agree))
            merged_flat = avg
        else:
            # Simply average the trimmed values
            merged_flat = torch.mean(flat, dim=0)  # (num_params,)

        # Reshape back to original shape
        merged[key] = merged_flat.view(stacked.shape[1:])

    return merged


# =============================================================================
# DARE Merging (Drop And REscale)
# =============================================================================
def dare_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    drop_rate: float = 0.1,
    rescale: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    DARE (Drop And REscale) merging for multiple models.

    Steps:
    1. Drop: for each parameter, randomly drop a fraction `drop_rate` of the values.
    2. Rescale: rescale the remaining values by 1 / (1 - drop_rate).
    3. Average: average the resulting values across models.

    Args:
        state_dicts: List of state dicts (same keys).
        drop_rate: Fraction of values to drop (0.0 to 1.0).
        rescale: If True, rescale remaining values.

    Returns:
        Merged state dict.
    """
    if not state_dicts:
        raise ValueError("No state dicts provided.")
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if set(sd.keys()) != keys:
            raise ValueError("State dicts have different keys.")

    merged = {}
    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)  # (num_models, ...)
        num_models = stacked.size(0)

        # Create dropout mask: for each model and each parameter, keep with prob (1 - drop_rate)
        mask = torch.rand_like(stacked) > drop_rate
        # Apply mask: zero out dropped values
        dropped = stacked * mask.float()
        # Rescale: multiply by 1 / (1 - drop_rate)
        if rescale:
            scale = 1.0 / (1.0 - drop_rate)
            dropped = dropped * scale
        # Average over models
        merged[key] = torch.mean(dropped, dim=0)

    return merged


# =============================================================================
# High‑level merge function
# =============================================================================
def merge_models(
    model_names: List[str],
    manager: ModelManager,
    method: str = "average",
    config: Optional[Config] = None,
    **kwargs,
) -> Optional[str]:
    """
    Merge a list of models using the specified method and register the result.

    Args:
        model_names: List of model names.
        manager: ModelManager instance.
        method: One of 'average', 'slerp', 'ties', 'dare'.
        config: Optional Config.
        **kwargs: Additional arguments for the specific method:
            - For 'slerp': t (default 0.5)
            - For 'ties': trim_ratio (default 0.2), elect_sign (default True)
            - For 'dare': drop_rate (default 0.1), rescale (default True)

    Returns:
        Name of the merged model, or None on failure.
    """
    if len(model_names) < 2:
        logger.info("Need at least two models to merge.")
        return model_names[0] if model_names else None

    logger.info(f"Merging {len(model_names)} models using method '{method}'...")

    # Load state dicts
    try:
        state_dicts = load_models(model_names, manager)
    except Exception as e:
        logger.error(f"Failed to load models for merging: {e}")
        return None

    # Perform merge
    try:
        if method == "average":
            keys = set(state_dicts[0].keys())
            merged_state = {}
            for key in keys:
                tensors = [sd[key].float() for sd in state_dicts]
                merged_state[key] = torch.mean(torch.stack(tensors), dim=0)
        elif method == "slerp":
            if len(state_dicts) != 2:
                logger.warning("SLERP requires exactly two models; falling back to average.")
                # Average all state dicts
                keys = set(state_dicts[0].keys())
                merged_state = {}
                for key in keys:
                    tensors = [sd[key].float() for sd in state_dicts]
                    merged_state[key] = torch.mean(torch.stack(tensors), dim=0)
            else:
                t = kwargs.get("t", 0.5)
                merged_state = slerp(state_dicts[0], state_dicts[1], t)
        elif method == "ties":
            trim_ratio = kwargs.get("trim_ratio", 0.2)
            elect_sign = kwargs.get("elect_sign", True)
            merged_state = ties_merge(state_dicts, trim_ratio, elect_sign)
        elif method == "dare":
            drop_rate = kwargs.get("drop_rate", 0.1)
            rescale = kwargs.get("rescale", True)
            merged_state = dare_merge(state_dicts, drop_rate, rescale)
        else:
            logger.error(f"Unknown merge method: {method}")
            return None
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return None

    # Create a new model from the first model's config
    try:
        first_info = manager.get_model(model_names[0])
        if not first_info or not first_info.path:
            raise ValueError(f"Model {model_names[0]} not found.")
        base_model = AutoModelForCausalLM.from_pretrained(first_info.path, low_cpu_mem_usage=True)
        # Create new model with same config
        new_model = AutoModelForCausalLM.from_config(base_model.config)
        new_model.load_state_dict(merged_state, strict=True)
        # Save
        merged_name = f"merged_{int(time.time())}"
        save_dir = manager.models_dir / merged_name
        save_dir.mkdir(parents=True, exist_ok=True)
        new_model.save_pretrained(save_dir)
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(first_info.path)
        tokenizer.save_pretrained(save_dir)

        # Register
        size_mb = sum(f.stat().st_size for f in save_dir.glob("*") if f.is_file()) / (1024 * 1024)
        info = ModelInfo(
            name=merged_name,
            original_size_mb=size_mb,
            path=str(save_dir),
            model_type="local",
            lazytorch_format=False,
            invalid=False,
        )
        with manager._lock:
            if merged_name in manager.registry:
                logger.warning(f"Overwriting existing registry entry for {merged_name}")
            manager.registry[merged_name] = info
            manager._save_registry()

        logger.info(f"Merged model saved as {merged_name}")
        return merged_name

    except Exception as e:
        logger.error(f"Failed to save merged model: {e}")
        return None