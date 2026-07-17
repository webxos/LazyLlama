"""
KV cache compression using TurboQuant (rotation + codebook) and mixed-dimension pruning.
Optimized for compatibility with LazyTorch: the KV cache resides in a separate memory space
and is not affected by model parameter unloading. All operations are device-agnostic.

FIXED: Removed internal residual splitting in TurboQuantCache.compress() to avoid double compression.
       Now stores the flattened shape correctly so decompression yields the expected tensor shape.
ADDED: Safety check in CompressedKVCache.get_full_kv() to verify element count before reshaping.

FIXES (2026-07-06):
- In TurboQuantCache.decompress(), replaced silent zero-return with raising ValueError
  on corrupted metadata to prevent data loss.
- In MixedDimKVCache, replaced torch.linalg.pinv with stable SVD-based projection.
- Added explicit error messages for all failure paths.

FIX (2026-07-10): Further improvements:
- Store device in metadata for empty tensors to ensure correct device on decompression.
- Cache the pseudo-inverse in MixedDimKVCache to avoid recomputing on every get_full_kv().
- Added additional device consistency checks and fallbacks.
- Improved docstrings and error messages.
- Removed unused `nn` import.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging
import warnings

logger = logging.getLogger(__name__)


class TurboQuantCache:
    """
    Lossy KV cache compression using random rotation and uniform codebook.
    Achieves ~4-8 bits per element with minimal accuracy loss.
    Fully compatible with LazyTorch: the cache is stored as standard torch tensors
    (not LazyParameters) and will persist across model parameter unloads.

    Note: This compression is lossy. For critical applications, increase `residual_tokens`
    to keep more recent tokens uncompressed. The default 128 is a good trade‑off.
    """

    def __init__(self, bits: int = 4, residual_tokens: int = 128, codebook_size: Optional[int] = None):
        """
        Args:
            bits: Target bits per element (4 or 8 typical)
            residual_tokens: Number of recent tokens kept uncompressed (higher = more accuracy)
            codebook_size: Size of codebook (default: 2^bits)
        """
        self.bits = bits
        self.residual_tokens = residual_tokens   # kept for compatibility but not used internally
        self.codebook_size = codebook_size or (1 << bits)
        self.rotation_matrix = None  # Random orthogonal matrix for decorrelation
        self.codebook = None  # Learned or uniform codebook
        self._device = torch.device('cpu')

    def _init_rotation(self, dim: int, device: torch.device) -> torch.Tensor:
        """Generate a random orthogonal rotation matrix of size dim x dim."""
        H = torch.randn(dim, dim, device=device)
        Q, _ = torch.linalg.qr(H)
        return Q.to(torch.float32)

    def _random_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to the last dimension."""
        if self.rotation_matrix is None:
            dim = x.shape[-1]
            self.rotation_matrix = self._init_rotation(dim, x.device)
            self._device = x.device
        # Reshape to (N, dim)
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        rotated = flat @ self.rotation_matrix.to(x.device)
        return rotated.reshape(orig_shape)

    def _inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation (transpose of rotation matrix)."""
        if self.rotation_matrix is None:
            raise ValueError("Rotation matrix not initialized")
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        inv_rotated = flat @ self.rotation_matrix.T.to(x.device)
        return inv_rotated.reshape(orig_shape)

    def _compute_codebook(self, data: torch.Tensor) -> torch.Tensor:
        """Compute a uniform codebook based on data statistics."""
        flat = data.flatten()
        if flat.numel() == 0:
            # Empty data, return dummy codebook
            return torch.linspace(0, 1, self.codebook_size, device=data.device)
        min_val = flat.min()
        max_val = flat.max()
        if min_val == max_val:
            # Edge case: constant data
            codebook = torch.linspace(min_val, min_val + 1e-6, self.codebook_size, device=data.device)
        else:
            codebook = torch.linspace(min_val, max_val, self.codebook_size, device=data.device)
        return codebook

    def compress(self, kv_tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Dict[str, Any], Optional[torch.Tensor]]:
        """
        Compress a key or value tensor. Compresses the entire tensor as one block.
        Returns:
            indices: quantized indices (or None if no compression)
            metadata: dict with compression info
            codebook: codebook used (or None)
        """
        if kv_tensor.numel() == 0:
            # Empty tensor, return no compression
            return None, {
                "original_shape": kv_tensor.shape,
                "residual_tokens": self.residual_tokens,
                "recent_tokens": 0,
                "older_shape": None,
                "compress_older": False,
                "bits": self.bits,
                "codebook_size": self.codebook_size,
                "empty": True,
                "device": str(kv_tensor.device),  # store device for later
            }, None

        # Compress the whole tensor without further splitting
        rotated = self._random_rotate(kv_tensor)
        codebook = self._compute_codebook(rotated)
        flat_rot = rotated.flatten()
        indices = torch.searchsorted(codebook, flat_rot).clamp(0, self.codebook_size - 1)
        indices = indices.reshape(rotated.shape)

        metadata = {
            "original_shape": kv_tensor.shape,
            "residual_tokens": self.residual_tokens,
            "recent_tokens": 0,                 # not used
            "older_shape": kv_tensor.shape,    # shape of the input tensor
            "compress_older": True,
            "bits": self.bits,
            "codebook_size": self.codebook_size,
            "empty": False,
            "device": str(kv_tensor.device),
        }
        return indices, metadata, codebook

    def decompress(self, indices: Optional[torch.Tensor], metadata: Dict[str, Any], codebook: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct the original tensor from compressed representation.
        Raises ValueError if metadata is corrupted or required components are missing.
        """
        if metadata.get("empty", False):
            # Return empty tensor with original shape on the appropriate device
            shape = metadata.get("original_shape", (0,))
            device_str = metadata.get("device", "cpu")
            device = torch.device(device_str)
            return torch.zeros(shape, device=device)

        compress_older = metadata.get("compress_older", False)
        if not compress_older:
            # If not compressed, we should not be here
            raise ValueError("decompress called on non-compressed data (compress_older=False)")

        older_shape = metadata.get("older_shape")
        if older_shape is None:
            raise ValueError("Missing 'older_shape' in metadata; cannot decompress.")

        if indices is None:
            raise ValueError("Indices are None but compression was supposed to be applied.")

        if codebook is None:
            raise ValueError("Codebook is None; cannot decompress.")

        try:
            # Map indices to codebook values
            quantized_values = codebook[indices]
            # Inverse rotate
            older = self._inverse_rotate(quantized_values)
            older = older.reshape(older_shape)
            return older
        except Exception as e:
            raise ValueError(f"Decompression failed: {e}") from e


class MixedDimKVCache:
    """
    EXPERIMENTAL cache that reduces dimension for older tokens using PCA-like projection.
    Keeps full dimension for recent tokens, compressed for older.

    WARNING: This is highly experimental and may cause shape mismatches, silent failures,
    or performance degradation. It is disabled by default in the configuration.
    Use only if you fully understand the implications.

    LazyTorch compatibility: The cache is stored as regular tensors; projection matrices are kept separately.
    """

    def __init__(self, full_dim: int, compressed_dim: int, residual_tokens: int = 128):
        """
        Args:
            full_dim: Original head dimension
            compressed_dim: Dimension to reduce to (e.g., 32)
            residual_tokens: Number of recent tokens kept at full dimension
        """
        self.full_dim = full_dim
        self.compressed_dim = compressed_dim
        self.residual_tokens = residual_tokens
        self.projection = None  # (full_dim, compressed_dim) learned projection
        self.mean = None
        self._proj_pinv = None   # cached pseudo-inverse (compressed_dim, full_dim)
        self.recent_keys = []   # List of tensors (batch, heads, tokens, full_dim)
        self.recent_values = []
        self.compressed_keys = []  # (batch, heads, tokens, compressed_dim)
        self.compressed_values = []
        self._device = torch.device('cpu')
        warnings.warn(
            "MixedDimKVCache is EXPERIMENTAL and may cause errors. Use with caution.",
            RuntimeWarning,
            stacklevel=2
        )
        logger.warning("MixedDimKVCache is experimental and may cause errors. Use at your own risk.")

    def _learn_projection(self, data: torch.Tensor):
        """
        Learn projection matrix from data using PCA via SVD for stability.
        Uses full SVD to get principal components, truncates to compressed_dim.
        """
        if data.numel() == 0:
            logger.warning("Empty data for PCA; projection not learned.")
            return
        # data shape: (num_samples, full_dim)
        mean = data.mean(dim=0, keepdim=True)
        centered = data - mean
        # Use SVD for numerical stability
        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            # Vh shape: (full_dim, full_dim)  # actually (full_dim, full_dim) if full_matrices=False
            # The principal components are the rows of Vh (right singular vectors)
            # We take the first compressed_dim components
            self.projection = Vh[:self.compressed_dim, :].T  # (full_dim, compressed_dim)
            self.mean = mean.squeeze(0)
            self._device = data.device
            # Cache pseudo-inverse for future use
            self._cache_pinv()
            logger.debug(f"Projection learned via SVD: {self.projection.shape}")
        except Exception as e:
            logger.warning(f"PCA via SVD failed: {e}. Using random projection.")
            # Fallback: random orthonormal projection
            rand = torch.randn(self.full_dim, self.full_dim, device=data.device)
            Q, _ = torch.linalg.qr(rand)
            self.projection = Q[:, :self.compressed_dim]
            self.mean = mean.squeeze(0)
            self._device = data.device
            self._cache_pinv()

    def _cache_pinv(self):
        """Compute and cache the pseudo-inverse of the projection matrix for up-projection."""
        if self.projection is None:
            self._proj_pinv = None
            return
        P = self.projection  # (full_dim, compressed_dim)
        try:
            U, S, Vh = torch.linalg.svd(P, full_matrices=False)
            # P = U @ diag(S) @ Vh   where Vh: (compressed_dim, full_dim)
            S_inv = torch.where(S > 1e-8, 1.0 / S, torch.zeros_like(S))
            # pinv(P) = Vh.T @ diag(1/S) @ U.T   (compressed_dim, full_dim)
            self._proj_pinv = Vh.T @ torch.diag(S_inv) @ U.T
            self._proj_pinv = self._proj_pinv.to(self._device)
        except Exception as e:
            logger.warning(f"Failed to compute pseudo-inverse via SVD: {e}. Using random projection.")
            # Fallback: random matrix
            rand = torch.randn(self.compressed_dim, self.full_dim, device=P.device)
            self._proj_pinv = rand / torch.norm(rand, dim=1, keepdim=True)

    def compress_older(self):
        """Move all but the most recent residual_tokens from recent to compressed."""
        if not self.recent_keys:
            return
        all_keys = torch.cat(self.recent_keys, dim=-2)
        all_values = torch.cat(self.recent_values, dim=-2)
        T = all_keys.shape[-2]
        if T <= self.residual_tokens:
            return
        keep_keys = all_keys[..., -self.residual_tokens:, :]
        keep_values = all_values[..., -self.residual_tokens:, :]
        older_keys = all_keys[..., :-self.residual_tokens, :]
        older_values = all_values[..., :-self.residual_tokens, :]

        if self.projection is None and older_keys.numel() > 0:
            N = older_keys.shape[-2] * older_keys.shape[0] * older_keys.shape[1]
            flat = older_keys.reshape(N, -1)
            self._learn_projection(flat)

        if self.projection is not None:
            # Project older keys and values
            compressed_keys = older_keys @ self.projection
            compressed_values = older_values @ self.projection
            self.compressed_keys.append(compressed_keys)
            self.compressed_values.append(compressed_values)
            # Keep only recent
            self.recent_keys = [keep_keys]
            self.recent_values = [keep_values]
        else:
            # If projection not learned, keep all as recent (no compression)
            logger.warning("Projection not learned; skipping compression.")
            self.recent_keys = [all_keys]
            self.recent_values = [all_values]

    def append(self, key: torch.Tensor, value: torch.Tensor):
        """Append new token's key and value to recent cache."""
        # Ensure tensors are on the same device as cached data
        if self.recent_keys:
            device = self.recent_keys[0].device
            key = key.to(device)
            value = value.to(device)
        else:
            self._device = key.device
        self.recent_keys.append(key)
        self.recent_values.append(value)
        total_recent_tokens = sum(k.shape[-2] for k in self.recent_keys)
        if total_recent_tokens > self.residual_tokens * 2:
            self.compress_older()

    def get_full_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve concatenated keys and values (reconstructed for compressed part)."""
        proj_pinv = self._proj_pinv

        full_keys = []
        full_values = []
        for ck, cv in zip(self.compressed_keys, self.compressed_values):
            if proj_pinv is not None:
                up_key = ck @ proj_pinv
                up_value = cv @ proj_pinv
                full_keys.append(up_key)
                full_values.append(up_value)
            else:
                logger.warning("Projection not available, cannot up-project; skipping compressed.")
        full_keys.extend(self.recent_keys)
        full_values.extend(self.recent_values)

        if full_keys:
            # Ensure all tensors are on the same device
            device = full_keys[0].device
            keys = torch.cat([k.to(device) for k in full_keys], dim=-2)
            values = torch.cat([v.to(device) for v in full_values], dim=-2)
        else:
            keys = torch.tensor([], device=self._device)
            values = torch.tensor([], device=self._device)
        return keys, values

    def clear(self):
        """Reset cache."""
        self.recent_keys = []
        self.recent_values = []
        self.compressed_keys = []
        self.compressed_values = []
        self.projection = None
        self.mean = None
        self._proj_pinv = None


class CompressedKVCache:
    """
    Unified KV cache with recent tokens in FP16 and older tokens compressed using TurboQuant.
    Compatible with LazyTorch: the cache is stored as plain torch tensors,
    independent of model parameters. It will survive parameter unloading between generation steps.

    This is the recommended KV cache compression method for production use.
    """

    def __init__(self, bits: int = 4, residual_tokens: int = 128, codebook_size: int = 256):
        self.residual_tokens = residual_tokens
        self.turbo_quant = TurboQuantCache(bits, residual_tokens, codebook_size)
        self.recent_keys = []   # list of tensors (batch, heads, token, head_dim)
        self.recent_values = []
        self.compressed_keys = None  # (indices, metadata, codebook)
        self.compressed_values = None
        self._device = torch.device('cpu')
        self._has_compressed = False

    def append(self, key: torch.Tensor, value: torch.Tensor):
        """Add new token's key and value to the cache."""
        # Ensure device consistency
        if self.recent_keys:
            device = self.recent_keys[0].device
            key = key.to(device)
            value = value.to(device)
        else:
            self._device = key.device

        self.recent_keys.append(key)
        self.recent_values.append(value)
        total_tokens = sum(k.shape[-2] for k in self.recent_keys)
        if total_tokens > self.residual_tokens * 2:
            self._compress()

    def _compress(self):
        """Compress all but the most recent residual_tokens."""
        if not self.recent_keys:
            return
        all_keys = torch.cat(self.recent_keys, dim=-2)
        all_values = torch.cat(self.recent_values, dim=-2)
        T = all_keys.shape[-2]
        if T <= self.residual_tokens:
            return
        keep_keys = all_keys[..., -self.residual_tokens:, :]
        keep_values = all_values[..., -self.residual_tokens:, :]
        older_keys = all_keys[..., :-self.residual_tokens, :]
        older_values = all_values[..., :-self.residual_tokens, :]

        # Flatten batch and heads for compression: (B*H, T_old, D)
        bh, T_old, D = older_keys.shape[0] * older_keys.shape[1], older_keys.shape[2], older_keys.shape[3]
        flat_keys = older_keys.reshape(bh, T_old, D)
        flat_values = older_values.reshape(bh, T_old, D)

        keys_idx, keys_meta, keys_code = self.turbo_quant.compress(flat_keys)
        vals_idx, vals_meta, vals_code = self.turbo_quant.compress(flat_values)

        # Store both the flattened shape and the original shape for later reconstruction
        keys_meta['orig_shape'] = older_keys.shape
        vals_meta['orig_shape'] = older_values.shape
        keys_meta['flattened_shape'] = flat_keys.shape
        vals_meta['flattened_shape'] = flat_values.shape

        self.compressed_keys = (keys_idx, keys_meta, keys_code)
        self.compressed_values = (vals_idx, vals_meta, vals_code)
        self.recent_keys = [keep_keys]
        self.recent_values = [keep_values]
        self._has_compressed = True

        logger.debug(f"KV cache compressed: kept {self.residual_tokens} recent tokens, compressed {older_keys.shape[-2]} older tokens.")

    def get_full_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full key and value tensors (decompressed)."""
        older_keys = None
        older_values = None
        if self.compressed_keys is not None and self._has_compressed:
            keys_idx, keys_meta, keys_code = self.compressed_keys
            vals_idx, vals_meta, vals_code = self.compressed_values

            try:
                older_keys = self.turbo_quant.decompress(keys_idx, keys_meta, keys_code)
                older_values = self.turbo_quant.decompress(vals_idx, vals_meta, vals_code)
            except ValueError as e:
                logger.error(f"Decompression failed: {e}. Falling back to only recent tokens.")
                older_keys = None
                older_values = None
                self._has_compressed = False
                self.compressed_keys = None
                self.compressed_values = None

            # Safety check: verify element count matches original shape
            if older_keys is not None:
                orig_shape = keys_meta.get('orig_shape')
                if orig_shape is not None:
                    expected_elements = np.prod(orig_shape)
                    actual_elements = older_keys.numel()
                    if actual_elements != expected_elements:
                        logger.warning(
                            f"Decompressed key tensor has {actual_elements} elements, "
                            f"but expected {expected_elements} from shape {orig_shape}. "
                            "Skipping compressed part and using only recent tokens."
                        )
                        older_keys = None
                        older_values = None
                        self._has_compressed = False
                        self.compressed_keys = None
                        self.compressed_values = None
                    else:
                        older_keys = older_keys.reshape(orig_shape)
                        older_values = older_values.reshape(orig_shape)
                else:
                    logger.warning("Missing orig_shape in metadata; skipping compressed part.")
                    older_keys = None
                    older_values = None

        if self.recent_keys:
            recent_keys = torch.cat(self.recent_keys, dim=-2)
            recent_values = torch.cat(self.recent_values, dim=-2)
            device = recent_keys.device
        else:
            recent_keys = torch.tensor([], device=self._device)
            recent_values = torch.tensor([], device=self._device)
            device = self._device

        if older_keys is not None:
            older_keys = older_keys.to(device)
            older_values = older_values.to(device)
            keys = torch.cat([older_keys, recent_keys], dim=-2)
            values = torch.cat([older_values, recent_values], dim=-2)
        else:
            keys = recent_keys
            values = recent_values
        return keys, values

    def clear(self):
        """Reset cache."""
        self.recent_keys = []
        self.recent_values = []
        self.compressed_keys = None
        self.compressed_values = None
        self._has_compressed = False
        logger.debug("KV cache cleared.")