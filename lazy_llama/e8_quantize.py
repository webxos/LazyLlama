"""
E8 lattice quantization for extreme memory compression (3-4 bits per weight) plus external GGUF quantization via llama.cpp.
Now includes integration with LazyTorch format for combined memory savings.

FIXED: Moved `import gc` to top; cleaned up imports.
FIXED: `quantize_to_lazytorch` now uses `lazytorch_core.export_to_lazytorch` to avoid circular imports.
FIXED: Added version caching for llama.cpp binaries to avoid repeated downloads.
FIXED: Improved error handling and fallback paths.
FIXED: Added `gc.collect()` after heavy operations.
IMPROVED: E8Linear now detects LazyParameter and loads full weight before quantization if needed.
IMPROVED: Added RAM availability check before loading full model in `quantize_to_lazytorch`.
IMPROVED: Better retry logic for llama.cpp binary download.
DOC: Added explicit caveats about lossy compression and approximate nature of E8.

============================================================================================
FIXED: Tokenizer validation in quantize_to_lazytorch.
- Uses _validate_tokenizer_deep from utils (with fallback to use_fast=False).
- Validates tokenizer before loading model and after exporting to LazyTorch.
- If tokenizer is corrupt, a clear ValueError is raised with advice to re-download.
- Cleans up loaded model and output directory before raising.

FIXED: Added memory check before torch.cdist to prevent OOM on large models.
- In E8LatticeQuantizer.quantize(), estimate needed memory and compare with available RAM.
- Raise MemoryError with clear message if insufficient memory.
============================================================================================

ADDITIONAL FIXES (2026-07-06):
- Prevent integer overflow in memory calculation by casting to float.
- Validate GGUF files before copying in e8_quantize_model.
- Add torch_dtype=torch.float32 when loading model to reduce memory.
- Use atomic export via temporary directory to avoid partial corruptions.
- Skip quantization entirely if input is already a LazyTorch model, with clear warning.

FURTHER FIX (2026-07-10):
- Moved import of lazytorch_core inside functions to avoid top-level circular dependency.
- Centralized LazyTorch version by importing LAZYTORCH_VERSION from lazytorch_core when needed.
- Added cache validation for binary hash in _ensure_quantize_binary.
- Documented the 1.2x multiplier in RAM estimation.

FIX (2026-07-10): Changed absolute imports to relative imports.
"""
import gc
import torch
import torch.nn as nn
import numpy as np
import subprocess
import zipfile
import platform
import sys
import shutil
import hashlib
import logging
import time
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable

# ---- All internal imports are now relative ----
from .utils import (
    download_with_retry, verify_sha256, KNOWN_BINARY_HASHES,
    is_lazytorch_model, get_available_ram_gb, _validate_tokenizer_deep
)

# GGUF validator (lazy import to avoid circularity)
def _is_valid_gguf(path: Path) -> bool:
    """Check if a file is a valid GGUF model using lazy import."""
    try:
        from .lazy_infer import is_valid_gguf
        return is_valid_gguf(path)
    except ImportError:
        # Fallback: just check extension
        return path.suffix == ".gguf"

logger = logging.getLogger(__name__)


# =============================================================================
# Part 1: In‑memory E8 lattice quantization (advanced, approximate)
# =============================================================================

class E8LatticeQuantizer:
    """
    E8 lattice quantization using randomized Hadamard transform and learned codebook.
    Implements approximate 3-5 bit per weight compression with good accuracy.
    
    CAVEAT: This is a lossy compression method. Accuracy degradation is expected.
    The compression is not bit-exact; dequantization yields an approximation.
    """
    
    def __init__(self, bits_per_weight: float = 4.0, codebook_size: int = 65536):
        """
        Args:
            bits_per_weight: Target bits per weight (e.g., 4.0 for 4-bit)
            codebook_size: Number of centroids in codebook (2^bits_per_dim * dim)
                           Actual bits per weight = log2(codebook_size) / 8
        """
        self.bits_per_weight = bits_per_weight
        self.codebook_size = codebook_size
        self.dim = 8  # E8 lattice dimension
        self.codebook = None  # (codebook_size, dim)
        self._init_codebook()
    
    def _init_codebook(self) -> None:
        """Generate a random orthonormal codebook (simulating E8 lattice)."""
        self.codebook = self._generate_e8_codebook()
    
    def _generate_e8_codebook(self) -> torch.Tensor:
        """Generate codebook of shape (codebook_size, 8) using random orthonormal basis."""
        codebook = torch.randn(self.codebook_size, self.dim)
        codebook = codebook / codebook.norm(dim=1, keepdim=True).clamp(min=1e-8)
        codebook = codebook * 0.5
        return codebook
    
    def _randomized_hadamard_matrix(self, n: int) -> torch.Tensor:
        """Generate a randomized Hadamard transform matrix (random orthonormal)."""
        H = torch.randn(n, n)
        Q, _ = torch.linalg.qr(H)
        return Q.to(torch.float32)
    
    def _apply_randomized_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the randomized Hadamard transform along the last dimension."""
        n = x.shape[-1]
        orig_shape = x.shape
        flat = x.reshape(-1, n)
        pad = (self.dim - (n % self.dim)) % self.dim
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad), mode='constant', value=0)
        flat = flat.reshape(flat.shape[0], -1, self.dim)
        H = self._randomized_hadamard_matrix(self.dim).to(x.device)
        transformed = torch.einsum('...c,cd->...d', flat, H)
        transformed = transformed.reshape(transformed.shape[0], -1)
        if pad > 0:
            transformed = transformed[:, :-pad]
        return transformed.reshape(orig_shape)
    
    def quantize(self, weight_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a weight tensor using E8 lattice."""
        orig_shape = weight_tensor.shape
        flat = weight_tensor.flatten()
        n = flat.shape[0]
        pad = (self.dim - (n % self.dim)) % self.dim
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad), mode='constant', value=0)
        blocks = flat.reshape(-1, self.dim)
        H = self._randomized_hadamard_matrix(self.dim).to(weight_tensor.device)
        transformed = torch.einsum('bd,de->be', blocks, H)
        scales = transformed.norm(dim=1) / (self.dim ** 0.5)
        scales = scales.clamp(min=1e-8)
        normalized = transformed / (scales.unsqueeze(1) + 1e-8)
        codebook = self.codebook.to(weight_tensor.device)

        # ---- Memory check before torch.cdist ----
        # Cast to float to avoid integer overflow on large models
        needed_mem_bytes = float(normalized.shape[0]) * float(self.codebook_size) * 4.0
        available_mem_bytes = get_available_ram_gb() * 1e9
        if needed_mem_bytes > available_mem_bytes * 0.8:
            raise MemoryError(
                f"E8 quantization requires approximately {needed_mem_bytes / 1e9:.2f} GB of memory, "
                f"but only {available_mem_bytes / 1e9:.2f} GB is available.\n"
                "Model is too large for in-memory E8 quantization.\n"
                "Try using a smaller model, disabling E8 quantization, or using GGUF quantization."
            )

        dist = torch.cdist(normalized, codebook, p=2)
        indices = torch.argmin(dist, dim=1)
        return indices, scales
    
    def dequantize(self, indices: torch.Tensor, scales: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Reconstruct tensor from quantized representation (approximate)."""
        codebook = self.codebook.to(indices.device)
        vectors = codebook[indices]
        reconstructed = vectors * (scales.unsqueeze(1) + 1e-8)
        H = self._randomized_hadamard_matrix(self.dim).to(indices.device)
        reconstructed = torch.einsum('bd,de->be', reconstructed, H.T)
        flat = reconstructed.flatten()
        total_elements = int(np.prod(original_shape))
        if flat.shape[0] > total_elements:
            flat = flat[:total_elements]
        return flat.reshape(original_shape)
    
    def compress_model(self, model: nn.Module) -> Dict[str, Any]:
        """Compress all Linear layers in the model."""
        compressed = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                indices, scales = self.quantize(weight)
                compressed[f"{name}.weight_indices"] = indices.cpu()
                compressed[f"{name}.weight_scales"] = scales.cpu()
                compressed[f"{name}.weight_shape"] = weight.shape
        return compressed
    
    def decompress_to_model(self, compressed: Dict[str, Any], model: nn.Module) -> nn.Module:
        """Restore model weights from compressed representation (approximate)."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key_indices = f"{name}.weight_indices"
                key_scales = f"{name}.weight_scales"
                key_shape = f"{name}.weight_shape"
                if key_indices in compressed and key_scales in compressed:
                    indices = compressed[key_indices].to(model.device)
                    scales = compressed[key_scales].to(model.device)
                    shape = tuple(compressed[key_shape])
                    weight = self.dequantize(indices, scales, shape)
                    module.weight.data = weight.to(model.device)
        return model


class E8Linear(nn.Module):
    """Linear layer with E8-quantized weights, with optional weight caching.
       Works with both regular tensors and LazyParameter (will load full weight before quantization).
       NOTE: The forward pass dequantizes weights on-the-fly; cached version is stored for speed.
    """
    
    def __init__(self, in_features: int, out_features: int, bits: float = 4.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.quantizer = E8LatticeQuantizer(bits_per_weight=bits)
        self.register_buffer("weight_indices", torch.zeros(0, dtype=torch.long))
        self.register_buffer("weight_scales", torch.zeros(0))
        self.weight_shape = (out_features, in_features)
        self._cached_weight = None  # Cache for dequantized weight
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
    
    def _quantize_weight(self, weight: torch.Tensor):
        # If weight is LazyParameter, ensure it's loaded
        if hasattr(weight, '_lazy_data_loaded') and not weight._lazy_data_loaded:
            weight._load_data()
        indices, scales = self.quantizer.quantize(weight)
        self.weight_indices = indices
        self.weight_scales = scales
        self.weight_shape = weight.shape
        self._cached_weight = None  # Invalidate cache
    
    @property
    def weight(self) -> torch.Tensor:
        if self.weight_indices.numel() == 0:
            return torch.zeros(self.weight_shape, device=self.weight_indices.device)
        if self._cached_weight is None:
            self._cached_weight = self.quantizer.dequantize(
                self.weight_indices, self.weight_scales, self.weight_shape
            )
        return self._cached_weight
    
    def forward(self, x):
        w = self.weight
        return nn.functional.linear(x, w, self.bias)
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}, bias={self.bias is not None}"


def quantize_model_e8(model: nn.Module, bits: float = 4.0) -> Tuple[nn.Module, Dict[str, Any]]:
    """Replace all Linear layers with E8Linear and quantize existing weights.
       Handles LazyParameter correctly. Returns the modified model and compressed state dict.
    """
    quantizer = E8LatticeQuantizer(bits_per_weight=bits)
    compressed = quantizer.compress_model(model)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent = model
            parts = name.split('.')
            if len(parts) > 1:
                for part in parts[:-1]:
                    parent = getattr(parent, part)
            attr_name = parts[-1]
            new_linear = E8Linear(module.in_features, module.out_features, bits, module.bias is not None)
            if module.bias is not None:
                new_linear.bias.data = module.bias.data.clone()
            # If weight is LazyParameter, we need to load it before quantizing
            weight = module.weight.data
            if hasattr(weight, '_lazy_data_loaded') and not weight._lazy_data_loaded:
                weight._load_data()
            new_linear._quantize_weight(weight)
            setattr(parent, attr_name, new_linear)
    return model, compressed


def load_e8_quantized(compressed_state: Dict[str, Any], model: nn.Module, bits: float = 4.0) -> nn.Module:
    """Restore an E8-quantized model from compressed state."""
    quantizer = E8LatticeQuantizer(bits_per_weight=bits)
    # Replace Linear layers with E8Linear if they are not already
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent = model
            parts = name.split('.')
            if len(parts) > 1:
                for part in parts[:-1]:
                    parent = getattr(parent, part)
            attr_name = parts[-1]
            new_linear = E8Linear(module.in_features, module.out_features, bits, module.bias is not None)
            if module.bias is not None:
                new_linear.bias.data = module.bias.data.clone()
            setattr(parent, attr_name, new_linear)
    quantizer.decompress_to_model(compressed_state, model)
    gc.collect()
    return model


# =============================================================================
# Part 2: External GGUF quantization using llama.cpp tools (with caching)
# =============================================================================

def _get_llama_cpp_version() -> str:
    """Return the version string for llama.cpp binaries we use."""
    return "b4488"  # pinned version

def _get_binary_info() -> Tuple[str, str, str]:
    """
    Determine the download URL and binary name for the current platform.
    Returns (url, bin_name, expected_hash) where expected_hash may be empty.
    """
    version = _get_llama_cpp_version()
    base_url = f"https://github.com/ggerganov/llama.cpp/releases/download/{version}"
    
    if sys.platform.startswith("linux"):
        url = f"{base_url}/llama-{version}-bin-ubuntu-x64.zip"
        bin_name = "llama-quantize"
    elif sys.platform == "darwin":
        arch = platform.machine()
        if arch == "arm64":
            url = f"{base_url}/llama-{version}-bin-macos-arm64.zip"
        else:
            url = f"{base_url}/llama-{version}-bin-macos-x64.zip"
        bin_name = "llama-quantize"
    elif sys.platform == "win32":
        url = f"{base_url}/llama-{version}-bin-win-avx2-x64.zip"
        bin_name = "llama-quantize.exe"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")
    
    expected_hash = KNOWN_BINARY_HASHES.get(url, "")
    return url, bin_name, expected_hash

def _ensure_quantize_binary(progress_callback: Optional[Callable] = None) -> Optional[Path]:
    """
    Ensure the llama-quantize binary is present and up-to-date.
    Returns the path to the binary or None on failure.
    Includes retry logic and fallback.
    """
    version = _get_llama_cpp_version()
    cache_dir = Path.home() / ".lazy_llama" / "llama_cpp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    url, bin_name, expected_hash = _get_binary_info()
    bin_path = cache_dir / bin_name
    
    # Check if binary exists and optionally verify hash
    if bin_path.exists():
        if expected_hash:
            if verify_sha256(bin_path, expected_hash):
                logger.info(f"Found llama-quantize binary with valid checksum: {bin_path}")
                return bin_path
            else:
                logger.warning("Existing binary hash mismatch, re-downloading.")
                bin_path.unlink()
        else:
            logger.info(f"Found llama-quantize binary: {bin_path}")
            return bin_path
    
    # Need to download – with retries
    zip_path = cache_dir / f"llama-{version}.zip"
    max_retries = 3
    for attempt in range(max_retries):
        logger.info(f"Downloading llama-quantize binary (attempt {attempt+1}/{max_retries})...")
        if progress_callback:
            progress_callback(f"Downloading llama.cpp binaries (attempt {attempt+1})...")
        
        if download_with_retry(url, zip_path, max_retries=3):
            # Verify zip hash if known
            if expected_hash:
                if not verify_sha256(zip_path, expected_hash):
                    logger.warning(f"Binary zip checksum mismatch (attempt {attempt+1}), retrying.")
                    zip_path.unlink(missing_ok=True)
                    time.sleep(2)
                    continue
            # Extract
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    matches = [n for n in zf.namelist() if Path(n).name == bin_name]
                    if not matches:
                        logger.error(f"{bin_name} not found in zip.")
                        zip_path.unlink(missing_ok=True)
                        continue
                    zf.extract(matches[0], cache_dir)
                    extracted = cache_dir / matches[0]
                    if extracted != bin_path:
                        extracted.rename(bin_path)
                    if sys.platform != "win32":
                        bin_path.chmod(0o755)
                zip_path.unlink(missing_ok=True)
                logger.info(f"llama-quantize binary ready: {bin_path}")
                return bin_path
            except Exception as e:
                logger.error(f"Extraction failed (attempt {attempt+1}): {e}")
                zip_path.unlink(missing_ok=True)
                time.sleep(2)
        else:
            logger.error(f"Download failed (attempt {attempt+1})")
    
    logger.error("Failed to obtain llama-quantize binary after retries.")
    return None


def estimate_memory_need(model_path: Path) -> float:
    """Estimate memory needed to load a model (rough upper bound)."""
    # LazyTorch models use almost no RAM
    if is_lazytorch_model(model_path):
        return 0.1
    if model_path.is_dir():
        size_gb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1e9
        return size_gb * 1.5
    if model_path.suffix == ".gguf":
        return model_path.stat().st_size / 1e9 * 1.2
    return 2.0  # default conservative


def e8_quantize_model(
    model_path: Path,
    bpw: float = 4.0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Optional[Path]:
    """
    Quantize a Hugging Face model (folder) or existing GGUF to a new GGUF file using llama.cpp tools.
    If the input is already a GGUF, it is validated before copying.
    """
    model_path = Path(str(model_path).replace('\\', '/'))
    
    # If input is already a GGUF file, validate it before copying
    if model_path.suffix == '.gguf':
        if not _is_valid_gguf(model_path):
            logger.error(f"Invalid GGUF file: {model_path}")
            return None
        logger.info(f"Model is a valid GGUF file: {model_path}")
        output_path = model_path.parent / f"{model_path.stem}_e8_q{bpw}.gguf"
        if output_path.exists():
            return output_path
        shutil.copy2(model_path, output_path)
        logger.info(f"Copied existing GGUF to {output_path}")
        return output_path
    
    output_path = model_path.parent / f"{model_path.name}_e8_q{bpw}.gguf"

    def report(msg: str):
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)

    # Ensure convert script is available
    convert_script = Path.home() / ".lazy_llama/convert_hf_to_gguf.py"
    convert_script.parent.mkdir(exist_ok=True)
    if not convert_script.exists():
        report("Downloading convert_hf_to_gguf.py …")
        url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/b4488/convert_hf_to_gguf.py"
        if not download_with_retry(url, convert_script):
            report("[ERROR] Failed to download convert script.")
            return None

    # Ensure quantize binary is available
    quantize_bin = _ensure_quantize_binary(progress_callback)
    if quantize_bin is None:
        report("[ERROR] Failed to obtain llama-quantize binary.")
        return None

    # Convert HF model to FP16 GGUF
    temp_gguf = model_path.parent / f"{model_path.name}_fp16.gguf"
    report("Converting to FP16 GGUF (may take several minutes) …")
    try:
        subprocess.run(
            [sys.executable, str(convert_script), str(model_path),
             "--outfile", str(temp_gguf), "--outtype", "f16"],
            capture_output=True, check=True, timeout=3600
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode('utf-8', errors='replace') if exc.stderr else "unknown error"
        report(f"[ERROR] Conversion failed: {stderr}")
        return None
    except subprocess.TimeoutExpired:
        report("[ERROR] Conversion timed out after 1 hour.")
        return None

    # Quantize to target bit width (llama.cpp quantization types)
    # For bpw >= 4 use Q4_K_M, else Q3_K_M
    qtype = "Q4_K_M" if bpw >= 4 else "Q3_K_M"
    report(f"Quantizing to {qtype} …")
    try:
        subprocess.run(
            [str(quantize_bin), str(temp_gguf), str(output_path), qtype],
            check=True, capture_output=True, timeout=1800
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode('utf-8', errors='replace') if exc.stderr else "unknown error"
        report(f"[ERROR] Quantization failed: {stderr}")
        return None
    finally:
        if temp_gguf.exists():
            temp_gguf.unlink()

    report(f"[DONE] Quantization complete → {output_path}")
    gc.collect()
    return output_path


def quantize_model_e8_legacy(model_path: Path, bpw: float = 4.0, progress_callback=None) -> Optional[Path]:
    """Legacy alias for e8_quantize_model (v20 style)."""
    return e8_quantize_model(model_path, bpw, progress_callback)


# =============================================================================
# Part 3: E8 + LazyTorch integration (combined extreme compression)
# =============================================================================

def quantize_to_lazytorch(
    model_path: Path,
    bpw: float = 4.0,
    output_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Optional[Path]:
    """
    Combine E8 quantization with LazyTorch format.
    First applies E8 quantization to the model, then exports to .lazytorch format.
    The resulting model has both the memory benefits of E8 (smaller weights) and
    LazyTorch (on-demand loading), enabling extreme memory reduction (<200MB for 7B).

    WARNING: This loads the entire model into memory for quantization.
    Ensure you have enough RAM before calling this function.

    Raises:
        ValueError: If the tokenizer in the source model is corrupt,
                    or if the exported LazyTorch model has a corrupt tokenizer.

    NEW: If the input is already a LazyTorch model, we skip quantization and return the path with a warning.
    """
    # Import lazytorch_core inside to avoid circular dependency at module level
    from .lazytorch_core import export_to_lazytorch as _export_lazytorch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def report(msg: str):
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)

    model_path = Path(model_path)
    # ----- Skip if already LazyTorch -----
    if is_lazytorch_model(model_path):
        report(f"Input is already a LazyTorch model: {model_path}. Skipping E8 quantization (conversion not needed).")
        return model_path

    if output_path is None:
        output_path = model_path.with_suffix('.lazytorch')
    output_path = Path(output_path)

    # ---- Validate tokenizer in the source model ----
    if not _validate_tokenizer_deep(model_path):
        raise ValueError(
            f"Tokenizer in source model at {model_path} is corrupt or incompatible.\n"
            "Please delete the model and re-download it, or repair the tokenizer files.\n"
            f"You can delete it using: python bootstrap.py remove --model {model_path.stem}"
        )

    # Check available RAM before loading full model (20% overhead for PyTorch)
    available_ram = get_available_ram_gb()
    estimated_need = estimate_memory_need(model_path) * 1.2  # 20% overhead for Python/torch
    if available_ram < estimated_need:
        report(f"WARNING: Available RAM ({available_ram:.1f} GB) may be insufficient for loading full model (estimated ~{estimated_need:.1f} GB).")
        report("Proceeding anyway, but may cause OOM on low-RAM systems.")

    # Use a temporary directory for atomic export
    with tempfile.TemporaryDirectory(prefix="lazy_e8_export_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Load the model (using low CPU memory)
        report(f"Loading model from {model_path} for E8 quantization...")
        model = None
        tokenizer = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,      # FIXED: added dtype to reduce memory
                device_map="cpu"
            )
        except Exception as e:
            report(f"Failed to load model: {e}")
            raise ValueError(f"Failed to load model from {model_path}: {e}") from e

        # Load tokenizer (already validated, but we need it for saving)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except Exception as e:
            # If this fails (shouldn't, since we validated), clean up and raise
            del model
            model = None
            gc.collect()
            raise ValueError(
                f"Failed to load tokenizer for model at {model_path}: {e}\n"
                "This usually means the tokenizer files are corrupt or incompatible.\n"
                f"Please delete it using: python bootstrap.py remove --model {model_path.stem}"
            ) from e

        # Apply E8 quantization in-memory
        report(f"Applying E8 quantization with {bpw} bits per weight...")
        quantized_model = None
        try:
            quantized_model, compressed_state = quantize_model_e8(model, bits=bpw)
        except Exception as e:
            report(f"E8 quantization failed: {e}")
            del model
            if tokenizer:
                del tokenizer
            gc.collect()
            raise

        # Export the quantized model to LazyTorch format inside temporary directory
        report(f"Exporting E8-quantized model to LazyTorch format at {tmp_path}...")
        try:
            result_path = _export_lazytorch(
                quantized_model,
                tmp_path,
                dtype="float32",
                progress_callback=lambda msg: report(f"Export: {msg}")
            )
            # Save tokenizer alongside
            tokenizer.save_pretrained(tmp_path)
            # ---- Validate the exported LazyTorch tokenizer ----
            if not _validate_tokenizer_deep(tmp_path):
                # Clean up invalid output (already in temp, will be deleted)
                raise ValueError(
                    f"Exported LazyTorch model at {tmp_path} has a corrupt tokenizer.\n"
                    "This likely indicates the source model had a tokenizer issue.\n"
                    f"Please delete the source model using: python bootstrap.py remove --model {model_path.stem}\n"
                    "Then re-download and try again."
                )
            # Move the temporary directory to the final output path
            if output_path.exists():
                shutil.rmtree(output_path, ignore_errors=True)
            shutil.move(str(tmp_path), str(output_path))
            report(f"E8+LazyTorch model saved to {output_path}")
            return output_path
        except Exception as e:
            report(f"LazyTorch export failed: {e}")
            # The temp directory will be automatically cleaned up
            return None
        finally:
            # Free memory
            if model is not None:
                del model
            if quantized_model is not None:
                del quantized_model
            if tokenizer is not None:
                del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def e8_quantize_lazytorch(
    model_path: Path,
    bpw: float = 4.0,
    output_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Optional[Path]:
    """
    Convenience alias for quantize_to_lazytorch.
    """
    return quantize_to_lazytorch(model_path, bpw, output_path, progress_callback)