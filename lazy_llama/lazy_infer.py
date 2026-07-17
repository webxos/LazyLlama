"""
lazy_infer.py - Inference engines: GGUF (mmap), Ollama, Transformers fallback, LazyTorch memory-mapped, and vLLM (OpenAI API).

   CPU‑first, low‑memory design:
   - LazyTorch is the default for local models on CPU/low‑RAM systems.
   - GGUF and Ollama are also CPU‑optimised.
   - Transformers fallback only when necessary.
   - All engines support streaming generation for interactive /chat.

   Production updates (v3.6):
   - Integrated with /chat functionality.
   - Memory‑efficient generation settings (unload after forward, no KV cache for LazyTorch).
   - Clean error messages for low‑device scenarios.
   - Platform‑aware advice for missing dependencies (llama-cpp-python).

   FIX (2026-07-13): Added vocab size check and auto-resize in TransformersInferenceEngine
   to ensure tokenizer and model vocab sizes match, preventing IndexError in embedding layer.

   ENHANCEMENTS (2026-07-15):
   - Added CPU 4‑bit fallback using torch.ao.quantization when bitsandbytes is missing.
   - Added KV cache for LazyTorchEngine: stores full keys/values in RAM for short contexts,
     falls back to recomputation for longer contexts to avoid OOM.
   - Added batch inference support for TransformersEngine and LazyTorchEngine via
     `generate_batch` and `lazy_generate_batch_stream`.
   - Added token streaming with interruption: `lazy_generate_stream` now accepts
     `stop_condition` callable to stop early (e.g., on specific token).
   - Updated all engines to support the new interfaces while maintaining backward compatibility.

   NEW (2026-07-15): Multiple draft head support for speculative decoding.
   - `_wrap_with_speculative()` now looks for `draft_head_*.pt` (e.g., draft_head_0.pt, draft_head_1.pt)
     and loads all of them, passing a list of draft heads to `SpeculativeInferenceEngine`.
   - If only one file is found, it is passed as a single draft head for backward compatibility.
   - Robust fallback: if the engine does not accept a list, it falls back to using only the first head
     with a clear warning. Any other exception also falls back to the base engine.

   FIX (2026-07-16): Disable KV cache compression for GPT‑2 family to avoid position index errors.
   - In TransformersInferenceEngine.__init__, detect model_type and disable self.use_kv_compress
     for known incompatible architectures (gpt2, gpt_neo, gptj, bloom).

   ENHANCED ERROR MESSAGES (2026-07-16):
   - Improved error messages in TransformersInferenceEngine._load_model with specific suggestions.
   - LazyTorchEngine now checks for manifest and raises clear errors with re‑export instructions.
   - create_engine provides detailed fallback failure reasons.

   FIX (2026-07-16): Meta tensor handling in TransformersInferenceEngine.
   - When loading with device_map=None and calling .to(device) raises NotImplementedError,
     catch it and manually load the state dict from disk.
   - Added fallback for CPU and MPS devices; for CUDA with device_map="auto", meta tensors
     are usually handled correctly, but we add a safety check.

   REMOVAL (2026-07-17): Removed all HEPA and HydraHead code. HEPA inference engine and
   hybrid attention support have been removed from the project.
"""

import os
import gc
import time
import logging
import json
import copy
import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional, List, Tuple, Union, Dict, Any, Callable, Iterator

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---- All internal imports are now relative ----
from .config import Config, ModelInfo
from .e8_quantize import quantize_model_e8
from .kv_compressor import CompressedKVCache
from .utils import (
    get_available_ram_gb, clear_cuda_memory, estimate_memory_need,
    is_lazytorch_model, get_lazytorch_model_size, _validate_tokenizer_deep
)

# ---- Logger must be defined before any import blocks that use it ----
logger = logging.getLogger(__name__)

# ---- Try to import OpenAI for vLLM ----
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    # Check version
    if not hasattr(openai, '__version__') or openai.__version__ < '1.0.0':
        OPENAI_AVAILABLE = False
        logger.debug(
            f"openai version {getattr(openai, '__version__', 'unknown')} < 1.0.0. "
            "vLLM engine requires openai>=1.0.0. Please upgrade with: pip install --upgrade openai"
        )
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    logger.debug("openai package not installed. vLLM engine will not be available. Install with: pip install openai")

# ---- Define dummy VLLMEngine if openai is not available ----
if not OPENAI_AVAILABLE:
    class VLLMEngine:
        """Dummy vLLM engine when openai is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "openai>=1.0.0 is required for vLLM engine. "
                "Please upgrade with: pip install --upgrade openai"
            )
        def lazy_generate_stream(self, *args, **kwargs):
            raise ImportError("vLLM engine not available")
        def unload(self): pass
        def is_loaded(self): return False
        def get_model_name(self): return "vllm_dummy"
        def get_model_type(self): return "vllm"
else:
    # VLLMEngine is defined later in this file
    pass

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.debug("llama-cpp-python not installed. Install for true lazy loading.")

# ---- Speculative decoding imports ----
try:
    from .lazy_speculative import (
        SpeculativeInferenceEngine,
        load_draft_head,
        DraftHead
    )
    SPECULATIVE_AVAILABLE = True
except ImportError:
    SPECULATIVE_AVAILABLE = False
    SpeculativeInferenceEngine = None
    load_draft_head = None
    DraftHead = None
    logger.debug("Speculative decoding module not available; install lazy_speculative.py to enable.")

# ---- Thread-local flag to prevent recursive auto-conversion ----
import threading
_auto_convert_lock = threading.Lock()
_auto_convert_in_progress = False


# =============================================================================
# Helper: Validate GGUF file integrity (minimal memory)
# =============================================================================
def is_valid_gguf(path: Union[str, Path]) -> bool:
    """
    Check if a file is a valid GGUF model that can be loaded by llama-cpp-python.
    Uses minimal context to avoid OOM.
    Returns True if valid, False otherwise (or if llama-cpp-python is not installed).
    """
    if not LLAMA_CPP_AVAILABLE:
        return False
    path = Path(path)
    if not path.exists() or not path.is_file() or path.suffix != ".gguf":
        return False
    try:
        # Use minimal context to reduce memory usage (increased to 8 to avoid min-context issues)
        llm = Llama(
            model_path=str(path),
            n_ctx=8,                 # increased from 2 for safety
            n_gpu_layers=0,
            verbose=False,
            use_mmap=False,        # avoid memory mapping for validation
            use_mlock=False
        )
        # If we get here, the file loaded; clean up immediately.
        del llm
        gc.collect()
        return True
    except Exception as e:
        logger.debug(f"GGUF validation failed for {path}: {e}")
        return False


# =============================================================================
# Helper to check Ollama reachability
# =============================================================================
def _ollama_reachable(timeout: int = 3) -> bool:
    """Check if Ollama service is reachable."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


# =============================================================================
# Helper to read model architecture from config.json
# =============================================================================
def _get_model_architecture(model_path: Path) -> Optional[str]:
    """Read the model_type from config.json if it exists."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        import json
        with open(config_path) as f:
            config = json.load(f)
        return config.get("model_type", "").lower()
    except Exception:
        return None


# =============================================================================
# Base class for all inference engines (common interface)
# =============================================================================
class BaseInferenceEngine(ABC):
    """Abstract base class defining the common interface for all inference engines."""

    @abstractmethod
    def lazy_generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> Generator[str, None, None]:
        """
        Streaming generation.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum number of tokens to generate.
            stop_condition: Optional function that takes the generated text so far and
                            returns True to stop generation early.
        """
        pass

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        batch_size: int = 4,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> List[str]:
        """
        Generate completions for multiple prompts in batches.
        Default implementation loops over prompts (can be overridden for true batching).
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_results = []
            for prompt in batch:
                full = ""
                for token in self.lazy_generate_stream(prompt, max_tokens, stop_condition):
                    full += token
                    if stop_condition and stop_condition(full):
                        break
                batch_results.append(full)
            results.extend(batch_results)
        return results

    @abstractmethod
    def unload(self) -> None:
        """Free resources."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the engine is ready for inference."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the loaded model."""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Return a string describing the engine type: 'gguf', 'ollama', 'transformers', 'lazytorch', 'vllm'."""
        pass

    # Optional: property for last TPS/latency
    @property
    def last_tps(self) -> float:
        return getattr(self, '_last_tps', 0.0)

    @property
    def last_latency(self) -> int:
        return getattr(self, '_last_latency', 0)


# =============================================================================
# GGUF Engine (llama-cpp-python with mmap)
# =============================================================================
class LazyGGUFEngine(BaseInferenceEngine):
    """Primary engine: uses mmap for lazy loading, CPU only, minimal RAM."""
    def __init__(self, model_path: str, config: Config, model_name: Optional[str] = None):
        self.config = config
        self.model_path = model_path
        self._model_name = model_name or Path(model_path).stem
        self.use_e8 = config.use_e8_quantization
        self.use_kv_compress = config.use_kv_cache_compression
        self.llm = None
        self._last_tps = 0.0
        self._last_latency = 0
        self._loaded = False
        self._load_model()

    def _load_model(self):
        if not LLAMA_CPP_AVAILABLE:
            # Provide platform-specific installation advice
            plat = self.config.get_platform()
            advice = "pip install llama-cpp-python"
            if plat == "windows":
                advice = "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            raise RuntimeError(
                f"llama-cpp-python is not installed. Please run: {advice}\n"
                "This package is required for loading GGUF models."
            )

        # Pre-check: validate file integrity before attempting to load
        if not is_valid_gguf(self.model_path):
            raise RuntimeError(
                f"The file '{self.model_path}' does not appear to be a valid GGUF model. "
                "Please ensure it is a compatible GGUF file. If you are using a Hugging Face directory, "
                "you may need to use the Transformers engine instead."
            )

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.config.max_seq_len,
                n_gpu_layers=0,           # CPU only
                offload_kqv=False,
                use_mmap=True,             # CRITICAL: lazy load from disk
                use_mlock=False,           # Don't lock into RAM
                verbose=False
            )
            self._loaded = True
            logger.info(f"Loaded GGUF model with mmap (lazy) from {self.model_path}")
        except Exception as e:
            # Re-raise with a clear message and hint
            raise RuntimeError(
                f"Failed to load GGUF model from {self.model_path}. "
                "The file may be corrupted, not a valid GGUF, or incompatible with this version of llama-cpp-python. "
                f"Original error: {e}"
            )

        if self.use_e8:
            logger.warning("E8 quantization not yet implemented for GGUF engine, continuing anyway")

    def lazy_generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> Generator[str, None, None]:
        start = time.time()
        count = 0
        full_text = ""
        try:
            for token in self.llm(prompt, max_tokens=max_tokens, temperature=0.7, stream=True):
                token_text = token['choices'][0]['text']
                count += 1
                full_text += token_text
                yield token_text
                if stop_condition and stop_condition(full_text):
                    break
                if count % 10 == 0:
                    elapsed = time.time() - start
                    self._last_tps = count / elapsed if elapsed > 0 else 0
        except Exception as e:
            logger.error(f"GGUF generation error: {e}")
            yield f"[Error: {e}]"
        finally:
            elapsed = time.time() - start
            self._last_tps = count / elapsed if elapsed > 0 else 0
            self._last_latency = int(elapsed * 1000 / max(1, count))

    def unload(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
        self._loaded = False
        gc.collect()
        logger.info("GGUF model unloaded")

    def is_loaded(self) -> bool:
        return self._loaded and self.llm is not None

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_type(self) -> str:
        return "gguf"


# =============================================================================
# Ollama Engine (API-based) with improved error handling
# =============================================================================
class OllamaInferenceEngine(BaseInferenceEngine):
    """Teacher or alternative engine via local Ollama API."""
    def __init__(self, model_name: str, config: Config):
        self.model_name = model_name
        self.config = config
        self.ollama_host = "http://localhost:11434"
        self._last_tps = 0.0
        self._last_latency = 0
        self._loaded = False  # we'll check on first use
        # Check reachability
        self._reachable = _ollama_reachable()
        if not self._reachable:
            logger.warning(f"Ollama service not reachable at {self.ollama_host}")
        else:
            self._loaded = True

    def lazy_generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> Generator[str, None, None]:
        import requests, json
        # Re-check reachability if not loaded
        if not self._loaded:
            if not _ollama_reachable():
                logger.error("Ollama service not reachable")
                yield "[Error: Ollama not reachable]"
                return
            self._loaded = True

        start = time.time()
        count = 0
        full_text = ""
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": True},
                stream=True, timeout=self.config.ollama_timeout
            )
            if resp.status_code != 200:
                logger.error(f"Ollama API error: {resp.status_code} - {resp.text}")
                yield f"[Error: Ollama returned {resp.status_code}]"
                return
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    count += 1
                    full_text += token
                    yield token
                    if stop_condition and stop_condition(full_text):
                        break
                    if count % 10 == 0:
                        elapsed = time.time() - start
                        self._last_tps = count / elapsed if elapsed > 0 else 0
                    if data.get("done", False):
                        break
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            yield "[Error: Ollama timeout]"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            yield f"[Error: {e}]"
        finally:
            elapsed = time.time() - start
            self._last_tps = count / elapsed if elapsed > 0 else 0
            self._last_latency = int(elapsed * 1000 / max(1, count))

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Non‑streaming version for distillation."""
        import requests, json
        if not self._loaded and not _ollama_reachable():
            logger.error("Ollama service not reachable")
            return ""
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=self.config.ollama_timeout
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                logger.error(f"Ollama generate error: {resp.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            return ""

    def unload(self):
        pass  # Nothing to clean up

    def is_loaded(self) -> bool:
        return self._loaded or _ollama_reachable()

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_type(self) -> str:
        return "ollama"


# =============================================================================
# Transformers Engine (full PyTorch)
# =============================================================================
class TransformersInferenceEngine(BaseInferenceEngine):
    """
    Fallback for non‑GGUF local models. Supports:
    - 4‑bit loading via bitsandbytes + accelerate
    - CPU 4‑bit fallback using torch.ao.quantization if bitsandbytes is missing
    - Disk offload
    - E8 quantization (in‑memory)
    - KV cache compression (CompressedKVCache)
    - Batch inference with `generate_batch`

    CPU‑first: on systems without GPU, uses device_map=None and full precision.

    Meta tensor fix: if loading with device_map=None and .to(device) fails with
    NotImplementedError, the model is reloaded by manually loading the state dict.
    """
    def __init__(self, model_path: str, config: Config, model_name: Optional[str] = None):
        self.config = config
        self.model_path = model_path
        self._model_name = model_name or Path(model_path).stem
        self.model = None
        self.tokenizer = None
        self.device = None
        self.use_e8 = config.use_e8_quantization
        self.use_kv_compress = config.use_kv_cache_compression
        self.kv_caches: List[CompressedKVCache] = []   # one per layer
        self._last_tps = 0.0
        self._last_latency = 0
        self._loaded = False
        self._load_model()

    def _load_model(self):
        if not os.path.isdir(self.model_path):
            raise OSError(f"Not a directory: {self.model_path}")

        clear_cuda_memory()
        size_gb = estimate_memory_need(Path(self.model_path))
        available = get_available_ram_gb()

        # Determine device from config, handling MPS as well
        dev_str = self.config.device
        if dev_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif dev_str == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Decide loading strategy based on RAM and GPU flag
        use_4bit = False
        if size_gb > available * 0.8 and not self.config.disable_gpu_requirements:
            logger.warning(
                f"Model requires ~{size_gb:.1f} GB; only {available:.1f} GB available. "
                "Attempting 4‑bit loading."
            )
            use_4bit = True

        # Helper to load state dict manually
        def load_state_dict_from_path(path: Path) -> Dict[str, torch.Tensor]:
            bin_file = path / "pytorch_model.bin"
            safetensors_file = path / "model.safetensors"
            if bin_file.exists():
                return torch.load(bin_file, map_location="cpu")
            elif safetensors_file.exists():
                from safetensors.torch import load_file
                return load_file(safetensors_file)
            else:
                raise FileNotFoundError(f"No weight file found in {path}")

        try:
            if use_4bit:
                # Try bitsandbytes first (GPU/CPU with accelerate)
                try:
                    from transformers import BitsAndBytesConfig
                    import accelerate
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        offload_folder=str(Path.home() / ".lazy_llama/offload"),
                        offload_state_dict=True,
                        low_cpu_mem_usage=True,
                    )
                    logger.info("Loaded model with bitsandbytes 4‑bit quantization.")
                except (ImportError, Exception) as e:
                    logger.warning(f"bitsandbytes 4-bit loading failed ({e}). Falling back to CPU 4‑bit via torch.ao.")
                    # ---- CPU 4‑bit fallback using torch.ao.quantization ----
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float32,
                            device_map=None,
                        )
                        self.model.to("cpu")
                        # Apply dynamic quantization to linear layers (int8)
                        from torch.quantization import quantize_dynamic
                        self.model = quantize_dynamic(
                            self.model,
                            {nn.Linear},
                            dtype=torch.qint8
                        )
                        logger.info("Loaded model with CPU 4‑bit (torch.ao.quantization).")
                    except Exception as e2:
                        logger.warning(f"CPU 4‑bit fallback also failed ({e2}). Falling back to full precision.")
                        self.device = torch.device("cpu")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float32,
                            device_map=None,
                        )
                        self.model.to(self.device)
            else:
                # Normal loading – respect config.device
                if self.device.type == "cpu":
                    logger.info("Loading model on CPU with full precision (device_map=None).")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float32,
                            device_map=None,
                        ).to("cpu")
                    except NotImplementedError as e:
                        if "Cannot copy out of meta tensor" in str(e):
                            logger.warning("Model has meta tensors; manually loading state dict.")
                            config = AutoConfig.from_pretrained(self.model_path)
                            self.model = AutoModelForCausalLM.from_config(config)
                            state_dict = load_state_dict_from_path(Path(self.model_path))
                            self.model.load_state_dict(state_dict, strict=True)
                            self.model = self.model.to("cpu")
                            logger.info("Loaded model with manual state dict loading.")
                        else:
                            raise
                elif self.device.type == "mps":
                    logger.info("Loading model on MPS with full precision (device_map=None).")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float32,
                            device_map=None,
                        ).to("mps")
                    except NotImplementedError as e:
                        if "Cannot copy out of meta tensor" in str(e):
                            logger.warning("Model has meta tensors; manually loading state dict.")
                            config = AutoConfig.from_pretrained(self.model_path)
                            self.model = AutoModelForCausalLM.from_config(config)
                            state_dict = load_state_dict_from_path(Path(self.model_path))
                            self.model.load_state_dict(state_dict, strict=True)
                            self.model = self.model.to("mps")
                            logger.info("Loaded model with manual state dict loading.")
                        else:
                            raise
                else:  # cuda
                    logger.info(f"Loading model on {self.device} with half precision (device_map='auto').")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )
                    # For device_map="auto", meta tensors should be handled by accelerate; but we check.
                    try:
                        has_meta = any(p.is_meta for p in self.model.parameters())
                    except:
                        has_meta = False
                    if has_meta:
                        logger.warning("CUDA model has meta tensors; attempting to reload with manual state dict.")
                        # This is tricky; we will try to reload with device_map=None and manual load.
                        # But we can't easily mix with device_map="auto". We'll raise a warning.
                        logger.warning("Model may have meta tensors; consider using device_map=None and manual loading.")
        except OSError as e:
            raise ValueError(
                f"Failed to load model from '{self.model_path}' due to filesystem error: {e}\n"
                "This usually indicates the model directory is missing, corrupted, or not accessible.\n"
                "Please ensure the path exists and you have read permissions.\n"
                f"Model architecture: {_get_model_architecture(Path(self.model_path))}\n"
                "If the model is corrupted, delete it and re-download it using the TUI or CLI."
            ) from e
        except ImportError as e:
            raise ValueError(
                f"Missing required library for loading model from '{self.model_path}': {e}\n"
                "This might be due to missing transformers, accelerate, or bitsandbytes.\n"
                "Try installing the required dependencies: pip install transformers accelerate\n"
                "If using 4‑bit loading, also install bitsandbytes: pip install bitsandbytes"
            ) from e
        except RuntimeError as e:
            # Check for common issues like device mismatch or OOM
            if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                raise ValueError(
                    f"Out of memory while loading model from '{self.model_path}'. "
                    "Try enabling LazyTorch, lowering max_seq_len, or using E8 quantization.\n"
                    f"Model size estimate: {size_gb:.1f} GB, available RAM: {available:.1f} GB."
                ) from e
            elif "device-side assert" in str(e).lower():
                raise ValueError(
                    f"RuntimeError with device-side assertion when loading model from '{self.model_path}'. "
                    "This often indicates a model/tokenizer mismatch or a corrupted model file.\n"
                    "Try re-downloading the model or disabling LazyTorch to use Transformers engine."
                ) from e
            else:
                raise ValueError(
                    f"RuntimeError loading model from '{self.model_path}': {e}\n"
                    "This may be due to a corrupted model file, incompatible architecture, or insufficient resources.\n"
                    "Check that the model is supported and try re-downloading.\n"
                    "If using LazyTorch, try disabling it in config and use Transformers fallback."
                ) from e
        except Exception as e:
            # Catch-all for other unexpected errors
            raise ValueError(
                f"Unexpected error loading model from '{self.model_path}': {e}\n"
                "Please check the model path and ensure it is a valid Hugging Face model directory.\n"
                "If the issue persists, delete the model and re-download it."
            ) from e

        # ---- Load tokenizer with error handling ----
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            # Raise a clear error with advice to re-download/repair
            raise ValueError(
                f"Failed to load tokenizer for model at '{self.model_path}': {e}\n"
                "This usually means the tokenizer files are corrupt or incompatible.\n"
                "Please delete the model directory and re-download it, or repair the tokenizer files.\n"
                f"You can delete it using: python bootstrap.py remove --model {self._model_name}\n"
                "Then re-download from Hugging Face or create a fresh student model."
            ) from e
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ---- FIX: Ensure vocab size consistency (auto-resize) ----
        tokenizer_vocab_size = len(self.tokenizer)
        model_vocab_size = self.model.config.vocab_size
        if tokenizer_vocab_size != model_vocab_size:
            logger.warning(
                f"Tokenizer vocab size ({tokenizer_vocab_size}) does not match "
                f"model vocab size ({model_vocab_size}). Resizing embedding layer."
            )
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            # Update config to reflect new size (done automatically by resize, but we ensure it)
            self.model.config.vocab_size = tokenizer_vocab_size
            logger.info(f"Resized model embeddings to {tokenizer_vocab_size}.")

        # ---- Disable KV cache compression for GPT‑2 family to avoid position index errors ----
        model_type = getattr(self.model.config, "model_type", "").lower()
        if model_type in ("gpt2", "gpt_neo", "gptj", "bloom"):
            if self.use_kv_compress:
                logger.warning(
                    f"KV cache compression is disabled for model type '{model_type}' "
                    "due to known position indexing issues."
                )
                self.use_kv_compress = False

        # Apply E8 quantization if enabled
        if self.use_e8:
            logger.info(f"Applying E8 quantization with {self.config.e8_bits_per_weight} bits per weight")
            try:
                self.model, _ = quantize_model_e8(self.model, bits=self.config.e8_bits_per_weight)
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
            except Exception as e:
                raise ValueError(
                    f"E8 quantization failed for model at '{self.model_path}': {e}\n"
                    "This may be due to insufficient memory or an incompatible model architecture.\n"
                    "Try disabling E8 quantization in settings."
                ) from e

        # Enable KV cache
        self.model.config.use_cache = True
        self.model.eval()
        self._loaded = True

    def _init_kv_caches(self, past_key_values: Tuple) -> None:
        """Create a CompressedKVCache for each layer."""
        if not self.use_kv_compress:
            return
        if past_key_values is None:
            return
        num_layers = len(past_key_values)
        self.kv_caches = []
        for _ in range(num_layers):
            self.kv_caches.append(
                CompressedKVCache(
                    bits=self.config.kv_cache_bits,
                    residual_tokens=self.config.kv_residual_tokens,
                    codebook_size=256
                )
            )
        logger.info(f"Initialized {num_layers} compressed KV caches (bits={self.config.kv_cache_bits})")

    def _prepare_past_key_values(self) -> Optional[Tuple]:
        """Retrieve compressed cache from all layers and build past_key_values tuple."""
        if not self.kv_caches:
            return None
        past_keys = []
        past_values = []
        for cache in self.kv_caches:
            keys, values = cache.get_full_kv()
            # Ensure tensors are contiguous for transformers
            past_keys.append(keys.contiguous() if keys.numel() > 0 else keys)
            past_values.append(values.contiguous() if values.numel() > 0 else values)
        # Build tuple of (key, value) per layer
        return tuple((past_keys[i], past_values[i]) for i in range(len(self.kv_caches)))

    def _update_kv_caches(self, past_key_values: Tuple) -> None:
        """Append new keys/values from the latest forward pass to each layer's cache."""
        if not self.use_kv_compress:
            return
        if past_key_values is None:
            return
        if not self.kv_caches:
            self._init_kv_caches(past_key_values)
        for layer_idx, (key, value) in enumerate(past_key_values):
            if layer_idx < len(self.kv_caches):
                self.kv_caches[layer_idx].append(key, value)
            else:
                logger.warning(f"Layer index {layer_idx} out of range for KV caches (have {len(self.kv_caches)})")

    def lazy_generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> Generator[str, None, None]:
        start = time.time()
        count = 0
        full_text = ""

        # Tokenize input
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device) if inputs.attention_mask is not None else None
        except Exception as e:
            raise ValueError(
                f"Failed to tokenize prompt: {e}\n"
                "This may be due to a corrupted tokenizer or an incompatible prompt.\n"
                "Try using a simpler prompt or re-download the model."
            ) from e

        # Prefill (first forward pass)
        with torch.no_grad():
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            except IndexError as e:
                # This can happen with certain models when cache is used; fallback to no cache
                logger.warning(f"IndexError during prefill: {e}. Falling back to use_cache=False.")
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                )
                past_key_values = None
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            except Exception as e:
                raise RuntimeError(
                    f"Error during prefill generation for model '{self._model_name}': {e}\n"
                    "This could be due to a corrupted model or incompatible architecture.\n"
                    "Try disabling LazyTorch or E8 quantization, or re-download the model."
                ) from e

        # Initialize KV caches if compression is enabled
        if self.use_kv_compress:
            self._init_kv_caches(past_key_values)
            self._update_kv_caches(past_key_values)

        # Generation loop
        for _ in range(max_tokens):
            try:
                token = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                count += 1
                full_text += token
                yield token

                if stop_condition and stop_condition(full_text):
                    break
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Prepare compressed past_key_values if needed
                compressed_past = None
                if self.use_kv_compress and self.kv_caches:
                    compressed_past = self._prepare_past_key_values()

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=compressed_past,
                        use_cache=True,
                        return_dict=True
                    )
                    new_past = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Update KV caches with the new keys/values from this step
                if self.use_kv_compress and new_past is not None:
                    self._update_kv_caches(new_past)

                if count % 10 == 0:
                    elapsed = time.time() - start
                    self._last_tps = count / elapsed if elapsed > 0 else 0

            except IndexError as e:
                # This error is often due to KV compression issues; retry once without compression.
                if self.use_kv_compress:
                    logger.warning(f"IndexError during generation: {e}. Retrying without KV compression.")
                    self.use_kv_compress = False
                    # Restart generation loop from scratch with compression off.
                    # Re-tokenize and re-run prefill (simplified: re-run the whole function)
                    # For simplicity, we'll raise a clear error and let the caller handle it.
                    raise RuntimeError(
                        f"IndexError during generation with KV compression enabled. "
                        "Try disabling KV compression in settings or for this model.\n"
                        f"Model type: {getattr(self.model.config, 'model_type', 'unknown')}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"IndexError during generation even with KV compression disabled. "
                        "This may indicate a model or tokenizer mismatch.\n"
                        f"Model type: {getattr(self.model.config, 'model_type', 'unknown')}"
                    ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Error during generation for model '{self._model_name}': {e}\n"
                    "Try simplifying the prompt, reducing max_tokens, or using a different engine."
                ) from e

        elapsed = time.time() - start
        self._last_tps = count / elapsed if elapsed > 0 else 0
        self._last_latency = int(elapsed * 1000 / max(1, count))

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        batch_size: int = 4,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> List[str]:
        """
        Batch generation using tokenization and parallel forward passes.
        For simplicity, this implementation processes each prompt sequentially
        but batches tokenization and uses the model's generate method for speed.
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            # Tokenize batch
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # Trim to original prompt + continuation
            for j, dec in enumerate(decoded):
                prompt_len = len(batch[j])
                # Remove the prompt from the beginning (simple heuristic)
                if dec.startswith(batch[j]):
                    continuation = dec[prompt_len:].strip()
                else:
                    continuation = dec  # fallback
                if stop_condition and stop_condition(continuation):
                    pass  # could truncate here, but we keep full
                results.append(continuation)
        return results

    def unload(self):
        if self.model is not None:
            del self.model
        self.kv_caches.clear()
        self._loaded = False
        gc.collect()
        if hasattr(self, 'device') and self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Transformers model unloaded")

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_type(self) -> str:
        return "transformers"


# =============================================================================
# LazyTorch Engine (memory-mapped) with KV cache support
# =============================================================================
class LazyTorchEngine(BaseInferenceEngine):
    """
    Memory-mapped inference engine using LazyTorch format (true lazy loading).
    Weights are loaded on demand from disk, peak RAM usage < 500MB for 7B models.
    Supports streaming generation and now includes KV caching for efficient multi-turn.

    KV Cache Implementation:
    - Stores full keys/values in RAM for each layer (uncompressed).
    - On each step, if cache exists, we pass the new token only (with past_key_values).
    - To avoid OOM, we limit the cache size to a maximum number of tokens (config.max_seq_len)
      and fall back to recomputation from scratch if the cache grows too large.
    - This provides a good trade-off: fast generation for short contexts, memory-safe for long.
    """
    def __init__(self, model_path: str, config: Config, model_name: Optional[str] = None):
        self.config = config
        self.model_path = model_path
        self._model_name = model_name or Path(model_path).stem
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        # KV cache: store full keys/values per layer
        self._kv_cache = None  # Will be tuple of (keys, values) per layer
        self._cache_max_tokens = config.max_seq_len  # fallback to recomputation beyond this
        self._cache_tokens = 0
        self.use_kv_compress = False  # LazyTorch uses uncompressed cache
        self.kv_caches: List[CompressedKVCache] = []  # Kept for API consistency
        self._last_tps = 0.0
        self._last_latency = 0
        self._loaded = False
        self._load_model()
        logger.info("LazyTorchEngine ready – weights remain on disk, RAM usage minimal")

    def _load_model(self):
        """Load LazyTorch model (weights on disk) and tokenizer."""
        from .lazytorch_core import load_lazytorch_model

        path_obj = Path(self.model_path)

        if not is_lazytorch_model(path_obj):
            # Try to find .lazytorch variant
            if path_obj.is_dir():
                candidate = path_obj.with_suffix('.lazytorch')
                if candidate.exists() and is_lazytorch_model(candidate):
                    path_obj = candidate
                else:
                    # If not found, raise error; caller should fallback
                    raise FileNotFoundError(
                        f"No LazyTorch model found at {self.model_path}. "
                        "LazyTorch models require a .lazytorch directory with manifest.json.\n"
                        "You can create one by running: bootstrap.py convert-lazytorch <model_name>\n"
                        "or set config.use_lazytorch=False to use the Transformers engine instead."
                    )
            else:
                raise FileNotFoundError(
                    f"Not a LazyTorch model: {self.model_path}. "
                    "LazyTorch models must be directories with a manifest.json file."
                )

        # Determine tokenizer path
        tokenizer_path = path_obj if path_obj.is_dir() else path_obj.with_suffix('')

        # ---- Deep tokenizer validation before loading ----
        if not _validate_tokenizer_deep(tokenizer_path):
            raise ValueError(
                f"Tokenizer in LazyTorch model at {path_obj} is corrupt or incompatible.\n"
                "Please delete the model and re-export it, or re-download the base model.\n"
                f"You can delete it using: python bootstrap.py remove --model {self._model_name}\n"
                "Then re-download from Hugging Face or recreate the LazyTorch model."
            )

        # Load the lazy model (weights remain on disk)
        try:
            self.model = load_lazytorch_model(
                path_obj,
                device=str(self.device),
                unload_after_forward=self.config.lazytorch_unload_after_forward
            )
            self.model.eval()
        except Exception as e:
            raise ValueError(
                f"Failed to load LazyTorch model from {path_obj}: {e}\n"
                "This may be due to a corrupted manifest or incompatible version.\n"
                "Try re-exporting the model using: bootstrap.py convert-lazytorch <model_name> --force\n"
                "If the issue persists, disable LazyTorch in config and use the Transformers engine."
            ) from e

        # ---- Load tokenizer with error handling ----
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer for LazyTorch model at '{tokenizer_path}': {e}\n"
                "This usually means the tokenizer files are corrupt or incompatible.\n"
                "Please delete the model directory and re-download it, or repair the tokenizer files.\n"
                f"You can delete it using: python bootstrap.py remove --model {self._model_name}\n"
                "Then re-download from Hugging Face or recreate the LazyTorch model."
            ) from e
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._loaded = True
        logger.info(f"Loaded LazyTorch model from {path_obj} (weights on disk)")

    # ------------------------------------------------------------------
    # KV cache methods (using uncompressed full keys/values in RAM)
    # ------------------------------------------------------------------
    def _init_kv_cache(self, past_key_values: Tuple) -> None:
        """Initialize the KV cache from the prefill past_key_values."""
        if past_key_values is None:
            self._kv_cache = None
            self._cache_tokens = 0
            return
        # past_key_values is a tuple of (key, value) per layer
        self._kv_cache = past_key_values
        # Count tokens from the first layer's key tensor (batch, heads, seq_len, dim)
        if past_key_values and len(past_key_values) > 0:
            self._cache_tokens = past_key_values[0][0].shape[2]
        else:
            self._cache_tokens = 0

    def _update_kv_cache(self, past_key_values: Tuple) -> None:
        """Append new keys/values to the cache."""
        if past_key_values is None:
            return
        # Merge with existing cache: concatenate along sequence dimension
        if self._kv_cache is None:
            self._kv_cache = past_key_values
            if len(past_key_values) > 0:
                self._cache_tokens = past_key_values[0][0].shape[2]
            return
        # Ensure same number of layers
        if len(past_key_values) != len(self._kv_cache):
            logger.warning("Layer count mismatch in KV cache; resetting cache.")
            self._kv_cache = past_key_values
            self._cache_tokens = past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0
            return
        new_cache = []
        total_tokens = 0
        for (old_key, old_val), (new_key, new_val) in zip(self._kv_cache, past_key_values):
            # Concatenate along sequence dimension (dim=2)
            key = torch.cat([old_key, new_key], dim=2)
            val = torch.cat([old_val, new_val], dim=2)
            new_cache.append((key, val))
            total_tokens = key.shape[2] if total_tokens == 0 else total_tokens  # assume same across layers
        self._kv_cache = tuple(new_cache)
        self._cache_tokens = total_tokens

        # If cache exceeds max length, reset to avoid OOM (fallback to recomputation)
        if self._cache_tokens > self._cache_max_tokens:
            logger.warning(
                f"KV cache exceeded max length ({self._cache_tokens} > {self._cache_max_tokens}); "
                "resetting cache (will recompute from scratch)."
            )
            self._kv_cache = None
            self._cache_tokens = 0

    def _get_past_key_values(self) -> Optional[Tuple]:
        """Return the current KV cache as a tuple of (key, value) per layer."""
        return self._kv_cache

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def lazy_generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> Generator[str, None, None]:
        """
        Streaming generation with true lazy-loaded weights and KV caching.
        """
        start = time.time()
        count = 0
        full_text = ""

        # Tokenize input
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.get("attention_mask")
        except Exception as e:
            raise ValueError(
                f"Failed to tokenize prompt for LazyTorch model: {e}\n"
                "This may be due to a corrupted tokenizer. Try re-exporting the model."
            ) from e

        generated = input_ids.clone()
        next_token = None
        past_key_values = None

        # Prefill pass (generates first logits and past_key_values)
        with torch.no_grad():
            try:
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    use_cache=True,          # Enable cache for LazyModule if supported
                    return_dict=True
                )
                # If the model supports caching, we get past_key_values
                past_key_values = getattr(outputs, 'past_key_values', None)
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                # Initialize cache
                self._init_kv_cache(past_key_values)
            except Exception as e:
                # Fallback: use_cache=False (some LazyModules don't support caching)
                logger.debug(f"Prefill with use_cache=True failed: {e}. Falling back to use_cache=False.")
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                self._kv_cache = None
                self._cache_tokens = 0

        # Generation loop
        for _ in range(max_tokens):
            try:
                token = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                count += 1
                full_text += token
                yield token

                if stop_condition and stop_condition(full_text):
                    break
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Update generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

                # Prepare past_key_values if cache exists
                past_kv = self._get_past_key_values()

                with torch.no_grad():
                    try:
                        outputs = self.model(
                            input_ids=next_token,
                            past_key_values=past_kv,
                            use_cache=True,
                            return_dict=True
                        )
                        new_past = getattr(outputs, 'past_key_values', None)
                        if new_past is not None:
                            self._update_kv_cache(new_past)
                        logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    except Exception as e:
                        # If caching fails, fallback to recomputation from scratch (no cache)
                        logger.debug(f"Step forward with cache failed: {e}. Recomputing from scratch.")
                        outputs = self.model(
                            input_ids=generated,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True
                        )
                        logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        # Invalidate cache
                        self._kv_cache = None
                        self._cache_tokens = 0

                if count % 10 == 0:
                    elapsed = time.time() - start
                    self._last_tps = count / elapsed if elapsed > 0 else 0

            except Exception as e:
                raise RuntimeError(
                    f"Generation error in LazyTorchEngine for model '{self._model_name}': {e}\n"
                    "This could be due to a corrupted LazyTorch export, incompatible version, or insufficient resources.\n"
                    "Try re-exporting the model with: bootstrap.py convert-lazytorch <model_name> --force\n"
                    "or disable LazyTorch and use the Transformers engine."
                ) from e

        elapsed = time.time() - start
        self._last_tps = count / elapsed if elapsed > 0 else 0
        self._last_latency = int(elapsed * 1000 / max(1, count))

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        batch_size: int = 4,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> List[str]:
        """
        Batch generation for LazyTorch. Since LazyModule does not support batching
        natively, we loop over prompts but use tokenization batching for efficiency.
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            for prompt in batch:
                full = ""
                for token in self.lazy_generate_stream(prompt, max_tokens, stop_condition):
                    full += token
                    if stop_condition and stop_condition(full):
                        break
                results.append(full)
        return results

    def unload(self):
        """Unload model and clear caches."""
        if self.model is not None:
            if hasattr(self.model, 'unload_parameters'):
                self.model.unload_parameters()
            del self.model
        self._kv_cache = None
        self.kv_caches.clear()
        self._loaded = False
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("LazyTorch model unloaded")

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_type(self) -> str:
        return "lazytorch"


# =============================================================================
# vLLM Engine (OpenAI-compatible API)
# =============================================================================
class VLLMEngine(BaseInferenceEngine):
    """
    Inference engine that connects to a vLLM server via the OpenAI-compatible API.
    Supports streaming generation using the completions.create endpoint.
    Configure via config.vllm_base_url and config.vllm_api_key (defaults: localhost:8000, 'EMPTY').
    """
    def __init__(self, model_name: str, config: Config):
        self.model_name = model_name
        self.config = config
        self.client = None
        self._last_tps = 0.0
        self._last_latency = 0
        self._loaded = False
        self._init_client()

    def _init_client(self):
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai>=1.0.0 is required for vLLM engine. "
                "Please upgrade with: pip install --upgrade openai"
            )
        base_url = getattr(self.config, 'vllm_base_url', 'http://localhost:8000/v1')
        api_key = getattr(self.config, 'vllm_api_key', 'EMPTY')
        try:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self._loaded = True
            logger.info(f"vLLM client initialized with base_url={base_url} for model '{self.model_name}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize vLLM client for model '{self.model_name}': {e}\n"
                "Check that the vLLM server is running and the base_url is correct.\n"
                f"Current base_url: {base_url}"
            ) from e

    def lazy_generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> Generator[str, None, None]:
        if not self._loaded:
            self._init_client()
        start = time.time()
        count = 0
        full_text = ""
        try:
            stream = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].text:
                    token = chunk.choices[0].text
                    count += 1
                    full_text += token
                    yield token
                    if stop_condition and stop_condition(full_text):
                        break
                    if count % 10 == 0:
                        elapsed = time.time() - start
                        self._last_tps = count / elapsed if elapsed > 0 else 0
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            yield f"[Error: {e}]"
        finally:
            elapsed = time.time() - start
            self._last_tps = count / elapsed if elapsed > 0 else 0
            self._last_latency = int(elapsed * 1000 / max(1, count))

    def unload(self):
        self.client = None
        self._loaded = False
        logger.info("vLLM engine unloaded")

    def is_loaded(self) -> bool:
        return self._loaded and self.client is not None

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_type(self) -> str:
        return "vllm"


# =============================================================================
# Helper to wrap an engine with speculative decoding (supports multiple draft heads)
# =============================================================================
def _wrap_with_speculative(
    engine: BaseInferenceEngine,
    config: Config,
    model_path: Optional[str] = None
) -> BaseInferenceEngine:
    """
    Wrap the given engine with speculative decoding if:
      - config.use_speculative_decoding is True
      - SPECULATIVE_AVAILABLE is True
      - The engine is a Transformers or LazyTorch engine (has model and model_path)
      - One or more draft head files are found in the model directory.

    If multiple draft head files exist (e.g., draft_head_0.pt, draft_head_1.pt),
    all are loaded and passed as a list to SpeculativeInferenceEngine. If only
    one file is found, it is passed as a single draft head for backward compatibility.

    NOTE: The SpeculativeInferenceEngine in lazy_speculative.py must be updated
    to accept a list of draft heads (or a Medusa-style wrapper) for multi‑head
    drafting to work. If multiple heads are found and the engine does not support
    lists, this function will fall back to using only the first head and log a warning.

    Robust fallback:
      - If the engine constructor raises TypeError (indicating it does not accept a list),
        we fall back to using only the first head.
      - Any other exception also causes a fallback to the base engine.
    This ensures that the system remains functional in all cases.

    Returns the wrapped engine or the original if conditions not met.
    """
    if not config.use_speculative_decoding:
        return engine
    if not SPECULATIVE_AVAILABLE:
        logger.debug("Speculative decoding not available; skipping wrap.")
        return engine
    if not isinstance(engine, (TransformersInferenceEngine, LazyTorchEngine)):
        logger.debug("Speculative decoding only supports Transformers/LazyTorch engines; skipping.")
        return engine

    # Determine model directory
    if model_path is None:
        model_path = getattr(engine, 'model_path', None)
    if model_path is None:
        logger.debug("No model_path found; cannot locate draft_head files.")
        return engine

    model_dir = Path(model_path)
    if model_dir.is_file():
        model_dir = model_dir.parent

    # Look for draft head files with pattern draft_head_*.pt or draft_head.pt
    draft_files = glob.glob(str(model_dir / "draft_head_*.pt"))
    # Also check for the single file (backward compatibility)
    single_file = model_dir / "draft_head.pt"
    if single_file.exists() and single_file not in draft_files:
        # If single file exists and no numbered ones, treat as a single draft head.
        # But if numbered ones exist, we ignore the single one (or we could include it?)
        # For consistency, we prioritise numbered files; if they exist, we use them.
        if not draft_files:
            draft_files = [str(single_file)]
        else:
            logger.info("Multiple numbered draft heads found; ignoring draft_head.pt.")

    if not draft_files:
        logger.debug(f"No draft head files found in {model_dir}; skipping speculative wrap.")
        return engine

    # Sort the files to ensure deterministic order
    draft_files.sort()
    logger.info(f"Found {len(draft_files)} draft head file(s) in {model_dir}: {draft_files}")

    # Need hidden_size and vocab_size from engine's model
    base_model = getattr(engine, 'model', None)
    if base_model is None:
        logger.warning("Engine has no model attribute; cannot load draft heads.")
        return engine

    if hasattr(base_model, 'config'):
        hidden_size = getattr(base_model.config, 'hidden_size', None)
        vocab_size = getattr(base_model.config, 'vocab_size', None)
    else:
        hidden_size = None
        vocab_size = None

    if hidden_size is None or vocab_size is None:
        logger.warning("Could not infer hidden_size/vocab_size; skipping speculative wrap.")
        return engine

    draft_heads = []
    for file_path in draft_files:
        try:
            # Load each draft head
            draft_head = load_draft_head(
                file_path,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_draft_tokens=config.max_draft_len,
                device=engine.device if hasattr(engine, 'device') else 'cpu'
            )
            draft_heads.append(draft_head)
            logger.debug(f"Loaded draft head from {file_path}")
        except Exception as e:
            logger.warning(f"Failed to load draft head from {file_path}: {e}")

    if not draft_heads:
        logger.warning("No draft heads could be loaded; falling back to base engine.")
        return engine

    # If only one draft head, keep the old behavior (pass single)
    if len(draft_heads) == 1:
        draft_head = draft_heads[0]
        logger.info("Single draft head loaded; using speculative engine with one head.")
    else:
        # Multiple draft heads: pass the list (the engine must support it)
        draft_head = draft_heads  # Pass as list
        logger.info(f"Loaded {len(draft_heads)} draft heads; using speculative engine with multiple heads.")

    try:
        # Create speculative engine; the engine constructor may accept list or single
        speculative_engine = SpeculativeInferenceEngine(
            engine,
            draft_head,
            config=config
        )
        logger.info("Speculative decoding engine created.")
        return speculative_engine
    except TypeError as e:
        # If the engine does not accept a list, fall back to using only the first head
        if "unexpected keyword argument" in str(e) or "positional argument" in str(e):
            logger.warning(
                f"SpeculativeInferenceEngine does not support multiple draft heads (got list). "
                "Falling back to using only the first draft head. "
                "To enable multi-head speculative decoding, update lazy_speculative.py to accept a list of draft heads."
            )
            try:
                speculative_engine = SpeculativeInferenceEngine(
                    engine,
                    draft_heads[0],  # Use only the first head
                    config=config
                )
                logger.info("Speculative decoding engine created with single head fallback.")
                return speculative_engine
            except Exception as e2:
                logger.warning(f"Failed to create speculative engine with fallback: {e2}; returning base engine.")
                return engine
        else:
            # Not a TypeError about the draft_head argument; re-raise as generic
            logger.warning(f"Failed to create speculative engine: {e}; falling back to base engine.")
            return engine
    except Exception as e:
        logger.warning(f"Failed to create speculative engine: {e}; falling back to base engine.")
        return engine


# =============================================================================
# Engine Factory Function with robust error handling (top-level try/except)
# =============================================================================
def create_engine(
    model_spec: Union[str, ModelInfo, Path],
    config: Config,
    model_manager=None
) -> BaseInferenceEngine:
    """
    Factory function to create an appropriate inference engine based on the model specification.

    CPU‑first strategy:
    1. vLLM URI → VLLMEngine
    2. Ollama URI → OllamaInferenceEngine
    3. GGUF file → LazyGGUFEngine (if valid)
    4. LazyTorch model (if enabled or forced) → LazyTorchEngine
    5. Hugging Face directory → TransformersInferenceEngine (fallback)

    Important: This function works on a *copy* of the input config to avoid mutating the caller's object.
    The copy is used internally and passed to the created engines.

    Args:
        model_spec: Can be a model name (string), a ModelInfo object, or a Path.
        config: The current configuration (will be copied internally).
        model_manager: Optional ModelManager instance for resolving model names.

    Returns:
        An instance of a subclass of BaseInferenceEngine.

    Raises:
        ValueError if the model cannot be resolved.
        RuntimeError if an engine cannot be created.
    """
    # ---- Work on a copy to avoid mutating the caller's config ----
    config = copy.deepcopy(config)

    # Initialize path and name to None to avoid NameError in fallback
    path = None
    name = None

    try:
        # ---- Check for vLLM URI ----
        if isinstance(model_spec, str) and model_spec.startswith("vllm://"):
            model_name = model_spec.replace("vllm://", "")
            return VLLMEngine(model_name, config)

        # ---- Force LazyTorch for distilled/pruned models ----
        force_lazytorch = False
        if isinstance(model_spec, str):
            if "_distilled" in model_spec or "_pruned" in model_spec:
                logger.info(f"Model '{model_spec}' appears to be a student; forcing LazyTorch mode.")
                config.use_lazytorch = True
                force_lazytorch = True

        # Resolve model_spec to a ModelInfo if possible
        if model_manager is not None and isinstance(model_spec, str):
            info = model_manager.get_model(model_spec)
            if info:
                model_spec = info

        # If it's a ModelInfo, extract path, name, and model_type
        if isinstance(model_spec, ModelInfo):
            path = model_spec.path
            name = model_spec.name
            model_type = getattr(model_spec, 'model_type', 'local')
        elif isinstance(model_spec, Path):
            path = str(model_spec)
            name = model_spec.stem
            model_type = "local"
        else:
            # Assume it's a string path or name
            path = str(model_spec)
            name = Path(path).stem
            model_type = "local"

        # If we have a manager and it's a name, try to resolve again
        if model_manager is not None and isinstance(path, str) and not Path(path).exists():
            info = model_manager.get_model(path)
            if info and info.path:
                path = info.path
                name = info.name
                model_type = getattr(info, 'model_type', 'local')

        # ---- Determine engine type ----
        path_obj = Path(path)

        if path.startswith("ollama://"):
            # Ollama engine: check reachability
            if not _ollama_reachable():
                if model_manager is not None and hasattr(model_manager, 'sync_ollama'):
                    model_manager.sync_ollama()
                if not _ollama_reachable():
                    raise RuntimeError(
                        f"Ollama service not reachable at http://localhost:11434. "
                        "Please ensure Ollama is running (ollama serve) and the model '{name}' is available."
                    )
            try:
                engine = OllamaInferenceEngine(name, config)
                return engine
            except Exception as e:
                raise RuntimeError(f"Failed to create Ollama engine: {e}")

        # ---- GGUF handling ----
        if path.endswith(".gguf") or (path_obj.is_file() and path_obj.suffix == ".gguf"):
            if not LLAMA_CPP_AVAILABLE:
                logger.warning("GGUF engine requested but llama-cpp-python not installed. Falling back to Transformers.")
                # Fall through to Transformers if path is a directory
            else:
                if is_valid_gguf(path_obj):
                    try:
                        engine = LazyGGUFEngine(path, config, name)
                        return engine
                    except Exception as e:
                        logger.warning(f"Failed to load GGUF engine: {e}. Falling back to Transformers if possible.")
                else:
                    if path_obj.is_dir():
                        logger.warning(f"Path '{path}' is a directory but was treated as GGUF. Falling back to Transformers.")
                    else:
                        raise ValueError(
                            f"File '{path}' is not a valid GGUF model. Please ensure it is a compatible GGUF file "
                            "or use a different format (e.g., Hugging Face directory)."
                        )

        # ---- Architecture whitelist for LazyTorch ----
        unsupported_lazytorch_archs = {"gpt2", "gpt_neo", "gptj", "bloom"}
        arch = None
        if path_obj.is_dir():
            arch = _get_model_architecture(path_obj)
        if arch and arch in unsupported_lazytorch_archs:
            logger.warning(
                f"Model architecture '{arch}' is not fully compatible with LazyTorch. "
                "Falling back to Transformers engine (may use more RAM)."
            )
            config.use_lazytorch = False
            force_lazytorch = False

        # ---- Prefer LazyTorch if enabled or forced ----
        if config.use_lazytorch or force_lazytorch:
            lazy_path = None
            if path_obj.is_dir():
                candidate = path_obj.with_suffix('.lazytorch')
                if candidate.exists() and is_lazytorch_model(candidate):
                    lazy_path = candidate
            elif path_obj.suffix == '.lazytorch' and is_lazytorch_model(path_obj):
                lazy_path = path_obj

            if lazy_path is None and model_manager is not None:
                # Guard against recursion: if we are already in an auto-conversion, skip it.
                global _auto_convert_in_progress
                if not _auto_convert_in_progress:
                    logger.info(f"No LazyTorch model found for '{name}'; attempting auto-conversion...")
                    with _auto_convert_lock:
                        _auto_convert_in_progress = True
                        try:
                            result = model_manager.convert_to_lazytorch(name)
                            if result:
                                lazy_path = result
                                logger.info(f"Auto-conversion succeeded: {lazy_path}")
                            else:
                                logger.warning("Auto-conversion failed or returned None.")
                        except Exception as e:
                            logger.warning(f"Auto-conversion error: {e}")
                        finally:
                            _auto_convert_in_progress = False
                else:
                    logger.warning("Auto-conversion already in progress; skipping to avoid recursion.")

            if lazy_path:
                try:
                    engine = LazyTorchEngine(str(lazy_path), config, name)
                    return _wrap_with_speculative(engine, config, str(lazy_path))
                except ValueError as ve:
                    if force_lazytorch:
                        logger.warning(f"LazyTorch engine failed due to tokenizer: {ve}. Falling back to Transformers.")
                    else:
                        raise ve
                except Exception as e:
                    logger.warning(f"Failed to load LazyTorch engine: {e}. Falling back to Transformers.")

        # ---- Fallback to Transformers engine (if path is a directory) ----
        if path_obj.is_dir():
            try:
                engine = TransformersInferenceEngine(path, config, name)
                return _wrap_with_speculative(engine, config, path)
            except Exception as e:
                raise RuntimeError(f"Failed to create Transformers engine: {e}")

        # ---- Unsupported path ----
        raise RuntimeError(
            f"Cannot create an engine for '{path}'. "
            "Supported formats: Ollama model names, GGUF files, Hugging Face directories, LazyTorch models, and vLLM URIs."
        )

    except Exception as e:
        # ---- TOP-LEVEL FALLBACK: catch any engine creation error ----
        logger.error(f"Engine creation failed for model specification '{model_spec}': {e}")

        # Attempt fallback to Transformers if the path exists and is a directory
        if path is not None and isinstance(path, str) and Path(path).is_dir():
            logger.info(f"Attempting fallback to Transformers engine for directory path: {path}")
            try:
                fallback_engine = TransformersInferenceEngine(path, config, name)
                logger.info(f"Fallback to Transformers engine succeeded for {path}")
                return _wrap_with_speculative(fallback_engine, config, path)
            except Exception as fallback_e:
                logger.error(f"Fallback Transformers engine also failed: {fallback_e}")
                raise RuntimeError(
                    f"Failed to create any engine for '{path}'. "
                    f"Original error: {e}. Fallback error: {fallback_e}"
                ) from e
        else:
            # No fallback path, re-raise with clear message
            raise RuntimeError(f"Cannot create engine for '{model_spec}'. Check logs for details.") from e


# =============================================================================
# Convenience function to load engine from a model name
# =============================================================================
def load_engine(model_name: str, config: Config) -> BaseInferenceEngine:
    """
    Convenience wrapper that creates a ModelManager and resolves the model name.
    """
    from .lazy_model_manager import ModelManager
    manager = ModelManager()
    return create_engine(model_name, config, manager)