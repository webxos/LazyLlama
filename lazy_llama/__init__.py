"""Lazy Llama - Low-end inference engine with distillation, pruning, E8 quantization,
KV compression, LazyTorch memory-mapped loading, and Endless RL self‑improvement.
"""
__version__ = "3.6.0"

# ----------------------------------------------------------------------
# Environment and dependency checks
# ----------------------------------------------------------------------
import sys
import warnings

# Python version check
if sys.version_info < (3, 10):
    warnings.warn(
        "Lazy Llama requires Python 3.10 or later. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}",
        RuntimeWarning,
        stacklevel=2,
    )

# NumPy compatibility check
try:
    import numpy as np
    np_version = np.__version__
    if np_version.startswith("2."):
        # NumPy 2.0 is supported
        pass
    elif np_version < "1.24.0":
        warnings.warn(
            f"NumPy version {np_version} is older than 1.24.0; some features may not work. "
            "Please upgrade to numpy>=2.0.0 or at least 1.24.0.",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError:
    warnings.warn(
        "NumPy is not installed. Lazy Llama will not work. "
        "Please install numpy>=2.0.0.",
        RuntimeWarning,
        stacklevel=2,
    )

# ----------------------------------------------------------------------
# Guarded import for bootstrap (CLI entry point) – silently skip if missing
# ----------------------------------------------------------------------
try:
    from .bootstrap import main as bootstrap_main
    from .bootstrap import run_reap_pipeline_checklist
except ImportError:
    bootstrap_main = None
    run_reap_pipeline_checklist = None

# ----------------------------------------------------------------------
# Core components – public API (all relative imports)
# ----------------------------------------------------------------------
from .config import (
    Config,
    load_config,
    auto_optimize_config,
    recommend_enhancements,
)
from .utils import (
    get_available_ram_gb,
    get_total_ram_gb,
    get_memory_usage_gb,
    clear_cuda_memory,
    get_gpu_memory_gb,
    retry,
    get_system_profile,
    export_to_ollama,
    download_with_retry,
    verify_sha256,
    get_lazytorch_model_size,
    is_lazytorch_model,
    convert_hf_to_lazytorch,
)
from .lazy_infer import (
    LazyGGUFEngine,
    OllamaInferenceEngine,
    TransformersInferenceEngine,
    LazyTorchEngine,
    # HEPAInferenceEngine removed
    create_engine,
)
from .lazy_model_manager import ModelManager
from .lazy_distill import LazyDistillationEngine
from .lazy_prune import Pruner, get_task_prompts, TASK_PROMPTS
from .benchmark import (
    benchmark_model,
    benchmark_student_models,
)
from .metrics_store import MetricsStore
from .lazy_tui import LazyTUI
from .e8_quantize import (
    E8LatticeQuantizer,
    quantize_model_e8,
    load_e8_quantized,
    e8_quantize_model,
)
from .kv_compressor import (
    TurboQuantCache,
    MixedDimKVCache,
    CompressedKVCache,
)
from .zero_shot_compensation import (
    ZeroShotAdapterCompensation,
    apply_zero_shot_compensation,
)
from .dashboard_server import start_dashboard
from .lazytorch_core import (
    LazyModule,
    LazyParameter,
    LazyLinear,
    export_to_lazytorch,
    load_lazytorch_model,
    lazy_model_context,
    export_to_standard_pytorch,
)

# ----------------------------------------------------------------------
# Optional Endless RL components (keep only those that don't depend on HEPA)
# ----------------------------------------------------------------------
try:
    from .endless_rl import (
        run_endless_distillation,
        run_endless_prune,
        run_endless_auto,          # auto may call finetune; ensure it's updated
    )
    ENDLESS_AVAILABLE = True
except ImportError:
    ENDLESS_AVAILABLE = False

# ----------------------------------------------------------------------
# Build __all__ list
# ----------------------------------------------------------------------
__all__ = [
    # Config & utils
    "Config",
    "load_config",
    "auto_optimize_config",
    "recommend_enhancements",
    "get_available_ram_gb",
    "get_total_ram_gb",
    "get_memory_usage_gb",
    "clear_cuda_memory",
    "get_gpu_memory_gb",
    "retry",
    "get_system_profile",
    "export_to_ollama",
    "download_with_retry",
    "verify_sha256",
    "get_lazytorch_model_size",
    "is_lazytorch_model",
    "convert_hf_to_lazytorch",
    # Inference engines
    "LazyGGUFEngine",
    "OllamaInferenceEngine",
    "TransformersInferenceEngine",
    "LazyTorchEngine",
    # "HEPAInferenceEngine" removed
    "create_engine",
    # Model management
    "ModelManager",
    # Distillation & pruning
    "LazyDistillationEngine",
    "Pruner",
    "get_task_prompts",
    "TASK_PROMPTS",
    # Benchmarking
    "benchmark_model",
    "benchmark_student_models",
    # Metrics & UI
    "MetricsStore",
    "LazyTUI",
    # E8 quantization
    "E8LatticeQuantizer",
    "quantize_model_e8",
    "load_e8_quantized",
    "e8_quantize_model",
    # KV cache compression
    "TurboQuantCache",
    "MixedDimKVCache",
    "CompressedKVCache",
    # Zero‑shot compensation
    "ZeroShotAdapterCompensation",
    "apply_zero_shot_compensation",
    # Dashboard
    "start_dashboard",
    # LazyTorch core
    "LazyModule",
    "LazyParameter",
    "LazyLinear",
    "export_to_lazytorch",
    "load_lazytorch_model",
    "lazy_model_context",
    "export_to_standard_pytorch",
]

# Conditionally add Endless RL components that are HEPA‑free
if ENDLESS_AVAILABLE:
    __all__.extend([
        "run_endless_distillation",
        "run_endless_prune",
        "run_endless_auto",
        # "run_endless_finetune" removed (depends on HEPA)
    ])

# Conditionally add bootstrap_main and run_reap_pipeline_checklist if available
if bootstrap_main is not None:
    __all__.append("bootstrap_main")
if run_reap_pipeline_checklist is not None:
    __all__.append("run_reap_pipeline_checklist")

# ----------------------------------------------------------------------
# Package metadata
# ----------------------------------------------------------------------
__author__ = "Lazy Llama Team"
__email__ = "team@lazy-llama.ai"
__license__ = "MIT"
__url__ = "https://github.com/lazy-llama/lazy-llama"
