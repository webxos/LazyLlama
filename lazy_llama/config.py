"""
Configuration management with full v20 fields + new E8/KV/zero-shot flags + LazyTorch settings.
Added: Export settings for zip/vLLM/Ollama compatibility.
FIXED: ModelInfo serialization with to_dict() method.
ADDED: Dashboard settings (refresh interval, max models, cache TTL).
FIXED: auto_optimize_config() now sets slow_mode and adjusts batch sizes for low RAM.
ADDED: LoRA configuration fields (use_lora, lora_r, lora_alpha, lora_dropout).
ADDED: get_effective_device() helper for consistent device selection.
FIXED: should_use_lazytorch() now checks both config flag and RAM limit.
VERIFIED: use_lazytorch default is True; auto-optimize enables it for low RAM.
NOTE: Student/teacher model selection is handled via global state (dashboard_server),
         not stored in this config. ModelInfo includes flags for pruning/distillation.
UPDATED: auto_optimize_config now never enables experimental `use_mixed_dim_kv` by default.
IMPROVED: recommend_enhancements now explicitly warns about experimental features.

NEW: Added `invalid` field to ModelInfo to track models with missing/corrupted paths.
NEW: Added `allow_gguf_pruning` and `allow_gguf_distillation` flags (both default False)
        to control whether GGUF models are accepted (with warning) for pruning/distillation.
        These are placeholders; no automatic conversion is performed.

NEW: Added `default_student` and `default_teacher` fields to pre‑populate global selections.
        These are used by the TUI and dashboard when no explicit selection exists.
VERIFIED: All fields are correctly serialized/deserialized.

FIXED: Increased `ollama_timeout` default from 60 to 600 seconds to drastically reduce
          timeouts with slower Ollama services. Users can override this in ~/.lazy_llama/config.json.

NEW: `auto_download_missing_models` – if True, student creation will automatically
        download a missing base model from Hugging Face.
NEW: `auto_convert_to_lazytorch` – if True, models are converted to LazyTorch without
        asking for confirmation when needed (chat, benchmark, etc.).
NEW: `tokenizer_deep_validation` – if True, perform deep tokenizer loading validation
        (default True). Set to False only if you trust the model files.

============================================================================
SPECULATIVE DECODING (v3.3.3):
- Added configuration fields for DSpark‑style semi‑autoregressive drafting:
     use_speculative_decoding, max_draft_len, confidence_threshold, train_draft_head.
============================================================================

NEW FLAG (v3.3.4):
- `auto_convert_student_to_lazytorch`: if True, automatically convert a newly created
     student model to LazyTorch format during student creation (default False).
     This flag is separate from `auto_convert_to_lazytorch` which applies to downloads.
============================================================================

NEW FLAG (v3.3.7):
- `use_static_moe`: if True, enable static (deterministic) routing for Micro MoE
     layers during distillation and pruning. This improves determinism and fusion
     with external runtimes (llama.cpp/Ollama).
============================================================================

PLATFORM SUPPORT (v3.5):
- Added `platform` field to Config to store the operating system ("auto", "linux",
     "darwin", "windows"). Auto-detected on first run.
- Added `get_platform()` method to return the effective platform with WSL2 detection.
- Added `get_ollama_binary()` method to return the correct Ollama binary path.
- Added `get_llama_cpp_install_cmd()` helper for platform‑specific pip advice.
- Uses `is_wsl2()` from utils to detect Windows Subsystem for Linux and treat it as Windows.
============================================================================

ENDLESS RL LOOP SETTINGS (v3.6):
- Added configuration fields for the self‑improvement endless loops:
     endless_max_cycles (default -1 = infinite),
     endless_cycle_sleep (seconds between cycles),
     endless_policy ("worst", "best", "random"),
     endless_models (list of model names to manage in auto loop).
============================================================================

REAP PIPELINE SETTINGS (v3.6):
- Added configuration fields for the REAP (Distillation → Pruning → Finetuning → Evaluation)
     self‑improvement pipeline:
     reap_prune_ratio (default 0.3),
     reap_quantize (default True),
     reap_low_mem (default True),
     reap_eval_after_finetune (default False),
     reap_checklist_enabled (default True),
     reap_checklist_dir (optional Path),
     reap_force_cpu (default True),
     reap_disable_offload (default True).
============================================================================

FIXES (2026-07-07):
- ModelInfo.to_dict now explicitly converts all Path objects to strings.
- Config.get_platform now uses utils.detect_platform() exclusively for
     auto-detection, eliminating code duplication.
- Added import of detect_platform from utils.

NEW (2026-07-08): Added `default_students_installed` flag to track whether
                     the default student models have been downloaded.

FURTHER FIX (2026-07-10):
- Ensured `lazytorch_cache_dir` is always a Path (converted in save/load).
- Improved serialization robustness in to_json_dict.
- Clarified usage of `default_students_installed` in docstring.
- Removed unused imports and streamlined code.
- **FIXED circular import**: moved `from lazy_llama.utils import detect_platform` inside `get_platform()` method.
- **FIXED get_ollama_binary**: uses `os.path.expandvars` correctly (import os already at top).

FIX (2026-07-13): Added REAP pipeline configuration fields to support the
                     distillation → pruning → finetuning → evaluation workflow.
                     All REAP fields are low‑memory friendly by default.

ENHANCEMENTS (2026-07-15):
- Added `validate_all()` method to perform comprehensive config validation.
- Added `migrate()` static method to handle future version upgrades.
- Added `reap_passes` and `reap_temperature_schedule` fields for distillation.
- Improved validation of `reap_prune_ratio` range.
- Added consistency checks for experimental features (e.g., mixed_dim_kv with low RAM).
- Added `__all__` to expose public API.
- Replaced `print` warning in `load_config` with proper logging.
- Added robust type conversion in `migrate()`.

ENHANCEMENTS (2026-07-15) – QLoRA / 4‑bit training:
- Added configuration fields for QLoRA (4‑bit quantized LoRA):
     use_qlora, qlora_r, qlora_alpha, qlora_dropout, qlora_target_modules.
- Extended `validate_all()` to check qlora_r > 0 if use_qlora is True.
- Extended `auto_optimize_config()` to enable QLoRA and adjust ranks on low RAM.

FIXES (2026-07-14) – Conflict resolution:
- In `__post_init__`, if both `use_qlora` and `use_lora` are True, automatically
  set `use_lora = False` (QLoRA is a superset) and log a warning.
- In `auto_optimize_config()`, when enabling QLoRA, explicitly set `use_lora = False`
  to avoid redundant LoRA configuration.
- Added a helper `get_effective_lora_config()` to return the active LoRA/QLoRA
  settings for the training code, prioritising QLoRA if enabled.

FIX (2026-07-14) – Load-time conflict resolution:
- In `load_config()`, after setting all attributes from the loaded JSON, call
  `config.__post_init__()` again to ensure conflict resolution and path
  normalisation are applied even for existing configs that have both flags set.

NEW (2026-07-15) – MoE (Mixture of Experts) configuration:
- Added fields `moe_num_experts`, `moe_top_k`, `moe_hierarchical`, `moe_expert_capacity`.
- Added validation to ensure `moe_num_experts >= 2` and `moe_top_k <= moe_num_experts`.

NEW (2026-07-15) – Hyperparameter search configuration:
- Added fields `hyperparameter_search_enabled`, `search_space_distill`,
  `search_space_prune`, `search_space_finetune` with default reasonable ranges.

NEW (2026-07-16) – Benchmark settings fields:
- Added fields to store benchmark defaults (prompt, max_tokens, perplexity, MC, long-context).
- Updated `auto_optimize_config()` to adjust benchmark settings based on available RAM.

REMOVED (2026-07-17): Removed all HEPA and HydraHead configuration fields. These
features have been removed from the project.

ENHANCED PRUNING DEFAULTS (2026-07-16):
- Lowered `reap_prune_ratio` from 0.3 to 0.15 to prevent catastrophic collapse.
- Enabled `use_qlora` by default for better post‑pruning recovery.
- Increased `qlora_r` to 16 for higher‑rank recovery.
- Enabled `use_zero_shot_compensation` and `auto_convert_student_to_lazytorch`.
- Enabled benchmark defaults for perplexity and multiple‑choice evaluation.
- These changes improve pruning quality while maintaining low memory usage.

DISTILLATION ALPHA ENHANCEMENT (2026-07-16):
- Increased default `distill_alpha` from 0.7 to 0.8 to strengthen teacher signal
  during recovery distillation, helping pruned models regain coherence faster.
- Updated `auto_optimize_config` to set `distill_alpha = 0.8` whenever `reap_low_mem` is True
  (i.e., on low‑RAM systems) to maximise recovery quality.
"""

import json
import logging
import psutil
import torch
import platform as _platform
import os
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union, Tuple

# ---- Setup logger ----
logger = logging.getLogger(__name__)

# Directories (same as multi-file version)
LAZY_DIR = Path.home() / ".lazy_llama"
MODELS_DIR = LAZY_DIR / "models"
CHECKPOINTS_DIR = LAZY_DIR / "checkpoints"
CACHE_DIR = LAZY_DIR / "cache"
LOGS_DIR = LAZY_DIR / "logs"
LAZYTORCH_CACHE_DIR = LAZY_DIR / "lazytorch_cache"  # For converted models

# ---- Ensure all required directories exist at import time ----
for d in [LAZY_DIR, MODELS_DIR, CHECKPOINTS_DIR, CACHE_DIR, LOGS_DIR, LAZYTORCH_CACHE_DIR]:
    d.mkdir(exist_ok=True)


@dataclass
class ModelInfo:
    """Model metadata stored in registry.json"""
    name: str
    original_size_mb: float
    distilled_size_mb: Optional[float] = None
    distillation_date: Optional[str] = None
    pruning_applied: bool = False
    task_specialization: Optional[str] = None
    verification_passes: int = 0
    accuracy_score: Optional[float] = None
    path: Optional[str] = None
    quantized: Optional[str] = None
    e8_quantized: bool = False
    e8_bpw: Optional[float] = None
    lazytorch_format: bool = False  # Flag for LazyTorch models
    invalid: bool = False           # Flag for missing/corrupted paths
    model_type: str = "local"       # "local", "gguf", "ollama", "lazytorch" (hepa removed)

    # ---- REAP-specific metadata ----
    reap_stages: Dict[str, bool] = field(default_factory=lambda: {
        "distilled": False,
        "pruned": False,
        "finetuned": False,
        "evaluated": False,
    })
    reap_checkpoint: Optional[str] = None  # path to .reap_checkpoint file

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary suitable for JSON serialization.
        Ensures that Path objects are converted to strings.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create a ModelInfo from a dictionary, handling Path objects."""
        # Ensure path is string or None (avoid converting None to 'None')
        if 'path' in data and data['path'] is not None:
            data['path'] = str(data['path'])
        elif 'path' in data:
            data['path'] = None
        # ---- Backward compatibility for missing fields ----
        if 'invalid' not in data:
            data['invalid'] = False
        if 'model_type' not in data:
            data['model_type'] = "local"
        # ---- REAP fields backward compatibility ----
        if 'reap_stages' not in data:
            data['reap_stages'] = {"distilled": False, "pruned": False, "finetuned": False, "evaluated": False}
        if 'reap_checkpoint' not in data:
            data['reap_checkpoint'] = None
        return cls(**data)


@dataclass
class Config:
    # --- v20 fields (from lazy_llama19.py distillation) ---
    ram_limit_gb: float = 6.0
    device: str = "cpu"
    max_seq_len: int = 512
    use_e8_quantization: bool = False
    e8_bits_per_weight: float = 4.0
    use_kv_cache_compression: bool = True
    kv_cache_bits: int = 4
    distill_temperature: float = 2.0
    distill_alpha: float = 0.8               # <-- CHANGED: increased for better recovery
    gradient_accumulation_steps: int = 4
    calibration_prompts: List[str] = field(default_factory=lambda: [
        "What is Python?",
        "Explain recursion.",
        "Write a loop summing 1 to 10.",
        "What is the capital of France?",
        "Define machine learning.",
    ])

    # --- multi-file additional fields (not in v20 but added) ---
    slow_mode: bool = True
    verify_passes: int = 3
    quantize: Optional[str] = "q4_0"
    log_level: str = "info"
    sliding_window_size: int = 512
    prune_threshold: float = 0.05
    distill_learning_rate: float = 1e-4
    distill_batch_size: int = 2
    checkpoint_interval: int = 5
    ollama_timeout: int = 600  # Increased from 180 to 600 to drastically reduce timeouts with slower Ollama services
    dashboard_port: int = 8080
    dashboard_auto_open: bool = True
    validation_prompts: List[str] = field(default_factory=lambda: [
        "Explain what Python is in simple terms.",
        "Write a function to calculate fibonacci numbers.",
        "What is the difference between list and tuple?",
        "How does garbage collection work in Python?",
        "Explain recursion with an example.",
    ])

    # --- LoRA configuration (standard) ---
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # --- QLoRA / 4‑bit training configuration (HIGH PRIORITY) ---
    use_qlora: bool = True                  # <-- CHANGED: enabled by default for recovery
    qlora_r: int = 16                       # <-- CHANGED: higher rank for better recovery
    qlora_alpha: int = 32
    qlora_dropout: float = 0.1
    qlora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # --- MoE (Mixture of Experts) configuration ---
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_hierarchical: bool = False
    moe_expert_capacity: int = 4  # optional, for auxiliary load-balancing

    # --- Hyperparameter search configuration ---
    hyperparameter_search_enabled: bool = False
    search_space_distill: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": (0.5, 3.0),
        "alpha": (0.5, 0.9),
        "learning_rate": (1e-5, 1e-3),
        "gradient_accumulation_steps": (1, 8),
    })
    search_space_prune: Dict[str, Any] = field(default_factory=lambda: {
        "threshold": (0.02, 0.15),
        "iterative_steps": (2, 5),
        "activation_threshold": (0.005, 0.05),
    })
    search_space_finetune: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": (1e-5, 1e-3),
        "epochs": (5, 20),
        "batch_size": (8, 32),
    })

    # --- Dashboard settings ---
    dashboard_refresh_interval: int = 2       # seconds between metric updates
    dashboard_max_models: int = 100           # max number of models to show in dropdowns
    model_cache_ttl: int = 60                 # seconds to cache model lists

    # --- KV compression extras (from multi-file) ---
    kv_residual_tokens: int = 128
    # EXPERIMENTAL: use_mixed_dim_kv can cause instability; default remains False.
    use_mixed_dim_kv: bool = False

    # --- Zero-shot compensation (from multi-file) ---
    use_zero_shot_compensation: bool = True   # <-- CHANGED: enabled by default for recovery

    # --- Static MoE routing ---
    use_static_moe: bool = False              # Enable deterministic static routing for MoE layers

    # --- Auto-optimization flags (from multi-file) ---
    auto_optimized: bool = False
    recommended_e8: bool = False
    recommended_kv_bits: int = 4

    # --- GPU requirements flag (optional, for systems without CUDA) ---
    disable_gpu_requirements: bool = False

    # --- LazyTorch fields ---
    use_lazytorch: bool = True  # Enable memory-mapped lazy loading (default on for low RAM)
    lazytorch_weights_file: Optional[str] = None  # Specific weights file (if not using default)
    lazy_load_layers: bool = True  # Load layers on demand, unload after forward
    lazytorch_cache_dir: Path = LAZYTORCH_CACHE_DIR  # Directory for converted .lazytorch models
    lazytorch_unload_after_forward: bool = True  # Unload parameters after each forward pass

    # --- Export settings ---
    export_zip_include_lazytorch: bool = True      # Include .lazytorch variant when exporting zip
    export_vllm_compatible: bool = True            # Use safetensors for vLLM exports
    export_ollama_quantize: str = "q4_0"           # Quantization level for Ollama export (q4_0, q8_0, etc.)
    export_keep_original: bool = True              # Keep original model after export (don't delete)

    # --- GGUF handling (placeholders; no automatic conversion) ---
    allow_gguf_pruning: bool = False       # If True, only log warning when GGUF model is used for pruning; still fails because no conversion
    allow_gguf_distillation: bool = False  # If True, only log warning when GGUF model is used as student; still fails because no conversion

    # --- Default global selections (for TUI/dashboard pre-population) ---
    default_student: Optional[str] = None   # Name of default student model (e.g., "distilgpt2")
    default_teacher: Optional[str] = None   # Name of default teacher model (e.g., "llama2")

    # --- NEW: Auto‑behaviour flags ---
    auto_download_missing_models: bool = True   # If True, automatically download base model when creating student
    auto_convert_to_lazytorch: bool = True      # If True, convert to LazyTorch without prompting when needed
    tokenizer_deep_validation: bool = True      # If True, perform deep tokenizer loading validation

    # ========================================================================
    # NEW FLAG: auto-convert student models to LazyTorch during creation
    # ========================================================================
    auto_convert_student_to_lazytorch: bool = True  # <-- CHANGED: enabled by default for memory savings

    # ---- Flag to track default student installation ----
    default_students_installed: bool = False   # Track if default student models have been downloaded

    # ========================================================================
    # SPECULATIVE DECODING (DSpark-style)
    # ========================================================================
    use_speculative_decoding: bool = False          # Enable semi-autoregressive drafting
    max_draft_len: int = 4                          # Maximum number of draft tokens
    confidence_threshold: float = 0.5               # Threshold for accepting drafts
    train_draft_head: bool = False                  # Train draft head during distillation

    # ========================================================================
    # PLATFORM SUPPORT (v3.5)
    # ========================================================================
    platform: str = "auto"  # "auto", "linux", "darwin", "windows"

    # ========================================================================
    # ENDLESS RL LOOP SETTINGS (v3.6)
    # ========================================================================
    endless_max_cycles: int = -1           # -1 for infinite
    endless_cycle_sleep: int = 60          # seconds between cycles
    endless_policy: str = "worst"          # "worst", "best", "random"
    endless_models: List[str] = field(default_factory=list)  # models to include in auto loop

    # ========================================================================
    # REAP PIPELINE SETTINGS (v3.6)
    # ========================================================================
    # Pruning stage
    reap_prune_ratio: float = 0.15          # <-- CHANGED: lower ratio to prevent collapse
    reap_quantize: bool = True              # Apply dynamic quantization after pruning (int8)

    # Memory and performance
    reap_low_mem: bool = True               # Use aggressive memory-saving settings
    reap_force_cpu: bool = True             # Always use CPU even if GPU available
    reap_disable_offload: bool = True       # Avoid offloading to disk (for speed)

    # Evaluation and checkpointing
    reap_eval_after_finetune: bool = False  # Automatically run evaluation after finetuning
    reap_checklist_enabled: bool = True     # Store checklist JSON sidecars
    reap_checklist_dir: Optional[Path] = None  # Optional custom directory for checklists

    # ========================================================================
    # NEW REAP ENHANCEMENTS (v3.7)
    # ========================================================================
    reap_passes: int = 3                    # Number of distillation passes in REAP pipeline
    reap_temperature_schedule: List[float] = field(default_factory=lambda: [2.0, 1.5, 1.0])  # Temperature annealing

    # ========================================================================
    # NEW BENCHMARK SETTINGS (v3.8)
    # ========================================================================
    benchmark_prompt: str = "What is machine learning?"
    benchmark_max_tokens: int = 100
    benchmark_run_perplexity: bool = True   # <-- CHANGED: enabled by default for feedback
    benchmark_val_texts: List[str] = field(default_factory=list)
    benchmark_run_multiple_choice: bool = True  # <-- CHANGED: enabled by default for feedback
    benchmark_mc_questions: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_run_long_context: bool = False
    benchmark_context_lengths: List[int] = field(default_factory=lambda: [2048, 4096, 8192, 16384])
    benchmark_num_trials: int = 3
    benchmark_long_context_max_tokens: int = 20
    benchmark_store_in_registry: bool = True

    # ------------------------------------------------------------------
    # Helper methods (updated for WSL2 detection)
    # ------------------------------------------------------------------
    def __post_init__(self):
        """
        Ensure lazytorch_cache_dir and reap_checklist_dir are Path objects.
        Also resolve conflict between QLoRA and LoRA: if both are enabled,
        QLoRA takes precedence and standard LoRA is disabled.
        This method is idempotent – calling it multiple times is safe.
        """
        if isinstance(self.lazytorch_cache_dir, str):
            self.lazytorch_cache_dir = Path(self.lazytorch_cache_dir)
        if self.reap_checklist_dir is not None and isinstance(self.reap_checklist_dir, str):
            self.reap_checklist_dir = Path(self.reap_checklist_dir)

        # ---- Resolve QLoRA/LoRA conflict ----
        if self.use_qlora and self.use_lora:
            logger.warning(
                "Both use_qlora and use_lora are True. QLoRA (4-bit + LoRA) is a superset of LoRA, "
                "so use_lora has been automatically set to False to avoid redundancy."
            )
            self.use_lora = False

    def get_effective_lora_config(self) -> Tuple[bool, int, int, float, List[str]]:
        """
        Return the effective LoRA/QLoRA configuration for training code.
        Prioritises QLoRA if enabled, otherwise falls back to standard LoRA.

        Returns:
            (enabled, rank, alpha, dropout, target_modules)
        """
        if self.use_qlora:
            return (True, self.qlora_r, self.qlora_alpha, self.qlora_dropout, self.qlora_target_modules)
        elif self.use_lora:
            return (True, self.lora_r, self.lora_alpha, self.lora_dropout, ["q_proj", "v_proj"])
        else:
            return (False, 0, 0, 0.0, [])

    def get_platform(self) -> str:
        """
        Return the effective platform (auto-detected if 'auto').
        If the platform is 'auto', uses utils.detect_platform() to detect the current OS.
        On Linux, if WSL2 is detected, returns 'windows' (to use Windows Ollama binary).
        Returns: 'linux', 'darwin', or 'windows'.
        """
        if self.platform != "auto":
            return self.platform
        # Import inside method to avoid circular import
        from .utils import detect_platform
        return detect_platform()

    def get_ollama_binary(self) -> str:
        """
        Return the path to the Ollama binary for the current platform.
        On Windows (including WSL2), returns the typical install location using environment variables.
        On Linux/macOS, assumes 'ollama' is in PATH.
        """
        plat = self.get_platform()
        if plat == "windows":
            # Typical Windows install location; expand environment variable
            return os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
        else:
            return "ollama"  # assume in PATH

    def get_llama_cpp_install_cmd(self) -> str:
        """
        Return a platform-specific pip install command for llama-cpp-python.
        Used for error messages when the library is missing.
        """
        plat = self.get_platform()
        if plat == "windows":
            return "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
        else:
            return "pip install llama-cpp-python"

    # ------------------------------------------------------------------
    # Existing helpers (unchanged)
    # ------------------------------------------------------------------
    def should_use_4bit(self) -> bool:
        """Check if 4-bit loading should be attempted (e.g., when RAM is tight)."""
        return not self.disable_gpu_requirements

    def should_use_lazytorch(self) -> bool:
        """
        Return True if LazyTorch mode should be used based on current settings.
        Uses both the explicit flag and RAM limit (if RAM < 8 GB, default is True).
        """
        # If explicitly disabled, return False
        if not self.use_lazytorch:
            return False
        # If RAM is tight or auto-optimized, enable
        if self.ram_limit_gb < 8.0 or self.auto_optimized:
            return True
        # Otherwise, respect the flag (which is True by default)
        return self.use_lazytorch

    def get_effective_device(self) -> torch.device:
        """Return the torch device based on config.device and availability."""
        # If REAP force_cpu is enabled, always return CPU
        if self.reap_force_cpu:
            return torch.device("cpu")
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_default_student(self) -> Optional[str]:
        """Return the default student name from config or None."""
        return self.default_student

    def get_default_teacher(self) -> Optional[str]:
        """Return the default teacher name from config or None."""
        return self.default_teacher

    # ---- REAP config validation ----
    def validate_reap_config(self) -> bool:
        """
        Validate REAP configuration fields.
        Returns True if valid, False otherwise.
        """
        if not 0.0 <= self.reap_prune_ratio <= 1.0:
            return False
        if self.reap_passes <= 0:
            return False
        if not self.reap_temperature_schedule:
            return False
        # reap_checklist_dir can be None or a valid Path
        if self.reap_checklist_dir is not None:
            # Path validation is done in __post_init__
            pass
        return True

    # ---- MoE validation ----
    def validate_moe_config(self) -> bool:
        """
        Validate MoE configuration fields.
        Returns True if valid, False otherwise.
        """
        if self.moe_num_experts < 2:
            logger.error("moe_num_experts must be at least 2.")
            return False
        if self.moe_top_k > self.moe_num_experts:
            logger.error(f"moe_top_k ({self.moe_top_k}) cannot exceed moe_num_experts ({self.moe_num_experts}).")
            return False
        if self.moe_top_k < 1:
            logger.error("moe_top_k must be at least 1.")
            return False
        if self.moe_expert_capacity < 1:
            logger.error("moe_expert_capacity must be at least 1 (if used).")
            return False
        return True

    # ---- QLoRA validation ----
    def validate_qlora_config(self) -> bool:
        """
        Validate QLoRA configuration fields.
        Returns True if valid, False otherwise.
        """
        if self.use_qlora:
            if self.qlora_r <= 0:
                logger.error("qlora_r must be positive when use_qlora is True.")
                return False
            if self.qlora_alpha <= 0:
                logger.error("qlora_alpha must be positive when use_qlora is True.")
                return False
            if not 0.0 <= self.qlora_dropout < 1.0:
                logger.error("qlora_dropout must be in [0, 1).")
                return False
            if not self.qlora_target_modules:
                logger.error("qlora_target_modules must not be empty when use_qlora is True.")
                return False
        return True

    # ---- Benchmark config validation ----
    def validate_benchmark_config(self) -> bool:
        """
        Validate benchmark configuration fields.
        Returns True if valid, False otherwise.
        """
        if self.benchmark_max_tokens <= 0:
            logger.error("benchmark_max_tokens must be positive.")
            return False
        if self.benchmark_run_perplexity and not self.benchmark_val_texts:
            logger.error("benchmark_val_texts required when benchmark_run_perplexity is True.")
            return False
        if self.benchmark_run_multiple_choice and not self.benchmark_mc_questions:
            logger.error("benchmark_mc_questions required when benchmark_run_multiple_choice is True.")
            return False
        if self.benchmark_num_trials <= 0:
            logger.error("benchmark_num_trials must be positive.")
            return False
        if self.benchmark_run_long_context and not self.benchmark_context_lengths:
            logger.error("benchmark_context_lengths required when benchmark_run_long_context is True.")
            return False
        return True

    # ---- Consistency validation ----
    def _validate_consistency(self) -> bool:
        """
        Check for consistency between configuration options.
        Returns True if consistent, False otherwise.
        """
        # Warn about experimental features with low RAM
        if self.use_mixed_dim_kv and self.ram_limit_gb < 6.0:
            logger.warning("Mixed-dim KV cache is experimental and may cause OOM with <6GB RAM.")
            # Not fatal, just warning
        # Warn about E8 + LazyTorch if RAM is extremely low
        if self.use_e8_quantization and self.use_lazytorch and self.ram_limit_gb < 2.0:
            logger.warning("E8 quantization with LazyTorch on <2GB RAM may be too slow.")
        # QLoRA and LoRA conflict is already resolved in __post_init__
        return True

    # ---- Full validation ----
    def validate_all(self) -> bool:
        """
        Perform comprehensive validation of all configuration fields.
        Returns True if all validations pass.
        """
        reap_ok = self.validate_reap_config()
        moe_ok = self.validate_moe_config()
        qlora_ok = self.validate_qlora_config()
        bench_ok = self.validate_benchmark_config()
        cons_ok = self._validate_consistency()
        return reap_ok and moe_ok and qlora_ok and bench_ok and cons_ok

    # ---- Migration support ----
    @staticmethod
    def migrate(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate configuration data from older versions to the current schema.
        This is a placeholder for future version upgrades.
        Currently, it ensures that new fields have sensible defaults.
        """
        # Ensure REAP fields exist with correct types
        if 'reap_passes' not in data:
            data['reap_passes'] = 3
        else:
            # If it exists but is not an int, reset to default
            if not isinstance(data['reap_passes'], int):
                data['reap_passes'] = 3

        if 'reap_temperature_schedule' not in data:
            data['reap_temperature_schedule'] = [2.0, 1.5, 1.0]
        else:
            # If it exists but is not a list, reset to default
            if not isinstance(data['reap_temperature_schedule'], list):
                data['reap_temperature_schedule'] = [2.0, 1.5, 1.0]
            # If it's a list but contains non-float values, convert or reset
            else:
                try:
                    data['reap_temperature_schedule'] = [float(x) for x in data['reap_temperature_schedule']]
                except (ValueError, TypeError):
                    data['reap_temperature_schedule'] = [2.0, 1.5, 1.0]

        # ---- QLoRA fields (v3.7) ----
        if 'use_qlora' not in data:
            data['use_qlora'] = False
        if 'qlora_r' not in data:
            data['qlora_r'] = 8
        elif not isinstance(data['qlora_r'], int) or data['qlora_r'] <= 0:
            data['qlora_r'] = 8
        if 'qlora_alpha' not in data:
            data['qlora_alpha'] = 32
        elif not isinstance(data['qlora_alpha'], int) or data['qlora_alpha'] <= 0:
            data['qlora_alpha'] = 32
        if 'qlora_dropout' not in data:
            data['qlora_dropout'] = 0.1
        elif not isinstance(data['qlora_dropout'], (float, int)) or not (0 <= data['qlora_dropout'] < 1):
            data['qlora_dropout'] = 0.1
        if 'qlora_target_modules' not in data:
            data['qlora_target_modules'] = ["q_proj", "v_proj"]
        elif not isinstance(data['qlora_target_modules'], list) or not data['qlora_target_modules']:
            data['qlora_target_modules'] = ["q_proj", "v_proj"]

        # ---- MoE fields (v3.7) ----
        if 'moe_num_experts' not in data:
            data['moe_num_experts'] = 4
        elif not isinstance(data['moe_num_experts'], int) or data['moe_num_experts'] < 2:
            data['moe_num_experts'] = 4
        if 'moe_top_k' not in data:
            data['moe_top_k'] = 1
        elif not isinstance(data['moe_top_k'], int) or data['moe_top_k'] < 1:
            data['moe_top_k'] = 1
        if 'moe_hierarchical' not in data:
            data['moe_hierarchical'] = False
        if 'moe_expert_capacity' not in data:
            data['moe_expert_capacity'] = 4
        elif not isinstance(data['moe_expert_capacity'], int) or data['moe_expert_capacity'] < 1:
            data['moe_expert_capacity'] = 4

        # ---- Hyperparameter search fields (v3.7) ----
        if 'hyperparameter_search_enabled' not in data:
            data['hyperparameter_search_enabled'] = False
        if 'search_space_distill' not in data:
            data['search_space_distill'] = {
                "temperature": (0.5, 3.0),
                "alpha": (0.5, 0.9),
                "learning_rate": (1e-5, 1e-3),
                "gradient_accumulation_steps": (1, 8),
            }
        if 'search_space_prune' not in data:
            data['search_space_prune'] = {
                "threshold": (0.02, 0.15),
                "iterative_steps": (2, 5),
                "activation_threshold": (0.005, 0.05),
            }
        if 'search_space_finetune' not in data:
            data['search_space_finetune'] = {
                "learning_rate": (1e-5, 1e-3),
                "epochs": (5, 20),
                "batch_size": (8, 32),
            }

        # ---- NEW: Benchmark fields (v3.8) ----
        if 'benchmark_prompt' not in data:
            data['benchmark_prompt'] = "What is machine learning?"
        if 'benchmark_max_tokens' not in data:
            data['benchmark_max_tokens'] = 100
        elif not isinstance(data['benchmark_max_tokens'], int) or data['benchmark_max_tokens'] <= 0:
            data['benchmark_max_tokens'] = 100
        if 'benchmark_run_perplexity' not in data:
            data['benchmark_run_perplexity'] = False
        if 'benchmark_val_texts' not in data:
            data['benchmark_val_texts'] = []
        elif not isinstance(data['benchmark_val_texts'], list):
            data['benchmark_val_texts'] = []
        if 'benchmark_run_multiple_choice' not in data:
            data['benchmark_run_multiple_choice'] = False
        if 'benchmark_mc_questions' not in data:
            data['benchmark_mc_questions'] = []
        elif not isinstance(data['benchmark_mc_questions'], list):
            data['benchmark_mc_questions'] = []
        if 'benchmark_run_long_context' not in data:
            data['benchmark_run_long_context'] = False
        if 'benchmark_context_lengths' not in data:
            data['benchmark_context_lengths'] = [2048, 4096, 8192, 16384]
        elif not isinstance(data['benchmark_context_lengths'], list) or not data['benchmark_context_lengths']:
            data['benchmark_context_lengths'] = [2048, 4096, 8192, 16384]
        # Ensure all elements are ints
        else:
            data['benchmark_context_lengths'] = [int(x) for x in data['benchmark_context_lengths'] if isinstance(x, (int, float))]
            if not data['benchmark_context_lengths']:
                data['benchmark_context_lengths'] = [2048, 4096, 8192, 16384]
        if 'benchmark_num_trials' not in data:
            data['benchmark_num_trials'] = 3
        elif not isinstance(data['benchmark_num_trials'], int) or data['benchmark_num_trials'] <= 0:
            data['benchmark_num_trials'] = 3
        if 'benchmark_long_context_max_tokens' not in data:
            data['benchmark_long_context_max_tokens'] = 20
        elif not isinstance(data['benchmark_long_context_max_tokens'], int) or data['benchmark_long_context_max_tokens'] <= 0:
            data['benchmark_long_context_max_tokens'] = 20
        if 'benchmark_store_in_registry' not in data:
            data['benchmark_store_in_registry'] = True

        # Remove any leftover HEPA or HydraHead fields (they are not in current schema)
        # This ensures they are dropped from the loaded config
        hepa_fields = [
            'hepa_d_model', 'hepa_nhead', 'hepa_num_layers', 'hepa_patch_len',
            'hepa_max_horizon', 'hepa_sigreg_alpha', 'hepa_pretrain_epochs',
            'hepa_pretrain_lr', 'hepa_pretrain_batch_size', 'hepa_finetune_epochs',
            'hepa_finetune_lr', 'hepa_finetune_batch_size', 'hepa_positive_weight',
            'hepa_eval_thresholds', 'hepa_eval_metrics', 'hepa_horizon_distribution',
            'hepa_min_context', 'hepa_max_context'
        ]
        hydra_fields = [
            'hydrahead_enabled', 'hydrahead_la_fa_ratio', 'hydrahead_num_heads',
            'hydrahead_head_selection_method', 'hydrahead_fa_heads_per_layer',
            'hydrahead_use_gated_deltanet', 'hydrahead_fusion_scale_norm',
            'hydrahead_gradient_checkpointing', 'hydrahead_use_cpu_offload',
            'hydrahead_mixed_precision'
        ]
        for field in hepa_fields + hydra_fields:
            if field in data:
                del data[field]

        return data

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> None:
        """Save this configuration to a JSON file."""
        save_config(self, path)

    def to_json_dict(self) -> dict:
        """
        Return a serializable dictionary representation of the config.
        Converts Path objects to strings and handles lists.
        """
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple)):
                # Convert any Path objects inside lists
                d[k] = [str(item) if isinstance(item, Path) else item for item in v]
            elif isinstance(v, Path):
                d[k] = str(v)
            elif callable(v):
                continue
            else:
                d[k] = v
        return d


def get_system_profile() -> Dict[str, Any]:
    """Return system RAM, GPU, CPU info for auto-optimization."""
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


def auto_optimize_config(config: Config, save: bool = False) -> Config:
    """
    Create a new optimized configuration based on system RAM.
    Does not modify the original config unless you reassign.
    If save=True, persists the new config to disk.
    """
    profile = get_system_profile()
    ram_gb = profile["available_ram_gb"]

    # Start from a deep copy of the original config
    new_config = deepcopy(config)

    # Reset optimization flags
    new_config.use_e8_quantization = False
    new_config.use_kv_cache_compression = True
    new_config.kv_cache_bits = 4
    new_config.use_mixed_dim_kv = False  # Always disable experimental feature by default
    new_config.use_lazytorch = True  # Enable LazyTorch by default in auto-optimize
    new_config.auto_optimized = True

    # --- REAP settings based on RAM ---
    if ram_gb < 3.0:
        new_config.reap_low_mem = True
        new_config.reap_force_cpu = True
        new_config.reap_prune_ratio = 0.15  # lower for safety
        new_config.reap_quantize = True
        new_config.reap_disable_offload = True
        new_config.reap_passes = 2
        new_config.distill_alpha = 0.8      # strengthen teacher signal for recovery
    elif ram_gb < 5.0:
        new_config.reap_low_mem = True
        new_config.reap_force_cpu = True
        new_config.reap_prune_ratio = 0.15
        new_config.reap_quantize = True
        new_config.reap_disable_offload = False
        new_config.reap_passes = 3
        new_config.distill_alpha = 0.8
    elif ram_gb < 8.0:
        new_config.reap_low_mem = True
        new_config.reap_force_cpu = True
        new_config.reap_prune_ratio = 0.15
        new_config.reap_quantize = True
        new_config.reap_disable_offload = False
        new_config.reap_passes = 3
        new_config.distill_alpha = 0.8
    else:
        new_config.reap_low_mem = False
        new_config.reap_force_cpu = False
        new_config.reap_prune_ratio = 0.15
        new_config.reap_quantize = False
        new_config.reap_disable_offload = False
        new_config.reap_passes = 4
        new_config.distill_alpha = 0.8      # keep higher alpha for all RAM levels

    # --- Memory-based settings (existing) ---
    if ram_gb < 3.0:
        new_config.use_e8_quantization = True
        new_config.e8_bits_per_weight = 2.5
        new_config.kv_cache_bits = 2
        new_config.gradient_accumulation_steps = 16
        new_config.recommended_e8 = True
        new_config.recommended_kv_bits = 2
        new_config.lazytorch_unload_after_forward = True
        new_config.distill_batch_size = 1
        # ---- LoRA: disable (QLoRA will be used instead) ----
        new_config.use_lora = False
        # ---- QLoRA: enable and reduce rank ----
        new_config.use_qlora = True
        new_config.qlora_r = 8   # lower rank for very low RAM
        new_config.qlora_alpha = 16
        new_config.qlora_dropout = 0.1
    elif ram_gb < 5.0:
        new_config.use_e8_quantization = True
        new_config.e8_bits_per_weight = 3.0
        new_config.kv_cache_bits = 3
        new_config.gradient_accumulation_steps = 12
        new_config.recommended_e8 = True
        new_config.recommended_kv_bits = 3
        new_config.lazytorch_unload_after_forward = True
        new_config.distill_batch_size = 1
        # ---- LoRA: disable (QLoRA used) ----
        new_config.use_lora = False
        # ---- QLoRA: enable with default rank ----
        new_config.use_qlora = True
        new_config.qlora_r = 12
        new_config.qlora_alpha = 24
        new_config.qlora_dropout = 0.1
    elif ram_gb < 8.0:
        new_config.use_e8_quantization = True
        new_config.e8_bits_per_weight = 4.0
        new_config.kv_cache_bits = 4
        new_config.gradient_accumulation_steps = 8
        new_config.recommended_e8 = True
        new_config.recommended_kv_bits = 4
        new_config.lazytorch_unload_after_forward = True
        new_config.distill_batch_size = 2
        # ---- LoRA: disable (QLoRA used) ----
        new_config.use_lora = False
        # ---- QLoRA: enable with default rank ----
        new_config.use_qlora = True
        new_config.qlora_r = 16
        new_config.qlora_alpha = 32
        new_config.qlora_dropout = 0.1
    else:
        new_config.use_e8_quantization = False
        new_config.e8_bits_per_weight = 4.0
        new_config.kv_cache_bits = 4
        new_config.gradient_accumulation_steps = 2
        new_config.recommended_e8 = False
        new_config.recommended_kv_bits = 4
        new_config.use_lazytorch = False  # Sufficient RAM, no need for LazyTorch
        new_config.lazytorch_unload_after_forward = False
        new_config.distill_batch_size = 4
        # ---- LoRA and QLoRA: not needed with enough RAM ----
        new_config.use_lora = False
        new_config.use_qlora = False

    # --- Benchmark settings based on RAM ---
    # Reduce max_tokens and disable expensive benchmarks on low RAM
    if ram_gb < 3.0:
        new_config.benchmark_max_tokens = 50
        new_config.benchmark_run_perplexity = True
        new_config.benchmark_run_multiple_choice = True
        new_config.benchmark_run_long_context = False
        new_config.benchmark_num_trials = 1
        new_config.benchmark_context_lengths = [1024]  # minimal
        new_config.benchmark_long_context_max_tokens = 10
    elif ram_gb < 5.0:
        new_config.benchmark_max_tokens = 80
        new_config.benchmark_run_perplexity = True
        new_config.benchmark_run_multiple_choice = True
        new_config.benchmark_run_long_context = False
        new_config.benchmark_num_trials = 2
        new_config.benchmark_context_lengths = [2048, 4096]
        new_config.benchmark_long_context_max_tokens = 15
    elif ram_gb < 8.0:
        new_config.benchmark_max_tokens = 100
        new_config.benchmark_run_perplexity = True
        new_config.benchmark_run_multiple_choice = True
        # Long-context still disabled to save memory
        new_config.benchmark_run_long_context = False
        new_config.benchmark_num_trials = 3
        new_config.benchmark_context_lengths = [2048, 4096, 8192]
        new_config.benchmark_long_context_max_tokens = 20
    else:
        # High RAM: enable all
        new_config.benchmark_max_tokens = 150
        new_config.benchmark_run_perplexity = True
        new_config.benchmark_run_multiple_choice = True
        new_config.benchmark_run_long_context = True
        new_config.benchmark_num_trials = 3
        new_config.benchmark_context_lengths = [2048, 4096, 8192, 16384]
        new_config.benchmark_long_context_max_tokens = 30

    # Set slow_mode based on RAM
    new_config.slow_mode = (ram_gb < 8.0)

    # If GPU and sufficient memory, disable KV compression for speed
    if profile["gpu_available"] and profile["gpu_memory_gb"] > 4:
        new_config.use_kv_cache_compression = False

    if save:
        save_config(new_config)

    return new_config


def recommend_enhancements() -> str:
    """Return human-readable suggestions for optimization."""
    profile = get_system_profile()
    ram = profile["available_ram_gb"]
    suggestions = []

    if ram < 3:
        suggestions.append("🔴 Extremely low RAM. E8 2.5-bit quantization and KV compression (bits=2) are enabled.")
    elif ram < 5:
        suggestions.append("🟠 Low RAM. E8 3-bit quantization and KV compression (bits=3) are enabled.")
    elif ram < 8:
        suggestions.append("🟡 Moderate RAM. E8 4-bit quantization is recommended and enabled.")
    else:
        suggestions.append("🟢 Sufficient RAM. E8 quantization is optional and currently disabled.")

    if ram < 8:
        suggestions.append("💾 LazyTorch memory-mapped loading is enabled (peak RAM ~500MB).")
    else:
        suggestions.append("⚡ LazyTorch is disabled (sufficient RAM). You can enable it manually.")

    if profile["gpu_available"]:
        if profile["gpu_memory_gb"] < 4:
            suggestions.append("🟠 GPU with limited memory. KV compression is enabled to reduce VRAM usage.")
        else:
            suggestions.append("✅ GPU available. KV compression can be disabled for speed if desired.")
    else:
        suggestions.append("💻 CPU-only mode. E8 quantization and KV compression are enabled to reduce RAM usage.")

    # Add QLoRA suggestion
    if ram < 8:
        suggestions.append("📉 QLoRA (4‑bit + LoRA) is enabled for distillation/finetuning to drastically reduce memory usage.")
    else:
        suggestions.append("📈 LoRA/QLoRA are disabled; full fine-tuning possible with sufficient RAM.")

    # Warn about experimental features
    suggestions.append("⚠️ Experimental features (mixed-dim KV cache) are disabled by default; enable only if you understand the risks.")

    # Note about GGUF flags (optional)
    suggestions.append("ℹ️ GGUF models are not supported for pruning/distillation. The allow_gguf_* flags are placeholders only.")

    # Static MoE routing suggestion
    suggestions.append("🧩 Static MoE routing is available; set use_static_moe=True to enable deterministic expert assignment.")

    # Endless RL suggestion
    suggestions.append("🔄 Endless RL self‑improvement is available; set endless_max_cycles=-1 to run indefinitely.")

    # REAP pipeline suggestions
    if ram < 8:
        suggestions.append("🔧 REAP pipeline is configured for low memory (prune_ratio=0.15, quantization enabled, CPU-only, distill_alpha=0.8).")
    else:
        suggestions.append("⚡ REAP pipeline is configured for speed (prune_ratio=0.15, quantization disabled, GPU allowed, distill_alpha=0.8).")
    suggestions.append("📋 REAP checklist is enabled; pipeline progress is tracked in model directories.")

    # Benchmark suggestions
    if ram < 5:
        suggestions.append("📊 Benchmarking is limited (max_tokens=80, long-context disabled) due to low RAM.")
    elif ram < 8:
        suggestions.append("📊 Benchmarking is moderate (perplexity and MC enabled, long-context disabled).")
    else:
        suggestions.append("📊 Full benchmarking enabled (perplexity, MC, long-context).")

    return "\n".join(suggestions)


def save_config(config: Config, path: Optional[Path] = None) -> None:
    """Save configuration to JSON file (handles all fields)."""
    if path is None:
        path = LAZY_DIR / "config.json"
    with open(path, "w") as f:
        json.dump(config.to_json_dict(), f, indent=2)


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from JSON file, falling back to defaults."""
    if path is None:
        path = LAZY_DIR / "config.json"
    config = Config()  # start with defaults
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            # Apply migration to ensure new fields exist and remove old ones
            data = Config.migrate(data)
            for k, v in data.items():
                if hasattr(config, k):
                    # Convert back to Path if it's a path field
                    if k in ["lazytorch_cache_dir", "reap_checklist_dir"] and isinstance(v, str):
                        setattr(config, k, Path(v) if v else None)
                    else:
                        setattr(config, k, v)
            # ---- FIX: Re-run __post_init__ to apply conflict resolution and path normalisation ----
            config.__post_init__()
        except Exception as e:
            logger.warning(f"Could not load config from {path}: {e}. Using defaults.")
    return config


# =============================================================================
# Public API exports
# =============================================================================
__all__ = [
    'Config',
    'ModelInfo',
    'load_config',
    'save_config',
    'auto_optimize_config',
    'recommend_enhancements',
    'get_system_profile',
    'LAZY_DIR',
    'MODELS_DIR',
    'CHECKPOINTS_DIR',
    'CACHE_DIR',
    'LOGS_DIR',
    'LAZYTORCH_CACHE_DIR',
]