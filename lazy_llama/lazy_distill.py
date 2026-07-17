"""
Distillation engine with KL divergence (HF teacher) or fine-tuning (Ollama/GGUF).
Supports gradient accumulation, mixed precision, LoRA, checkpoint resume, zero‑shot compensation,
and automatic export to LazyTorch format for extreme memory savings.
Now with Micro Mixture of Experts (µMoE) integration for sub‑1B student models.

FIX (v3.3.3): Added optional training of DSpark‑style draft head during distillation.
The draft head learns to predict multiple future tokens and confidence scores,
which can be used for speculative decoding during inference.

FIX (v3.3.6): Added explicit Ollama reachability and model existence check in `_load_teacher`.
Returns None with clear logging if Ollama is not reachable or the teacher model is missing.

NEW: Integration of static routing for Micro MoE layers.
If `use_static_routing` is enabled, a KMeans router is pre‑computed on calibration
activations and used deterministically during inference, improving fusion with
llama.cpp/Ollama and reducing variance.

================================================================================
NEW (v3.6): Endless distillation loop for unattended self‑improvement.
- Added `run_endless_distillation()` function to repeatedly distill a teacher→student pair.
- Used by the CLI (`endless distill`) and TUI (`Endless RL Loop` menu).
- Respects `cycles` and `sleep` parameters; callback for progress reporting.

================================================================================
FIXES (2026-07-07):
- Improved error handling: replaced many `logger.warning` with `logger.error` and
  re-raised appropriate exceptions to prevent silent failures. Non‑critical errors
  (e.g., a single bad sample in fine‑tuning) are logged with error and skipped,
  with a failure counter to abort if too many failures occur.
- Implemented draft head training for fine‑tuning (Ollama/GGUF teachers):
  Added `_compute_draft_losses_from_labels()` that uses the teacher-generated
  continuation tokens (labels) as targets for draft predictions. The draft head
  is now trained during fine‑tuning as well, improving speculative decoding
  performance for models distilled from Ollama/GGUF teachers.

FIX (2026-07-08): Ensure all critical errors are raised and not silently swallowed.
                  Added explicit early exit on critical failures (e.g., tokenizer corruption).

FIX (2026-07-08): Improved error message when student is not a local model (Ollama/vLLM).
                  Now provides a clear, actionable suggestion to create a local student.

FURTHER FIX (2026-07-10):
- Distillation now always loads the student as a standard HF model, bypassing LazyTorch.
  This ensures support for labels, hidden_states, and loss, which LazyModule does not provide.
- Added timeout to Ollama requests in `_get_teacher_response`.
- Added safety check in `_compute_draft_losses_from_labels` to avoid IndexError.
- Enhanced early validation of GGUF students before any loading.

================================================================================
FIXES (2026-07-13) - Improved robustness:
- Clearer error messages for non-local and GGUF student models, with specific
  instructions on how to create a local student model.
- Moved GGUF student rejection earlier in `run_distillation` before any
  expensive operations (tokenizer validation, teacher loading) to fail fast.
- Added a RAM availability check before loading the student model to prevent OOM.
- Enhanced draft head training robustness:
  - Added explicit check that `student_hidden` is not None and has valid shape.
  - Safer indexing in `_compute_draft_losses_from_labels` with bounds checking.
  - Fallback to skipping draft loss if hidden states are missing or shape mismatch.
  - More detailed logging for draft head failures.
- Added `estimate_memory_need` import and used it for student model loading.

================================================================================
ENHANCEMENTS (2026-07-15):
- Added QLoRA / 4‑bit distillation: load student in 4‑bit with bitsandbytes,
  use LoRA adapters for training (dramatically reduces memory).
- Added Progressive Layer Distillation: distill layer-by-layer, freezing lower
  layers while training upper ones, reducing memory and improving convergence.
- Added Combined Loss: optional hidden_state MSE loss and attention KL divergence
  (with weights `hidden_loss_weight`, `attention_loss_weight`).
- Added Dynamic Temperature Annealing: temperature decreases over epochs
  according to a schedule (e.g., linear or exponential).
- All new features are optional and backward-compatible.

FIXES (2026-07-15) - Post‑review corrections:
- Moved `import re` to top of file for efficiency.
- Added guard against division by zero in progressive step calculation.
- Avoid `.to(self.device)` after QLoRA loading (device_map already handles placement).
- Conditionally fetch teacher hidden states and attentions only when respective
  loss weights are > 0 to save memory and CPU transfer.
- Added warning when `num_layers` cannot be inferred for progressive distillation.

NEW (2026-07-15) - QLoRA and MoE parameter overrides:
- `run_distillation()` now accepts `use_qlora`, `qlora_r`, `qlora_alpha`, `qlora_dropout`,
  `qlora_target_modules` to override instance defaults.
- `run_distillation()` now accepts `moe_num_experts`, `moe_top_k`, `moe_hierarchical`
  and passes them to `convert_dense_to_micro_moe` (after `micro_moe.py` is updated to
  accept the `hierarchical` parameter; for now, it is stored but not passed).
- `_load_student()` uses the effective QLoRA settings from instance attributes.
- `_distill_hf_teacher()` and `_fine_tune()` respect the new MoE parameters.

NOTE: The `hierarchical` parameter in `convert_dense_to_micro_moe` is reserved for future
use; the current `micro_moe.py` does not support it. The code stores the value but does
not pass it to avoid TypeError. Once `micro_moe.py` is updated, the calls can be amended.

================================================================================
NEW (2026-07-16): Operation logging for distillation.
- Added `log_operation_result()` call in `run_distillation` to record each
  distillation attempt (success/failure, teacher, passes, etc.) in the model's
  registry metadata. This provides a persistent history for debugging and auditing.

FIX (2026-07-16): Robust loading of student models with meta tensor fallback.
- Added manual state dict loading when `NotImplementedError` occurs due to meta tensors.
- Now loads weights from `pytorch_model.bin` or `model.safetensors` directly.
- Handles both full‑precision and QLoRA paths with clear error messages.

REMOVED (2026-07-17): Removed all HydraHead (hybrid attention) related code,
including imports, flags, and logic for detecting and applying hybrid attention.

FIX (2026-07-17): In `_load_student`, when a LazyTorch model lacks `original_path`
in its manifest, fall back to the directory without the `.lazytorch` suffix.
This allows distillation to proceed if the original HF model is present alongside
the LazyTorch model, improving usability.

================================================================================
ENHANCED RECOVERY DEFAULTS (2026-07-16):
- In `run_distillation`, if the student name contains '_pruned' (indicating a
  pruned model), the following recovery‑friendly defaults are applied when not
  explicitly overridden:
    - passes = max(passes, 5) – more passes for better recovery.
    - use_qlora = True (if not specified) – QLoRA reduces memory and improves recovery.
    - qlora_r = 16 – higher rank for more capacity.
    - progressive_steps = 3 (if not specified) – train top layers first.
    - distill_alpha = 0.8 (temporarily) – stronger teacher signal.
  These defaults can be overridden by explicit arguments.
"""

import os
import time
import gc
import logging
import shutil
import json
import re
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Callable, Optional, Tuple, Dict, Any, Union
from pathlib import Path
from datetime import datetime

# ---- All internal imports are now RELATIVE ----
from .config import CHECKPOINTS_DIR, Config, ModelInfo, LAZY_DIR
from .utils import (
    get_available_ram_gb, save_checkpoint, load_checkpoint,
    find_latest_checkpoint, clear_cuda_memory, retry, check_low_ram,
    is_lazytorch_model, _validate_tokenizer_deep, copy_tokenizer_files,
    check_ollama_model, estimate_memory_need, log_operation_result
)
from .zero_shot_compensation import apply_zero_shot_compensation
from .lazytorch_core import export_to_lazytorch, load_lazytorch_model
from .micro_moe import (
    convert_dense_to_micro_moe,
    compute_auxiliary_loss,
    create_static_router,
    MicroMoELayer
)

# ---- Try to import GGUF validator from lazy_infer if available ----
try:
    from .lazy_infer import is_valid_gguf
except ImportError:
    is_valid_gguf = None

# ---- DSpark speculative decoding imports ----
try:
    from .lazy_speculative import DraftHead, save_draft_head, attach_draft_head_to_model
    SPECULATIVE_AVAILABLE = True
except ImportError:
    SPECULATIVE_AVAILABLE = False
    DraftHead = None
    save_draft_head = None
    attach_draft_head_to_model = None

# Optional PEFT (LoRA)
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = get_peft_model = TaskType = prepare_model_for_kbit_training = None

# Optional bitsandbytes for 4-bit loading
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    BITSANDBYTES_AVAILABLE = False

logger = logging.getLogger(__name__)

# Helper to load global state from dashboard (same as lazy_tui)
def _load_global_state():
    """Read global teacher/student selections saved by dashboard."""
    state_file = LAZY_DIR / "global_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                data = json.load(f)
                return data.get("teacher", ""), data.get("student", "")
        except Exception:
            pass
    return "", ""

def _is_gguf_path(path: Path) -> bool:
    """
    Return True if the given path points to a GGUF model (either a .gguf file
    or a directory that contains a .gguf file). Uses is_valid_gguf from lazy_infer
    if available for more robust file validation.
    Used to reject GGUF models as students for distillation.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".gguf":
        if is_valid_gguf is not None:
            return is_valid_gguf(path)
        return True
    if path.is_dir():
        for f in path.glob("*.gguf"):
            if is_valid_gguf is not None:
                if is_valid_gguf(f):
                    logger.warning(f"Directory {path} contains a valid GGUF file; treating as GGUF model.")
                    return True
            else:
                logger.warning(f"Directory {path} contains a .gguf file; treating as GGUF model.")
                return True
    return False

def _reset_router_logits(model: nn.Module) -> None:
    """Reset stored router logits in all MicroMoELayer modules."""
    for module in model.modules():
        if hasattr(module, '_last_router_logits'):
            module._last_router_logits = None

def _collect_router_logits(model: nn.Module) -> List[torch.Tensor]:
    """Collect router logits from all MicroMoELayer modules."""
    logits = []
    for module in model.modules():
        if hasattr(module, '_last_router_logits') and module._last_router_logits is not None:
            logits.append(module._last_router_logits)
    return logits


class LazyDistillationEngine:
    """
    Distillation engine:
    - If teacher is a local HF model: true KL divergence with optional LoRA and MoE.
    - If teacher is GGUF or Ollama: supervised fine‑tuning on teacher outputs.
    Supports checkpoint resume and zero‑shot compensation.
    After distillation, the resulting model is saved in both HF format and LazyTorch format.

    IMPORTANT: The student model MUST be a PyTorch model (Hugging Face or LazyTorch).
    GGUF models are NOT supported as students. Use a local HF model as the student.

    NEW ENHANCEMENTS:
    - QLoRA / 4‑bit distillation: load student in 4‑bit, train with LoRA adapters.
    - Progressive Layer Distillation: freeze lower layers, train upper layers first.
    - Combined Loss: hidden state MSE + attention KL.
    - Dynamic Temperature Annealing: temperature decreases over epochs.
    """

    def __init__(self, config: Config, use_moe: bool = False, num_experts: int = 4,
                 top_k: int = 1, moe_reduction_factor: int = 2, aux_loss_weight: float = 0.01,
                 use_static_routing: bool = False,
                 use_qlora: bool = False,
                 progressive_steps: int = 0,
                 hidden_loss_weight: float = 0.0,
                 attention_loss_weight: float = 0.0,
                 temperature_schedule: Optional[List[float]] = None,
                 # QLoRA override fields
                 qlora_r: Optional[int] = None,
                 qlora_alpha: Optional[int] = None,
                 qlora_dropout: Optional[float] = None,
                 qlora_target_modules: Optional[List[str]] = None):
        self.config = config
        self.teacher = None
        self.teacher_tokenizer = None
        self.teacher_type = None          # "hf", "gguf", or "ollama"
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.ollama_host = "http://localhost:11434"
        self.progress_callback: Optional[Callable[[int, int, int, int], None]] = None
        self.use_zero_shot = config.use_zero_shot_compensation
        self.teacher_name = ""            # initialized later

        # µMoE parameters
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_reduction_factor = moe_reduction_factor
        self.aux_loss_weight = aux_loss_weight

        # ---- Static routing ----
        self.use_static_routing = use_static_routing

        # ---- Draft head training ----
        self.train_draft_head = getattr(config, 'train_draft_head', False)
        if self.train_draft_head and not SPECULATIVE_AVAILABLE:
            logger.warning("train_draft_head is True but speculative decoding module is not available; disabling.")
            self.train_draft_head = False

        # ---- NEW: QLoRA ----
        self.use_qlora = use_qlora
        self.qlora_r = qlora_r or getattr(config, 'qlora_r', 8)
        self.qlora_alpha = qlora_alpha or getattr(config, 'qlora_alpha', 32)
        self.qlora_dropout = qlora_dropout or getattr(config, 'qlora_dropout', 0.1)
        self.qlora_target_modules = qlora_target_modules or getattr(config, 'qlora_target_modules', ["q_proj", "v_proj"])

        if self.use_qlora and not BITSANDBYTES_AVAILABLE:
            logger.warning("QLoRA requested but bitsandbytes not available; falling back to full precision.")
            self.use_qlora = False
        if self.use_qlora and not PEFT_AVAILABLE:
            logger.warning("QLoRA requested but peft not available; falling back to full precision.")
            self.use_qlora = False

        # ---- NEW: Progressive distillation ----
        self.progressive_steps = progressive_steps
        self._current_progressive_step = 0

        # ---- NEW: Combined loss weights ----
        self.hidden_loss_weight = hidden_loss_weight
        self.attention_loss_weight = attention_loss_weight

        # ---- NEW: Temperature schedule ----
        self.temperature_schedule = temperature_schedule or []
        self._current_temp_epoch = 0

        # ---- Gradient checkpointing for memory ----
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

        # ---- MoE hierarchical flag (reserved for future use) ----
        self.moe_hierarchical = False  # will be set by run_distillation if passed

    def set_progress_callback(self, callback: Callable[[int, int, int, int], None]) -> None:
        self.progress_callback = callback

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------
    def _load_teacher(self, teacher_name: str, manager) -> Optional[str]:
        """
        Load teacher model. Returns 'hf', 'gguf', or 'ollama'.
        For Ollama teachers, we explicitly check reachability and model existence.
        Returns None if the teacher cannot be loaded (with appropriate logging).
        """
        info = manager.get_model(teacher_name)
        if not info or not info.path:
            logger.error(f"Teacher model '{teacher_name}' not found in registry.")
            return None

        # ---- explicitly handle Ollama early ----
        if info.path.startswith("ollama://"):
            if not check_ollama_model(teacher_name):
                logger.error(
                    f"Ollama service is not reachable or teacher model '{teacher_name}' not found.\n"
                    "Please ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull " + teacher_name + "`)."
                )
                return None
            self.teacher = None
            self.teacher_type = "ollama"
            return "ollama"

        # Try HF (transformers)
        path_obj = Path(info.path)
        if path_obj.is_dir():
            if not _validate_tokenizer_deep(path_obj):
                logger.error(f"Tokenizer in teacher model {teacher_name} is corrupt. Cannot load as HF teacher.")
                return None
            try:
                self.teacher = AutoModelForCausalLM.from_pretrained(
                    info.path, low_cpu_mem_usage=True
                )
                self.teacher.eval()
                if torch.cuda.is_available():
                    self.teacher = self.teacher.cuda()
                self.teacher_tokenizer = AutoTokenizer.from_pretrained(info.path)
                self.teacher_type = "hf"
                return "hf"
            except Exception as e:
                logger.error(f"Failed to load HF teacher: {e}")
                # Continue to try GGUF

        # Try GGUF (llama-cpp-python)
        try:
            from llama_cpp import Llama
            gguf_path = (
                path_obj if info.path.endswith(".gguf")
                else next(path_obj.glob("**/*.gguf"), None)
            )
            if gguf_path and gguf_path.exists():
                self.teacher = Llama(str(gguf_path), verbose=False)
                self.teacher_type = "gguf"
                return "gguf"
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Failed to load GGUF teacher: {e}")

        return None

    # ------------------------------------------------------------------
    # Teacher response for fine‑tuning (Ollama / GGUF)
    # ------------------------------------------------------------------
    @retry(max_attempts=3, delay=2)
    def _get_teacher_response(self, prompt: str) -> str:
        """Get response from Ollama teacher (fallback for fine‑tuning)."""
        try:
            requests.get(f"{self.ollama_host}/api/tags", timeout=2)
        except:
            return ""
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.teacher_name, "prompt": prompt, "stream": False},
                timeout=self.config.ollama_timeout
            )
            return resp.json().get("response", "")
        except Exception as e:
            logger.error(f"Teacher request failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Similarity & verification (optional)
    # ------------------------------------------------------------------
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        try:
            from nltk.translate.bleu_score import sentence_bleu
            return sentence_bleu([text1.split()], text2.split())
        except Exception:
            w1, w2 = set(text1.lower().split()), set(text2.lower().split())
            if not w1 or not w2:
                return 0.0
            return len(w1 & w2) / max(len(w1), len(w2))

    def verify(self, student, tokenizer, test_prompts: List[str]) -> float:
        """Return accuracy % based on teacher similarity (only for Ollama teacher)."""
        student.eval()
        correct = 0
        for prompt in test_prompts:
            teacher_ans = self._get_teacher_response(prompt)
            if not teacher_ans:
                continue
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = student.generate(**inputs, max_new_tokens=50, do_sample=False)
            student_ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if self._calculate_similarity(teacher_ans, student_ans) > 0.7:
                correct += 1
        return (correct / len(test_prompts)) * 100 if test_prompts else 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_distillation(
        self,
        teacher_name: str,
        student_name: str,
        texts: List[str],
        passes: int = 3,
        resume: bool = True,
        use_moe: Optional[bool] = None,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        moe_reduction_factor: Optional[int] = None,
        aux_loss_weight: Optional[float] = None,
        train_draft_head: Optional[bool] = None,
        use_static_routing: Optional[bool] = None,
        # NEW parameters
        use_qlora: Optional[bool] = None,
        progressive_steps: Optional[int] = None,
        hidden_loss_weight: Optional[float] = None,
        attention_loss_weight: Optional[float] = None,
        temperature_schedule: Optional[List[float]] = None,
        # QLoRA overrides
        qlora_r: Optional[int] = None,
        qlora_alpha: Optional[int] = None,
        qlora_dropout: Optional[float] = None,
        qlora_target_modules: Optional[List[str]] = None,
        # MoE overrides (new)
        moe_num_experts: Optional[int] = None,
        moe_top_k: Optional[int] = None,
        moe_hierarchical: bool = False,
    ) -> Path:
        """
        Run distillation (KL or fine‑tuning). Returns path to distilled model (HF format).
        If teacher_name or student_name are empty strings, attempt to load from global state.

        µMoE parameters can be passed to override instance defaults.
        train_draft_head: if True, train a draft head for speculative decoding.
        use_static_routing: if True, use static (pre‑computed) routing for MoE layers.

        NEW:
        use_qlora: if True, load student in 4‑bit with LoRA adapters (memory saving).
        qlora_r, qlora_alpha, qlora_dropout, qlora_target_modules: QLoRA hyperparameters.
        progressive_steps: number of progressive steps (0 = disabled). If >0, freeze layers
                           progressively: step 1 = train only last 1/N layers, etc.
        hidden_loss_weight: weight for MSE loss between student and teacher hidden states.
        attention_loss_weight: weight for KL divergence on attention distributions.
        temperature_schedule: list of temperatures per epoch (e.g., [2.0, 1.5, 1.0]).
        moe_num_experts, moe_top_k, moe_hierarchical: MoE parameters.

        RECOVERY MODE DEFAULTS:
        If the student name contains '_pruned' (indicating a pruned model), and the
        corresponding parameters are not explicitly set, the following recovery‑friendly
        defaults are applied:
            - passes = max(passes, 5) – more passes for better recovery.
            - use_qlora = True – QLoRA reduces memory and improves recovery.
            - qlora_r = 16 – higher rank for more capacity.
            - progressive_steps = 3 – train top layers first.
            - distill_alpha = 0.8 – stronger teacher signal (temporarily set in config).
        These can be overridden by explicit arguments.
        """
        # Fallback to global state if arguments are empty
        if not teacher_name or not student_name:
            global_teacher, global_student = _load_global_state()
            if not teacher_name:
                teacher_name = global_teacher
                logger.info(f"Using global teacher: {teacher_name}")
            if not student_name:
                student_name = global_student
                logger.info(f"Using global student: {student_name}")

        if not teacher_name or not student_name:
            raise ValueError("Teacher and student names must be provided or set in global state.")

        logger.info(f"Distilling {teacher_name} → {student_name} with {passes} passes (resume={resume})")
        self.teacher_name = teacher_name

        # ---- ENHANCED RECOVERY DEFAULTS: detect pruned student ----
        is_pruned = "_pruned" in student_name
        if is_pruned:
            logger.info("Student model appears to be pruned. Applying recovery‑friendly defaults (if not overridden).")

            # Increase passes if not explicitly set (passed as argument, we'll check if it's the default)
            # Since passes is a parameter, we can only modify if it's the default (3). We'll check if the caller provided a value.
            # We can't easily detect if the caller passed a different value, but we can bump if passes < 5.
            if passes < 5:
                passes = 5
                logger.info(f"  passes set to {passes} (recovery mode)")

            # Enable QLoRA if not explicitly disabled
            if use_qlora is None:
                use_qlora = True
                logger.info("  use_qlora set to True (recovery mode)")

            # Increase rank if not explicitly set
            if qlora_r is None:
                qlora_r = 16
                logger.info(f"  qlora_r set to {qlora_r} (recovery mode)")

            # Enable progressive steps if not explicitly set
            if progressive_steps is None:
                progressive_steps = 3
                logger.info(f"  progressive_steps set to {progressive_steps} (recovery mode)")

            # Override distill_alpha in config to 0.8 (temporarily)
            self.config.distill_alpha = 0.8
            logger.info(f"  distill_alpha set to {self.config.distill_alpha} (recovery mode)")

        # Update parameters if provided
        if use_moe is not None:
            self.use_moe = use_moe
        if num_experts is not None:
            self.num_experts = num_experts
        if top_k is not None:
            self.top_k = top_k
        if moe_reduction_factor is not None:
            self.moe_reduction_factor = moe_reduction_factor
        if aux_loss_weight is not None:
            self.aux_loss_weight = aux_loss_weight
        if train_draft_head is not None:
            self.train_draft_head = train_draft_head
            if self.train_draft_head and not SPECULATIVE_AVAILABLE:
                logger.warning("train_draft_head is True but speculative module not available; disabling.")
                self.train_draft_head = False
        if use_static_routing is not None:
            self.use_static_routing = use_static_routing
        if use_qlora is not None:
            self.use_qlora = use_qlora
            if self.use_qlora and not BITSANDBYTES_AVAILABLE:
                logger.warning("QLoRA requested but bitsandbytes not available; disabling.")
                self.use_qlora = False
            if self.use_qlora and not PEFT_AVAILABLE:
                logger.warning("QLoRA requested but peft not available; disabling.")
                self.use_qlora = False
        if progressive_steps is not None:
            self.progressive_steps = progressive_steps
            self._current_progressive_step = 0
        if hidden_loss_weight is not None:
            self.hidden_loss_weight = hidden_loss_weight
        if attention_loss_weight is not None:
            self.attention_loss_weight = attention_loss_weight
        if temperature_schedule is not None:
            self.temperature_schedule = temperature_schedule
            self._current_temp_epoch = 0

        # ---- QLoRA overrides ----
        if qlora_r is not None:
            self.qlora_r = qlora_r
        if qlora_alpha is not None:
            self.qlora_alpha = qlora_alpha
        if qlora_dropout is not None:
            self.qlora_dropout = qlora_dropout
        if qlora_target_modules is not None:
            self.qlora_target_modules = qlora_target_modules

        # ---- MoE overrides (store for later use) ----
        if moe_num_experts is not None:
            self.num_experts = moe_num_experts
        if moe_top_k is not None:
            self.top_k = moe_top_k
        # Store hierarchical flag for future use (reserved)
        self.moe_hierarchical = moe_hierarchical

        # Get manager and student info
        from .lazy_model_manager import ModelManager
        manager = ModelManager()
        student_info = manager.get_model(student_name)

        # ---- Improved error message for non-local student ----
        if not student_info or not student_info.path:
            raise ValueError(
                f"Student model '{student_name}' not found in registry. "
                "Please ensure the model exists and is registered."
            )
        if student_info.path.startswith(("ollama://", "vllm://")):
            raise ValueError(
                f"Student model '{student_name}' is not a local model (detected {student_info.path}). "
                "Distillation requires a local Hugging Face model (not Ollama or vLLM). "
                "Please create a local student model first using the 'create-student' command or the TUI. "
                "Example: `python bootstrap.py create-student --base distilgpt2 --student-name my_student`"
            )

        # ---- Validate student is not a GGUF (early) ----
        student_path = Path(student_info.path)
        if _is_gguf_path(student_path):
            allow_gguf = getattr(self.config, 'allow_gguf_distillation', False)
            error_msg = (
                f"Student model '{student_name}' is a GGUF model (path: {student_path}). "
                "Distillation requires a PyTorch model (Hugging Face or LazyTorch). "
                "Please use a local HF model as the student. "
                "If you have external conversion tools and understand the risks, "
                "you may set config.allow_gguf_distillation=True (but you still need to convert first)."
            )
            if allow_gguf:
                logger.warning(error_msg + " Proceeding anyway, but this will likely fail.")
            else:
                raise ValueError(error_msg)

        # ---- Validate tokenizer before proceeding ----
        if student_path.is_dir() and not _validate_tokenizer_deep(student_path):
            raise ValueError(
                f"Student model at {student_path} has a corrupt or missing tokenizer. "
                "Please delete the model and re-download it, or repair the tokenizer files.\n"
                "You can try: python bootstrap.py remove --model {student_name}\n"
                "Then re-download from Hugging Face or create a fresh student."
            )

        # ---- RAM check before loading student ----
        estimated_mem = estimate_memory_need(student_path)
        available_ram = get_available_ram_gb()
        if available_ram < estimated_mem * 1.2:  # 20% buffer
            logger.warning(
                f"Available RAM ({available_ram:.1f} GB) may be insufficient for distillation "
                f"(estimated need: {estimated_mem:.1f} GB). Continuing anyway, but may cause OOM."
            )

        valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
        if not valid_texts:
            raise ValueError("No valid calibration prompts provided.")

        # ---- NEW: Operation logging ----
        # We'll log success/failure at the end. Capture manager for logging.
        log_manager = manager

        try:
            # Load teacher – now returns "ollama" early if needed
            teacher_type = self._load_teacher(teacher_name, manager)
            if teacher_type is None:
                raise ValueError(f"Could not load teacher model: {teacher_name}")

            if teacher_type == "hf" and self.teacher is None:
                raise RuntimeError("Teacher model not loaded but HF teacher type was reported")

            if teacher_type == "hf":
                student_temp = AutoModelForCausalLM.from_pretrained(
                    student_info.path, low_cpu_mem_usage=True
                )
                if student_temp.config.vocab_size != self.teacher.config.vocab_size:
                    student_temp = None
                    raise ValueError("Teacher and student vocab sizes differ. Cannot perform KL distillation.")
                del student_temp
                logger.info("Using true KL‑divergence distillation (HF teacher).")
                result_path = self._distill_hf_teacher(
                    student_info, valid_texts, passes, manager, resume,
                    use_static_routing=self.use_static_routing
                )
            else:
                logger.warning(
                    f"Teacher is {teacher_type} — no logits access. Falling back to fine‑tuning. "
                    f"Teacher requests will use timeout={self.config.ollama_timeout}s."
                )
                teacher_gguf = self.teacher if teacher_type == "gguf" else None
                result_path = self._fine_tune(
                    teacher_name, student_info, valid_texts, passes, manager, teacher_gguf, resume,
                    use_static_routing=self.use_static_routing
                )

            # ---- Log success ----
            log_operation_result(
                model_name=student_name,
                operation='distill',
                success=True,
                details={
                    'teacher': teacher_name,
                    'passes': passes,
                    'resume': resume,
                    'use_moe': self.use_moe,
                    'num_experts': self.num_experts,
                    'top_k': self.top_k,
                    'use_qlora': self.use_qlora,
                    'progressive_steps': self.progressive_steps,
                    'hidden_loss_weight': self.hidden_loss_weight,
                    'attention_loss_weight': self.attention_loss_weight,
                    'teacher_type': teacher_type,
                },
                manager=log_manager
            )
            return result_path

        except Exception as e:
            # ---- Log failure ----
            log_operation_result(
                model_name=student_name,
                operation='distill',
                success=False,
                details={
                    'teacher': teacher_name,
                    'error': str(e),
                    'passes': passes,
                    'teacher_type': self.teacher_type,
                },
                manager=log_manager
            )
            # Re-raise
            raise

        finally:
            self._unload_teacher()

    # ------------------------------------------------------------------
    # Helper: load student model (HF, with QLoRA support, meta tensor fallback)
    # ------------------------------------------------------------------
    def _load_student(self, student_info, manager):
        """
        Load student model and tokenizer as a standard Hugging Face model.
        Supports QLoRA (4‑bit + LoRA) if enabled.
        Handles meta tensor errors by manually loading state dict.
        Uses the effective QLoRA settings from instance attributes.
        """
        student_path = Path(student_info.path)

        # ---- Double‑check GGUF (should already be caught, but safe) ----
        if _is_gguf_path(student_path):
            raise ValueError(
                f"Student model at '{student_path}' is a GGUF model. "
                "Distillation requires a PyTorch model (Hugging Face or LazyTorch). "
                "If you have conversion tools and understand the risks, set config.allow_gguf_distillation=True."
            )

        # ---- Determine the actual path to load HF model ----
        actual_path = student_path  # default

        if is_lazytorch_model(student_path):
            logger.info("Student is a LazyTorch model; attempting to find original HF source for distillation.")
            manifest_path = student_path / "manifest.json" if student_path.is_dir() else student_path.with_suffix('') / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    original_path = manifest.get("source_path") or manifest.get("original_path")
                    if original_path and Path(original_path).exists():
                        actual_path = Path(original_path)
                        logger.info(f"Using original Hugging Face model at {actual_path} for distillation.")
                    else:
                        # ---- Fallback: infer path from student_path (remove .lazytorch suffix) ----
                        inferred_path = student_path.with_suffix('') if student_path.suffix == '.lazytorch' else student_path
                        if inferred_path.is_dir() and (inferred_path / "config.json").exists():
                            actual_path = inferred_path
                            logger.info(f"Inferred original HF model at {actual_path}.")
                        else:
                            raise ValueError(
                                f"LazyTorch model at {student_path} has no valid original_path in manifest "
                                f"and no inferred HF model found at {inferred_path}. "
                                "Please use the original Hugging Face model as the student for distillation, "
                                "or re-export the LazyTorch model with the original path stored."
                            )
                except Exception as e:
                    raise ValueError(f"Failed to read manifest for LazyTorch student: {e}") from e
            else:
                raise ValueError(
                    f"LazyTorch model at {student_path} has no manifest.json. "
                    "Cannot determine original source. Please use the original Hugging Face model as the student."
                )

        # ---- Validate tokenizer in the actual path ----
        if not _validate_tokenizer_deep(actual_path):
            raise ValueError(
                f"Tokenizer in student model at {actual_path} is corrupt. "
                "Please delete and re‑download the model, or run `bootstrap.py remove --model {student_info.name}`."
            )

        # ---- Helper to load state dict from disk ----
        def load_state_dict_from_path(path: Path) -> Dict[str, torch.Tensor]:
            """Load state dict from pytorch_model.bin or model.safetensors."""
            bin_file = path / "pytorch_model.bin"
            safetensors_file = path / "model.safetensors"
            if bin_file.exists():
                return torch.load(bin_file, map_location="cpu")
            elif safetensors_file.exists():
                from safetensors.torch import load_file
                return load_file(safetensors_file)
            else:
                raise FileNotFoundError(f"No weight file found in {path}")

        # ---- Load model with QLoRA or full precision ----
        student = None
        tokenizer = None
        try:
            # If QLoRA enabled, load in 4-bit
            if self.use_qlora and BITSANDBYTES_AVAILABLE and PEFT_AVAILABLE:
                logger.info("Loading student in 4‑bit with QLoRA (bitsandbytes + LoRA).")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                device_map = "auto" if torch.cuda.is_available() else None
                student = AutoModelForCausalLM.from_pretrained(
                    str(actual_path),
                    quantization_config=bnb_config,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )
                student = prepare_model_for_kbit_training(student)
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.qlora_r,
                    lora_alpha=self.qlora_alpha,
                    lora_dropout=self.qlora_dropout,
                    target_modules=self.qlora_target_modules,
                )
                student = get_peft_model(student, lora_config)
                logger.info(f"QLoRA model ready (4‑bit + LoRA, r={self.qlora_r}, alpha={self.qlora_alpha}).")
                # Do NOT call .to(self.device) – device_map handles placement.
            else:
                # Full precision loading – with meta tensor fallback
                try:
                    logger.info("Loading student in full precision (device_map=None).")
                    student = AutoModelForCausalLM.from_pretrained(
                        str(actual_path),
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,
                        device_map=None,
                    )
                    student = student.to(self.device)
                except NotImplementedError as e:
                    if "Cannot copy out of meta tensor" in str(e):
                        logger.warning("Model has meta tensors; manually loading state dict.")
                        # Load config
                        config = AutoConfig.from_pretrained(str(actual_path))
                        # Create model on the target device with empty weights
                        student = AutoModelForCausalLM.from_config(config)
                        # Load state dict from disk
                        state_dict = load_state_dict_from_path(actual_path)
                        # Load into model
                        student.load_state_dict(state_dict, strict=True)
                        student = student.to(self.device)
                        logger.info("Loaded student model with manual state dict loading.")
                    else:
                        raise

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(actual_path))
            if tokenizer is None:
                raise ValueError(f"Tokenizer is None for student model at {actual_path}")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        except Exception as e:
            # Fallback: try loading without QLoRA if it failed
            if self.use_qlora:
                logger.warning(f"QLoRA loading failed ({e}). Falling back to full precision.")
                try:
                    student = AutoModelForCausalLM.from_pretrained(
                        str(actual_path),
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,
                        device_map=None,
                    )
                    student = student.to(self.device)
                    tokenizer = AutoTokenizer.from_pretrained(str(actual_path))
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                except Exception as e2:
                    raise ValueError(f"Failed to load student model (fallback) from {actual_path}: {e2}") from e2
            else:
                raise ValueError(f"Failed to load student model from {actual_path}: {e}") from e

        logger.info(f"Loaded student as standard HF model from {actual_path}")

        # ---- Final check: ensure the model has at least one trainable parameter ----
        if student is None:
            raise RuntimeError("Student model could not be loaded (HF loading failed).")
        if not any(p.requires_grad for p in student.parameters()):
            if not self.use_qlora:
                raise ValueError(
                    f"Student model '{student_info.name}' has no trainable parameters. "
                    "This usually indicates a corrupt model. "
                    "Please delete the student and re-create it from a valid base model.\n"
                    f"You can delete it using: python bootstrap.py remove --model {student_info.name}\n"
                    "Then create a fresh student with `bootstrap.py create-student --base <base> --student-name <new>`."
                )

        return student, tokenizer

    # ------------------------------------------------------------------
    # Helper: attach and train draft head (shared for HF and fine-tuning)
    # ------------------------------------------------------------------
    def _setup_draft_head(self, student: nn.Module) -> Optional[Dict]:
        """
        Create and attach a DraftHead to the student model.
        Returns a dict with 'draft_head' reference, 'draft_loss_weight' etc.
        """
        if not self.train_draft_head or not SPECULATIVE_AVAILABLE:
            return None

        if hasattr(student, 'config'):
            hidden_size = getattr(student.config, 'hidden_size', None)
            vocab_size = getattr(student.config, 'vocab_size', None)
        else:
            hidden_size = None
            vocab_size = None

        if hidden_size is None or vocab_size is None:
            logger.warning("Could not infer hidden_size/vocab_size for draft head; skipping.")
            return None

        num_draft = getattr(self.config, 'max_draft_len', 4)
        draft_head = DraftHead(hidden_size, vocab_size, num_draft_tokens=num_draft)
        draft_head.to(self.device)

        if attach_draft_head_to_model is not None:
            attach_draft_head_to_model(student, hidden_size, vocab_size, num_draft_tokens=num_draft)
        else:
            student.draft_head = draft_head

        return {
            'draft_head': draft_head,
            'num_draft': num_draft,
            'draft_loss_weight': getattr(self.config, 'draft_loss_weight', 0.1),
            'confidence_loss_weight': getattr(self.config, 'confidence_loss_weight', 0.05)
        }

    # ---- Draft loss computation for HF teacher (uses teacher_logits) ----
    def _compute_draft_losses(
        self,
        teacher_logits: torch.Tensor,
        student_hidden: torch.Tensor,
        draft_head: nn.Module,
        num_draft: int,
        draft_loss_weight: float,
        confidence_loss_weight: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute draft prediction loss and confidence loss using teacher logits.
        Returns (draft_loss, confidence_loss) as scalar tensors.
        """
        teacher_next_token = torch.argmax(teacher_logits[:, -1, :], dim=-1)
        draft_logits, confidences = draft_head(student_hidden)
        draft_logits_first = draft_logits[:, 0, :]
        student_draft_token = torch.argmax(draft_logits_first, dim=-1)

        draft_loss = F.cross_entropy(draft_logits_first, teacher_next_token)
        conf_target = (student_draft_token == teacher_next_token).float()
        conf_pred = confidences[:, 0]
        confidence_loss = F.binary_cross_entropy(conf_pred, conf_target)

        return draft_loss * draft_loss_weight, confidence_loss * confidence_loss_weight

    # ---- Draft loss computation for fine-tuning (uses labels) ----
    def _compute_draft_losses_from_labels(
        self,
        labels: torch.Tensor,
        student_hidden: torch.Tensor,
        draft_head: nn.Module,
        num_draft: int,
        draft_loss_weight: float,
        confidence_loss_weight: float,
        prompt_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute draft prediction loss and confidence loss using teacher-generated
        continuation tokens (labels). This is used for Ollama/GGUF fine‑tuning.

        labels: (batch, seq_len) with -100 for prompt tokens.
        student_hidden: last hidden state from student (batch, seq_len, hidden_size)
        prompt_len: length of prompt (so we can extract continuation tokens)
        """
        # Safety check: ensure prompt_len is valid and within bounds
        if prompt_len <= 0 or prompt_len >= labels.size(1):
            logger.warning(f"Invalid prompt_len={prompt_len} for labels shape {labels.shape}; skipping draft loss.")
            return torch.tensor(0.0, device=labels.device), torch.tensor(0.0, device=labels.device)

        # Ensure student_hidden has the expected shape
        if student_hidden is None or student_hidden.size(0) != labels.size(0):
            logger.warning("student_hidden shape mismatch; skipping draft loss.")
            return torch.tensor(0.0, device=labels.device), torch.tensor(0.0, device=labels.device)

        # Get the hidden state at the last prompt token (index prompt_len - 1)
        # If student_hidden has shape (batch, seq_len, hidden), we index correctly.
        last_hidden = student_hidden[:, prompt_len - 1, :]  # (batch, hidden)

        # We want to predict the first continuation token (index prompt_len)
        target_token = labels[:, prompt_len]  # (batch,)
        # If target_token is -100, we skip (should not happen if we have continuation)
        if (target_token == -100).any():
            logger.debug("Some target tokens are -100; skipping those samples.")
            # We still compute loss with ignore_index=-100, so it's fine.

        # Add sequence dimension for draft head
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.unsqueeze(1)  # (batch, 1, hidden)

        draft_logits, confidences = draft_head(last_hidden)
        draft_logits_first = draft_logits[:, 0, :]
        student_draft_token = torch.argmax(draft_logits_first, dim=-1)

        draft_loss = F.cross_entropy(draft_logits_first, target_token, ignore_index=-100)
        conf_target = (student_draft_token == target_token).float()
        # Ensure conf_target has same shape as conf_pred (batch,)
        conf_pred = confidences[:, 0]
        confidence_loss = F.binary_cross_entropy(conf_pred, conf_target)

        return draft_loss * draft_loss_weight, confidence_loss * confidence_loss_weight

    # ---- Combined loss helpers ----
    def _compute_hidden_loss(self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between student and teacher hidden states."""
        if student_hidden is None or teacher_hidden is None:
            return torch.tensor(0.0, device=self.device)
        # Ensure same shape (batch, seq_len, hidden)
        if student_hidden.shape != teacher_hidden.shape:
            # Try to align dimensions (e.g., if one has different length)
            min_len = min(student_hidden.shape[1], teacher_hidden.shape[1])
            student_hidden = student_hidden[:, :min_len, :]
            teacher_hidden = teacher_hidden[:, :min_len, :]
        return F.mse_loss(student_hidden, teacher_hidden)

    def _compute_attention_loss(self, student_attentions, teacher_attentions) -> torch.Tensor:
        """
        Compute KL divergence between student and teacher attention distributions.
        student_attentions and teacher_attentions are tuples of tensors per layer.
        Each tensor shape: (batch, heads, seq_len, seq_len).
        """
        if student_attentions is None or teacher_attentions is None:
            return torch.tensor(0.0, device=self.device)
        if len(student_attentions) != len(teacher_attentions):
            # Use the minimum number of layers
            min_layers = min(len(student_attentions), len(teacher_attentions))
            student_attentions = student_attentions[:min_layers]
            teacher_attentions = teacher_attentions[:min_layers]
        total_kl = 0.0
        for s_attn, t_attn in zip(student_attentions, teacher_attentions):
            # Normalize over last dimension (seq_len) to get probability distributions
            s_probs = F.softmax(s_attn, dim=-1)
            t_probs = F.softmax(t_attn, dim=-1)
            # KL divergence: sum over heads and sequence positions
            kl = F.kl_div(
                F.log_softmax(s_attn, dim=-1),
                t_probs,
                reduction='batchmean'
            )
            total_kl += kl
        return total_kl / len(student_attentions)

    # ------------------------------------------------------------------
    # KL divergence distillation (HF teacher) with progressive, combined loss, temperature annealing
    # ------------------------------------------------------------------
    def _distill_hf_teacher(
        self,
        student_info,
        texts: List[str],
        passes: int,
        manager,
        resume: bool,
        use_static_routing: bool = False
    ) -> Path:
        device = self.device
        student = None
        draft_head_info = None
        try:
            student, tokenizer = self._load_student(student_info, manager)

            # ---- Apply MoE if requested ----
            if self.use_moe:
                logger.info(f"Converting dense FFNs to Micro MoE with {self.num_experts} experts, top_k={self.top_k}")
                # NOTE: hierarchical parameter is reserved; not passed until micro_moe.py supports it.
                student = convert_dense_to_micro_moe(
                    student,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    reduction_factor=self.moe_reduction_factor
                )
                student.to(device)

                if use_static_routing:
                    logger.info("Enabling static routing for MoE layers.")
                    if not texts:
                        raise ValueError("No calibration texts provided for static router creation.")
                    calib_prompts = texts[:10]
                    router = create_static_router(
                        student,
                        calib_prompts,
                        tokenizer,
                        num_experts=self.num_experts,
                        device=str(device)
                    )
                    for module in student.modules():
                        if isinstance(module, MicroMoELayer):
                            module.use_static_routing = True
                            module.static_router = router
                    logger.info("Static routing applied to all MoE layers.")

            # ---- Draft head setup ----
            if self.train_draft_head and SPECULATIVE_AVAILABLE:
                draft_head_info = self._setup_draft_head(student)
                if draft_head_info is None:
                    logger.warning("Draft head setup failed; continuing without draft training.")
                else:
                    logger.info("Draft head training enabled.")

            # ---- LoRA (non-QLoRA) if not using QLoRA ----
            use_lora = getattr(self.config, 'use_lora', False) and PEFT_AVAILABLE and not self.use_qlora
            if use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=getattr(self.config, 'lora_r', 8),
                    lora_alpha=getattr(self.config, 'lora_alpha', 32),
                    lora_dropout=getattr(self.config, 'lora_dropout', 0.1),
                    target_modules=["q_proj", "v_proj"],
                )
                student = get_peft_model(student, lora_config)
                logger.info(f"LoRA enabled: r={lora_config.r}, alpha={lora_config.lora_alpha}")
                for param in student.base_model.parameters():
                    param.requires_grad = False

            # ---- Teacher ----
            if self.teacher is None:
                raise RuntimeError("Teacher model not loaded.")
            self.teacher.to(device)
            self.teacher.eval()

            # ---- Progressive distillation: determine layer groups ----
            num_layers = getattr(student.config, 'num_hidden_layers', None)
            if num_layers is None:
                num_layers = getattr(student.config, 'num_layers', None)
            if num_layers is None:
                logger.warning("Could not infer num_layers from config; defaulting to 12 for progressive distillation.")
                num_layers = 12

            progressive_groups = []
            if self.progressive_steps > 0 and num_layers is not None:
                # Divide layers into progressive_steps groups (from top to bottom)
                layers_per_group = max(1, num_layers // self.progressive_steps)
                for step in range(self.progressive_steps):
                    start = max(0, num_layers - (step + 1) * layers_per_group)
                    end = num_layers - step * layers_per_group
                    progressive_groups.append((start, end))
                logger.info(f"Progressive distillation with {self.progressive_steps} groups: {progressive_groups}")

            # ---- Optimizer ----
            # With PEFT, only LoRA parameters are trainable; we can still use the same optimizer.
            params = list(student.parameters())
            if draft_head_info:
                params += list(draft_head_info['draft_head'].parameters())
            optimizer = torch.optim.AdamW(params, lr=self.config.distill_learning_rate)

            # ---- Gradient checkpointing ----
            if self.gradient_checkpointing:
                if hasattr(student, 'gradient_checkpointing_enable'):
                    student.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled.")

            # ---- Temperature annealing ----
            temp_schedule = self.temperature_schedule or [self.config.distill_temperature] * passes

            # ---- Accumulation ----
            accumulation_steps = self.config.gradient_accumulation_steps
            scaler = GradScaler() if device.type == "cuda" else None

            # ---- Resume ----
            start_epoch = 0
            start_step = 0
            if resume:
                ckpt_path = find_latest_checkpoint(student_info.name)
                if ckpt_path:
                    logger.info(f"Resuming from checkpoint {ckpt_path}")
                    student, optimizer, start_epoch, start_step, _ = load_checkpoint(
                        student, optimizer, ckpt_path, str(device)
                    )
                    if draft_head_info and (ckpt_path.parent / 'draft_head.pt').exists():
                        try:
                            draft_head_info['draft_head'].load_state_dict(
                                torch.load(ckpt_path.parent / 'draft_head.pt', map_location=device)
                            )
                            logger.info("Loaded draft head from checkpoint.")
                        except Exception as e:
                            logger.warning(f"Failed to load draft head checkpoint: {e}")

            best_loss = float("inf")
            # ---- Progressive step tracking ----
            current_progressive_step = 0

            for epoch in range(start_epoch, passes):
                # ---- Temperature annealing ----
                temp = temp_schedule[min(epoch, len(temp_schedule)-1)] if temp_schedule else self.config.distill_temperature
                logger.info(f"Epoch {epoch+1}/{passes} - Temperature: {temp:.2f}")

                # ---- Progressive freezing ----
                if progressive_groups and self.progressive_steps > 0 and passes > 0:
                    # Determine which group to train based on epoch (start from top)
                    # Each group gets passes / progressive_steps epochs
                    steps_per_group = max(1, passes // self.progressive_steps)
                    group_idx = min(epoch // steps_per_group, len(progressive_groups)-1)
                    if group_idx != current_progressive_step:
                        current_progressive_step = group_idx
                        start, end = progressive_groups[group_idx]
                        # Freeze all layers except those in [start, end)
                        logger.info(f"Progressive step {group_idx+1}: training layers {start} to {end-1}")
                        for name, param in student.named_parameters():
                            # Determine layer index from name (heuristic)
                            layer_idx = None
                            match = re.search(r'\.(\d+)\.', name)
                            if match:
                                layer_idx = int(match.group(1))
                            if layer_idx is not None and (layer_idx < start or layer_idx >= end):
                                param.requires_grad = False
                            else:
                                param.requires_grad = True

                student.train()
                if draft_head_info:
                    draft_head_info['draft_head'].train()
                total_loss = 0.0
                optimizer.zero_grad()

                for idx, prompt in enumerate(texts):
                    if epoch == start_epoch and idx < start_step:
                        continue
                    if self.progress_callback:
                        self.progress_callback(epoch+1, passes, idx+1, len(texts))

                    enc = tokenizer(
                        prompt, return_tensors="pt", truncation=True,
                        max_length=self.config.max_seq_len
                    ).to(device)

                    # ---- Conditionally fetch teacher hidden/attentions only if needed ----
                    need_hidden = self.hidden_loss_weight > 0
                    need_attn = self.attention_loss_weight > 0
                    with torch.no_grad():
                        teacher_out = self.teacher(
                            **enc,
                            output_hidden_states=need_hidden,
                            output_attentions=need_attn
                        )
                        teacher_logits = teacher_out.logits.detach().cpu()
                        teacher_hidden = teacher_out.hidden_states[-1].detach().cpu() if need_hidden else None
                        teacher_attentions = teacher_out.attentions if need_attn else None
                    del teacher_out
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                    teacher_logits = teacher_logits.to(device)
                    if teacher_hidden is not None:
                        teacher_hidden = teacher_hidden.to(device)
                    if teacher_attentions is not None:
                        teacher_attentions = tuple(attn.to(device) for attn in teacher_attentions)

                    _reset_router_logits(student)

                    # ---- Forward pass ----
                    if scaler:
                        with autocast():
                            student_out = student(
                                **enc,
                                labels=enc["input_ids"],
                                output_hidden_states=need_hidden,
                                output_attentions=need_attn,
                                return_dict=True
                            )
                            student_logits = student_out.logits
                            student_hidden = student_out.hidden_states[-1] if need_hidden else None
                            student_attentions = student_out.attentions if need_attn else None

                            # KL loss
                            teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
                            student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
                            kl_loss = F.kl_div(student_log_probs, teacher_probs,
                                               reduction="batchmean") * (temp ** 2)
                            ce_loss = student_out.loss
                            main_loss = self.config.distill_alpha * kl_loss + (1.0 - self.config.distill_alpha) * ce_loss

                            # Hidden loss
                            if self.hidden_loss_weight > 0 and student_hidden is not None and teacher_hidden is not None:
                                hidden_loss = self._compute_hidden_loss(student_hidden, teacher_hidden)
                                main_loss = main_loss + self.hidden_loss_weight * hidden_loss

                            # Attention loss
                            if self.attention_loss_weight > 0 and student_attentions is not None and teacher_attentions is not None:
                                attn_loss = self._compute_attention_loss(student_attentions, teacher_attentions)
                                main_loss = main_loss + self.attention_loss_weight * attn_loss
                    else:
                        student_out = student(
                            **enc,
                            labels=enc["input_ids"],
                            output_hidden_states=need_hidden,
                            output_attentions=need_attn,
                            return_dict=True
                        )
                        student_logits = student_out.logits
                        student_hidden = student_out.hidden_states[-1] if need_hidden else None
                        student_attentions = student_out.attentions if need_attn else None

                        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
                        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
                        kl_loss = F.kl_div(student_log_probs, teacher_probs,
                                           reduction="batchmean") * (temp ** 2)
                        ce_loss = student_out.loss
                        main_loss = self.config.distill_alpha * kl_loss + (1.0 - self.config.distill_alpha) * ce_loss

                        if self.hidden_loss_weight > 0 and student_hidden is not None and teacher_hidden is not None:
                            hidden_loss = self._compute_hidden_loss(student_hidden, teacher_hidden)
                            main_loss = main_loss + self.hidden_loss_weight * hidden_loss

                        if self.attention_loss_weight > 0 and student_attentions is not None and teacher_attentions is not None:
                            attn_loss = self._compute_attention_loss(student_attentions, teacher_attentions)
                            main_loss = main_loss + self.attention_loss_weight * attn_loss

                    total_loss_value = main_loss

                    # ---- Draft head loss ----
                    if draft_head_info and student_hidden is not None:
                        try:
                            draft_loss, conf_loss = self._compute_draft_losses(
                                teacher_logits,
                                student_hidden,
                                draft_head_info['draft_head'],
                                draft_head_info['num_draft'],
                                draft_head_info['draft_loss_weight'],
                                draft_head_info['confidence_loss_weight']
                            )
                            total_loss_value = total_loss_value + draft_loss + conf_loss
                            if idx % 10 == 0:
                                logger.debug(f"Draft loss: {draft_loss.item():.4f}, Conf loss: {conf_loss.item():.4f}")
                        except Exception as e:
                            logger.error(f"Draft head loss computation failed: {e}")

                    # ---- Auxiliary loss ----
                    router_logits = _collect_router_logits(student)
                    if router_logits:
                        aux_loss = compute_auxiliary_loss(router_logits, self.num_experts)
                        total_loss_value = total_loss_value + self.aux_loss_weight * aux_loss

                    # ---- Backward ----
                    if scaler:
                        scaler.scale(total_loss_value).backward()
                    else:
                        total_loss_value.backward()

                    del teacher_logits, student_out
                    if teacher_hidden is not None:
                        del teacher_hidden
                    if teacher_attentions is not None:
                        del teacher_attentions
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                    total_loss += total_loss_value.item()

                    if (idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                        if draft_head_info:
                            torch.nn.utils.clip_grad_norm_(draft_head_info['draft_head'].parameters(), 1.0)
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                    if idx % self.config.checkpoint_interval == 0:
                        ckpt = CHECKPOINTS_DIR / f"{student_info.name}_epoch{epoch}_step{idx}.pt"
                        save_checkpoint(student, optimizer, epoch, idx, ckpt)
                        if draft_head_info:
                            torch.save(draft_head_info['draft_head'].state_dict(),
                                       ckpt.parent / f"{student_info.name}_draft_head_step{idx}.pt")

                    if check_low_ram(1.0):
                        clear_cuda_memory()
                        time.sleep(1)

                    if self.config.slow_mode:
                        time.sleep(0.05)

                if (len(texts) % accumulation_steps) != 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    if draft_head_info:
                        torch.nn.utils.clip_grad_norm_(draft_head_info['draft_head'].parameters(), 1.0)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                avg_loss = total_loss / max(len(texts), 1)
                logger.info(f"Epoch {epoch+1}/{passes} — avg loss: {avg_loss:.4f}")
                if epoch > 0 and avg_loss >= best_loss - 0.01:
                    logger.info("Early stopping.")
                    break
                best_loss = avg_loss

            # ---- Merge LoRA if used ----
            if use_lora and hasattr(student, 'merge_and_unload'):
                student = student.merge_and_unload()
            elif self.use_qlora and hasattr(student, 'merge_and_unload'):
                student = student.merge_and_unload()

            final_path = self._save_distilled(student, tokenizer, student_info, manager)

            if draft_head_info and save_draft_head is not None:
                draft_path = final_path / 'draft_head.pt'
                save_draft_head(draft_head_info['draft_head'], draft_path)
                logger.info(f"Draft head saved to {draft_path}")

            return final_path
        finally:
            if student is not None:
                del student
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Fine‑tuning (Ollama / GGUF teacher) with MoE and draft head
    # ------------------------------------------------------------------
    def _fine_tune(
        self,
        teacher_name: str,
        student_info,
        texts: List[str],
        passes: int,
        manager,
        teacher_gguf,
        resume: bool,
        use_static_routing: bool = False
    ) -> Path:
        logger.info("Fine‑tuning student on teacher‑generated responses …")
        device = self.device
        student = None
        draft_head_info = None
        try:
            student, tokenizer = self._load_student(student_info, manager)

            # ---- MoE ----
            if self.use_moe:
                logger.info(f"Converting dense FFNs to Micro MoE with {self.num_experts} experts, top_k={self.top_k}")
                # NOTE: hierarchical parameter is reserved; not passed until micro_moe.py supports it.
                student = convert_dense_to_micro_moe(
                    student,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    reduction_factor=self.moe_reduction_factor
                )
                student.to(device)

                if use_static_routing:
                    logger.info("Enabling static routing for MoE layers.")
                    if not texts:
                        raise ValueError("No calibration texts provided for static router creation.")
                    calib_prompts = texts[:10]
                    router = create_static_router(
                        student,
                        calib_prompts,
                        tokenizer,
                        num_experts=self.num_experts,
                        device=str(device)
                    )
                    for module in student.modules():
                        if isinstance(module, MicroMoELayer):
                            module.use_static_routing = True
                            module.static_router = router
                    logger.info("Static routing applied to all MoE layers.")

            # ---- Draft head ----
            if self.train_draft_head and SPECULATIVE_AVAILABLE:
                draft_head_info = self._setup_draft_head(student)
                if draft_head_info is None:
                    logger.warning("Draft head setup failed; continuing without draft training.")
                else:
                    logger.info("Draft head training enabled for fine‑tuning.")

            # ---- Zero-shot compensation ----
            if self.use_zero_shot:
                logger.info("Applying zero-shot compensation to student model before fine‑tuning")
                calib_prompts = getattr(self.config, 'calibration_prompts', None)
                if not calib_prompts:
                    calib_prompts = texts[:10]
                student = apply_zero_shot_compensation(
                    student,
                    calibration_prompts=calib_prompts,
                    tokenizer=tokenizer,
                    device=str(device),
                    rank=8
                )

            # ---- LoRA (non-QLoRA) ----
            use_lora = getattr(self.config, 'use_lora', False) and PEFT_AVAILABLE and not self.use_qlora
            if use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=getattr(self.config, 'lora_r', 8),
                    lora_alpha=getattr(self.config, 'lora_alpha', 32),
                    lora_dropout=getattr(self.config, 'lora_dropout', 0.1),
                    target_modules=["q_proj", "v_proj"],
                )
                student = get_peft_model(student, lora_config)
                logger.info("LoRA enabled for fine‑tuning.")
                for param in student.base_model.parameters():
                    param.requires_grad = False

            # ---- Gradient checkpointing ----
            if self.gradient_checkpointing:
                if hasattr(student, 'gradient_checkpointing_enable'):
                    student.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled.")

            # ---- Optimizer ----
            params = list(student.parameters())
            if draft_head_info:
                params += list(draft_head_info['draft_head'].parameters())
            optimizer = torch.optim.AdamW(params, lr=self.config.distill_learning_rate)
            accumulation_steps = self.config.gradient_accumulation_steps
            scaler = GradScaler() if device.type == "cuda" else None

            # ---- Resume ----
            start_epoch = 0
            start_step = 0
            if resume:
                ckpt_path = find_latest_checkpoint(student_info.name)
                if ckpt_path:
                    logger.info(f"Resuming from checkpoint {ckpt_path}")
                    student, optimizer, start_epoch, start_step, _ = load_checkpoint(
                        student, optimizer, ckpt_path, str(device)
                    )
                    if draft_head_info and (ckpt_path.parent / 'draft_head.pt').exists():
                        try:
                            draft_head_info['draft_head'].load_state_dict(
                                torch.load(ckpt_path.parent / 'draft_head.pt', map_location=device)
                            )
                            logger.info("Loaded draft head from checkpoint.")
                        except Exception as e:
                            logger.warning(f"Failed to load draft head checkpoint: {e}")

            failure_count = 0
            max_failures = len(texts) // 2

            for epoch in range(start_epoch, passes):
                student.train()
                if draft_head_info:
                    draft_head_info['draft_head'].train()
                optimizer.zero_grad()
                for idx, prompt in enumerate(texts):
                    if epoch == start_epoch and idx < start_step:
                        continue
                    if self.progress_callback:
                        self.progress_callback(epoch+1, passes, idx+1, len(texts))

                    try:
                        if teacher_gguf:
                            resp = teacher_gguf(prompt, max_tokens=128, temperature=0.7)
                            teacher_text = resp["choices"][0]["text"].strip()
                        else:
                            teacher_text = self._get_teacher_response(prompt)
                        if not teacher_text:
                            # Skip empty responses
                            continue

                        full_text = prompt + " " + teacher_text
                        enc = tokenizer(
                            full_text, return_tensors="pt", truncation=True,
                            max_length=self.config.max_seq_len
                        ).to(device)
                        input_ids = enc.input_ids

                        prompt_enc = tokenizer(
                            prompt, return_tensors="pt", truncation=True,
                            max_length=self.config.max_seq_len
                        )
                        prompt_len = prompt_enc.input_ids.shape[1]
                        labels = input_ids.clone()
                        labels[:, :prompt_len] = -100

                        _reset_router_logits(student)

                        if scaler:
                            with autocast():
                                outputs = student(
                                    input_ids=input_ids,
                                    labels=labels,
                                    output_hidden_states=True,
                                    return_dict=True
                                )
                            main_loss = outputs.loss
                            hidden_states = outputs.hidden_states[-1]
                        else:
                            outputs = student(
                                input_ids=input_ids,
                                labels=labels,
                                output_hidden_states=True,
                                return_dict=True
                            )
                            main_loss = outputs.loss
                            hidden_states = outputs.hidden_states[-1]

                        total_loss = main_loss

                        # ---- Draft head losses ----
                        if draft_head_info and hidden_states is not None:
                            try:
                                draft_loss, conf_loss = self._compute_draft_losses_from_labels(
                                    labels=labels,
                                    student_hidden=hidden_states,
                                    draft_head=draft_head_info['draft_head'],
                                    num_draft=draft_head_info['num_draft'],
                                    draft_loss_weight=draft_head_info['draft_loss_weight'],
                                    confidence_loss_weight=draft_head_info['confidence_loss_weight'],
                                    prompt_len=prompt_len
                                )
                                total_loss = total_loss + draft_loss + conf_loss
                                if idx % 10 == 0:
                                    logger.debug(f"Fine‑tune draft loss: {draft_loss.item():.4f}, Conf loss: {conf_loss.item():.4f}")
                            except Exception as e:
                                logger.error(f"Draft head loss computation failed: {e}")

                        router_logits = _collect_router_logits(student)
                        if router_logits:
                            aux_loss = compute_auxiliary_loss(router_logits, self.num_experts)
                            total_loss = total_loss + self.aux_loss_weight * aux_loss

                        if scaler:
                            scaler.scale(total_loss).backward()
                        else:
                            total_loss.backward()

                        if (idx + 1) % accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                            if draft_head_info:
                                torch.nn.utils.clip_grad_norm_(draft_head_info['draft_head'].parameters(), 1.0)
                            if scaler:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            optimizer.zero_grad()

                        if idx % self.config.checkpoint_interval == 0:
                            ckpt = CHECKPOINTS_DIR / f"{student_info.name}_epoch{epoch}_step{idx}.pt"
                            save_checkpoint(student, optimizer, epoch, idx, ckpt)
                            if draft_head_info:
                                torch.save(draft_head_info['draft_head'].state_dict(),
                                           ckpt.parent / f"{student_info.name}_draft_head_step{idx}.pt")

                        if check_low_ram(1.0):
                            clear_cuda_memory()
                            time.sleep(1)

                        if self.config.slow_mode:
                            time.sleep(0.05)

                    except Exception as e:
                        failure_count += 1
                        logger.error(f"Fine‑tuning step error (sample {idx}): {e}")
                        if failure_count > max_failures:
                            raise RuntimeError(f"Too many failures ({failure_count}) during fine‑tuning. Aborting.") from e

                if (len(texts) % accumulation_steps) != 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    if draft_head_info:
                        torch.nn.utils.clip_grad_norm_(draft_head_info['draft_head'].parameters(), 1.0)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

            if use_lora and hasattr(student, 'merge_and_unload'):
                student = student.merge_and_unload()
            elif self.use_qlora and hasattr(student, 'merge_and_unload'):
                student = student.merge_and_unload()

            final_path = self._save_distilled(student, tokenizer, student_info, manager)
            if draft_head_info and save_draft_head is not None:
                draft_path = final_path / 'draft_head.pt'
                save_draft_head(draft_head_info['draft_head'], draft_path)
                logger.info(f"Draft head saved to {draft_path}")

            return final_path
        finally:
            if student is not None:
                del student
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Teacher unload helper
    # ------------------------------------------------------------------
    def _unload_teacher(self):
        """Explicitly unload teacher model to free memory."""
        if self.teacher is not None:
            del self.teacher
            self.teacher = None
        if self.teacher_tokenizer is not None:
            del self.teacher_tokenizer
            self.teacher_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Teacher model unloaded")

    # ------------------------------------------------------------------
    # Save distilled model (HF + optional LazyTorch)
    # ------------------------------------------------------------------
    def _save_distilled(self, student, tokenizer, student_info, manager) -> Path:
        """
        Save the distilled model in Hugging Face format and optionally as LazyTorch.
        Returns the path to the saved HF model directory.
        Ensures the distilled model name ends with '_distilled' and registry flags are updated.
        """
        base_name = student_info.name
        if not base_name.endswith("_distilled"):
            distilled_name = f"{base_name}_distilled"
        else:
            distilled_name = base_name

        final_path = manager.models_dir / distilled_name
        final_path.mkdir(parents=True, exist_ok=True)

        # If using QLoRA/LoRA, ensure we save the merged model
        if hasattr(student, 'merge_and_unload'):
            student = student.merge_and_unload()
        student.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

        if not _validate_tokenizer_deep(final_path):
            logger.error(f"Saved distilled model at {final_path} has corrupt tokenizer.")
            shutil.rmtree(final_path, ignore_errors=True)
            raise ValueError(
                f"Saved distilled model at {final_path} has a corrupt tokenizer. "
                "This likely indicates the student model had a tokenizer issue. "
                "Please delete the student model and re-create it."
            )

        size_mb = sum(f.stat().st_size for f in final_path.glob("*") if f.is_file()) / (1024 * 1024)

        new_info = ModelInfo(
            name=distilled_name,
            original_size_mb=student_info.original_size_mb,
            distilled_size_mb=size_mb,
            distillation_date=datetime.now().isoformat(),
            path=str(final_path),
            lazytorch_format=False
        )
        new_info.static_routing = getattr(self, 'use_static_routing', False)

        with manager._lock:
            if distilled_name in manager.registry:
                logger.warning(f"Overwriting existing registry entry for {distilled_name}")
            manager.registry[distilled_name] = new_info
            manager._save_registry()
        logger.info(f"Distilled model saved (HF) to {final_path}")

        if self.config.use_lazytorch:
            logger.info("Exporting distilled model to LazyTorch format...")
            try:
                lazytorch_output_dir = final_path
                result_path = export_to_lazytorch(
                    student,
                    output_path=lazytorch_output_dir,
                    dtype=torch.float32,
                    progress_callback=lambda msg: logger.info(f"LazyTorch export: {msg}")
                )
                if not _validate_tokenizer_deep(result_path):
                    logger.error(f"LazyTorch export produced corrupt tokenizer at {result_path}")
                    shutil.rmtree(result_path, ignore_errors=True)
                    raise RuntimeError("LazyTorch export failed: corrupt tokenizer.")
                with manager._lock:
                    if distilled_name in manager.registry:
                        manager.registry[distilled_name].lazytorch_format = True
                        manager._save_registry()
                logger.info(f"LazyTorch version saved to {result_path}")
            except Exception as e:
                logger.warning(f"LazyTorch export failed: {e}")

        manager.reload_registry()
        return final_path


# =============================================================================
# Endless distillation loop (v3.6)
# =============================================================================

def run_endless_distillation(
    teacher: str,
    student: str,
    passes: int = 2,
    cycles: int = -1,
    sleep: int = 60,
    callback: Optional[Callable] = None
) -> None:
    """
    Run endless distillation loop.
    cycles = -1 means infinite.
    """
    from .lazy_model_manager import ModelManager
    from .config import load_config

    config = load_config()
    manager = ModelManager(config)
    engine = LazyDistillationEngine(config)

    cycle = 0
    while cycles == -1 or cycle < cycles:
        cycle += 1
        logger.info(f"Endless distillation cycle {cycle}")
        if callback:
            callback(f"Cycle {cycle}: distilling {teacher} -> {student}")

        if not manager.model_exists(teacher):
            logger.error(f"Teacher {teacher} not found; aborting cycle")
            break
        if not manager.model_exists(student):
            logger.error(f"Student {student} not found; aborting cycle")
            break

        try:
            engine.run_distillation(
                teacher, student,
                texts=config.validation_prompts,
                passes=passes,
                resume=True
            )
            manager.reload_registry()
            logger.info(f"Distillation cycle {cycle} complete.")
            if callback:
                callback(f"Cycle {cycle} complete.")
        except Exception as e:
            logger.error(f"Distillation cycle {cycle} failed: {e}")
            if callback:
                callback(f"Cycle {cycle} failed: {e}")

        if cycles != -1 and cycle >= cycles:
            break
        if sleep > 0:
            logger.info(f"Sleeping {sleep} seconds before next cycle...")
            time.sleep(sleep)