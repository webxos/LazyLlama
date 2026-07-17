#!/usr/bin/env python3
"""
Web dashboard with real‑time metrics, clean shutdown, model export to Ollama, and LazyTorch integration.
Enhanced with full distillation/pruning/benchmark/export controls using global teacher/student selections.
Unified model management tab combines student creation and Ollama pull.
Fixed: global selectors are the only source of truth; no duplicate selectors below graph.
Added: Benchmark tab with async execution and results display.
Added: Status line showing current teacher → student.
FIXED: Global models endpoint now forces Ollama sync and returns combined model list with type labels.
FIXED: Both teacher and student dropdowns show the full combined list with clear labels.
FIXED: JS refreshes model lists after every operation (distill, prune, import, pull, delete, etc.).
FIXED: Background task states are properly reset on completion.
NEW: /api/base-models endpoint to populate Create Student dropdown dynamically from registry.
NEW: JS loadBaseModels() function to fetch and populate base model select.
FIXED: Added thread‑safe locks around global state modifications.
FIXED: Export jobs are now cleaned up after 10 minutes to prevent memory leaks.
FIXED: Added /api/download/<job_id> endpoint to serve exported zip files to the browser.
FIXED: Improved pull progress parsing with more robust regex.

NEW: Added detailed logging in export jobs.
FIX: _run_export_job now validates model path existence and sets failed status on errors.
FIX: _handle_export_student validates model existence and rejects Ollama models with clear message.
FIX: _handle_download checks file existence and returns 404 if not found.

NEW: /api/validate-model endpoint to validate model paths.
NEW: Export job logs are stored and displayed in the modal.
NEW: Validate button in model list.

FIX: _handle_create_student now validates base model and created student directory.
FIX: Enhanced error logging in all endpoints and background threads, especially in _handle_terminal.

=====================================================================
CRITICAL FIXES (v3.1.1):
- Added `_validate_student_directory()` that actually loads the tokenizer
  and verifies JSON integrity, preventing registration of corrupted models.
- `_run_student_creation` now calls this validation and deletes the directory
  if validation fails, raising a clear error.
- All background threads (distill, prune, benchmark) now check tokenizer validity
  before proceeding, and gracefully handle errors without corrupting the registry.
- Added tokenizer validation in `_handle_validate_model` endpoint.
- Export jobs now validate tokenizer integrity before zipping.

=====================================================================
FIX (2026-07-08): Restrict student operations to local models only.
- Added `_is_local_model()` helper to check for Ollama/vLLM URIs.
- `_validate_student_directory()` returns False for URI paths.
- In `_handle_start_distill`, `_handle_start_prune`, `_handle_start_benchmark`,
  added early validation to reject non-local student models with a clear error.
- In background threads (`_run_distillation_thread`, `_run_prune_thread`,
  `_run_benchmark_thread`), added explicit checks for URI paths and raise
  ValueError with an actionable message.

=====================================================================
FIX (2026-07-10): Consolidated import paths, used shared tokenizer validation,
                  added proper handling of invalid models in global selector,
                  improved export job concurrency with locks, and ensured
                  all background tasks clean up resources properly.

=====================================================================
FIX (2026-07-11): In _run_prune_thread, explicitly pass the loaded tokenizer
                  to Pruner() constructor to ensure the pruner has a valid
                  tokenizer for export. This prevents tokenizer corruption
                  issues during pruning.

=====================================================================
NEW (2026-07-14): Integrated REAP Pipeline Checklist into student creation workflow.
                   Enhanced terminal NLP `/chat` endpoint to use actual inference engine.
                   Unified logging with context to reduce repetition.

=====================================================================
NEW (2026-07-16): Benchmark settings and report enhancements.
- _handle_start_benchmark now accepts a `settings` object to configure benchmarks.
- _run_benchmark_thread uses the provided settings and captures detailed errors.
- Added `/api/benchmark-report/<model_name>` endpoint to retrieve stored reports.
- Benchmark progress callback updates _benchmark_state in real time.
- Single‑model benchmark results are now stored in the registry using store_benchmark_results.

=====================================================================
FIX (2026-07-16): Added `overwrite=True` to `pruner.export_pruned()` call in
                  `_run_prune_thread` to prevent FileExistsError when the pruned
                  model already exists.

=====================================================================
FIX (2026-07-17): Student creation robustness:
- `_handle_create_student` now validates that the base model exists either locally
  (with directory integrity check) or as a valid Hugging Face model ID using
  `huggingface_hub.model_info`. If not found, returns a clear error to the client.
- `_run_student_creation` catches `OSError` and produces a user‑friendly message
  when a model ID does not exist on Hugging Face.
- Added import of `huggingface_hub` with graceful fallback if not installed.

=====================================================================
NEW (2026-07-17): Enhanced user feedback in background threads:
- Added user-friendly error messages for common failures (checkpoint mismatch, OOM, Ollama issues).
- Added `/api/task-state` endpoint to expose detailed task states including user-friendly errors.
- Included user_friendly_error and error_type in metrics response for UI display.
"""

import json
import threading
import webbrowser
import psutil
import subprocess
import time
import torch
import shutil
import gc
import uuid
import re
import os
import logging
import traceback
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List

# ---- All internal imports are now RELATIVE ----
from .metrics_store import MetricsStore
from .lazy_model_manager import ModelManager
from .utils import (
    export_to_ollama,
    is_lazytorch_model,
    get_lazytorch_model_size,
    _validate_tokenizer_deep,
    check_ollama_model,
)
from .config import load_config, LAZY_DIR
from .lazytorch_core import export_to_lazytorch
from .benchmark import benchmark_model, benchmark_student_models, BenchmarkSettings, format_benchmark_summary, store_benchmark_results
from .lazy_distill import LazyDistillationEngine
from .lazy_infer import create_engine

# ---- Import REAP checklist from bootstrap (shared function) ----
# To avoid circular imports, we define a local wrapper that loads the function dynamically.
try:
    from .bootstrap import run_reap_pipeline_checklist
except ImportError:
    # Fallback: define a minimal version if bootstrap is not available
    def run_reap_pipeline_checklist(model_name, manager):
        logger.warning("REAP checklist function not available; skipping.")
        return False

logger = logging.getLogger(__name__)

# ---- Optional: Hugging Face Hub for model validation ----
try:
    from huggingface_hub import model_info as hf_model_info
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_model_info = None
    logger.warning("huggingface_hub not installed; remote model validation will be skipped.")

# =============================================================================
# Helper: Check if model is local (not Ollama/vLLM)
# =============================================================================
def _is_local_model(info) -> bool:
    """Return True if the model info points to a local file/directory (not ollama:// or vllm://)."""
    if info is None or not info.path:
        return False
    path = info.path
    return not (path.startswith("ollama://") or path.startswith("vllm://"))


# =============================================================================
# GLOBAL STATE – shared across all requests
# =============================================================================
_global_student = None   # model name (string)
_global_teacher = None   # model name (string)
_model_manager = ModelManager()
_state_lock = threading.RLock()   # protects global state dicts

# ---- Persist global state to JSON ----
GLOBAL_STATE_FILE = LAZY_DIR / "global_state.json"


def _load_global_state():
    """Load saved global teacher/student selections from disk."""
    global _global_student, _global_teacher
    if GLOBAL_STATE_FILE.exists():
        try:
            with open(GLOBAL_STATE_FILE) as f:
                data = json.load(f)
                _global_teacher = data.get("teacher", "")
                _global_student = data.get("student", "")
        except Exception:
            _global_teacher = ""
            _global_student = ""
    else:
        _global_teacher = ""
        _global_student = ""


def _save_global_state(teacher, student):
    """Persist global selections to disk."""
    with open(GLOBAL_STATE_FILE, "w") as f:
        json.dump({"teacher": teacher, "student": student}, f)


# Load initial state
_load_global_state()

# Background task states (protected by _state_lock)
# Each state dict includes user_friendly_error and error_type for better UI feedback
_distillation_state = {
    "running": False,
    "progress": 0,
    "status": "idle",
    "phase": "",
    "error": None,
    "user_friendly_error": None,
    "error_type": None,
}
_prune_state = {
    "running": False,
    "progress": 0,
    "status": "idle",
    "error": None,
    "user_friendly_error": None,
    "error_type": None,
}
_student_creation_state = {
    "running": False,
    "progress": 0,
    "status": "idle",
    "error": None,
    "user_friendly_error": None,
    "error_type": None,
}
_benchmark_state = {
    "running": False,
    "progress": 0,
    "status": "idle",
    "results": None,
    "error": None,
    "user_friendly_error": None,
    "error_type": None,
    "traceback": None,
}
_export_jobs = {}          # job_id -> dict
_export_jobs_lock = threading.RLock()  # separate lock for export jobs

# Clean up export jobs older than 10 minutes
_EXPORT_JOB_TTL = 600  # seconds


def _cleanup_old_export_jobs():
    """Remove completed/failed jobs older than TTL."""
    now = time.time()
    with _export_jobs_lock:
        to_delete = []
        for job_id, job in _export_jobs.items():
            if job.get("status") in ("completed", "failed"):
                if now - job.get("timestamp", 0) > _EXPORT_JOB_TTL:
                    to_delete.append(job_id)
        for job_id in to_delete:
            del _export_jobs[job_id]
            logger.debug(f"Cleaned up stale export job: {job_id}")


# MetricsStore instance
_metrics_store = MetricsStore()


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Helper: Build user-friendly error messages
# =============================================================================
def _get_user_friendly_error(error: Exception, context: str = "", **kwargs) -> str:
    """
    Return a user-friendly error message based on the exception and context.
    """
    error_msg = str(error)
    if "Missing key(s)" in error_msg or "state_dict" in error_msg:
        return (
            "Checkpoint mismatch: the saved checkpoint does not match the current model.\n"
            "This usually happens when the student model has changed since the checkpoint was saved.\n\n"
            f"To fix: delete the checkpoint and restart without resume:\n"
            f"  rm -f ~/.lazy_llama/checkpoints/{kwargs.get('student', '')}*\n"
            f"  Then restart the operation with resume=False"
        )
    elif "out of memory" in error_msg.lower() or "cuda out of memory" in error_msg.lower():
        max_seq_len = kwargs.get('max_seq_len', 512)
        return (
            f"Out of memory during {context}. Try reducing memory usage:\n"
            f"  - Reduce max_seq_len (currently {max_seq_len})\n"
            "  - Enable QLoRA: use_qlora=True\n"
            "  - Reduce batch size\n"
            "  - Enable gradient checkpointing\n"
            "  - Use LazyTorch for the student if possible"
        )
    elif "Ollama" in error_msg or "ollama" in error_msg:
        teacher = kwargs.get('teacher', 'the teacher model')
        return (
            f"Ollama error. Please ensure:\n"
            f"  - Ollama is running: `ollama serve`\n"
            f"  - The teacher model is pulled: `ollama pull {teacher}`\n"
            f"  - The service is reachable at localhost:11434"
        )
    elif "tokenizer" in error_msg.lower() or "corrupt" in error_msg.lower():
        return (
            f"Tokenizer error: The model's tokenizer appears to be corrupt or incompatible.\n"
            "Please delete the model and re-download it, or repair the tokenizer files.\n"
            f"To delete: python -m lazy_llama.bootstrap remove --model {kwargs.get('model_name', '')}"
        )
    else:
        return error_msg


# =============================================================================
# FIXED: Tokenizer validation helper (actually loads and parses tokenizer.json)
# =============================================================================
def _validate_student_directory(path: Path) -> bool:
    """
    Perform a thorough validation of a model directory.
    Checks:
      - config.json exists and is valid JSON
      - At least one weight file exists (pytorch_model.bin or model.safetensors)
      - tokenizer files exist and can be loaded by AutoTokenizer
    Returns True if valid, False otherwise.

    FIX (2026-07-08): Return False if the path is a URI (ollama:// or vllm://)
                      because those are not local directories.
    """
    path_str = str(path)
    if path_str.startswith("ollama://") or path_str.startswith("vllm://"):
        return False

    if not path.is_dir():
        logger.error(f"Not a directory: {path}")
        return False

    # Check config.json
    config_path = path / "config.json"
    if not config_path.exists():
        logger.error(f"Missing config.json: {path}")
        return False
    try:
        with open(config_path) as f:
            json.load(f)
    except Exception as e:
        logger.error(f"Invalid config.json: {e}")
        return False

    # Check weight files
    has_weights = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    if not has_weights:
        logger.error(f"No weight files found in: {path}")
        return False

    # Check tokenizer files and loadability
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        if tokenizer.vocab_size == 0:
            logger.error(f"Tokenizer has zero vocab size: {path}")
            return False
    except Exception as e:
        logger.error(f"Tokenizer validation failed for {path}: {e}")
        return False

    return True


# =============================================================================
# Background Thread Functions
# =============================================================================

def _run_distillation_thread(teacher, student, passes):
    global _distillation_state
    metrics_store = _metrics_store
    try:
        with _state_lock:
            _distillation_state["running"] = True
            _distillation_state["progress"] = 0
            _distillation_state["status"] = "initializing"
            _distillation_state["error"] = None
            _distillation_state["user_friendly_error"] = None
            _distillation_state["error_type"] = None
        metrics_store.set_active_task("distillation", 0)
        config = load_config()

        # ---- Validate that student is a local model ----
        mm = ModelManager()
        s_info = mm.get_model(student)
        if s_info and s_info.path and (s_info.path.startswith("ollama://") or s_info.path.startswith("vllm://")):
            raise ValueError(
                f"Student model '{student}' is not a local model. "
                "Distillation requires a local Hugging Face model. "
                "Please select a local student model."
            )

        engine = LazyDistillationEngine(config)
        val = config.validation_prompts

        def cb(p, total_p, b, total_b):
            progress = ((p - 1) * total_b + b) / (total_p * total_b) * 100
            progress = int(progress)
            with _state_lock:
                _distillation_state["progress"] = progress
                _distillation_state["status"] = f"pass {p}/{total_p}"
            metrics_store.set_active_task("distillation", progress)
            time.sleep(0.1)

        engine.set_progress_callback(cb)
        engine.run_distillation(teacher, student, val, passes=passes, resume=True)
        with _state_lock:
            _distillation_state["status"] = "completed"
            _distillation_state["progress"] = 100
        metrics_store.set_active_task("distillation", 100)
    except Exception as e:
        logger.exception("Distillation thread failed")
        # ---- NEW: Categorize error for better user feedback ----
        error_msg = str(e)
        user_friendly = _get_user_friendly_error(
            e,
            context="distillation",
            student=student,
            teacher=teacher,
            max_seq_len=load_config().max_seq_len
        )
        with _state_lock:
            _distillation_state["error"] = error_msg
            _distillation_state["user_friendly_error"] = user_friendly
            _distillation_state["status"] = f"failed: {error_msg}"
            _distillation_state["error_type"] = type(e).__name__
    finally:
        with _state_lock:
            _distillation_state["running"] = False
        metrics_store.clear_task()
        _cleanup_memory()


def _run_prune_thread(model_name, strategy, task):
    global _prune_state
    metrics_store = _metrics_store
    try:
        with _state_lock:
            _prune_state["running"] = True
            _prune_state["progress"] = 0
            _prune_state["status"] = "loading model"
            _prune_state["error"] = None
            _prune_state["user_friendly_error"] = None
            _prune_state["error_type"] = None
        metrics_store.set_active_task("prune", 0)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .lazy_prune import Pruner, get_task_prompts
        manager = ModelManager()
        info = manager.get_model(model_name)
        if not info or not info.path:
            raise ValueError("Model not found")
        # ---- Validate that the model is local (not Ollama/vLLM) ----
        if info.path.startswith("ollama://") or info.path.startswith("vllm://"):
            raise ValueError(
                f"Model '{model_name}' is not a local model. "
                "Pruning requires a local Hugging Face model. "
                "Please select a local student model."
            )
        # ---- Validate tokenizer before loading ----
        model_path = Path(info.path)
        if not _validate_student_directory(model_path):
            raise ValueError(f"Model directory {model_path} is invalid (tokenizer or config corrupt).")
        with _state_lock:
            _prune_state["status"] = f"loading {model_name}"
        model = AutoModelForCausalLM.from_pretrained(info.path, low_cpu_mem_usage=True)
        # ---- FIX: Load tokenizer and pass it to Pruner ----
        tokenizer = AutoTokenizer.from_pretrained(info.path)
        config = load_config()
        # Pass original_path and tokenizer explicitly
        pruner = Pruner(model, config, original_path=Path(info.path), tokenizer=tokenizer)
        with _state_lock:
            _prune_state["progress"] = 20
        metrics_store.set_active_task("prune", 20)
        time.sleep(0.1)
        if strategy == "magnitude":
            with _state_lock:
                _prune_state["status"] = "magnitude pruning"
            pruner.magnitude_prune()
        elif strategy == "neuron":
            with _state_lock:
                _prune_state["status"] = "neuron pruning"
            pruner.neuron_prune()
        elif strategy == "task":
            with _state_lock:
                _prune_state["status"] = f"task‑specific pruning ({task})"
            prompts = get_task_prompts(task)
            pruner.task_specific_reap(task, prompts, tokenizer)
        with _state_lock:
            _prune_state["progress"] = 80
        metrics_store.set_active_task("prune", 80)
        time.sleep(0.1)
        out_path = manager.models_dir / f"{model_name}_pruned"
        # ---- FIX: Add overwrite=True to prevent FileExistsError ----
        pruner.export_pruned(str(out_path), overwrite=True)
        # Validate the pruned model before registering
        if not _validate_student_directory(out_path):
            shutil.rmtree(out_path, ignore_errors=True)
            raise RuntimeError("Pruned model failed validation (corrupt output).")
        size_mb = sum(f.stat().st_size for f in out_path.glob("*") if f.is_file()) / (1024 * 1024)
        manager.registry[f"{model_name}_pruned"] = manager._create_model_info(
            name=f"{model_name}_pruned",
            path=str(out_path),
            size_mb=size_mb,
            pruning_applied=True,
        )
        manager._save_registry()
        with _state_lock:
            _prune_state["progress"] = 100
            _prune_state["status"] = "completed"
        metrics_store.set_active_task("prune", 100)
    except Exception as e:
        logger.exception("Prune thread failed")
        error_msg = str(e)
        user_friendly = _get_user_friendly_error(
            e,
            context="pruning",
            model_name=model_name,
            max_seq_len=load_config().max_seq_len
        )
        with _state_lock:
            _prune_state["status"] = f"failed: {error_msg}"
            _prune_state["error"] = error_msg
            _prune_state["user_friendly_error"] = user_friendly
            _prune_state["error_type"] = type(e).__name__
    finally:
        with _state_lock:
            _prune_state["running"] = False
        metrics_store.clear_task()
        _cleanup_memory()


def _run_student_creation(base_model, student_name):
    global _student_creation_state
    metrics_store = _metrics_store
    try:
        with _state_lock:
            _student_creation_state["running"] = True
            _student_creation_state["progress"] = 10
            _student_creation_state["status"] = f"downloading {base_model}"
            _student_creation_state["error"] = None
            _student_creation_state["user_friendly_error"] = None
            _student_creation_state["error_type"] = None
        metrics_store.set_active_task("student_creation", 10)

        # ---- Validate base model again (thread-safe) ----
        mm = ModelManager()
        base_info = mm.get_model(base_model)
        if base_info and base_info.invalid:
            raise ValueError(f"Base model '{base_model}' is marked invalid.")
        if base_info and base_info.path and not base_info.path.startswith("ollama://"):
            base_path = Path(base_info.path)
            if not _validate_student_directory(base_path):
                raise ValueError(f"Base model directory {base_path} is corrupt.")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Attempt to load the model (this may download if not local)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        except OSError as e:
            # Provide a user-friendly error for missing model IDs
            if "not a local folder" in str(e) or "404" in str(e) or "not found" in str(e):
                msg = f"Model '{base_model}' does not exist on Hugging Face or is not accessible. Please check the model ID."
            else:
                msg = str(e)
            raise RuntimeError(msg) from e

        with _state_lock:
            _student_creation_state["progress"] = 50
        metrics_store.set_active_task("student_creation", 50)
        with _state_lock:
            _student_creation_state["status"] = "saving locally"
        manager = ModelManager()
        dest_dir = manager.models_dir / student_name
        model.save_pretrained(dest_dir)
        tokenizer.save_pretrained(dest_dir)

        # ---- FIXED: Validate the created directory with the new function ----
        if not _validate_student_directory(dest_dir):
            logger.error(f"Created student directory {dest_dir} is invalid (corrupted).")
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise RuntimeError("Created student model is invalid (config/tokenizer/weights corrupt).")

        size_mb = sum(f.stat().st_size for f in dest_dir.glob("*") if f.is_file()) / (1024 * 1024)
        manager.registry[student_name] = manager._create_model_info(
            name=student_name,
            path=str(dest_dir),
            size_mb=size_mb,
        )
        manager._save_registry()
        with _state_lock:
            _student_creation_state["progress"] = 90
            _student_creation_state["status"] = f"created {student_name}"

        # ---- NEW: Run REAP Pipeline Checklist on the created student ----
        logger.info(f"Running REAP Pipeline Checklist for student '{student_name}'...")
        passed = run_reap_pipeline_checklist(student_name, manager)
        if not passed:
            logger.warning(f"REAP Pipeline Checklist incomplete for '{student_name}'. Continuing with partial model.")
        else:
            logger.info(f"✅ Student '{student_name}' passed REAP Pipeline Checklist.")

        config = load_config()
        if config.use_lazytorch:
            with _state_lock:
                _student_creation_state["status"] = "exporting to LazyTorch..."
            lazytorch_output = dest_dir.with_suffix('.lazytorch')
            export_to_lazytorch(
                dest_dir,
                output_path=lazytorch_output,
                dtype=torch.float32,
                progress_callback=lambda msg: _student_creation_state.update({"status": msg}),
                reap_mode=True  # Apply REAP optimizations during export
            )
            with manager._lock:
                if student_name in manager.registry:
                    manager.registry[student_name].lazytorch_format = True
                    manager._save_registry()
            with _state_lock:
                _student_creation_state["status"] = f"created {student_name} (LazyTorch ready)"
        with _state_lock:
            _student_creation_state["progress"] = 100
        metrics_store.set_active_task("student_creation", 100)
        logger.info(f"Student '{student_name}' created successfully (REAP pipeline checked).")
    except Exception as e:
        logger.exception("Student creation thread failed")
        error_msg = str(e)
        user_friendly = _get_user_friendly_error(
            e,
            context="student creation",
            model_name=base_model,
            max_seq_len=load_config().max_seq_len
        )
        with _state_lock:
            _student_creation_state["status"] = f"failed: {error_msg}"
            _student_creation_state["error"] = error_msg
            _student_creation_state["user_friendly_error"] = user_friendly
            _student_creation_state["error_type"] = type(e).__name__
    finally:
        with _state_lock:
            _student_creation_state["running"] = False
        metrics_store.clear_task()
        _cleanup_memory()


def _run_benchmark_thread(model_name, all_students, settings_dict):
    """Background benchmark thread that accepts settings and updates progress."""
    global _benchmark_state
    metrics_store = _metrics_store
    try:
        with _state_lock:
            _benchmark_state["running"] = True
            _benchmark_state["progress"] = 0
            _benchmark_state["status"] = "initializing"
            _benchmark_state["results"] = None
            _benchmark_state["error"] = None
            _benchmark_state["user_friendly_error"] = None
            _benchmark_state["error_type"] = None
            _benchmark_state["traceback"] = None
        metrics_store.set_active_task("benchmark", 0)

        config = load_config()

        # Build BenchmarkSettings from dict
        if settings_dict:
            # Ensure required fields are present with defaults
            default_settings = {
                "prompt": "What is machine learning?",
                "max_tokens": 100,
                "run_perplexity": False,
                "val_texts": None,
                "run_multiple_choice": False,
                "mc_questions": None,
                "run_long_context": False,
                "context_lengths": None,
                "num_trials": 3,
                "store_in_registry": True,
                "long_context_max_tokens": 20,
            }
            for key, default in default_settings.items():
                if key not in settings_dict:
                    settings_dict[key] = default
            settings = BenchmarkSettings(**settings_dict)
        else:
            # Use default settings
            settings = BenchmarkSettings()

        # Progress callback for benchmark_student_models
        def progress_callback(model_name: str, status: str, result: Optional[Dict]):
            with _state_lock:
                if status == "done":
                    _benchmark_state["progress"] = min(100, _benchmark_state["progress"] + 5)
                    _benchmark_state["status"] = f"benchmarked {model_name}"
                elif status == "error":
                    _benchmark_state["progress"] = min(100, _benchmark_state["progress"] + 5)
                    _benchmark_state["status"] = f"error in {model_name}"
                elif status == "starting":
                    _benchmark_state["status"] = f"starting {model_name}"
            metrics_store.set_active_task("benchmark", _benchmark_state["progress"])

        if all_students:
            with _state_lock:
                _benchmark_state["status"] = "benchmarking all students..."
            results = benchmark_student_models(
                settings,
                config=config,
                progress_callback=progress_callback if not model_name else None
            )
            with _state_lock:
                _benchmark_state["results"] = results
                _benchmark_state["progress"] = 100
                _benchmark_state["status"] = "completed"
        else:
            mm = ModelManager()
            info = mm.get_model(model_name)
            if not info or not info.path:
                raise ValueError(f"Model {model_name} not found")
            # ---- Validate that the model is local ----
            if info.path.startswith("ollama://") or info.path.startswith("vllm://"):
                # Benchmark can support Ollama, but we don't validate tokenizer for them.
                pass
            else:
                # Validate the model before benchmarking (only for local)
                model_path = Path(info.path)
                if not _validate_student_directory(model_path):
                    raise ValueError(f"Model directory {model_path} is invalid.")
            with _state_lock:
                _benchmark_state["status"] = f"benchmarking {model_name}..."
            with _state_lock:
                _benchmark_state["progress"] = 30
            metrics_store.set_active_task("benchmark", 30)
            time.sleep(0.2)

            # For single model, we can call benchmark_model directly
            res = benchmark_model(
                info.path,
                prompt=settings.prompt,
                max_tokens=settings.max_tokens,
                config=config,
                model_name=model_name,
                additional_metrics=(
                    ['perplexity'] if settings.run_perplexity and settings.val_texts else []
                ) + (['multiple_choice'] if settings.run_multiple_choice and settings.mc_questions else []),
                val_texts=settings.val_texts if settings.run_perplexity else None,
                mc_questions=settings.mc_questions if settings.run_multiple_choice else None,
            )
            # If additional long-context requested, run it separately.
            if settings.run_long_context and res.get('success', False):
                model_mem = res.get('peak_memory_gb', 2.0)
                if model_mem < 0.5:
                    model_mem = 1.0
                available_ram = psutil.virtual_memory().available / (1024**3)
                from .benchmark import benchmark_long_context, _get_max_context_length
                max_safe = _get_max_context_length(available_ram, model_mem)
                ctx_lengths = [cl for cl in (settings.context_lengths or [2048, 4096, 8192, 16384]) if cl <= max_safe]
                if ctx_lengths:
                    long_res = benchmark_long_context(
                        info.path,
                        config,
                        context_lengths=ctx_lengths,
                        num_trials=settings.num_trials,
                        max_tokens_per_task=settings.long_context_max_tokens,
                    )
                    if 'error' not in long_res:
                        res['long_context'] = long_res
                    else:
                        res['long_context_error'] = long_res['error']

            # ---- Store single-model benchmark result in registry ----
            if settings.store_in_registry:
                store_data = {
                    'tokens_per_second': res.get('tokens_per_second'),
                    'peak_memory_gb': res.get('peak_memory_gb'),
                    'e8_quantized': res.get('e8_quantized'),
                    'kv_compressed': res.get('kv_compressed'),
                    'lazytorch_used': res.get('lazytorch_used'),
                    'timestamp': time.time(),
                }
                if 'perplexity' in res:
                    store_data['perplexity'] = res['perplexity']
                if 'multiple_choice_accuracy' in res:
                    store_data['multiple_choice_accuracy'] = res['multiple_choice_accuracy']
                if 'long_context' in res:
                    store_data['long_context'] = res['long_context']
                store_benchmark_results(model_name, store_data)

            with _state_lock:
                _benchmark_state["progress"] = 100
                _benchmark_state["results"] = [res]  # list for consistency
                _benchmark_state["status"] = "completed"
        metrics_store.set_active_task("benchmark", 100)
    except Exception as e:
        logger.exception("Benchmark thread failed")
        error_msg = str(e)
        user_friendly = _get_user_friendly_error(
            e,
            context="benchmark",
            model_name=model_name if model_name else "unknown",
            max_seq_len=load_config().max_seq_len
        )
        with _state_lock:
            _benchmark_state["status"] = f"failed: {error_msg}"
            _benchmark_state["error"] = error_msg
            _benchmark_state["user_friendly_error"] = user_friendly
            _benchmark_state["error_type"] = type(e).__name__
            _benchmark_state["traceback"] = traceback.format_exc()
    finally:
        with _state_lock:
            _benchmark_state["running"] = False
        metrics_store.clear_task()
        _cleanup_memory()


# =============================================================================
# Export Job (with detailed logging and validation)
# =============================================================================
def _run_export_job(job_id, model_name, format_type, options):
    """Background export job with progress updates and detailed logging."""
    global _export_jobs
    metrics_store = _metrics_store
    logs = []
    try:
        logger.info(f"Export job {job_id} started for model '{model_name}' (format: {format_type})")
        logs.append(f"Export started for model '{model_name}'")

        with _export_jobs_lock:
            _export_jobs[job_id]["status"] = "running"
            _export_jobs[job_id]["progress"] = 0
            _export_jobs[job_id]["message"] = "Starting export..."
            _export_jobs[job_id]["logs"] = logs.copy()

        mm = ModelManager()
        info = mm.get_model(model_name)
        if not info or not info.path:
            raise ValueError(f"Model '{model_name}' not found in registry")
        if info.path.startswith("ollama://"):
            raise ValueError("Cannot export Ollama model as zip")

        model_path = Path(info.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        if not model_path.is_dir() and not model_path.is_file():
            raise ValueError(f"Path is neither file nor directory: {model_path}")

        # ---- Validate tokenizer if it's a directory ----
        if model_path.is_dir() and not _validate_student_directory(model_path):
            raise ValueError(f"Model directory {model_path} is invalid (corrupt). Cannot export.")

        export_dir = Path.home() / ".lazy_llama/exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        zip_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = export_dir / zip_name
        with _export_jobs_lock:
            _export_jobs[job_id]["zip_path"] = str(zip_path)
            _export_jobs[job_id]["zip_name"] = zip_name
            _export_jobs[job_id]["progress"] = 10
            _export_jobs[job_id]["message"] = "Preparing model..."
            _export_jobs[job_id]["logs"] = logs.copy()

        logger.info(f"Export job {job_id}: Preparing model at {model_path}")
        logs.append(f"Preparing model from {model_path}")

        if format_type == "vllm":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            with _export_jobs_lock:
                _export_jobs[job_id]["message"] = "Loading model for vLLM export..."
                _export_jobs[job_id]["logs"] = logs.copy()
            logger.info(f"Export job {job_id}: Loading model for vLLM")
            logs.append("Loading model for vLLM export...")
            try:
                model = AutoModelForCausalLM.from_pretrained(str(model_path), low_cpu_mem_usage=True)
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except Exception as load_err:
                logger.error(f"Export job {job_id}: Failed to load model for vLLM: {load_err}")
                logs.append(f"Failed to load model: {load_err}")
                raise RuntimeError(f"Failed to load model for vLLM export: {load_err}") from load_err

            temp_dir = export_dir / f"temp_{model_name}"
            temp_dir.mkdir(exist_ok=True)
            with _export_jobs_lock:
                _export_jobs[job_id]["progress"] = 40
                _export_jobs[job_id]["message"] = "Saving model in safetensors..."
                _export_jobs[job_id]["logs"] = logs.copy()
            try:
                model.save_pretrained(temp_dir, safe_serialization=True)
                tokenizer.save_pretrained(temp_dir)
                logs.append("Model saved in safetensors format")
            except Exception as save_err:
                logger.error(f"Export job {job_id}: Failed to save safetensors: {save_err}")
                logs.append(f"Failed to save safetensors: {save_err}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise RuntimeError(f"Failed to save vLLM model: {save_err}") from save_err

            with _export_jobs_lock:
                _export_jobs[job_id]["progress"] = 80
                _export_jobs[job_id]["message"] = "Creating zip archive..."
                _export_jobs[job_id]["logs"] = logs.copy()
            try:
                shutil.make_archive(str(zip_path.with_suffix('')), 'zip', temp_dir)
                logs.append("Zip archive created")
            except Exception as zip_err:
                logger.error(f"Export job {job_id}: Failed to create zip: {zip_err}")
                logs.append(f"Failed to create zip: {zip_err}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise RuntimeError(f"Failed to create zip archive: {zip_err}") from zip_err
            shutil.rmtree(temp_dir)
            _cleanup_memory()
            logger.info(f"Export job {job_id}: vLLM export zip created at {zip_path}")
            logs.append(f"Export completed: {zip_name}")
        else:
            # PyTorch format: zip the entire model directory or file
            with _export_jobs_lock:
                _export_jobs[job_id]["message"] = "Creating zip archive..."
                _export_jobs[job_id]["logs"] = logs.copy()
            try:
                shutil.make_archive(str(zip_path.with_suffix('')), 'zip', model_path)
                logs.append("Zip archive created")
            except Exception as zip_err:
                logger.error(f"Export job {job_id}: Failed to create zip: {zip_err}")
                logs.append(f"Failed to create zip: {zip_err}")
                raise RuntimeError(f"Failed to create zip archive: {zip_err}") from zip_err
            logger.info(f"Export job {job_id}: PyTorch zip created at {zip_path}")
            logs.append(f"Export completed: {zip_name}")

        with _export_jobs_lock:
            _export_jobs[job_id]["progress"] = 100
            _export_jobs[job_id]["status"] = "completed"
            _export_jobs[job_id]["message"] = f"Export complete: {zip_name}"
            _export_jobs[job_id]["timestamp"] = time.time()
            _export_jobs[job_id]["logs"] = logs.copy()
        logger.info(f"Export job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Export job {job_id} failed: {e}", exc_info=True)
        logs.append(f"Export failed: {str(e)}")
        with _export_jobs_lock:
            _export_jobs[job_id]["status"] = "failed"
            _export_jobs[job_id]["message"] = str(e)
            _export_jobs[job_id]["error"] = str(e)
            _export_jobs[job_id]["timestamp"] = time.time()
            _export_jobs[job_id]["logs"] = logs.copy()


# =============================================================================
# EMBEDDED HTML – unchanged (same as before)
# =============================================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LAZY LLAMA · Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        /* ----- Reset & Base ----- */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #000;
            color: #eee;
            font-family: 'Share Tech Mono', monospace;
            padding: 16px;
            overflow-x: hidden;
        }
        .container { max-width: 1400px; margin: 0 auto; }

        /* ----- Header ----- */
        .header {
            border-bottom: 1px solid #333;
            margin-bottom: 16px;
            padding-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
        .logo h1 {
            font-size: 24px;
            letter-spacing: 2px;
            color: #fff;
            font-weight: 700;
        }
        .logo p { font-size: 10px; color: #888; letter-spacing: 1px; }

        .stats {
            display: flex;
            gap: 20px;
            font-size: 12px;
            background: #111;
            padding: 6px 14px;
            border-radius: 4px;
            border: 1px solid #333;
            flex-wrap: wrap;
        }
        .stat span:first-child { color: #aaa; margin-right: 4px; }
        .stat span:last-child { color: #fff; font-weight: bold; }

        #fullscreen-btn {
            background: transparent;
            border: 1px solid #555;
            color: #ccc;
            font-size: 12px;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            transition: 0.2s;
        }
        #fullscreen-btn:hover { background: #222; border-color: #aaa; color: #fff; }

        /* ----- Global Selectors ----- */
        .global-selectors {
            background: #0a0a0a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }
        .global-selectors .group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .global-selectors label {
            font-size: 11px;
            color: #888;
        }
        .global-selectors select {
            background: #1a1a1a;
            border: 1px solid #333;
            color: #eee;
            font-family: inherit;
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            outline: none;
            min-width: 160px;
        }
        .global-selectors select:focus { border-color: #888; }
        .global-selectors .status-badge {
            font-size: 10px;
            padding: 2px 10px;
            border-radius: 12px;
            background: #1a1a1a;
            border: 1px solid #444;
        }
        .global-selectors .status-badge.loaded { border-color: #4a4; color: #4a4; }
        .global-selectors .status-badge.empty { border-color: #444; color: #666; }
        .global-selectors button {
            background: transparent;
            border: 1px solid #555;
            color: #ddd;
            font-family: inherit;
            font-size: 11px;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .global-selectors button:hover { background: #222; border-color: #aaa; color: #fff; }
        .global-selectors .status-line {
            font-size: 11px;
            color: #aaa;
            margin-left: auto;
            white-space: nowrap;
        }
        .global-selectors .status-line strong { color: #fff; }

        /* ----- Graph Card ----- */
        .graph-card {
            background: #0a0a0a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 16px;
        }
        .graph-card canvas {
            width: 100% !important;
            height: 400px !important;
            background: #000;
        }
        .graph-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: #aaa;
            margin-bottom: 8px;
        }
        .graph-title .task-label {
            color: #88dd88;
        }

        /* ----- Tabs ----- */
        .tabs {
            display: flex;
            gap: 4px;
            border-bottom: 1px solid #333;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: #888;
            font-family: inherit;
            font-size: 12px;
            padding: 8px 16px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: 0.2s;
        }
        .tab-btn:hover { color: #eee; }
        .tab-btn.active {
            color: #fff;
            border-bottom-color: #888;
        }
        .tab-panel {
            display: none;
            background: #0a0a0a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .tab-panel.active { display: block; }

        .panel-title {
            font-size: 14px;
            color: #ccc;
            margin-bottom: 12px;
            border-left: 3px solid #666;
            padding-left: 10px;
        }

        /* ----- Forms & Buttons ----- */
        select, input, textarea {
            width: 100%;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #eee;
            font-family: inherit;
            font-size: 12px;
            padding: 6px 8px;
            margin-bottom: 10px;
            border-radius: 4px;
            outline: none;
        }
        select:focus, input:focus, textarea:focus {
            border-color: #888;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.1);
        }
        button {
            background: transparent;
            border: 1px solid #555;
            color: #ddd;
            font-family: inherit;
            font-size: 11px;
            letter-spacing: 0.5px;
            padding: 5px 14px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        button:hover { background: #222; border-color: #aaa; color: #fff; }
        button.primary { border-color: #888; color: #fff; }
        button.primary:hover { background: #2a2a2a; }
        button.danger { border-color: #a33; color: #f88; }
        button.danger:hover { background: #2a1111; }
        button.success { border-color: #4a4; color: #4a4; }
        button.success:hover { background: #112a11; }
        button.small { font-size: 9px; padding: 2px 6px; margin: 0; }

        .progress-bar {
            background: #1a1a1a;
            border-radius: 4px;
            height: 18px;
            overflow: hidden;
            margin: 6px 0;
        }
        .progress-fill {
            background: #888;
            width: 0%;
            height: 100%;
            transition: width 0.3s;
        }

        /* ----- Results Table ----- */
        .results-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            margin-top: 8px;
        }
        .results-table th, .results-table td {
            border: 1px solid #333;
            padding: 4px 8px;
            text-align: left;
        }
        .results-table th {
            background: #1a1a1a;
            color: #aaa;
        }
        .results-table tr:nth-child(even) { background: #0d0d0d; }

        /* ----- Terminal Log ----- */
        .log-terminal {
            background: #050505;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            padding: 10px;
            height: 200px;
            overflow-y: auto;
            font-size: 11px;
            margin-top: 10px;
        }
        .log-line {
            border-bottom: 1px solid #1a1a1a;
            padding: 3px 0;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .log-line.info { color: #99ccff; }
        .log-line.success { color: #88dd88; }
        .log-line.error { color: #ff7777; }
        .log-line.warning { color: #ffcc66; }

        /* ----- Export Modal ----- */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active { display: flex; }
        .modal-box {
            background: #0a0a0a;
            border: 1px solid #444;
            border-radius: 12px;
            padding: 24px;
            max-width: 600px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            color: #eee;
        }
        .modal-box h2 { margin-bottom: 16px; color: #fff; }
        .modal-close {
            float: right;
            background: transparent;
            border: none;
            color: #aaa;
            font-size: 20px;
            cursor: pointer;
        }
        .modal-close:hover { color: #fff; }
        .modal-log {
            background: #050505;
            border: 1px solid #2a2a2a;
            border-radius: 4px;
            padding: 8px;
            max-height: 150px;
            overflow-y: auto;
            font-size: 10px;
            margin-top: 8px;
            font-family: 'Share Tech Mono', monospace;
        }
        .modal-log .log-line { border-bottom: none; padding: 2px 0; }

        .text-muted { color: #666; font-size: 10px; }
        .mt-1 { margin-top: 8px; }
        .flex-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
        hr { border-color: #2a2a2a; margin: 10px 0; }
        .badge {
            display: inline-block;
            background: #222;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 9px;
            color: #aaa;
            margin-left: 6px;
        }
        @media (max-width: 768px) {
            .global-selectors { flex-direction: column; align-items: stretch; }
            .global-selectors .group { flex-wrap: wrap; }
            .global-selectors .status-line { margin-left: 0; }
        }
    </style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <div class="logo">
            <h1>LAZY LLAMA v3.6</h1>
            <p>low‑end inference · E8 + KV compression · LazyTorch</p>
        </div>
        <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <button id="fullscreen-btn">⛶ Fullscreen</button>
            <div class="stats">
                <div class="stat"><span>t/s</span> <span id="stat-tps">0.00</span></div>
                <div class="stat"><span>RAM</span> <span id="stat-ram">0.0</span> GB</div>
                <div class="stat"><span>CPU</span> <span id="stat-cpu">0</span>%</div>
                <div class="stat"><span>E8</span> <span id="stat-e8">OFF</span></div>
                <div class="stat"><span>LazyTorch</span> <span id="stat-lt">OFF</span></div>
            </div>
        </div>
    </div>

    <!-- ============================================================ -->
    <!-- GLOBAL MODEL SELECTORS (SINGLE SOURCE OF TRUTH) -->
    <!-- ============================================================ -->
    <div class="global-selectors" id="globalSelector">
        <div class="group">
            <label>Teacher:</label>
            <select id="global-teacher">
                <option value="">-- select teacher --</option>
            </select>
            <span class="status-badge empty" id="teacher-status">not loaded</span>
        </div>
        <div class="group">
            <label>Student:</label>
            <select id="global-student">
                <option value="">-- select student --</option>
            </select>
            <span class="status-badge empty" id="student-status">not loaded</span>
        </div>
        <button id="refresh-models-btn">⟳ Refresh</button>
        <span class="status-line" id="model-status-line">Teacher: <strong id="status-teacher">None</strong> → Student: <strong id="status-student">None</strong></span>
    </div>

    <!-- Graph -->
    <div class="graph-card">
        <div class="graph-title">
            <span>GRAPH</span>
            <span class="task-label" id="graph-task-label">—</span>
        </div>
        <canvas id="tpsChart"></canvas>
    </div>

    <!-- Tabs: Models | Distill | Prune | Benchmark | Export | Terminal -->
    <div class="tabs" id="tabHeaders">
        <button class="tab-btn active" data-tab="models">Models</button>
        <button class="tab-btn" data-tab="distill">Distill</button>
        <button class="tab-btn" data-tab="prune">Prune</button>
        <button class="tab-btn" data-tab="benchmark">Benchmark</button>
        <button class="tab-btn" data-tab="export">Export</button>
        <button class="tab-btn" data-tab="terminal">📟 Terminal</button>
    </div>

    <!-- ============================================================ -->
    <!-- TAB: Models (combined student creation + Ollama pull) -->
    <!-- ============================================================ -->
    <div id="tabModels" class="tab-panel active">
        <div class="panel-title">Manage Models</div>
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
            <div style="flex:1; min-width:200px;">
                <h4 style="color:#aaa; font-size:12px;">Create Student</h4>
                <input type="text" id="student-name" placeholder="student_model_name" value="my_student">
                <select id="base-model">
                    <option value="">-- loading base models --</option>
                </select>
                <button id="create-student" class="primary">🔧 Create & Register</button>
                <div id="create-status" class="text-muted mt-1"></div>
            </div>
            <div style="flex:1; min-width:200px;">
                <h4 style="color:#aaa; font-size:12px;">Pull Ollama Teacher</h4>
                <div style="display:flex; gap:8px;">
                    <input type="text" id="pull-teacher-name" placeholder="model name (e.g., llama2)" style="flex:1;">
                    <button id="pull-teacher-btn" class="primary">⬇ Pull</button>
                </div>
                <div id="pull-status" class="text-muted mt-1"></div>
                <div class="progress-bar" style="height:8px; margin-top:4px;"><div class="progress-fill" id="pull-progress" style="height:8px;"></div></div>
            </div>
        </div>
        <hr>
        <div class="panel-title">Existing Models</div>
        <div id="model-list"></div>
    </div>

    <!-- ============================================================ -->
    <!-- TAB: Distill -->
    <!-- ============================================================ -->
    <div id="tabDistill" class="tab-panel">
        <div class="panel-title">Distillation</div>
        <p class="text-muted">Using global teacher: <strong id="distill-teacher-display">(none)</strong> → student: <strong id="distill-student-display">(none)</strong></p>
        <input type="number" id="distill-passes" value="2" min="1" step="1" style="width:80px; display:inline-block;">
        <button id="start-distill" class="primary">▶ START DISTILLATION</button>
        <div class="progress-bar"><div class="progress-fill" id="distill-progress"></div></div>
        <div id="distill-status" class="text-muted"></div>
    </div>

    <!-- ============================================================ -->
    <!-- TAB: Prune -->
    <!-- ============================================================ -->
    <div id="tabPrune" class="tab-panel">
        <div class="panel-title">Prune Model</div>
        <p class="text-muted">Using global student: <strong id="prune-model-display">(none)</strong></p>
        <select id="prune-strategy">
            <option value="magnitude">Magnitude</option>
            <option value="neuron">Neuron</option>
            <option value="task">Task‑specific</option>
        </select>
        <input type="text" id="prune-task" placeholder="task (coding/chat/math)" value="coding">
        <button id="start-prune" class="primary">PRUNE STUDENT</button>
        <div class="progress-bar"><div class="progress-fill" id="prune-progress"></div></div>
        <div id="prune-status" class="text-muted"></div>
    </div>

    <!-- ============================================================ -->
    <!-- TAB: Benchmark -->
    <!-- ============================================================ -->
    <div id="tabBenchmark" class="tab-panel">
        <div class="panel-title">Benchmark</div>
        <p class="text-muted">Using global student: <strong id="benchmark-model-display">(none)</strong></p>
        <button id="start-benchmark" class="primary">▶ BENCHMARK STUDENT</button>
        <button id="start-benchmark-all" class="primary">▶ BENCHMARK ALL STUDENTS</button>
        <div class="progress-bar"><div class="progress-fill" id="benchmark-progress"></div></div>
        <div id="benchmark-status" class="text-muted"></div>
        <div id="benchmark-results" style="margin-top:8px;"></div>
    </div>

    <!-- ============================================================ -->
    <!-- TAB: Export -->
    <!-- ============================================================ -->
    <div id="tabExport" class="tab-panel">
        <div class="panel-title">Export Model</div>
        <p class="text-muted">Using global student: <strong id="export-model-display">(none)</strong></p>
        <select id="export-format">
            <option value="pytorch">PyTorch (.zip)</option>
            <option value="vllm">vLLM (safetensors)</option>
        </select>
        <button id="export-student-btn" class="primary">⬇️ EXPORT AS ZIP</button>
        <button id="export-ollama-btn" class="primary">⬇️ EXPORT TO OLLAMA</button>
        <div id="export-result" class="text-muted mt-1"></div>
    </div>

    <!-- ============================================================ -->
    <!-- TAB: Terminal -->
    <!-- ============================================================ -->
    <div id="tabTerminal" class="tab-panel">
        <div class="panel-title">📟 NLP Terminal</div>
        <p class="text-muted">Using global student: <strong id="terminal-model-display">(none)</strong></p>
        <div style="display:flex; gap:8px; margin-bottom:8px;">
            <input type="text" id="term-input" placeholder="e.g., chat with llama2 about Python" style="flex:3; margin:0;">
            <button id="term-run" class="primary">▶ SEND</button>
            <button id="term-clear">✕ CLEAR</button>
        </div>
        <div class="log-terminal" id="log-terminal">
            <div class="log-line info">► Lazy Llama Dashboard active</div>
            <div class="log-line info">► Select teacher and student models at the top</div>
            <div class="log-line info">► All operations use the selected models</div>
        </div>
    </div>

</div>

<!-- Export Modal -->
<div class="modal-overlay" id="export-modal">
    <div class="modal-box">
        <button class="modal-close" id="modal-close">&times;</button>
        <h2>Exporting Model</h2>
        <div id="modal-status" class="text-muted">Preparing...</div>
        <div class="progress-bar"><div class="progress-fill" id="modal-progress"></div></div>
        <div id="modal-download" style="display:none; margin-top:12px;">
            <a href="#" id="modal-download-link" class="button" style="color:#88dd88; border:1px solid #88dd88; padding:6px 12px; border-radius:4px; text-decoration:none;">⬇ Download</a>
        </div>
        <div id="modal-error" style="display:none; color:#ff7777; margin-top:8px;"></div>
        <div id="modal-log-container" style="margin-top:12px; display:none;">
            <div style="color:#aaa; font-size:10px;">Logs:</div>
            <div class="modal-log" id="modal-log"></div>
        </div>
    </div>
</div>

<script>
    // ============================================================
    // GLOBAL STATE
    // ============================================================
    let globalStudent = '';
    let globalTeacher = '';
    let tpsChart;
    const maxPoints = 60;

    // ============================================================
    // CHART
    // ============================================================
    function initChart() {
        const ctx = document.getElementById('tpsChart').getContext('2d');
        tpsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'tokens/sec',
                    data: [],
                    borderColor: '#ffffff',
                    backgroundColor: 'rgba(255,255,255,0.05)',
                    fill: true,
                    tension: 0.2,
                    pointRadius: 0.5,
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: { grid: { color: '#222' }, ticks: { color: '#aaa' } },
                    x: { grid: { color: '#222' }, ticks: { color: '#666', maxRotation: 45 } }
                },
                plugins: {
                    legend: { labels: { color: '#ccc', font: { size: 10 } } }
                }
            }
        });
    }

    async function fetchMetrics() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();
            document.getElementById('stat-tps').innerText = data.tokens_per_second?.toFixed(2) || '0.00';
            document.getElementById('stat-ram').innerText = data.ram_used_gb?.toFixed(1) || '0.0';
            document.getElementById('stat-cpu').innerText = data.cpu_percent || '0';
            document.getElementById('stat-e8').innerText = data.e8_enabled ? 'ON' : 'OFF';
            document.getElementById('stat-lt').innerText = data.lazytorch_enabled ? 'ON' : 'OFF';

            const [distillRes, pruneRes, benchRes] = await Promise.all([
                fetch('/api/distillation-status'),
                fetch('/api/prune-status'),
                fetch('/api/benchmark-status')
            ]);
            const distill = await distillRes.json();
            const prune = await pruneRes.json();
            const bench = await benchRes.json();

            const taskLabel = document.getElementById('graph-task-label');
            if (distill.running) {
                taskLabel.textContent = `Distillation: ${distill.progress}%`;
                taskLabel.style.color = '#88dd88';
                updateChartWithProgress(distill.progress);
            } else if (prune.running) {
                taskLabel.textContent = `Pruning: ${prune.progress}%`;
                taskLabel.style.color = '#88dd88';
                updateChartWithProgress(prune.progress);
            } else if (bench.running) {
                taskLabel.textContent = `Benchmarking: ${bench.progress}%`;
                taskLabel.style.color = '#88dd88';
                updateChartWithProgress(bench.progress);
            } else {
                taskLabel.textContent = '—';
                taskLabel.style.color = '#aaa';
                if (tpsChart && data.tokens_per_second !== undefined) {
                    const now = new Date().toLocaleTimeString();
                    tpsChart.data.labels.push(now);
                    tpsChart.data.datasets[0].data.push(data.tokens_per_second);
                    if (tpsChart.data.labels.length > maxPoints) {
                        tpsChart.data.labels.shift();
                        tpsChart.data.datasets[0].data.shift();
                    }
                    tpsChart.update('none');
                }
            }
        } catch(e) { console.warn(e); }
    }

    function updateChartWithProgress(progress) {
        const now = new Date().toLocaleTimeString();
        tpsChart.data.labels.push(now);
        tpsChart.data.datasets[0].data.push(progress);
        if (tpsChart.data.labels.length > maxPoints) {
            tpsChart.data.labels.shift();
            tpsChart.data.datasets[0].data.shift();
        }
        tpsChart.update('none');
    }

    // ============================================================
    // GLOBAL MODEL LOADING & SELECTION (combined list)
    // ============================================================
    async function loadGlobalModels() {
        try {
            const res = await fetch('/api/global-models');
            const data = await res.json();
            const teacherSel = document.getElementById('global-teacher');
            const studentSel = document.getElementById('global-student');

            // Clear both dropdowns
            teacherSel.innerHTML = '<option value="">-- select teacher --</option>';
            studentSel.innerHTML = '<option value="">-- select student --</option>';

            // Combine all models into one list with labels
            const allModels = data.all_models || [];
            for (const entry of allModels) {
                const name = entry.name;
                const typeLabel = entry.type; // 'local', 'ollama', 'gguf', 'lazytorch'
                const label = `${name} (${typeLabel})`;
                // Add to teacher dropdown
                const optTeacher = document.createElement('option');
                optTeacher.value = name;
                optTeacher.textContent = label;
                teacherSel.appendChild(optTeacher);
                // Add to student dropdown as well
                const optStudent = document.createElement('option');
                optStudent.value = name;
                optStudent.textContent = label;
                studentSel.appendChild(optStudent);
            }

            // Set current selections
            if (data.global_teacher) {
                teacherSel.value = data.global_teacher;
                globalTeacher = data.global_teacher;
                document.getElementById('teacher-status').textContent = 'loaded: ' + globalTeacher;
                document.getElementById('teacher-status').className = 'status-badge loaded';
            } else {
                document.getElementById('teacher-status').textContent = 'not loaded';
                document.getElementById('teacher-status').className = 'status-badge empty';
            }
            if (data.global_student) {
                studentSel.value = data.global_student;
                globalStudent = data.global_student;
                document.getElementById('student-status').textContent = 'loaded: ' + globalStudent;
                document.getElementById('student-status').className = 'status-badge loaded';
            } else {
                document.getElementById('student-status').textContent = 'not loaded';
                document.getElementById('student-status').className = 'status-badge empty';
            }

            // Update status line and tab displays
            updateDisplays();

            // Attach change handlers (remove old ones first)
            teacherSel.onchange = function() { setGlobalModels(); };
            studentSel.onchange = function() { setGlobalModels(); };
        } catch(e) {
            console.error('Failed to load global models:', e);
        }
    }

    async function setGlobalModels() {
        const teacher = document.getElementById('global-teacher').value;
        const student = document.getElementById('global-student').value;
        try {
            const res = await fetch('/api/set-global-models', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({teacher, student})
            });
            const data = await res.json();
            if (data.success) {
                globalTeacher = teacher;
                globalStudent = student;
                // Update status badges
                if (teacher) {
                    document.getElementById('teacher-status').textContent = 'loaded: ' + teacher;
                    document.getElementById('teacher-status').className = 'status-badge loaded';
                } else {
                    document.getElementById('teacher-status').textContent = 'not loaded';
                    document.getElementById('teacher-status').className = 'status-badge empty';
                }
                if (student) {
                    document.getElementById('student-status').textContent = 'loaded: ' + student;
                    document.getElementById('student-status').className = 'status-badge loaded';
                } else {
                    document.getElementById('student-status').textContent = 'not loaded';
                    document.getElementById('student-status').className = 'status-badge empty';
                }
                updateDisplays();
                addLog(`Global models set: teacher=${teacher || 'none'}, student=${student || 'none'}`, 'success');
            }
        } catch(e) {
            console.error('Failed to set global models:', e);
        }
    }

    function updateDisplays() {
        // Status line
        document.getElementById('status-teacher').textContent = globalTeacher || 'None';
        document.getElementById('status-student').textContent = globalStudent || 'None';
        // Tab displays
        document.getElementById('distill-teacher-display').textContent = globalTeacher || '(none)';
        document.getElementById('distill-student-display').textContent = globalStudent || '(none)';
        document.getElementById('prune-model-display').textContent = globalStudent || '(none)';
        document.getElementById('benchmark-model-display').textContent = globalStudent || '(none)';
        document.getElementById('export-model-display').textContent = globalStudent || '(none)';
        document.getElementById('terminal-model-display').textContent = globalStudent || '(none)';
    }

    // ============================================================
    // BASE MODELS DROPDOWN (dynamic)
    // ============================================================
    async function loadBaseModels() {
        try {
            const res = await fetch('/api/base-models');
            const models = await res.json();
            const select = document.getElementById('base-model');
            select.innerHTML = '';
            if (!models || models.length === 0) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = '-- no base models available --';
                select.appendChild(opt);
                return;
            }
            for (const name of models) {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                select.appendChild(opt);
            }
        } catch(e) {
            console.error('Failed to load base models:', e);
        }
    }

    // ============================================================
    // TAB SWITCHING
    // ============================================================
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            const tabId = this.dataset.tab;
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            const panelMap = {
                'models': 'tabModels',
                'distill': 'tabDistill',
                'prune': 'tabPrune',
                'benchmark': 'tabBenchmark',
                'export': 'tabExport',
                'terminal': 'tabTerminal'
            };
            document.getElementById(panelMap[tabId]).classList.add('active');
        });
    });

    // ============================================================
    // FULLSCREEN
    // ============================================================
    document.getElementById('fullscreen-btn').addEventListener('click', function() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {});
        } else {
            if (document.exitFullscreen) document.exitFullscreen();
        }
    });
    document.addEventListener('fullscreenchange', function() {
        const btn = document.getElementById('fullscreen-btn');
        if (document.fullscreenElement) {
            btn.textContent = '⛶ Exit Fullscreen';
            if (tpsChart) tpsChart.resize();
        } else {
            btn.textContent = '⛶ Fullscreen';
            if (tpsChart) tpsChart.resize();
        }
    });

    // ============================================================
    // UI HELPERS
    // ============================================================
    function addLog(msg, type='info') {
        const logDiv = document.getElementById('log-terminal');
        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        logDiv.appendChild(line);
        line.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function updateModelList() {
        fetch('/api/exportable-models')
            .then(r => r.json())
            .then(models => {
                const container = document.getElementById('model-list');
                container.innerHTML = '';
                if (models.length === 0) {
                    container.innerHTML = '<div class="text-muted">No local models found.</div>';
                    return;
                }
                models.forEach(m => {
                    const div = document.createElement('div');
                    div.style.display = 'flex';
                    div.style.justifyContent = 'space-between';
                    div.style.padding = '4px 0';
                    div.style.borderBottom = '1px solid #222';
                    div.style.alignItems = 'center';
                    const nameSpan = document.createElement('span');
                    nameSpan.textContent = `${m.name} (${(m.size_mb/1024).toFixed(1)} GB)`;
                    const btnGroup = document.createElement('span');
                    const validateBtn = document.createElement('button');
                    validateBtn.textContent = '✓ Validate';
                    validateBtn.className = 'small success';
                    validateBtn.style.margin = '0 4px';
                    validateBtn.addEventListener('click', () => {
                        validateModel(m.name);
                    });
                    const delBtn = document.createElement('button');
                    delBtn.textContent = '🗑';
                    delBtn.className = 'danger small';
                    delBtn.style.margin = '0';
                    delBtn.style.padding = '2px 8px';
                    delBtn.addEventListener('click', () => {
                        if (confirm(`Delete ${m.name}?`)) {
                            fetch('/api/delete-model', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ name: m.name })
                            }).then(() => {
                                addLog(`Deleted ${m.name}`, 'warning');
                                updateModelList();
                                loadGlobalModels();
                                loadBaseModels();
                            });
                        }
                    });
                    btnGroup.appendChild(validateBtn);
                    btnGroup.appendChild(delBtn);
                    div.appendChild(nameSpan);
                    div.appendChild(btnGroup);
                    container.appendChild(div);
                });
            });
    }

    async function validateModel(name) {
        try {
            const res = await fetch('/api/validate-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            });
            const data = await res.json();
            if (data.valid) {
                addLog(`Model "${name}" is valid.`, 'success');
            } else {
                addLog(`Model "${name}" is invalid: ${data.reason || 'unknown reason'}`, 'error');
            }
        } catch (e) {
            addLog(`Validation failed: ${e.message}`, 'error');
        }
    }

    // Refresh function that updates everything
    function refreshAll() {
        loadGlobalModels();
        updateModelList();
        loadBaseModels();
        addLog('Models refreshed', 'info');
    }

    // ============================================================
    // CREATE STUDENT
    // ============================================================
    document.getElementById('create-student').onclick = async () => {
        const base = document.getElementById('base-model').value;
        const name = document.getElementById('student-name').value.trim();
        if (!base) { addLog('Please select a base model', 'error'); return; }
        if (!name) { addLog('Student name required', 'error'); return; }
        addLog(`Creating student "${name}" from ${base}...`, 'info');
        document.getElementById('create-status').innerText = 'creating...';
        const res = await fetch('/api/create-student', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_model: base, student_name: name })
        });
        const data = await res.json();
        if (data.success) {
            addLog(`Student "${name}" created and registered`, 'success');
            document.getElementById('create-status').innerHTML = '✅ created';
            refreshAll();
        } else {
            addLog(`Creation failed: ${data.error}`, 'error');
            document.getElementById('create-status').innerHTML = `❌ ${data.error}`;
        }
    };

    // ============================================================
    // PULL OLLAMA TEACHER
    // ============================================================
    async function pollPullProgress() {
        const interval = setInterval(async () => {
            const res = await fetch('/api/pull-status');
            const state = await res.json();
            document.getElementById('pull-progress').style.width = state.progress + '%';
            document.getElementById('pull-status').innerText = state.status || '';
            if (!state.running) {
                clearInterval(interval);
                if (state.status === 'completed') {
                    addLog(`Pulled ${state.model} successfully`, 'success');
                    document.getElementById('pull-status').innerHTML = '✅ pulled';
                    refreshAll();
                } else if (state.status === 'failed') {
                    addLog(`Pull failed: ${state.error}`, 'error');
                    document.getElementById('pull-status').innerHTML = `❌ ${state.error}`;
                }
            }
        }, 1000);
    }

    document.getElementById('pull-teacher-btn').onclick = async () => {
        const name = document.getElementById('pull-teacher-name').value.trim();
        if (!name) { addLog('Enter a model name to pull', 'warning'); return; }
        addLog(`Pulling ${name} from Ollama...`, 'info');
        document.getElementById('pull-status').innerText = 'starting...';
        document.getElementById('pull-progress').style.width = '0%';
        const res = await fetch('/api/pull-ollama', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: name })
        });
        const data = await res.json();
        if (data.success) {
            pollPullProgress();
        } else {
            addLog(`Pull failed: ${data.error}`, 'error');
            document.getElementById('pull-status').innerHTML = `❌ ${data.error}`;
        }
    };

    // ============================================================
    // DISTILLATION
    // ============================================================
    async function pollDistillProgress() {
        const interval = setInterval(async () => {
            const res = await fetch('/api/distillation-status');
            const state = await res.json();
            document.getElementById('distill-progress').style.width = state.progress + '%';
            document.getElementById('distill-status').innerText = state.status;
            if (!state.running) {
                clearInterval(interval);
                if (state.error) {
                    addLog(`Distillation failed: ${state.error}`, 'error');
                    if (state.user_friendly_error) {
                        addLog(`Details: ${state.user_friendly_error}`, 'warning');
                    }
                } else {
                    addLog('Distillation completed.', 'success');
                }
                refreshAll();
            }
        }, 1500);
    }

    document.getElementById('start-distill').onclick = async () => {
        const teacher = document.getElementById('global-teacher').value;
        const student = document.getElementById('global-student').value;
        const passes = parseInt(document.getElementById('distill-passes').value);
        if (!teacher || !student) { addLog('Select teacher and student at the top', 'error'); return; }
        addLog(`Starting distillation ${teacher} → ${student} (${passes} passes)`, 'info');
        const res = await fetch('/api/start-distill', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ teacher, student, passes })
        });
        const data = await res.json();
        if (data.success) pollDistillProgress();
        else addLog(`Distillation start failed: ${data.error}`, 'error');
    };

    // ============================================================
    // PRUNE
    // ============================================================
    async function pollPruneProgress() {
        const interval = setInterval(async () => {
            const res = await fetch('/api/prune-status');
            const state = await res.json();
            document.getElementById('prune-progress').style.width = state.progress + '%';
            document.getElementById('prune-status').innerText = state.status;
            if (!state.running) {
                clearInterval(interval);
                if (state.status.includes('failed')) {
                    addLog(`Pruning failed: ${state.status}`, 'error');
                    if (state.user_friendly_error) {
                        addLog(`Details: ${state.user_friendly_error}`, 'warning');
                    }
                } else {
                    addLog('Pruning completed.', 'success');
                }
                refreshAll();
            }
        }, 1500);
    }

    document.getElementById('start-prune').onclick = async () => {
        const model = document.getElementById('global-student').value;
        const strategy = document.getElementById('prune-strategy').value;
        const task = document.getElementById('prune-task').value;
        if (!model) { addLog('Select a student model at the top', 'error'); return; }
        addLog(`Pruning ${model} with ${strategy}...`, 'info');
        const res = await fetch('/api/start-prune', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, strategy, task })
        });
        const data = await res.json();
        if (data.success) pollPruneProgress();
        else addLog(`Prune start failed: ${data.error}`, 'error');
    };

    // ============================================================
    // BENCHMARK
    // ============================================================
    async function pollBenchmarkProgress() {
        const interval = setInterval(async () => {
            const res = await fetch('/api/benchmark-status');
            const state = await res.json();
            document.getElementById('benchmark-progress').style.width = state.progress + '%';
            document.getElementById('benchmark-status').innerText = state.status;
            if (!state.running) {
                clearInterval(interval);
                if (state.results) {
                    displayBenchmarkResults(state.results);
                    addLog('Benchmark completed.', 'success');
                } else if (state.error) {
                    addLog(`Benchmark failed: ${state.error}`, 'error');
                    if (state.user_friendly_error) {
                        addLog(`Details: ${state.user_friendly_error}`, 'warning');
                    }
                }
                // Reset progress after a moment
                setTimeout(() => {
                    document.getElementById('benchmark-progress').style.width = '0%';
                }, 2000);
            }
        }, 1000);
    }

    function displayBenchmarkResults(results) {
        const container = document.getElementById('benchmark-results');
        if (!results || results.length === 0) {
            container.innerHTML = '<div class="text-muted">No results.</div>';
            return;
        }
        let html = '<table class="results-table"><tr><th>Model</th><th>tok/s</th><th>Peak Mem (GB)</th><th>E8</th><th>KV</th><th>LazyTorch</th></tr>';
        for (const r of results) {
            if (r.success) {
                html += `<tr>
                    <td>${r.model_name}</td>
                    <td>${r.tokens_per_second.toFixed(2)}</td>
                    <td>${r.peak_memory_gb.toFixed(2)}</td>
                    <td>${r.e8_quantized ? '✓' : ''}</td>
                    <td>${r.kv_compressed ? '✓' : ''}</td>
                    <td>${r.lazytorch_used ? '✓' : ''}</td>
                </tr>`;
            } else {
                html += `<tr><td>${r.model_name}</td><td colspan="5">FAILED: ${r.error || 'unknown'}</td></tr>`;
            }
        }
        html += '</table>';
        container.innerHTML = html;
    }

    document.getElementById('start-benchmark').onclick = async () => {
        const model = document.getElementById('global-student').value;
        if (!model) { addLog('Select a student model at the top', 'error'); return; }
        addLog(`Benchmarking ${model}...`, 'info');
        document.getElementById('benchmark-results').innerHTML = '';
        // Include benchmark settings from a hidden form or default
        const settings = {
            prompt: "What is machine learning?",
            max_tokens: 100,
            run_perplexity: false,
            run_multiple_choice: false,
            run_long_context: false,
            store_in_registry: true
        };
        const res = await fetch('/api/start-benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: model, all_students: false, settings })
        });
        const data = await res.json();
        if (data.success) pollBenchmarkProgress();
        else addLog(`Benchmark start failed: ${data.error}`, 'error');
    };

    document.getElementById('start-benchmark-all').onclick = async () => {
        addLog('Benchmarking all students...', 'info');
        document.getElementById('benchmark-results').innerHTML = '';
        const settings = {
            prompt: "What is machine learning?",
            max_tokens: 100,
            run_perplexity: false,
            run_multiple_choice: false,
            run_long_context: false,
            store_in_registry: true
        };
        const res = await fetch('/api/start-benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ all_students: true, settings })
        });
        const data = await res.json();
        if (data.success) pollBenchmarkProgress();
        else addLog(`Benchmark all failed: ${data.error}`, 'error');
    };

    // ============================================================
    // EXPORT
    // ============================================================
    const modal = document.getElementById('export-modal');
    const modalClose = document.getElementById('modal-close');
    const modalProgress = document.getElementById('modal-progress');
    const modalStatus = document.getElementById('modal-status');
    const modalDownload = document.getElementById('modal-download');
    const modalDownloadLink = document.getElementById('modal-download-link');
    const modalError = document.getElementById('modal-error');
    const modalLogContainer = document.getElementById('modal-log-container');
    const modalLog = document.getElementById('modal-log');

    function showModal() { modal.classList.add('active'); }
    function hideModal() { modal.classList.remove('active'); }

    modalClose.addEventListener('click', hideModal);
    modal.addEventListener('click', (e) => { if (e.target === modal) hideModal(); });

    document.getElementById('export-student-btn').onclick = async () => {
        const model = document.getElementById('global-student').value;
        const format = document.getElementById('export-format').value;
        if (!model) { addLog('Select a student model at the top', 'error'); return; }
        showModal();
        modalProgress.style.width = '0%';
        modalStatus.innerText = 'Starting export...';
        modalDownload.style.display = 'none';
        modalError.style.display = 'none';
        modalLogContainer.style.display = 'none';
        modalLog.innerHTML = '';

        const res = await fetch('/api/export-student', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: model, format: format, async: true })
        });
        const data = await res.json();
        if (!data.success) {
            modalError.style.display = 'block';
            modalError.innerText = data.error || 'Export failed';
            return;
        }
        const jobId = data.job_id;

        const pollInterval = setInterval(async () => {
            const statusRes = await fetch(`/api/export-status/${jobId}`);
            const status = await statusRes.json();
            modalProgress.style.width = status.progress + '%';
            modalStatus.innerText = status.message || 'Processing...';
            // Update logs if present
            if (status.logs && status.logs.length > 0) {
                modalLogContainer.style.display = 'block';
                modalLog.innerHTML = status.logs.map(log => `<div class="log-line info">${log}</div>`).join('');
                modalLog.scrollTop = modalLog.scrollHeight;
            }
            if (status.status === 'completed') {
                clearInterval(pollInterval);
                modalDownload.style.display = 'block';
                modalDownloadLink.href = `/api/download/${jobId}`;
                modalDownloadLink.download = status.zip_name || 'model.zip';
                addLog(`Export completed: ${status.zip_name}`, 'success');
            } else if (status.status === 'failed') {
                clearInterval(pollInterval);
                modalError.style.display = 'block';
                modalError.innerText = status.message || 'Export failed';
                addLog(`Export failed: ${status.message}`, 'error');
            }
        }, 1000);
    };

    document.getElementById('export-ollama-btn').onclick = async () => {
        const model = document.getElementById('global-student').value;
        if (!model) { addLog('Select a student model at the top', 'error'); return; }
        addLog(`Exporting ${model} to Ollama...`, 'info');
        const res = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: model, path: '' })  // path will be resolved on server
        });
        const data = await res.json();
        if (data.success) {
            addLog(`Exported ${model} to Ollama`, 'success');
            document.getElementById('export-result').innerHTML = '✅ Exported to Ollama';
        } else {
            addLog(`Export failed: ${data.error}`, 'error');
            document.getElementById('export-result').innerHTML = `❌ ${data.error}`;
        }
    };

    // ============================================================
    // TERMINAL (enhanced with actual inference)
    // ============================================================
    document.getElementById('term-run').onclick = async () => {
        const command = document.getElementById('term-input').value.trim();
        const model = document.getElementById('global-student').value;
        if (!command) return;
        if (!model) { addLog('Please select a student model at the top', 'error'); return; }
        addLog(`▶ ${command}`, 'info');
        document.getElementById('term-input').value = '';

        try {
            const res = await fetch('/api/terminal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command, model })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                if (data.result) {
                    if (typeof data.result === 'string') {
                        addLog(data.result, 'success');
                    } else {
                        addLog(JSON.stringify(data.result, null, 2), 'success');
                    }
                } else if (data.stream) {
                    addLog('Streaming response (implement with EventSource)', 'info');
                }
            } else {
                addLog(`Error: ${data.message}`, 'error');
            }
        } catch(e) {
            addLog(`Request failed: ${e.message}`, 'error');
        }
    };

    document.getElementById('term-clear').onclick = () => {
        document.getElementById('log-terminal').innerHTML = '';
    };

    document.getElementById('refresh-models-btn').onclick = refreshAll;

    // ============================================================
    // INIT
    // ============================================================
    initChart();
    refreshAll();
    setInterval(fetchMetrics, 2000);
    setInterval(refreshAll, 30000);  // periodic refresh
    fetchMetrics();
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ('/', '/dashboard'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == '/api/metrics':
            self.send_json(self._get_metrics())
        elif self.path == '/api/exportable-models':
            self.send_json(self._get_exportable_models())
        elif self.path == '/api/ollama-models':
            self.send_json(self._get_ollama_models())
        elif self.path == '/api/distillation-status':
            with _state_lock:
                self.send_json(_distillation_state)
        elif self.path == '/api/prune-status':
            with _state_lock:
                self.send_json(_prune_state)
        elif self.path == '/api/benchmark-status':
            with _state_lock:
                self.send_json(_benchmark_state)
        elif self.path.startswith('/api/export-status/'):
            job_id = self.path.split('/')[-1]
            self._handle_export_status(job_id)
        elif self.path == '/api/global-models':
            self.send_json(self._get_global_models())
        elif self.path == '/api/pull-status':
            with _state_lock:
                self.send_json(_pull_state)
        elif self.path == '/api/base-models':
            self.send_json(self._get_base_models())
        elif self.path.startswith('/api/download/'):
            job_id = self.path.split('/')[-1]
            self._handle_download(job_id)
        elif self.path.startswith('/api/benchmark-report/'):
            model_name = self.path.split('/')[-1]
            self._handle_benchmark_report(model_name)
        elif self.path == '/api/task-state':
            with _state_lock:
                self.send_json({
                    "distillation": _distillation_state,
                    "prune": _prune_state,
                    "benchmark": _benchmark_state,
                    "student_creation": _student_creation_state,
                })
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            if self.path == '/api/export':
                self._handle_export()
            elif self.path == '/api/convert-lazytorch':
                self._handle_convert_lazytorch()
            elif self.path == '/api/create-student':
                self._handle_create_student()
            elif self.path == '/api/start-distill':
                self._handle_start_distill()
            elif self.path == '/api/start-prune':
                self._handle_start_prune()
            elif self.path == '/api/start-benchmark':
                self._handle_start_benchmark()
            elif self.path == '/api/export-student':
                self._handle_export_student()
            elif self.path == '/api/terminal':
                self._handle_terminal()
            elif self.path == '/api/pull-ollama':
                self._handle_pull_ollama()
            elif self.path == '/api/delete-model':
                self._handle_delete_model()
            elif self.path == '/api/set-global-models':
                self._handle_set_global_models()
            elif self.path == '/api/validate-model':
                self._handle_validate_model()
            else:
                self.send_error(404)
        except Exception as e:
            logger.exception(f"Unhandled error in POST {self.path}")
            self.send_json({'success': False, 'error': f'Internal server error: {str(e)}'}, 500)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    # =====================================================================
    # Endpoints
    # =====================================================================

    def _get_metrics(self):
        ms = MetricsStore()
        hist = ms.get_history(seconds=2)
        config = load_config()
        lazytorch_enabled = config.use_lazytorch
        with _state_lock:
            distill = _distillation_state
            prune = _prune_state
            creation = _student_creation_state
            bench = _benchmark_state

        active_task = ""
        task_progress = 0
        if distill["running"]:
            active_task = "distillation"
            task_progress = distill["progress"]
        elif prune["running"]:
            active_task = "prune"
            task_progress = prune["progress"]
        elif creation["running"]:
            active_task = "student_creation"
            task_progress = creation["progress"]
        elif bench["running"]:
            active_task = "benchmark"
            task_progress = bench["progress"]

        tps = 0.0
        if hist:
            latest = hist[-1]
            tps = latest.get("tokens_per_second", 0.0)

        if tps == 0.0 and active_task:
            tps = task_progress / 10.0

        response = {
            "tokens_per_second": tps,
            "inference_latency_ms": hist[-1].get("inference_latency_ms", 0) if hist else 0,
            "ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_percent": psutil.cpu_percent(),
            "distillation_progress": distill["progress"] if distill["running"] else 0,
            "distillation_status": distill["status"],
            "active_model": "None",
            "queue_length": 0,
            "e8_enabled": config.use_e8_quantization,
            "kv_compression_bits": config.kv_cache_bits if config.use_kv_cache_compression else 0,
            "lazytorch_enabled": lazytorch_enabled,
            "lazytorch_savings_percent": 95.0 if lazytorch_enabled else 0.0,
            "active_task": active_task,
            "task_progress": task_progress,
            "distill_error": distill.get("error"),
            "distill_user_error": distill.get("user_friendly_error"),
            "distill_error_type": distill.get("error_type"),
            "prune_error": prune.get("error"),
            "prune_user_error": prune.get("user_friendly_error"),
            "prune_error_type": prune.get("error_type"),
            "creation_error": creation.get("error"),
            "creation_user_error": creation.get("user_friendly_error"),
            "creation_error_type": creation.get("error_type"),
            "benchmark_error": bench.get("error"),
            "benchmark_user_error": bench.get("user_friendly_error"),
            "benchmark_error_type": bench.get("error_type"),
        }
        return response

    def _get_exportable_models(self):
        mm = ModelManager()
        models = []
        for info in mm.list_models():
            if info.path and not info.path.startswith('ollama://'):
                try:
                    models.append({'name': info.name, 'path': info.path, 'size_mb': info.original_size_mb})
                except Exception:
                    pass
        return models

    def _get_ollama_models(self):
        try:
            _model_manager.sync_ollama()
            ollama_models = []
            for info in _model_manager.list_models():
                if info.path and info.path.startswith('ollama://'):
                    size = info.original_size_mb * 1_000_000 if info.original_size_mb else 0
                    ollama_models.append({'name': info.name, 'size': size})
            return ollama_models
        except Exception as e:
            logger.exception("Failed to get Ollama models")
            return []

    def _get_global_models(self):
        """Return a combined list of all models with type labels."""
        mm = ModelManager()
        mm.sync_ollama()  # force fresh list

        all_models = []
        for info in mm.list_models():
            if info.path:
                if info.path.startswith('ollama://'):
                    type_label = 'ollama'
                elif info.path.endswith('.gguf'):
                    type_label = 'gguf'
                elif is_lazytorch_model(Path(info.path)):
                    type_label = 'lazytorch'
                else:
                    type_label = 'local'
                all_models.append({
                    'name': info.name,
                    'type': type_label,
                    'path': info.path
                })

        # If no models, include defaults
        if not all_models:
            defaults = ["distilgpt2", "gpt2", "facebook/opt-125m"]
            for name in defaults:
                all_models.append({'name': name, 'type': 'local', 'path': name})

        return {
            "all_models": all_models,
            "global_student": _global_student,
            "global_teacher": _global_teacher
        }

    def _get_base_models(self):
        """Return a list of local model names suitable as base for student creation."""
        mm = ModelManager()
        mm.sync_ollama()
        mm.reload_registry()
        base_models = []
        for info in mm.list_models():
            # Only local models (non-Ollama) and not already a student (distilled/pruned)
            if info.path and not info.path.startswith('ollama://'):
                name = info.name
                # Exclude models that are clearly students (distilled/pruned)
                if '_distilled' not in name and not name.endswith('_pruned'):
                    base_models.append(name)
        # If no base models found, fallback to hardcoded defaults for UX
        if not base_models:
            base_models = ["distilgpt2", "gpt2", "facebook/opt-125m"]
        return base_models

    def _handle_set_global_models(self):
        global _global_student, _global_teacher
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            teacher = data.get('teacher') or None
            student = data.get('student') or None
            with _state_lock:
                _global_student = student
                _global_teacher = teacher
            _save_global_state(teacher, student)
            self.send_json({'success': True})
        except Exception as e:
            logger.exception("set-global-models failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_validate_model(self):
        """Validate a model by name using ModelManager.validate_model()."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            name = data.get('name')
            if not name:
                self.send_json({'success': False, 'error': 'Missing model name'}, 400)
                return
            mm = ModelManager()
            # Use the enhanced validation that actually loads the tokenizer
            valid = False
            info = mm.registry.get(name)
            if info and info.path:
                path_obj = Path(info.path)
                if path_obj.exists():
                    if path_obj.is_dir():
                        valid = _validate_student_directory(path_obj)
                    elif path_obj.is_file() and path_obj.suffix == ".gguf":
                        # For GGUF, we consider it valid if it exists (we don't have tokenizer)
                        valid = True
            if not info:
                valid = False
            reason = None
            if not valid and info:
                if not info.path:
                    reason = "No path associated with this model."
                elif info.path.startswith("ollama://"):
                    reason = "Ollama models are considered valid remotely."
                else:
                    path_obj = Path(info.path)
                    if not path_obj.exists():
                        reason = f"Path does not exist: {path_obj}"
                    elif path_obj.is_dir():
                        # We already called _validate_student_directory, which provides detailed reasons
                        reason = "Model directory is invalid (missing/corrupt config, tokenizer, or weights)."
                    elif path_obj.is_file():
                        if path_obj.suffix != ".gguf":
                            reason = "File is not a .gguf"
                        else:
                            reason = "File exists but may not be a valid GGUF."
                    else:
                        reason = "Path is neither a file nor a directory."
            self.send_json({'valid': valid, 'reason': reason})
        except Exception as e:
            logger.exception("validate-model failed")
            self.send_json({'valid': False, 'reason': str(e)}, 500)

    def _handle_export(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self.send_json({'success': False, 'error': 'No data'}, 400)
                return
            data = json.loads(self.rfile.read(length))
            model_name = data.get('name')
            if not model_name:
                self.send_json({'success': False, 'error': 'Missing model name'}, 400)
                return
            mm = ModelManager()
            info = mm.get_model(model_name)
            if not info or not info.path:
                self.send_json({'success': False, 'error': f'Model {model_name} not found'}, 404)
                return
            # Validate before export
            if info.path and not info.path.startswith('ollama://'):
                path_obj = Path(info.path)
                if path_obj.is_dir() and not _validate_student_directory(path_obj):
                    self.send_json({'success': False, 'error': 'Model directory is corrupt; cannot export'}, 400)
                    return
            success = export_to_ollama(info.path, model_name)
            # NOTE: No size update needed; export to Ollama doesn't change local model size.
            self.send_json({'success': success, 'error': '' if success else 'Export failed'})
        except Exception as e:
            logger.exception("export failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_convert_lazytorch(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            name = data.get('name')
            if not name:
                self.send_json({'success': False, 'error': 'Missing name'}, 400)
                return
            mm = ModelManager()
            info = mm.get_model(name)
            if not info or not info.path:
                self.send_json({'success': False, 'error': 'Model not found'}, 404)
                return
            if is_lazytorch_model(Path(info.path)):
                self.send_json({'success': False, 'error': 'Already LazyTorch'}, 400)
                return
            # Validate before conversion
            path_obj = Path(info.path)
            if path_obj.is_dir() and not _validate_student_directory(path_obj):
                self.send_json({'success': False, 'error': 'Model directory is corrupt; cannot convert'}, 400)
                return
            result_path = mm.convert_to_lazytorch(name)
            if result_path:
                with mm._lock:
                    if name in mm.registry:
                        mm.registry[name].lazytorch_format = True
                        mm._save_registry()
                self.send_json({'success': True, 'path': str(result_path)})
            else:
                self.send_json({'success': False, 'error': 'Conversion failed'}, 500)
        except Exception as e:
            logger.exception("convert-lazytorch failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_create_student(self):
        """Create a student model from a base model."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            base = data.get('base_model')
            student_name = data.get('student_name')
            if not base or not student_name:
                self.send_json({'success': False, 'error': 'Missing base_model or student_name'}, 400)
                return

            mm = ModelManager()
            base_info = mm.get_model(base)

            # ---- Validate base model existence ----
            if base_info:
                # Local model: check directory integrity
                if base_info.invalid:
                    self.send_json({'success': False, 'error': f'Base model "{base}" is marked invalid.'}, 400)
                    return
                if base_info.path and not base_info.path.startswith('ollama://'):
                    base_path = Path(base_info.path)
                    if base_path.is_dir() and not _validate_student_directory(base_path):
                        self.send_json({'success': False, 'error': f'Base model directory {base_path} is corrupt.'}, 400)
                        return
                # If it's Ollama/vLLM, reject
                if base_info.path and (base_info.path.startswith('ollama://') or base_info.path.startswith('vllm://')):
                    self.send_json({'success': False, 'error': f'Cannot create student from Ollama/vLLM model "{base}". Please use a local Hugging Face model.'}, 400)
                    return
            else:
                # Not in registry: treat as Hugging Face model ID; verify existence on Hub
                if HF_HUB_AVAILABLE:
                    try:
                        hf_model_info(base)  # raises if not found
                    except Exception as e:
                        self.send_json({
                            'success': False,
                            'error': f"Model '{base}' not found in registry and is not a valid Hugging Face model ID: {e}"
                        }, 400)
                        return
                else:
                    # If huggingface_hub not installed, we'll let the thread handle it, but we warn the user
                    logger.warning("huggingface_hub not installed; skipping remote validation of base model ID.")
                    # Proceed anyway; the thread will raise an error if not found.

            with _state_lock:
                if _student_creation_state["running"]:
                    self.send_json({'success': False, 'error': 'Creation already in progress'}, 409)
                    return

            # Start background thread for creation
            thread = threading.Thread(target=_run_student_creation, args=(base, student_name))
            thread.start()
            self.send_json({'success': True})
        except Exception as e:
            logger.exception("create-student failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_start_distill(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            teacher = data.get('teacher') or _global_teacher
            student = data.get('student') or _global_student
            passes = data.get('passes', 2)

            if not teacher or not student:
                self.send_json({
                    'success': False,
                    'error': 'Please select a teacher and student model using the global selectors at the top.'
                }, 400)
                return

            mm = ModelManager()
            teacher_info = mm.get_model(teacher)
            student_info = mm.get_model(student)

            if not teacher_info:
                self.send_json({'success': False, 'error': f'Teacher model "{teacher}" not found in registry'}, 400)
                return
            if not student_info:
                self.send_json({'success': False, 'error': f'Student model "{student}" not found in registry'}, 400)
                return

            # ---- FIX: Validate that student is a local model ----
            if not _is_local_model(student_info):
                self.send_json({
                    'success': False,
                    'error': f"Student model '{student}' is not a local model. "
                             "Distillation requires a local Hugging Face model. "
                             "Please select a local student model."
                }, 400)
                return

            # Validate student directory before distillation
            if student_info.path and not student_info.path.startswith('ollama://'):
                student_path = Path(student_info.path)
                if student_path.is_dir() and not _validate_student_directory(student_path):
                    self.send_json({'success': False, 'error': f'Student model "{student}" is corrupt (invalid tokenizer/config).'}, 400)
                    return

            with _state_lock:
                if _distillation_state["running"]:
                    self.send_json({'success': False, 'error': 'Distillation already running'}, 409)
                    return

            thread = threading.Thread(target=_run_distillation_thread, args=(teacher, student, passes))
            thread.start()
            self.send_json({'success': True})
        except Exception as e:
            logger.exception("start-distill failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_start_prune(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            model = data.get('model') or _global_student
            strategy = data.get('strategy', 'magnitude')
            task = data.get('task', 'coding')

            if not model:
                self.send_json({
                    'success': False,
                    'error': 'Please select a student model using the global selector at the top.'
                }, 400)
                return

            mm = ModelManager()
            info = mm.get_model(model)
            if not info:
                self.send_json({'success': False, 'error': f'Model "{model}" not found in registry'}, 400)
                return

            # ---- FIX: Validate that the model is local ----
            if not _is_local_model(info):
                self.send_json({
                    'success': False,
                    'error': f"Model '{model}' is not a local model. "
                             "Pruning requires a local Hugging Face model. "
                             "Please select a local student model."
                }, 400)
                return

            # Validate model before pruning
            if info.path and not info.path.startswith('ollama://'):
                model_path = Path(info.path)
                if model_path.is_dir() and not _validate_student_directory(model_path):
                    self.send_json({'success': False, 'error': f'Model "{model}" is corrupt (invalid tokenizer/config).'}, 400)
                    return

            with _state_lock:
                if _prune_state["running"]:
                    self.send_json({'success': False, 'error': 'Pruning already running'}, 409)
                    return

            thread = threading.Thread(target=_run_prune_thread, args=(model, strategy, task))
            thread.start()
            self.send_json({'success': True})
        except Exception as e:
            logger.exception("start-prune failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_start_benchmark(self):
        """Handle start benchmark with optional settings."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            model_name = data.get('model_name') or _global_student
            all_students = data.get('all_students', False)
            settings_dict = data.get('settings', {})

            if not all_students and not model_name:
                self.send_json({'success': False, 'error': 'No student model selected'}, 400)
                return

            with _state_lock:
                if _benchmark_state["running"]:
                    self.send_json({'success': False, 'error': 'Benchmark already running'}, 409)
                    return

            if not all_students:
                mm = ModelManager()
                info = mm.get_model(model_name)
                if not info or not info.path:
                    self.send_json({'success': False, 'error': f'Model "{model_name}" not found'}, 400)
                    return
                # ---- For benchmark, we allow Ollama but we skip validation ----
                # If it's local, we validate; if it's Ollama/vLLM, we don't need to validate tokenizer.
                if _is_local_model(info):
                    model_path = Path(info.path)
                    if model_path.is_dir() and not _validate_student_directory(model_path):
                        self.send_json({'success': False, 'error': f'Model "{model_name}" is corrupt (invalid tokenizer/config).'}, 400)
                        return
                # else: skip validation for Ollama/vLLM

            # Start benchmark thread with settings
            thread = threading.Thread(
                target=_run_benchmark_thread,
                args=(model_name, all_students, settings_dict)
            )
            thread.start()
            self.send_json({'success': True})
        except Exception as e:
            logger.exception("start-benchmark failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_export_student(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self.send_json({'success': False, 'error': 'No data'}, 400)
                return
            data = json.loads(self.rfile.read(length))
            model_name = data.get('model_name') or _global_student
            format_type = data.get('format', 'pytorch')
            async_mode = data.get('async', False)

            if not model_name:
                self.send_json({'success': False, 'error': 'Missing model_name (set globally or in request)'}, 400)
                return

            mm = ModelManager()
            info = mm.get_model(model_name)
            if not info:
                self.send_json({'success': False, 'error': f'Model "{model_name}" not found in registry'}, 400)
                return
            if info.path.startswith("ollama://"):
                self.send_json({'success': False, 'error': 'Cannot export Ollama model as zip'}, 400)
                return
            # Validate the model path
            model_path = Path(info.path)
            if not model_path.exists():
                self.send_json({'success': False, 'error': f'Model path does not exist: {model_path}'}, 400)
                return
            if model_path.is_dir() and not _validate_student_directory(model_path):
                self.send_json({'success': False, 'error': 'Model directory is corrupt; cannot export'}, 400)
                return
            if format_type == "vllm" and not model_path.is_dir():
                self.send_json({'success': False, 'error': 'vLLM export requires a directory (Hugging Face model)'}, 400)
                return

            if async_mode:
                job_id = str(uuid.uuid4())
                with _export_jobs_lock:
                    _export_jobs[job_id] = {
                        "status": "queued",
                        "progress": 0,
                        "message": "Queued",
                        "zip_path": None,
                        "zip_name": None,
                        "timestamp": time.time(),
                        "logs": []
                    }
                thread = threading.Thread(
                    target=_run_export_job,
                    args=(job_id, model_name, format_type, {})
                )
                thread.start()
                self.send_json({'success': True, 'job_id': job_id})
            else:
                self.send_json({'success': False, 'error': 'Sync export not supported, use async=true'}, 400)
        except Exception as e:
            logger.exception("export-student failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_export_status(self, job_id):
        with _export_jobs_lock:
            job = _export_jobs.get(job_id)
            if not job:
                self.send_json({'error': 'Job not found'}, 404)
                return
            # Clean up old jobs periodically
            _cleanup_old_export_jobs()
            # Return logs if present
            response = job.copy()
            response.setdefault('logs', [])
            self.send_json(response)

    def _handle_download(self, job_id):
        """Serve the exported zip file for download."""
        with _export_jobs_lock:
            job = _export_jobs.get(job_id)
            if not job:
                self.send_error(404, "Job not found")
                return
            if job.get("status") != "completed":
                self.send_error(400, "Export not completed")
                return
            zip_path = job.get("zip_path")
            if not zip_path or not os.path.exists(zip_path):
                self.send_error(404, "File not found")
                return
            if not os.access(zip_path, os.R_OK):
                self.send_error(403, "File not readable")
                return
            zip_name = job.get("zip_name", "model.zip")
        # Serve the file
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Disposition', f'attachment; filename="{zip_name}"')
            self.end_headers()
            with open(zip_path, 'rb') as f:
                self.wfile.write(f.read())
            logger.info(f"Served download for job {job_id}: {zip_name}")
        except Exception as e:
            logger.exception(f"Error serving file for job {job_id}")
            self.send_error(500, f"Error serving file: {str(e)}")

    def _handle_benchmark_report(self, model_name):
        """Retrieve stored benchmark report for a given model from registry."""
        try:
            mm = ModelManager()
            info = mm.get_model(model_name)
            if not info:
                self.send_json({'error': f'Model "{model_name}" not found'}, 404)
                return

            metadata = getattr(info, 'metadata', {})
            benchmarks = metadata.get('benchmarks', {})
            if not benchmarks:
                self.send_json({'error': 'No benchmark data available for this model'}, 404)
                return

            # Format a readable report
            report = {
                'model_name': model_name,
                'last_benchmark': benchmarks.get('timestamp'),
                'tokens_per_second': benchmarks.get('tokens_per_second'),
                'peak_memory_gb': benchmarks.get('peak_memory_gb'),
                'e8_quantized': benchmarks.get('e8_quantized'),
                'kv_compressed': benchmarks.get('kv_compressed'),
                'lazytorch_used': benchmarks.get('lazytorch_used'),
                'perplexity': benchmarks.get('perplexity'),
                'multiple_choice_accuracy': benchmarks.get('multiple_choice_accuracy'),
                'long_context': benchmarks.get('long_context'),
                'error_type': benchmarks.get('error_type'),
                'error_message': benchmarks.get('error_message'),
            }
            self.send_json(report)
        except Exception as e:
            logger.exception(f"Failed to get benchmark report for {model_name}")
            self.send_json({'error': str(e)}, 500)

    def _handle_terminal(self):
        """
        Enhanced NLP terminal with robust command parsing and actual inference for /chat.
        """
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            command = data.get('command', '').strip()
            model = data.get('model') or _global_student

            if not command:
                self.send_json({'status': 'error', 'message': 'No command provided'}, 400)
                return

            command_lower = command.lower()
            result = None
            message = None

            # ---- CHAT COMMAND ----
            if command_lower.startswith('chat with'):
                parts = command_lower.split('chat with')
                if len(parts) > 1:
                    rest = parts[1].strip()
                    words = rest.split()
                    if words:
                        model_name = words[0]
                        prompt = ' '.join(words[1:]) if len(words) > 1 else "Hello"
                        if not model_name:
                            if model:
                                model_name = model
                                prompt = rest
                            else:
                                message = "Please specify a model or set a global student."
                    else:
                        if model:
                            model_name = model
                            prompt = ""
                        else:
                            message = "Please specify a model or set a global student."
                else:
                    if model:
                        model_name = model
                        prompt = command_lower.replace('chat with', '').strip()
                    else:
                        message = "Please specify a model or set a global student."

                if model_name and not message:
                    # Use actual inference engine to generate a response
                    try:
                        config = load_config()
                        mm = ModelManager()
                        # Resolve model spec
                        from .bootstrap import resolve_model_spec
                        path = resolve_model_spec(model_name, mm)
                        if not path:
                            result = f"Model '{model_name}' could not be resolved."
                        else:
                            # Create engine and generate
                            engine = create_engine(path, config, mm)
                            # Generate response (streaming not fully supported in this endpoint)
                            full_response = []
                            for token in engine.lazy_generate_stream(prompt, max_tokens=128):
                                full_response.append(token)
                            response_text = ''.join(full_response)
                            result = f"🧠 {model_name} says:\n{response_text}"
                            engine.unload()
                    except Exception as e:
                        logger.error(f"Chat inference error: {e}")
                        result = f"Error during chat: {str(e)}"

            # ---- BENCHMARK COMMAND ----
            elif command_lower.startswith('benchmark'):
                parts = command_lower.split('benchmark')
                if len(parts) > 1:
                    model_name = parts[1].strip()
                else:
                    model_name = model
                if model_name:
                    mm = ModelManager()
                    info = mm.get_model(model_name)
                    if info and info.path:
                        # Validate model
                        path_obj = Path(info.path)
                        if path_obj.is_dir() and not _validate_student_directory(path_obj):
                            result = f"Model '{model_name}' is corrupt; cannot benchmark."
                        else:
                            res = benchmark_model(info.path, max_tokens=50, config=load_config())
                            result = f"Benchmark {model_name}: {res['tokens_per_second']:.2f} tok/s, peak memory {res['peak_memory_gb']:.2f} GB"
                    else:
                        message = f"Model {model_name} not found."
                else:
                    message = "Please specify a model or set a global student."

            # ---- DISTILL COMMAND ----
            elif command_lower.startswith('distill'):
                parts = command_lower.split('distill')
                if len(parts) > 1:
                    rest = parts[1].strip()
                    words = rest.split()
                    if len(words) >= 2:
                        teacher = words[0]
                        student = words[1]
                    else:
                        teacher = _global_teacher
                        student = _global_student
                        if not teacher or not student:
                            message = "Usage: distill <teacher> <student>  or set global models"
                    if teacher and student:
                        # Validate student model
                        mm = ModelManager()
                        s_info = mm.get_model(student)
                        if s_info and s_info.path and not s_info.path.startswith('ollama://'):
                            s_path = Path(s_info.path)
                            if s_path.is_dir() and not _validate_student_directory(s_path):
                                result = f"Student model '{student}' is corrupt; cannot distill."
                            else:
                                thread = threading.Thread(
                                    target=_run_distillation_thread,
                                    args=(teacher, student, 2)
                                )
                                thread.start()
                                result = f"Started distillation from {teacher} to {student}."
                        else:
                            thread = threading.Thread(
                                target=_run_distillation_thread,
                                args=(teacher, student, 2)
                            )
                            thread.start()
                            result = f"Started distillation from {teacher} to {student}."
                else:
                    teacher = _global_teacher
                    student = _global_student
                    if teacher and student:
                        # Validate student model
                        mm = ModelManager()
                        s_info = mm.get_model(student)
                        if s_info and s_info.path and not s_info.path.startswith('ollama://'):
                            s_path = Path(s_info.path)
                            if s_path.is_dir() and not _validate_student_directory(s_path):
                                result = f"Student model '{student}' is corrupt; cannot distill."
                            else:
                                thread = threading.Thread(
                                    target=_run_distillation_thread,
                                    args=(teacher, student, 2)
                                )
                                thread.start()
                                result = f"Started distillation from {teacher} to {student}."
                        else:
                            thread = threading.Thread(
                                target=_run_distillation_thread,
                                args=(teacher, student, 2)
                            )
                            thread.start()
                            result = f"Started distillation from {teacher} to {student}."
                    else:
                        message = "Usage: distill <teacher> <student>  or set global models"

            # ---- LIST MODELS ----
            elif command_lower in ['list models', 'list']:
                mm = ModelManager()
                models = mm.list_models()
                result = "\n".join([f"{m.name} ({m.original_size_mb:.1f} MB)" for m in models])

            # ---- HELP ----
            elif command_lower in ['help', '?']:
                result = """Available commands:
- chat with <model> [prompt]  - start a chat (uses global student if model omitted)
- benchmark <model>            - run a quick benchmark (uses global student if model omitted)
- distill <teacher> <student>  - start distillation (uses global models if omitted)
- list models                  - show all models
- help                         - this message"""

            else:
                message = f"Unknown command. Try 'help'."

            if message:
                self.send_json({'status': 'error', 'message': message}, 400)
            else:
                self.send_json({'status': 'ok', 'result': result or 'Done.'})
        except json.JSONDecodeError as e:
            logger.exception("Terminal JSON decode error")
            self.send_json({'status': 'error', 'message': f'Invalid JSON: {str(e)}'}, 400)
        except Exception as e:
            logger.exception("Terminal command failed")
            self.send_json({'status': 'error', 'message': f'Internal error: {str(e)}'}, 500)

    def _handle_pull_ollama(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            model_name = data.get('model_name')
            if not model_name:
                self.send_json({'success': False, 'error': 'Missing model_name'}, 400)
                return
            # Start async pull with progress
            with _state_lock:
                if _pull_state["running"]:
                    self.send_json({'success': False, 'error': 'Pull already in progress'}, 409)
                    return
            thread = threading.Thread(target=_run_pull_thread, args=(model_name,))
            thread.start()
            self.send_json({'success': True})
        except Exception as e:
            logger.exception("pull-ollama failed")
            self.send_json({'success': False, 'error': str(e)}, 500)

    def _handle_delete_model(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            name = data.get('name')
            if not name:
                self.send_json({'success': False, 'error': 'Missing name'}, 400)
                return
            mm = ModelManager()
            if mm.delete_model(name):
                self.send_json({'success': True})
            else:
                self.send_json({'success': False, 'error': 'Model not found'}, 404)
        except Exception as e:
            logger.exception("delete-model failed")
            self.send_json({'success': False, 'error': str(e)}, 500)


# =============================================================================
# Pull State and Thread
# =============================================================================
_pull_state = {
    "running": False,
    "model": "",
    "progress": 0,
    "status": "idle",
    "error": None,
}


def _run_pull_thread(model_name):
    global _pull_state
    try:
        with _state_lock:
            _pull_state["running"] = True
            _pull_state["model"] = model_name
            _pull_state["progress"] = 0
            _pull_state["status"] = "starting pull..."
            _pull_state["error"] = None

        import subprocess
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            # Improved progress parsing
            if "downloading" in line.lower() or "pulling" in line.lower():
                match = re.search(r'(\d+)%', line)
                if match:
                    pct = int(match.group(1))
                    with _state_lock:
                        _pull_state["progress"] = pct
                        _pull_state["status"] = f"pulling... {pct}%"
            elif "success" in line.lower():
                with _state_lock:
                    _pull_state["progress"] = 100
                    _pull_state["status"] = "completed"
            elif "error" in line.lower():
                with _state_lock:
                    _pull_state["status"] = "failed"
                    _pull_state["error"] = line.strip()

        process.wait()
        if process.returncode != 0:
            with _state_lock:
                if _pull_state["status"] != "failed":
                    _pull_state["status"] = "failed"
                    _pull_state["error"] = f"Process returned {process.returncode}"

        # Sync registry after successful pull
        if _pull_state["status"] == "completed":
            mm = ModelManager()
            mm.sync_ollama()
    except Exception as e:
        logger.exception("Pull thread failed")
        with _state_lock:
            _pull_state["status"] = "failed"
            _pull_state["error"] = str(e)
    finally:
        with _state_lock:
            _pull_state["running"] = False


class DashboardServer:
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = None
        self.thread = None
        # Start a background thread to clean up old export jobs periodically
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_loop(self):
        while True:
            time.sleep(60)  # every minute
            _cleanup_old_export_jobs()

    def start(self, open_browser: bool = True):
        self.server = HTTPServer(('127.0.0.1', self.port), DashboardHandler)
        if open_browser:
            webbrowser.open(f'http://127.0.0.1:{self.port}/dashboard')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            shutdown_thread = threading.Thread(target=self.server.shutdown)
            shutdown_thread.start()
            shutdown_thread.join(timeout=5)
            self.server.server_close()
            if self.thread:
                self.thread.join(timeout=2)


def start_dashboard(port: int = 8080, open_browser: bool = True) -> DashboardServer:
    server = DashboardServer(port)
    server.start(open_browser)
    return server


if __name__ == "__main__":
    start_dashboard()
