#!/usr/bin/env python3
"""Main entry point with CLI commands, rotating logs, and TUI fallback.
   Integrates LazyTorch for memory-mapped, on-demand layer loading.

   Fixed (2026-07-13): Improved logging, reduced repeated warnings,
   added REAP Pipeline Checklist hook, fixed --force-student flag,
   restored full _add_hf_metadata implementation.
   Fixed (2026-07-14): Check both global --force-student and subparser --force flags.

   NEW (2026-07-14): Integrated REAP pipeline checklist with file-based
   persistence. All pipeline stages (distill, prune, finetune, eval) now
   update the checklist file in the model directory. The `run_reap_pipeline_checklist`
   function reads both registry flags and the file-based checklist for
   maximum compatibility.

   NEW (2026-07-16): Enhanced `benchmark-students` command with full benchmark
   settings support. Added CLI arguments for prompt, max_tokens, perplexity,
   multiple-choice, long-context, context lengths, number of trials, and registry
   storage. Uses the new BenchmarkSettings dataclass and displays a formatted
   summary table via format_benchmark_summary.

   FIX (2026-07-16): Changed `--store-registry` to `--no-store-registry` so users
   can explicitly disable registry storage. Added JSON file loading for MC questions.

   ENHANCEMENT (2026-07-17): In `cmd_create_student`, added validation to check
   if the base model exists locally or on Hugging Face Hub before attempting
   download. Uses `ModelManager.is_valid_hf_model()` to give a clear error
   message for invalid model IDs, improving user experience.

   REMOVED (2026-07-17): Removed all HEPA and HydraHead related code, including
   HEPA commands, hybrid attention flag, and endless finetune subcommand.

   ENHANCEMENT (2026-07-16): Added `recover` subcommand for one‑click prune + recovery.
   Added `--with-recovery` flag to `create-student` to automatically recover new students.
   Added `--prune-ratio` argument to endless subcommands (prune and auto) to allow
   overriding the default 15% ratio.

   NEW (2026-07-16): Recovery pipeline uses QLoRA (4‑bit + LoRA) for efficient
   fine‑tuning after pruning, with 3 passes by default. The recovered model is
   registered with a `_recovered` suffix.

   FIX (2026-07-16): In `cmd_endless_auto`, instead of passing non-existent
   `hyperparam_overrides` to `run_endless_auto`, we temporarily set the config's
   `reap_prune_ratio` when `--prune-ratio` is provided, ensuring compatibility
   without modifying `endless_rl.py`.

   FIX (2026-07-16): In `run_recovery`, added validation of the teacher's tokenizer
   to catch corruption early, and improved error handling.

   SIMPLIFIED (2026-07-16): In `cmd_create_student`, removed redundant `hasattr` checks
   for `args.prune_ratio` and `args.passes` as they are always defined.

   ENHANCED (2026-07-16): Added explicit `gc.collect()` and CUDA cache clearing
   in the recovery pipeline after pruning and after distillation to reduce memory
   pressure on low‑RAM systems.

   FIX (2026-07-16): Restored working `run_recovery` implementation with correct
   ordering: export pruned model before deleting model/pruner, ensuring memory
   cleanup without losing the pruned weights.

   FIX (2026-07-16): Added missing import of `Config` to resolve NameError in
   `run_recovery` function signature.

   NEW (2026-07-17): Added `health-check` command to check all models for issues.

   FIX (2026-07-17): Corrected import of CHECKPOINTS_DIR from config instead of utils.
"""

# ----------------------------------------------------------------------
# Ensure the package is importable when running from source
# ----------------------------------------------------------------------
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_package_root = _script_dir.parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

# ----------------------------------------------------------------------
# Imports – all absolute from lazy_llama (now guaranteed to be found)
# ----------------------------------------------------------------------
import argparse
import logging
import time
import shutil
import json
import gc
import numpy as np
import torch
import re  # for health-check command
from logging.handlers import RotatingFileHandler
from typing import Optional, Union, List

from rich.logging import RichHandler
from rich.prompt import Confirm

from lazy_llama.config import (
    load_config,
    LOGS_DIR,
    auto_optimize_config,
    recommend_enhancements,
    save_config,
    Config,
    CHECKPOINTS_DIR,   # <-- FIX: imported from config, not utils
)
from lazy_llama.lazy_tui import LazyTUI
from lazy_llama.lazy_model_manager import ModelManager
from lazy_llama.benchmark import (
    benchmark_student_models,
    BenchmarkSettings,
    format_benchmark_summary
)
from lazy_llama.utils import (
    export_to_ollama, get_lazytorch_model_size, is_lazytorch_model,
    _validate_tokenizer_deep, copy_tokenizer_files, check_ollama_model,
    update_stage_status, log_stage_summary, read_checklist, write_checklist,
    log_operation_result, clear_cuda_memory,
    # CHECKPOINTS_DIR removed from here - now imported from config
)
from lazy_llama.lazy_infer import create_engine, is_valid_gguf
from lazy_llama.lazy_prune import Pruner, get_task_prompts
from lazy_llama.lazy_distill import LazyDistillationEngine

# ----------------------------------------------------------------------
# Centralised logging setup (once, with rotating file and rich console)
# ----------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    log_file = LOGS_DIR / "lazy_llama.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=10_485_760, backupCount=5)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    console_handler = RichHandler(rich_tracebacks=True)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[console_handler, file_handler]
    )

# Call early so logger is available
setup_logging()
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# REAP Pipeline Checklist (with file-based persistence)
# ----------------------------------------------------------------------
def run_reap_pipeline_checklist(model_name: str, manager: ModelManager) -> bool:
    """
    Verify whether a given student model has passed the full REAP pipeline.
    Combines registry flags and the file‑based checklist for maximum compatibility.
    Returns True if all steps are complete, False otherwise.
    Logs a detailed checklist to INFO.
    """
    info = manager.get_model(model_name)
    if not info:
        logger.warning(f"Model '{model_name}' not found in registry.")
        return False

    # Read file-based checklist (if exists)
    file_checklist = read_checklist(model_name)

    # Build checklist from registry flags and file
    checklist = {
        "distilled": False,
        "pruned": False,
        "lazytorch_exported": False,
        "benchmark_nlp_passed": False
    }

    # 1. Distillation: check name, registry flag, or file checklist
    if "_distilled" in model_name:
        checklist["distilled"] = True
    if file_checklist.get("stages", {}).get("distillation", {}).get("completed", False):
        checklist["distilled"] = True

    # 2. Pruning: check name, registry flag, or file checklist
    if model_name.endswith("_pruned"):
        checklist["pruned"] = True
    if getattr(info, 'pruning_applied', False):
        checklist["pruned"] = True
    if file_checklist.get("stages", {}).get("pruning", {}).get("completed", False):
        checklist["pruned"] = True

    # 3. LazyTorch export: registry flag or file checklist
    if getattr(info, 'lazytorch_format', False):
        checklist["lazytorch_exported"] = True

    # 4. Benchmark NLP passed: accuracy_score > 0.5 or file checklist eval score
    if hasattr(info, 'accuracy_score') and info.accuracy_score is not None and info.accuracy_score > 0.5:
        checklist["benchmark_nlp_passed"] = True
    eval_stage = file_checklist.get("stages", {}).get("evaluation", {})
    if eval_stage.get("completed", False) and eval_stage.get("score", 0) > 0.5:
        checklist["benchmark_nlp_passed"] = True

    logger.info("=== REAP PIPELINE CHECKLIST ===")
    for step, done in checklist.items():
        status = "✅" if done else "❌"
        logger.info(f"  {step}: {status}")
    passed = all(checklist.values())
    if passed:
        logger.info(f"✅ Model '{model_name}' has passed the full REAP pipeline.")
    else:
        logger.info(f"⚠️ Model '{model_name}' has not completed all REAP steps.")
    return passed


# ----------------------------------------------------------------------
# Helper: resolve model spec (uses logger)
# ----------------------------------------------------------------------
def resolve_model_spec(model_spec: str, model_manager: ModelManager) -> Optional[Path]:
    """
    Resolve a model specification (name from registry, path, ollama://, or vllm://)
    to a Path object that can be passed to create_engine().
    """
    if model_spec.startswith("ollama://"):
        return Path(model_spec)
    if model_spec.startswith("vllm://"):
        return Path(model_spec)

    path = Path(model_spec)
    if path.exists():
        if path.is_dir():
            try:
                if _validate_tokenizer_deep(path):
                    return path
                else:
                    logger.warning(f"Directory {path} is not a valid Hugging Face model (tokenizer corrupt).")
                    return None
            except Exception as e:
                logger.warning(f"Error validating tokenizer for {path}: {e}")
                return None
        if path.is_file() and path.suffix == ".gguf":
            return path
        logger.warning(f"File {path} is not a .gguf file.")
        return None

    info = model_manager.get_model(model_spec)
    if info and info.path:
        p = Path(info.path)
        if p.exists() or p.as_posix().startswith("ollama://") or p.as_posix().startswith("vllm://"):
            if p.is_dir() and not _validate_tokenizer_deep(p):
                logger.warning(f"Model '{model_spec}' in registry points to directory with corrupt tokenizer.")
                return None
            return p
        else:
            logger.warning(f"Model '{model_spec}' in registry points to missing path '{p}'.")

    if check_ollama_model(model_spec):
        logger.info(f"Model '{model_spec}' found in Ollama; using ollama://{model_spec}")
        return Path(f"ollama://{model_spec}")

    if '/' in model_spec:
        logger.info(f"Model '{model_spec}' contains a slash; assuming vLLM model.")
        return Path(f"vllm://{model_spec}")

    logger.warning(f"Could not resolve model specification: '{model_spec}'")
    return None


# ----------------------------------------------------------------------
# Helper: Ensure default student models are installed (with improved logging)
# ----------------------------------------------------------------------
def ensure_default_students(force: bool = False) -> None:
    config = load_config()
    if getattr(config, 'default_students_installed', False) and not force:
        return

    manager = ModelManager(config)
    default_models = [
        "distilgpt2",
        "gpt2",
        "facebook/opt-125m"
    ]

    logger.info("Checking for default student models...")

    # Quick network reachability check
    try:
        import requests
        requests.get("https://huggingface.co", timeout=3)
    except Exception:
        logger.warning("Could not reach Hugging Face; skipping default student download (offline).")
        config.default_students_installed = True
        save_config(config)
        return

    for model_name in default_models:
        if not manager.model_exists(model_name):
            logger.info(f"Downloading default student '{model_name}'...")
            try:
                manager.download_from_hf(model_name)
                logger.info(f"Downloaded and registered '{model_name}'.")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
        else:
            logger.info(f"'{model_name}' already exists.")

    config.default_students_installed = True
    save_config(config)
    logger.info("Default student models ready (some may have failed; check logs).")


# ----------------------------------------------------------------------
# Helper: Add Hugging Face metadata (README.md, .gitattributes)
# ----------------------------------------------------------------------
def _add_hf_metadata(
    dest_dir: Path,
    model_name: str,
    model_info,
    config,
    format_type: str,
    lazytorch_present: bool
) -> None:
    """
    Write README.md and .gitattributes to dest_dir to make it a complete
    Hugging Face repository.
    """
    # Gather metadata
    description = f"Lazy Llama model: {model_name}"
    if model_info:
        if hasattr(model_info, 'task_specialization') and model_info.task_specialization:
            description += f" (specialized for {model_info.task_specialization})"
        if hasattr(model_info, 'pruning_applied') and model_info.pruning_applied:
            description += " – pruned"
        if hasattr(model_info, 'distillation_date') and model_info.distillation_date:
            description += f" (distilled on {model_info.distillation_date})"
    else:
        description += " (exported from Lazy Llama)"

    # License: look for LICENSE file in source, else default to MIT
    license_text = "MIT"
    # Try to find a license file in the dest_dir (if copied) or in source
    license_files = list(dest_dir.glob("LICENSE*")) + list(dest_dir.glob("license*"))
    if license_files:
        try:
            license_text = license_files[0].read_text(encoding='utf-8', errors='ignore').strip()
        except:
            pass

    # Tags
    tags = ["lazy-llama", "pytorch", "transformers"]
    if model_info:
        if hasattr(model_info, 'task_specialization') and model_info.task_specialization:
            tags.append(model_info.task_specialization)
    if format_type:
        tags.append(format_type)
    if lazytorch_present:
        tags.append("lazytorch")
    tags_str = ", ".join(tags)

    # Evaluation metrics
    eval_metrics = ""
    if model_info and hasattr(model_info, 'accuracy_score') and model_info.accuracy_score is not None:
        eval_metrics += f"- Accuracy: {model_info.accuracy_score:.4f}\n"
    if model_info and hasattr(model_info, 'perplexity') and model_info.perplexity is not None:
        eval_metrics += f"- Perplexity: {model_info.perplexity:.4f}\n"
    if not eval_metrics:
        eval_metrics = "No evaluation metrics available."

    # Usage example
    if lazytorch_present:
        usage = f"""```python
from lazy_llama.lazytorch_core import load_lazytorch_model
model = load_lazytorch_model("{dest_dir}")
# model is a LazyModule; use it like a regular PyTorch model
```"""
    else:
        usage = f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{dest_dir}")
tokenizer = AutoTokenizer.from_pretrained("{dest_dir}")
```"""

    # Citation
    citation = """@misc{lazy_llama2025,
  title = {Lazy Llama: Low-End Inference Engine},
  author = {Lazy Llama Team},
  year = {2025},
  url = {https://github.com/lazy-llama/lazy-llama}
}"""

    # Build README content using a template
    readme_template = """# {model_name}

{description}

## License
{license_text}

## Tags
{tags_str}

## Evaluation Metrics
{eval_metrics}

## How to Use
{usage}

## Citation

If you use this model, please cite:

{citation}
"""

    readme_content = readme_template.format(
        model_name=model_name,
        description=description,
        license_text=license_text,
        tags_str=tags_str,
        eval_metrics=eval_metrics,
        usage=usage,
        citation=citation
    )
    (dest_dir / "README.md").write_text(readme_content, encoding="utf-8")

    # Write .gitattributes
    gitattributes = """*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.gguf filter=lfs diff=lfs merge=lfs -text
"""
    (dest_dir / ".gitattributes").write_text(gitattributes, encoding="utf-8")


# ----------------------------------------------------------------------
# Recovery Pipeline (one-shot prune + distillation) - RESTORED & CORRECTED
# ----------------------------------------------------------------------
def run_recovery(
    model_name: str,
    teacher: Optional[str] = None,
    prune_ratio: Optional[float] = None,
    passes: int = 3,
    output_name: Optional[str] = None,
    manager: Optional[ModelManager] = None,
    config: Optional[Config] = None,  # <-- Config is now imported
) -> bool:
    """
    Run a one‑shot recovery pipeline: prune the given model (15% by default),
    then distill it (self‑distillation if no teacher specified) with QLoRA
    to recover performance.

    The recovered model is registered with a `_recovered` suffix.

    Args:
        model_name: Name of the model to recover.
        teacher: Optional teacher model for distillation. If None, self‑distillation is used.
        prune_ratio: Fraction of weights to prune (default from config.reap_prune_ratio).
        passes: Number of distillation passes.
        output_name: Optional name for the recovered model (default: model_name + "_recovered").
        manager: ModelManager instance (created if None).
        config: Config instance (loaded if None).

    Returns:
        True if successful, False otherwise.
    """
    if config is None:
        config = load_config()
    if manager is None:
        manager = ModelManager(config)

    if prune_ratio is None:
        prune_ratio = config.reap_prune_ratio

    # Ensure prune_ratio is valid
    if not 0.0 < prune_ratio < 1.0:
        logger.error(f"Invalid prune_ratio: {prune_ratio}. Must be between 0 and 1.")
        return False

    # Get model info
    info = manager.get_model(model_name)
    if not info or not info.path:
        logger.error(f"Model '{model_name}' not found or missing path.")
        return False

    # Validate it's a local model (not Ollama/vLLM)
    if info.path.startswith("ollama://") or info.path.startswith("vllm://"):
        logger.error("Recovery only works with local Hugging Face models.")
        return False

    # Validate tokenizer of the model to recover
    path_obj = Path(info.path)
    if path_obj.is_dir() and not _validate_tokenizer_deep(path_obj):
        logger.error(f"Model '{model_name}' has a corrupt tokenizer. Cannot recover.")
        return False

    # Validate teacher tokenizer if provided and different from model
    if teacher and teacher != model_name:
        teacher_info = manager.get_model(teacher)
        if not teacher_info or not teacher_info.path:
            logger.error(f"Teacher model '{teacher}' not found or missing path.")
            return False
        teacher_path = Path(teacher_info.path)
        if teacher_path.is_dir() and not _validate_tokenizer_deep(teacher_path):
            logger.error(f"Teacher model '{teacher}' has a corrupt tokenizer. Cannot recover.")
            return False

    # Determine output name
    if output_name is None:
        if model_name.endswith("_pruned") or model_name.endswith("_distilled"):
            base = model_name.rsplit("_", 1)[0]
            output_name = f"{base}_recovered"
        else:
            output_name = f"{model_name}_recovered"

    # Ensure we don't overwrite an existing model
    if manager.model_exists(output_name):
        if not Confirm.ask(f"Model '{output_name}' already exists. Overwrite?"):
            logger.info("Recovery cancelled.")
            return False
        manager.delete_model(output_name)

    logger.info(f"Starting recovery for '{model_name}' -> '{output_name}' with prune_ratio={prune_ratio}, passes={passes}")

    # Temporary names and paths
    temp_pruned_path = manager.models_dir / f"{output_name}_temp_pruned"
    temp_student_name = None  # initialize for safe cleanup

    try:
        # 1. Load the model for pruning
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading model for pruning...")
        model = AutoModelForCausalLM.from_pretrained(str(path_obj), low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(str(path_obj))

        # 2. Create pruner and apply pruning (modifies model in-place)
        pruner = Pruner(model, config, original_path=path_obj, tokenizer=tokenizer)
        logger.info(f"Applying magnitude pruning with ratio {prune_ratio}...")
        pruner.magnitude_prune(threshold=prune_ratio, iterative_steps=1)  # single-step prune

        # 3. Export pruned model to temporary directory (BEFORE deleting model/pruner)
        logger.info(f"Exporting pruned model to {temp_pruned_path}...")
        pruner.export_pruned(temp_pruned_path, overwrite=True, export_to_lazytorch=False, register=False)
        logger.info(f"Pruned model saved to {temp_pruned_path}")

        # 4. Clean up memory
        del model, pruner
        gc.collect()
        clear_cuda_memory()
        logger.debug("Memory cleaned after pruning.")

        # 5. Register the pruned model as a temporary student
        temp_student_name = f"{output_name}_temp_student"
        temp_student_info = manager._create_model_info(
            name=temp_student_name,
            path=temp_pruned_path,
            size_mb=sum(f.stat().st_size for f in temp_pruned_path.glob("*") if f.is_file()) / (1024 * 1024),
        )
        with manager._lock:
            manager.registry[temp_student_name] = temp_student_info
            manager._save_registry()

        # Determine teacher: if not provided, use the original model itself (self-distillation)
        teacher_name = teacher if teacher else model_name
        logger.info(f"Distilling from teacher '{teacher_name}' to pruned model...")

        # Ensure the teacher exists in registry (if not self)
        if teacher_name != model_name:
            teacher_info = manager.get_model(teacher_name)
            if not teacher_info:
                logger.error(f"Teacher model '{teacher_name}' not found.")
                manager.delete_model(temp_student_name)
                return False

        # 6. Run distillation with QLoRA for recovery
        distill_engine = LazyDistillationEngine(config)
        distill_engine.use_qlora = True
        distill_engine.qlora_r = 16  # higher rank for recovery

        val_texts = config.validation_prompts
        if not val_texts:
            logger.warning("No validation prompts in config; using default ones.")
            val_texts = [
                "What is Python?",
                "Explain recursion.",
                "Write a loop summing 1 to 10.",
                "What is the capital of France?",
                "Define machine learning.",
            ]

        logger.info(f"Running distillation with {passes} passes...")
        distill_engine.run_distillation(
            teacher_name,
            temp_student_name,
            texts=val_texts,
            passes=passes,
            resume=False,
            use_qlora=True,
            qlora_r=16,
        )

        # 7. After distillation, the distilled model is saved with "_distilled" suffix
        distilled_name = f"{temp_student_name}_distilled"
        distilled_info = manager.get_model(distilled_name)
        if not distilled_info:
            logger.error("Distillation produced no registered model.")
            manager.delete_model(temp_student_name)
            return False

        # 8. Rename the distilled model to the desired output name
        distilled_path = Path(distilled_info.path)
        if not distilled_path.exists():
            logger.error(f"Distilled model path {distilled_path} does not exist.")
            return False

        final_path = manager.models_dir / output_name
        if final_path.exists():
            shutil.rmtree(final_path, ignore_errors=True)
        distilled_path.rename(final_path)

        # 9. Update registry: remove temp student and distilled, add recovered model
        with manager._lock:
            if temp_student_name in manager.registry:
                del manager.registry[temp_student_name]
            if distilled_name in manager.registry:
                del manager.registry[distilled_name]
            new_info = manager._create_model_info(
                name=output_name,
                path=final_path,
                size_mb=sum(f.stat().st_size for f in final_path.glob("*") if f.is_file()) / (1024 * 1024),
                pruning_applied=True,
                task_specialization="recovered",
            )
            manager.registry[output_name] = new_info
            manager._save_registry()

        # 10. Clean up temp pruned directory
        shutil.rmtree(temp_pruned_path, ignore_errors=True)

        # Log success
        log_operation_result(
            model_name=output_name,
            operation='recover',
            success=True,
            details={
                'source_model': model_name,
                'teacher': teacher_name,
                'prune_ratio': prune_ratio,
                'passes': passes,
            },
            manager=manager
        )
        logger.info(f"Recovery successful! New model: '{output_name}'")
        return True

    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        # Clean up any temp entries
        if temp_student_name is not None:
            try:
                manager.delete_model(temp_student_name)
            except:
                pass
        try:
            shutil.rmtree(temp_pruned_path, ignore_errors=True)
        except:
            pass
        return False


# ----------------------------------------------------------------------
# CLI command implementations (with logger instead of print where appropriate)
# ----------------------------------------------------------------------
def cmd_chat(args):
    config = load_config()
    manager = ModelManager(config)
    model_name = None
    if args.student:
        model_name = args.student
    elif args.model:
        model_name = args.model
    elif args.teacher:
        model_name = args.teacher
    else:
        global_file = Path.home() / ".lazy_llama/global_state.json"
        if global_file.exists():
            try:
                with open(global_file) as f:
                    data = json.load(f)
                    model_name = data.get("student") or data.get("teacher")
            except:
                pass
        if not model_name:
            logger.error("No model specified. Use --student, --model, --teacher, or set a global model.")
            return
    path = resolve_model_spec(model_name, manager)
    if not path:
        logger.error(f"Could not resolve model: {model_name}")
        return
    engine = create_engine(path, config, manager)
    tui = LazyTUI()
    tui.current_model = engine
    tui._chat()


def cmd_download(args):
    config = load_config()
    manager = ModelManager(config)
    if args.source == "ollama":
        logger.info(f"Pulling Ollama model: {args.name}")
        manager.download_from_ollama(args.name)
    else:
        convert = args.auto_convert_lazytorch or config.auto_convert_to_lazytorch
        logger.info(f"Downloading HF model: {args.name} (convert to LazyTorch: {convert})")
        manager.download_from_hf(args.name, gguf_file=args.gguf, convert_to_lazytorch_after=convert)
    manager.reload_registry(sync_ollama=True)
    logger.info("Done.")


def cmd_create_student(args):
    config = load_config()
    manager = ModelManager(config)

    # ---- Validate base model existence before proceeding ----
    base_path = Path(args.base)
    base_info = manager.get_model(args.base)

    if not base_info and not base_path.exists():
        # Not local; check if it's a valid Hugging Face model ID
        if not manager.is_valid_hf_model(args.base):
            logger.error(f"Base model '{args.base}' does not exist locally and is not a valid Hugging Face model ID.")
            logger.error("Please check the model ID or ensure the model is available locally.")
            return
        # If it is valid, we'll download it later (auto_download will handle it)
    elif base_info and base_info.path:
        # Local model: validate tokenizer
        base_path = Path(base_info.path)
        if base_path.is_dir() and not _validate_tokenizer_deep(base_path):
            logger.error(f"Base model '{args.base}' has a corrupt tokenizer. Please re-download or repair the model.")
            logger.error(f"You can delete it using: python bootstrap.py remove --model {args.base}")
            logger.error("Then re-download from Hugging Face.")
            return

    # If base not found and auto-download is off, we'll handle later
    if not base_info and not base_path.exists() and not config.auto_download_missing_models:
        logger.error(f"Base model '{args.base}' not found. Use --auto-download or download it first.")
        return

    # Proceed with creation (may download if needed)
    success = manager.create_student(
        args.base,
        args.student_name,
        auto_download=config.auto_download_missing_models,
        use_hybrid_heads=False,  # hybrid heads removed
        use_lazytorch=args.convert_student_lazytorch or config.auto_convert_student_to_lazytorch
    )
    if not success:
        logger.error(f"Failed to create student '{args.student_name}'.")
        return

    logger.info(f"Student model '{args.student_name}' created successfully.")
    # Run REAP checklist on the new student (may not be complete yet)
    run_reap_pipeline_checklist(args.student_name, manager)

    # ---- NEW: if --with-recovery, run recovery pipeline ----
    if args.with_recovery:
        logger.info("Running recovery on the new student (prune + distill)...")
        # Use the base model as teacher
        teacher = args.base
        # Ensure teacher exists in registry (it should)
        if not manager.model_exists(teacher):
            logger.error(f"Teacher '{teacher}' not found. Cannot run recovery.")
            return
        # Run recovery using the provided prune_ratio and passes (they exist)
        success = run_recovery(
            model_name=args.student_name,
            teacher=teacher,
            prune_ratio=args.prune_ratio,
            passes=args.passes,
            manager=manager,
            config=config,
        )
        if success:
            logger.info(f"Recovery completed for '{args.student_name}'. New model registered.")
        else:
            logger.error(f"Recovery failed for '{args.student_name}'.")


def cmd_export_zip(args):
    config = load_config()
    manager = ModelManager(config)
    info = manager.get_model(args.model)
    if not info or not info.path:
        logger.error(f"Model '{args.model}' not found.")
        return
    src_path = Path(info.path)
    if not src_path.exists():
        logger.error(f"Model path does not exist: {src_path}")
        return
    if src_path.is_dir() and not _validate_tokenizer_deep(src_path):
        logger.error(f"Model directory {src_path} has corrupt tokenizer; cannot export.")
        return
    dest_dir = Path.home() / ".lazy_llama/exports"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = dest_dir / zip_name

    logger.info(f"Exporting {args.model} to {zip_path} ...")
    if args.zip_only:
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', src_path)
        logger.info("Zip created (without Hugging Face metadata).")
        return

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        shutil.copytree(src_path, tmp_path / args.model, symlinks=False, ignore_dangling_symlinks=True)
        model_dir = tmp_path / args.model
        lazytorch_present = is_lazytorch_model(src_path)
        _add_hf_metadata(model_dir, args.model, info, config, "pytorch", lazytorch_present)
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', model_dir)
    logger.info(f"Export complete: {zip_path}")


def cmd_import_zip(args):
    config = load_config()
    manager = ModelManager(config)
    zip_path = Path(args.zip)
    if not zip_path.exists():
        logger.error(f"Zip file not found: {zip_path}")
        return
    dest_dir = manager.models_dir / args.name
    if dest_dir.exists():
        if not Confirm.ask(f"Model directory {dest_dir} already exists. Overwrite?"):
            return
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(dest_dir))
    if not manager.validate_model_directory(dest_dir):
        logger.error("Imported model is invalid (missing config/tokenizer/weights). Removing.")
        shutil.rmtree(dest_dir)
        return
    size_mb = sum(f.stat().st_size for f in dest_dir.glob("*") if f.is_file()) / (1024 * 1024)
    manager.registry[args.name] = manager._create_model_info(args.name, dest_dir, size_mb)
    manager._save_registry()
    manager.reload_registry()
    logger.info(f"Model '{args.name}' imported successfully.")


def cmd_rename(args):
    config = load_config()
    manager = ModelManager(config)
    if manager.rename_model(args.old, args.new):
        logger.info(f"Renamed '{args.old}' to '{args.new}'.")
    else:
        logger.error("Rename failed.")


def cmd_remove(args):
    config = load_config()
    manager = ModelManager(config)
    if manager.delete_model(args.model):
        logger.info(f"Deleted '{args.model}'.")
    else:
        logger.error("Delete failed.")


def cmd_benchmark_students(args):
    """Run benchmarks on all student models with enhanced settings."""
    config = load_config()

    # Build BenchmarkSettings from command-line arguments
    settings = BenchmarkSettings(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        run_perplexity=args.perplexity,
        val_texts=args.val_texts if args.val_texts else None,
        run_multiple_choice=args.mc,
        mc_questions=args.mc_questions if args.mc_questions else None,
        run_long_context=args.long_context,
        context_lengths=args.context_lengths if args.context_lengths else None,
        num_trials=args.num_trials,
        store_in_registry=not args.no_store_registry,
        long_context_max_tokens=args.lc_max_tokens if hasattr(args, 'lc_max_tokens') else 20,
    )

    # ---- FIX: Load MC questions from JSON file if provided ----
    if args.mc_questions and Path(args.mc_questions).exists():
        try:
            with open(args.mc_questions, 'r') as f:
                loaded_questions = json.load(f)
            if isinstance(loaded_questions, list):
                settings.mc_questions = loaded_questions
                logger.info(f"Loaded {len(settings.mc_questions)} MC questions from {args.mc_questions}")
            else:
                logger.error(f"MC questions file must contain a JSON list of questions; got {type(loaded_questions)}")
                settings.run_multiple_choice = False
        except Exception as e:
            logger.error(f"Failed to load MC questions from {args.mc_questions}: {e}")
            settings.run_multiple_choice = False

    # If perplexity is enabled but no val_texts provided, use default validation prompts
    if settings.run_perplexity and not settings.val_texts:
        settings.val_texts = config.validation_prompts
        logger.info(f"Using default validation prompts for perplexity: {len(settings.val_texts)} texts")

    # If MC is enabled but no mc_questions, provide a small default set
    if settings.run_multiple_choice and not settings.mc_questions:
        settings.mc_questions = [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": 1
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
                "answer": 1
            },
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1
            }
        ]
        logger.info("Using default multiple-choice questions.")

    logger.info(f"Starting benchmark with settings: prompt='{settings.prompt}', max_tokens={settings.max_tokens}, "
                f"perplexity={settings.run_perplexity}, MC={settings.run_multiple_choice}, "
                f"long-context={settings.run_long_context}")

    try:
        results = benchmark_student_models(settings, config=config)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return

    # Print summary
    print("\n" + format_benchmark_summary(results['results']))

    # Also log the summary
    summary = results['summary']
    logger.info(f"Benchmark complete: {summary['succeeded']}/{summary['total']} succeeded, "
                f"avg TPS: {summary['avg_tps']:.2f}, avg peak mem: {summary['avg_peak_mem_gb']:.2f} GB")


def cmd_recover(args):
    """One‑click recovery: prune + distill."""
    config = load_config()
    manager = ModelManager(config)
    success = run_recovery(
        model_name=args.model,
        teacher=args.teacher,
        prune_ratio=args.prune_ratio,
        passes=args.passes,
        output_name=args.output_name,
        manager=manager,
        config=config,
    )
    if success:
        logger.info("Recovery completed successfully.")
    else:
        logger.error("Recovery failed.")


def cmd_endless_distill(args):
    try:
        from lazy_llama.endless_rl import run_endless_distillation
        teacher = args.teacher
        student = args.student
        run_endless_distillation(teacher, student, args.passes, args.cycles, args.sleep)
        update_stage_status(student, "distillation", True, {"teacher": teacher, "passes": args.passes})
        log_stage_summary("Distillation", student, True, f"teacher={teacher}")
    except Exception as e:
        student = args.student
        update_stage_status(student, "distillation", False, {"error": str(e)})
        log_stage_summary("Distillation", student, False, f"error={e}")
        logger.error(f"Endless distillation failed: {e}")


def cmd_endless_prune(args):
    try:
        from lazy_llama.endless_rl import run_endless_prune
        model = args.model
        strategies = args.strategies
        cycles = args.cycles
        sleep = args.sleep
        hyperparams = {}
        if args.prune_ratio is not None:
            hyperparams['threshold'] = args.prune_ratio
        run_endless_prune(model, strategies, cycles, sleep, hyperparams=hyperparams)
        update_stage_status(model, "pruning", True, {"strategies": strategies})
        log_stage_summary("Pruning", model, True, f"strategies={strategies}")
    except Exception as e:
        model = args.model
        update_stage_status(model, "pruning", False, {"error": str(e)})
        log_stage_summary("Pruning", model, False, f"error={e}")
        logger.error(f"Endless pruning failed: {e}")


def cmd_endless_auto(args):
    """
    Global endless auto loop. If --prune-ratio is provided, we temporarily override
    config.reap_prune_ratio for this run, then restore the original value.
    This avoids modifying endless_rl.py to accept hyperparam_overrides.
    """
    config = load_config()
    original_prune_ratio = config.reap_prune_ratio
    try:
        if args.prune_ratio is not None:
            logger.info(f"Temporarily setting config.reap_prune_ratio to {args.prune_ratio} for this auto cycle.")
            config.reap_prune_ratio = args.prune_ratio
            save_config(config)  # persist for the loop

        from lazy_llama.endless_rl import run_endless_auto
        models = args.models if args.models else None
        run_endless_auto(
            models=models,
            cycles=args.cycles,
            sleep=args.sleep,
            policy=args.policy
        )
        logger.info("Endless auto loop completed.")
    except Exception as e:
        logger.error(f"Endless auto loop failed: {e}")
    finally:
        # Restore original prune ratio
        if args.prune_ratio is not None:
            config.reap_prune_ratio = original_prune_ratio
            save_config(config)
            logger.info(f"Restored config.reap_prune_ratio to {original_prune_ratio}")


# =============================================================================
# Health Check Command (NEW)
# =============================================================================
def cmd_health_check(args):
    """Run health check on models in the registry."""
    config = load_config()
    manager = ModelManager(config)

    if args.model:
        info = manager.get_model(args.model)
        if not info:
            logger.error(f"Model '{args.model}' not found.")
            return
        models = [info]
    else:
        models = manager.list_models(include_invalid=True)

    if not models:
        logger.info("No models found in registry.")
        return

    issues = []
    warnings_list = []
    fixable = []

    print("\n" + "=" * 60)
    print("LAZY LLAMA HEALTH CHECK")
    print("=" * 60 + "\n")

    for info in models:
        if not info:
            continue
        name = info.name
        path = Path(info.path) if info.path else None

        print(f"📦 {name}:")

        # 1. Check if path exists
        if not path or not path.exists():
            issue = f"  ❌ Path does not exist: {path}"
            print(issue)
            issues.append((name, "Path does not exist", path))
            continue

        # 2. Check if it's a directory
        if path.is_dir():
            # Check config.json
            config_path = path / "config.json"
            if not config_path.exists():
                issue = f"  ❌ Missing config.json"
                print(issue)
                issues.append((name, "Missing config.json", path))
                continue

            # Check tokenizer
            if not _validate_tokenizer_deep(path):
                issue = f"  ❌ Tokenizer is corrupt or incompatible"
                print(issue)
                issues.append((name, "Corrupt tokenizer", path))
                continue

            # Check for weight files
            has_bin = (path / "pytorch_model.bin").exists()
            has_safetensors = (path / "model.safetensors").exists()
            if not has_bin and not has_safetensors:
                issue = f"  ❌ No weight files (pytorch_model.bin or model.safetensors)"
                print(issue)
                issues.append((name, "Missing weight files", path))
                continue

            # LazyTorch model check
            if is_lazytorch_model(path):
                manifest_path = path / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                        # Check weight files
                        missing = []
                        for mod_name, mod_info in manifest.get("modules", {}).items():
                            if "weight_file" in mod_info:
                                weight_path = path / mod_info["weight_file"]
                                if not weight_path.exists():
                                    missing.append(f"{mod_name}: {mod_info['weight_file']}")
                        if missing:
                            print(f"  ⚠️ Missing weight files:")
                            for m in missing:
                                print(f"      - {m}")
                            warnings_list.append((name, f"Missing {len(missing)} weight files", path))
                    except Exception as e:
                        print(f"  ❌ Invalid manifest: {e}")
                        issues.append((name, f"Invalid manifest: {e}", path))
                else:
                    print(f"  ⚠️ Missing manifest.json")
                    warnings_list.append((name, "Missing manifest.json", path))

            print(f"  ✅ Model directory appears valid")

        elif path.is_file():
            # Check GGUF file
            if path.suffix == ".gguf":
                if is_valid_gguf is not None:
                    if is_valid_gguf(path):
                        print(f"  ✅ GGUF file appears valid")
                    else:
                        print(f"  ❌ Invalid GGUF file")
                        issues.append((name, "Invalid GGUF file", path))
                else:
                    print(f"  ⚠️ GGUF validator not available; cannot verify")
                    warnings_list.append((name, "GGUF validator not available", path))
            else:
                print(f"  ⚠️ Unknown file type: {path.suffix}")
                warnings_list.append((name, f"Unknown file type: {path.suffix}", path))

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total models checked: {len(models)}")
    print(f"Issues found: {len(issues)}")
    print(f"Warnings: {len(warnings_list)}")

    if issues:
        print("\n❌ ISSUES FOUND (fix recommended):")
        for name, issue, path in issues:
            print(f"  - {name}: {issue}")

        print("\nSuggested actions:")
        print("  - Delete invalid models:     python -m lazy_llama.bootstrap remove --model <name>")
        print("  - Re-export LazyTorch model: python -m lazy_llama.bootstrap convert-lazytorch <name> --force")
        print("  - Re-download from HF:        python -m lazy_llama.bootstrap download huggingface <name>")
        print("  - Re-create student:          python -m lazy_llama.bootstrap create-student --base <base> --student-name <name>")

    if warnings_list and not issues:
        print("\n⚠️ WARNINGS (may not be critical):")
        for name, warning, path in warnings_list:
            print(f"  - {name}: {warning}")

    if not issues and not warnings_list:
        print("\n✅ All models are healthy!")

    # Check for stale checkpoints
    checkpoint_dir = CHECKPOINTS_DIR
    if checkpoint_dir.exists():
        ckpts = list(checkpoint_dir.glob("*.pt"))
        if ckpts:
            print(f"\n📁 Checkpoints found: {len(ckpts)}")
            # Check if they correspond to existing models
            model_names = {m.name for m in manager.list_models()}
            orphaned = []
            for ckpt in ckpts:
                # Extract model name (assumes format: model_name_epoch*.pt)
                stem = ckpt.stem
                # Try to find model name (everything before _epoch or _step)
                match = re.match(r'^(.+?)(?:_epoch|_step|_distilled)', stem)
                if match:
                    model_name = match.group(1)
                    if model_name not in model_names:
                        orphaned.append((model_name, ckpt))
            if orphaned:
                print(f"  ⚠️ Orphaned checkpoints ({len(orphaned)}):")
                for model_name, ckpt in orphaned:
                    print(f"    - {ckpt.name} (model '{model_name}' not in registry)")
            else:
                print("  ✅ All checkpoints correspond to registered models.")

    print("\n" + "=" * 60)


# ----------------------------------------------------------------------
# Main argument parser and entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Lazy Llama v3.6 + Endless RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="For more details, see the documentation or use --help on subcommands."
    )
    parser.add_argument("--platform", choices=["linux", "darwin", "windows", "auto"],
                        default="auto", help="Override platform detection")
    parser.add_argument("--auto-optimize", action="store_true", help="Auto-optimize config and exit")
    parser.add_argument("--auto-convert-lazytorch", action="store_true",
                        help="Auto-convert to LazyTorch without confirmation")
    parser.add_argument("--no-default-students", action="store_true",
                        help="Skip automatic download of default student models on startup")
    parser.add_argument("--force-student", action="store_true",
                        help="Force re-creation of student model even if name exists")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ------------------------------------------------------------------
    # Existing commands
    # ------------------------------------------------------------------
    # chat
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--model", help="Model name or path")
    chat_parser.add_argument("--teacher", help="Use as teacher (Ollama)")
    chat_parser.add_argument("--student", help="Use as student (local)")

    # download
    dl_parser = subparsers.add_parser("download", help="Download a model")
    dl_parser.add_argument("source", choices=["huggingface", "ollama"])
    dl_parser.add_argument("name", help="Model name")
    dl_parser.add_argument("--gguf", help="GGUF filename (for Hugging Face)")
    dl_parser.add_argument("--auto-convert-lazytorch", action="store_true",
                           help="Convert to LazyTorch after download")

    # create-student (updated with new flags)
    cs_parser = subparsers.add_parser("create-student", help="Create a student model")
    cs_parser.add_argument("--base", required=True, help="Base model name")
    cs_parser.add_argument("--student-name", required=True, help="Name for the student")
    cs_parser.add_argument("--convert-student-lazytorch", action="store_true",
                           help="Convert student to LazyTorch after creation")
    cs_parser.add_argument("--force", action="store_true",
                           help="Force re-creation even if student name already exists")
    cs_parser.add_argument("--with-recovery", action="store_true",
                           help="Run recovery (prune + distill) on the new student")
    cs_parser.add_argument("--prune-ratio", type=float, default=None,
                           help="Prune ratio for recovery (default: from config)")
    cs_parser.add_argument("--passes", type=int, default=3,
                           help="Number of distillation passes for recovery (default: 3)")

    # export-zip
    ez_parser = subparsers.add_parser("export-zip", help="Export model as zip")
    ez_parser.add_argument("--model", required=True, help="Model name")
    ez_parser.add_argument("--zip-only", action="store_true",
                           help="Skip Hugging Face metadata")

    # import-zip
    iz_parser = subparsers.add_parser("import-zip", help="Import model from zip")
    iz_parser.add_argument("--zip", required=True, help="Zip file path")
    iz_parser.add_argument("--name", required=True, help="Model name")

    # rename
    rename_parser = subparsers.add_parser("rename", help="Rename a model")
    rename_parser.add_argument("--old", required=True)
    rename_parser.add_argument("--new", required=True)

    # remove
    remove_parser = subparsers.add_parser("remove", help="Delete a model")
    remove_parser.add_argument("--model", required=True)

    # benchmark-students (enhanced)
    bs_parser = subparsers.add_parser("benchmark-students", help="Benchmark all student models with full settings")
    bs_parser.add_argument("--prompt", default="What is machine learning?",
                           help="Prompt for generation benchmark")
    bs_parser.add_argument("--max-tokens", type=int, default=100,
                           help="Maximum tokens to generate")
    bs_parser.add_argument("--perplexity", action="store_true",
                           help="Compute perplexity (requires val_texts or uses default validation prompts)")
    bs_parser.add_argument("--val-texts", nargs="+", default=None,
                           help="List of validation texts for perplexity (space-separated)")
    bs_parser.add_argument("--mc", action="store_true",
                           help="Run multiple-choice accuracy benchmark")
    bs_parser.add_argument("--mc-questions", type=str, default=None,
                           help="Path to JSON file with multiple-choice questions")
    bs_parser.add_argument("--long-context", action="store_true",
                           help="Run long-context (RULER-style) benchmarks")
    bs_parser.add_argument("--context-lengths", nargs="+", type=int, default=None,
                           help="Context lengths for long-context tests (space-separated)")
    bs_parser.add_argument("--num-trials", type=int, default=3,
                           help="Number of trials per benchmark")
    bs_parser.add_argument("--no-store-registry", action="store_true",
                           help="Do not store results in registry (default: store)")
    bs_parser.add_argument("--lc-max-tokens", type=int, default=20,
                           help="Max tokens to generate per long-context task")

    # NEW: recover subcommand
    recover_parser = subparsers.add_parser("recover", help="Run one‑shot recovery (prune + distill) on a model")
    recover_parser.add_argument("--model", required=True, help="Model name to recover")
    recover_parser.add_argument("--teacher", help="Teacher model for distillation (default: self-distillation)")
    recover_parser.add_argument("--prune-ratio", type=float, default=None,
                                help="Prune ratio (default: from config, typically 0.15)")
    recover_parser.add_argument("--passes", type=int, default=3,
                                help="Number of distillation passes (default: 3)")
    recover_parser.add_argument("--output-name", help="Name for the recovered model (default: model_name + '_recovered')")

    # ------------------------------------------------------------------
    # Endless RL commands (removed finetune)
    # ------------------------------------------------------------------
    endless_parser = subparsers.add_parser("endless", help="Run endless RL learning loops")
    endless_subparsers = endless_parser.add_subparsers(dest="endless_mode", required=True)

    edistill = endless_subparsers.add_parser("distill", help="Endless distillation loop")
    edistill.add_argument("--teacher", required=True, help="Teacher model name")
    edistill.add_argument("--student", required=True, help="Student model name")
    edistill.add_argument("--passes", type=int, default=2, help="Passes per cycle")
    edistill.add_argument("--cycles", type=int, default=-1, help="Number of cycles (-1 = infinite)")
    edistill.add_argument("--sleep", type=int, default=60, help="Sleep between cycles (seconds)")

    eprune = endless_subparsers.add_parser("prune", help="Endless pruning loop")
    eprune.add_argument("--model", required=True, help="Model to prune")
    eprune.add_argument("--strategies", nargs="+", default=["magnitude", "neuron", "task"],
                        help="Strategies to cycle through")
    eprune.add_argument("--cycles", type=int, default=-1, help="Number of cycles (-1 = infinite)")
    eprune.add_argument("--sleep", type=int, default=60, help="Sleep between cycles (seconds)")
    eprune.add_argument("--prune-ratio", type=float, default=None,
                        help="Prune ratio override (default: from config)")

    eauto = endless_subparsers.add_parser("auto", help="Global endless auto‑improvement loop")
    eauto.add_argument("--models", nargs="+", help="Models to manage (default: all students)")
    eauto.add_argument("--cycles", type=int, default=-1, help="Number of cycles (-1 = infinite)")
    eauto.add_argument("--sleep", type=int, default=120, help="Sleep between cycles (seconds)")
    eauto.add_argument("--policy", choices=["random", "best", "worst"], default="worst",
                       help="Action selection policy")
    eauto.add_argument("--prune-ratio", type=float, default=None,
                       help="Prune ratio override for pruning actions (default: from config)")

    # ------------------------------------------------------------------
    # NEW: health-check command
    # ------------------------------------------------------------------
    health_parser = subparsers.add_parser("health-check", help="Check all models for issues")
    health_parser.add_argument("--model", help="Check a specific model (optional)")
    health_parser.add_argument("--fix", action="store_true", help="Attempt to fix minor issues (experimental)")

    # ------------------------------------------------------------------
    # Parse and dispatch
    # ------------------------------------------------------------------
    args = parser.parse_args()

    # Set platform in config if provided
    if args.platform != "auto":
        config = load_config()
        config.platform = args.platform
        save_config(config)

    # Auto-optimize and exit
    if args.auto_optimize:
        config = load_config()
        config = auto_optimize_config(config, save=True)
        logger.info("Auto-optimization applied. Config saved.")
        return

    # ---- Conditionally ensure default student models ----
    if not args.no_default_students:
        commands_needing_students = {
            None,  # TUI (no command)
            "chat",
            "create-student",
            "benchmark-students",
            "endless",
            "recover",
        }
        if args.command in commands_needing_students:
            try:
                ensure_default_students()
            except Exception as e:
                logger.error(f"Failed to ensure default students: {e}")

    # ---- If no command is given, launch the TUI ----
    if not args.command:
        tui = LazyTUI()
        tui.run()
        return

    # Dispatch commands
    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "create-student":
        force = args.force_student or getattr(args, 'force', False)
        if force:
            manager = ModelManager()
            if manager.model_exists(args.student_name):
                logger.info(f"Student '{args.student_name}' exists. Removing due to --force...")
                manager.delete_model(args.student_name)
        cmd_create_student(args)
    elif args.command == "export-zip":
        cmd_export_zip(args)
    elif args.command == "import-zip":
        cmd_import_zip(args)
    elif args.command == "rename":
        cmd_rename(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "benchmark-students":
        cmd_benchmark_students(args)
    elif args.command == "recover":
        cmd_recover(args)
    elif args.command == "endless":
        if args.endless_mode == "distill":
            cmd_endless_distill(args)
        elif args.endless_mode == "prune":
            cmd_endless_prune(args)
        elif args.endless_mode == "auto":
            cmd_endless_auto(args)
        else:
            logger.error(f"Unknown endless mode: {args.endless_mode}")
    elif args.command == "health-check":
        cmd_health_check(args)
    else:
        # Fallback to TUI (just in case)
        tui = LazyTUI()
        tui.run()


if __name__ == "__main__":
    main()