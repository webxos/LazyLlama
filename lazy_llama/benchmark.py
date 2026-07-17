"""Model benchmarking: tokens per second, memory usage, latency, perplexity, multiple-choice accuracy,
and long-context (RULER-style) evaluations. Supports E8, KV compression, and LazyTorch flags.
Results can be stored in the registry metadata.

ENHANCEMENTS (v3.8):
- Dedicated BenchmarkSettings dataclass for flexible configuration.
- Enhanced benchmark_student_models() returns rich per-model results and summary.
- Format_benchmark_summary() generates a human-readable table.
- Improved error categorisation (KV cache, OOM, tokenizer, etc.).
- All results include model_type, architecture, error_type and error_message.
- Optional progress_callback for real-time updates.
- Registry storage for both successes and failures.
- Configurable long-context generation tokens (long_context_max_tokens).

NEW (v3.9):
- Perplexity threshold-based viability check: pruned models with perplexity > threshold are rejected.
- Repetition rate metric (percentage of repeated n‑grams) to detect over‑pruning.
- Viability column in benchmark summary (✅ / ❌).
- All new features are configurable via BenchmarkSettings.
- Optimised to avoid double‑computing perplexity when both run_perplexity and check_viability are True.
- `is_model_viable()` helper kept for standalone use.
"""
import time
import logging
import gc
import psutil
import random
import math
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import copy
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- All internal imports are now relative ----
from .lazy_infer import (
    LazyGGUFEngine,
    OllamaInferenceEngine,
    TransformersInferenceEngine,
    LazyTorchEngine,
    VLLMEngine,
    is_valid_gguf,
    create_engine,
)
from .config import load_config, Config, ModelInfo
from .utils import (
    get_memory_usage_gb, is_lazytorch_model, get_available_ram_gb,
    estimate_memory_need, _validate_tokenizer_deep, validate_tokenizer_cached
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper: compute repetition rate
# ----------------------------------------------------------------------
def compute_repetition_rate(text: str, n: int = 4) -> float:
    """
    Compute the proportion of repeated n‑grams in the given text.
    Returns a float between 0 and 1, where 0 means no repetition and 1 means all n‑grams are repeated.
    """
    if not text or len(text) < n:
        return 0.0
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(1 for cnt in counts.values() if cnt > 1)
    return repeated / len(ngrams) if ngrams else 0.0


# ----------------------------------------------------------------------
# Helper: check model viability based on perplexity threshold
# ----------------------------------------------------------------------
def is_model_viable(
    model_path: str,
    threshold: float = 80.0,
    val_texts: Optional[List[str]] = None,
    config: Optional[Config] = None,
    max_length: int = 512,
) -> Tuple[bool, float]:
    """
    Compute perplexity on a validation set and return (viable, perplexity).
    Viable if perplexity < threshold.
    If val_texts not provided, uses config.validation_prompts if available.
    Returns (False, inf) on error.

    This helper is kept for standalone use; for batch benchmarking, the logic
    is integrated into benchmark_student_models to avoid double computation.
    """
    if config is None:
        config = load_config()
    if val_texts is None:
        val_texts = getattr(config, 'validation_prompts', None)
        if not val_texts:
            logger.warning("No validation texts available for viability check; using defaults.")
            val_texts = [
                "What is Python?",
                "Explain recursion.",
                "Write a loop summing 1 to 10.",
                "What is the capital of France?",
                "Define machine learning.",
            ]
    try:
        ppl_result = benchmark_perplexity(
            model_path,
            val_texts,
            config=config,
            max_length=max_length,
            stride=max_length,
        )
        if 'perplexity' in ppl_result and not math.isnan(ppl_result['perplexity']):
            ppl = ppl_result['perplexity']
            viable = ppl < threshold
            return viable, ppl
        else:
            return False, float('inf')
    except Exception as e:
        logger.error(f"Viability check failed for {model_path}: {e}")
        return False, float('inf')


# ----------------------------------------------------------------------
# Benchmark Settings
# ----------------------------------------------------------------------
@dataclass
class BenchmarkSettings:
    """Configuration for benchmarking student models."""
    prompt: str = "What is machine learning?"
    max_tokens: int = 100
    run_perplexity: bool = False
    val_texts: Optional[List[str]] = None
    run_multiple_choice: bool = False
    mc_questions: Optional[List[Dict[str, Any]]] = None
    run_long_context: bool = False
    context_lengths: Optional[List[int]] = None
    num_trials: int = 3
    store_in_registry: bool = True
    long_context_max_tokens: int = 20  # max tokens to generate per long-context task
    # Optional overrides
    e8_quantization: Optional[bool] = None
    kv_compression: Optional[bool] = None
    lazytorch: Optional[bool] = None

    # ---- NEW: viability and repetition ----
    check_viability: bool = True               # if True, reject models with perplexity above threshold
    perplexity_threshold: float = 80.0         # threshold for viability (perplexity must be < this)
    viability_val_texts: Optional[List[str]] = None  # optional separate validation set for viability
    compute_repetition_rate: bool = True       # if True, compute repetition rate of generated text

    def __post_init__(self):
        if self.run_perplexity and not self.val_texts:
            raise ValueError("val_texts required when run_perplexity=True")
        if self.run_multiple_choice and not self.mc_questions:
            raise ValueError("mc_questions required when run_multiple_choice=True")
        if self.context_lengths is None:
            self.context_lengths = [2048, 4096, 8192, 16384]
        # If viability enabled but no val_texts for viability, we'll use config.validation_prompts later
        # If both run_perplexity and check_viability are True, we can share the perplexity result
        # by setting viability_val_texts to val_texts if not explicitly provided.
        if self.check_viability and self.viability_val_texts is None and self.run_perplexity:
            self.viability_val_texts = self.val_texts


# ----------------------------------------------------------------------
# Helper: Check if a model can be loaded (including tokenizer)
# ----------------------------------------------------------------------
def _is_model_loadable(model_info: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if the model path exists and is a valid HF directory (with tokenizer)
    or a valid GGUF file.
    Returns (True, "") if loadable, otherwise (False, reason).
    """
    path = model_info.get('path')
    if not path:
        return False, "Model info missing 'path'"
    path_obj = Path(path)
    if not path_obj.exists():
        return False, f"Path does not exist: {path}"

    if path_obj.is_dir():
        has_config = (path_obj / "config.json").exists()
        has_weights = (path_obj / "pytorch_model.bin").exists() or (path_obj / "model.safetensors").exists()
        if not (has_config and has_weights):
            return False, "Missing config.json or weight files"

        # LazyTorch
        if (path_obj / "manifest.json").exists():
            if not validate_tokenizer_cached(path_obj):
                return False, "LazyTorch directory has corrupt tokenizer"
            return True, ""

        # Standard HF
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "vocab.json"]
        if not any((path_obj / f).exists() for f in tokenizer_files):
            return False, "Missing tokenizer files"
        if not validate_tokenizer_cached(path_obj):
            return False, "Tokenizer validation failed"
        return True, ""

    # File: must be .gguf and valid
    if path_obj.is_file() and path_obj.suffix == ".gguf":
        if is_valid_gguf(path_obj):
            return True, ""
        else:
            return False, "Invalid GGUF file"

    return False, "Unsupported file type"


# ----------------------------------------------------------------------
# Context generator for long‑context benchmarks (memory-efficient)
# ----------------------------------------------------------------------
def _generate_needle_haystack_context_generator(
    needle: str,
    context_length: int,
    haystack_template: str = "The sky is blue. ",
    needle_position: Optional[float] = None,
    chunk_size: int = 1024
) -> str:
    """Generate long context with needle insertion (memory‑efficient)."""
    template_len = len(haystack_template)
    repeat_count = context_length // template_len + 2
    chunks = []
    remaining = context_length
    while remaining > 0:
        chunk = haystack_template * min(repeat_count, (remaining // template_len) + 2)
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        remaining -= len(chunk)
        if remaining <= 0:
            break
    full_haystack = ''.join(chunks)

    if needle_position is None:
        needle_position = random.uniform(0.2, 0.8)
    insert_idx = int(len(full_haystack) * needle_position)
    insert_idx = full_haystack.rfind(' ', 0, insert_idx) + 1
    if insert_idx == 0:
        insert_idx = 1
    return full_haystack[:insert_idx] + needle + " " + full_haystack[insert_idx:]


def _get_max_context_length(available_ram_gb: float, model_memory_gb: float) -> int:
    """Determine safe maximum context length based on RAM and model size."""
    base_ram_per_8k = 1.0
    scale_factor = max(0.1, model_memory_gb / 14.0)
    ram_per_8k = base_ram_per_8k * scale_factor
    usable_ram = max(0.5, available_ram_gb - 2.0) * 0.7
    max_tokens = int((usable_ram / ram_per_8k) * 8192)
    return max(1024, min(32768, max_tokens))


# =============================================================================
# Perplexity Benchmark
# =============================================================================
def benchmark_perplexity(
    model_path: str,
    val_texts: List[str],
    config: Optional[Config] = None,
    max_length: int = 512,
    stride: int = 512,
    batch_size: int = 1,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Compute perplexity on validation texts using sliding window."""
    if config is None:
        config = load_config()

    try:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        logger.error(f"Failed to load model for perplexity: {e}")
        return {"error": str(e), "perplexity": float('nan')}

    total_loss = 0.0
    total_tokens = 0

    try:
        with torch.no_grad():
            for text in val_texts:
                if not text or not isinstance(text, str):
                    continue
                enc = tokenizer(text, return_tensors="pt", truncation=False)
                input_ids = enc.input_ids[0]
                seq_len = input_ids.size(0)
                if seq_len == 0:
                    continue

                for start in range(0, seq_len, stride):
                    end = min(start + max_length, seq_len)
                    if end - start < 1:
                        continue
                    input_ids_window = input_ids[start:end].unsqueeze(0).to(device)
                    outputs = model(input_ids_window, labels=input_ids_window)
                    loss = outputs.loss
                    total_loss += loss.item() * (end - start - 1)
                    total_tokens += (end - start - 1)
                    del outputs, input_ids_window
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
        else:
            perplexity = float('nan')
    except Exception as e:
        logger.error(f"Perplexity computation failed: {e}")
        perplexity = float('nan')
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "perplexity": perplexity,
        "avg_loss": total_loss / total_tokens if total_tokens > 0 else float('nan'),
        "total_tokens": total_tokens,
        "num_texts": len(val_texts),
    }


# =============================================================================
# Multiple-Choice Accuracy
# =============================================================================
def benchmark_multiple_choice(
    model_path: str,
    questions: List[Dict[str, Any]],
    config: Optional[Config] = None,
    max_new_tokens: int = 5,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate multiple-choice accuracy via log-probability comparison."""
    if config is None:
        config = load_config()

    try:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        logger.error(f"Failed to load model for multiple-choice: {e}")
        return {"error": str(e), "accuracy": 0.0}

    correct = 0
    total = len(questions)
    results = []

    try:
        with torch.no_grad():
            for q in questions:
                question = q.get('question', '')
                choices = q.get('choices', [])
                answer_idx = q.get('answer', -1)
                if not choices or answer_idx < 0 or answer_idx >= len(choices):
                    continue

                log_probs = []
                for choice in choices:
                    full_prompt = question + " " + choice
                    enc = tokenizer(full_prompt, return_tensors="pt").to(device)
                    input_ids = enc.input_ids
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    log_probs.append(-loss.item() * input_ids.size(1))

                if log_probs:
                    pred_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
                    is_correct = (pred_idx == answer_idx)
                    if is_correct:
                        correct += 1
                    results.append({
                        "question": question,
                        "choices": choices,
                        "answer": answer_idx,
                        "predicted": pred_idx,
                        "correct": is_correct,
                        "log_probs": log_probs,
                    })
    except Exception as e:
        logger.error(f"Multiple-choice evaluation failed: {e}")
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# =============================================================================
# RULER-style Long-Context Evaluation
# =============================================================================
def benchmark_ruler(
    model_path: str,
    config: Optional[Config] = None,
    context_lengths: Optional[List[int]] = None,
    num_trials: int = 2,
    tasks: Optional[List[str]] = None,
    max_tokens_per_task: int = 20,
) -> Dict[str, Any]:
    """RULER-style long-context evaluation (needle, QA, summarisation)."""
    if config is None:
        config = load_config()
    if tasks is None:
        tasks = ["needle", "qa"]

    available_ram = get_available_ram_gb()
    if available_ram < 4.0:
        return {"error": "Insufficient RAM for RULER benchmark (need at least 4 GB)"}

    model_mem_estimate = estimate_memory_need(Path(model_path))
    if model_mem_estimate < 0.1:
        model_mem_estimate = 2.0
    if available_ram < model_mem_estimate * 1.5:
        return {"error": f"Insufficient RAM: need ~{model_mem_estimate*1.5:.1f} GB, have {available_ram:.1f}"}

    if context_lengths is None:
        context_lengths = [2048, 4096, 8192]
    max_safe = _get_max_context_length(available_ram, model_mem_estimate)
    filtered_lengths = [cl for cl in context_lengths if cl <= max_safe]
    if not filtered_lengths:
        return {"error": "No context length fits in available RAM"}

    logger.info(f"Running RULER-style evaluation with lengths {filtered_lengths} and tasks {tasks}")

    engine = None
    results = {}
    try:
        config_copy = copy.deepcopy(config)
        config_copy.max_seq_len = max(filtered_lengths) + 512
        engine = create_engine(model_path, config_copy)

        for task in tasks:
            task_results = {}
            for ctx_len in filtered_lengths:
                if task == "needle":
                    successes = 0
                    total_time = 0
                    total_tokens = 0
                    needle = "The secret passphrase is 'RULER-2024'."
                    for trial in range(num_trials):
                        context = _generate_needle_haystack_context_generator(needle, ctx_len)
                        prompt = f"Extract the secret passphrase from the text:\n\n{context}\n\nSecret passphrase:"
                        start = time.time()
                        response_tokens = []
                        for token in engine.lazy_generate_stream(prompt, max_tokens=max_tokens_per_task):
                            response_tokens.append(token)
                            if needle in ''.join(response_tokens):
                                break
                        elapsed = time.time() - start
                        response = ''.join(response_tokens)
                        total_time += elapsed
                        total_tokens += len(response_tokens)
                        if needle.lower() in response.lower():
                            successes += 1
                    success_rate = successes / num_trials if num_trials > 0 else 0
                    tps = total_tokens / total_time if total_time > 0 else 0
                    task_results[f"{ctx_len}"] = {
                        "success_rate": success_rate,
                        "tokens_per_second": tps,
                        "num_trials": num_trials,
                    }
                elif task == "qa":
                    long_text = " ".join(["The quick brown fox jumps over the lazy dog. " for _ in range(ctx_len // 50)])
                    question = "What animal jumps over the lazy dog?"
                    answer_expected = "fox"
                    prompt = f"Read the following text and answer the question:\n\n{long_text}\n\nQuestion: {question}\nAnswer:"
                    start = time.time()
                    response_tokens = []
                    for token in engine.lazy_generate_stream(prompt, max_tokens=max_tokens_per_task):
                        response_tokens.append(token)
                    elapsed = time.time() - start
                    response = ''.join(response_tokens)
                    total_tokens = len(response_tokens)
                    tps = total_tokens / elapsed if elapsed > 0 else 0
                    is_correct = answer_expected.lower() in response.lower()
                    task_results[f"{ctx_len}"] = {
                        "correct": is_correct,
                        "tokens_per_second": tps,
                        "response": response[:100],
                    }
            results[task] = task_results
    except Exception as e:
        logger.error(f"RULER benchmark failed: {e}")
        results["error"] = str(e)
    finally:
        if engine:
            engine.unload()
        gc.collect()

    return results


# =============================================================================
# Store benchmark results in registry
# =============================================================================
def store_benchmark_results(
    model_name: str,
    results: Dict[str, Any],
    manager=None,
    overwrite: bool = True
) -> bool:
    """Store benchmark results in the model's registry metadata, adding a timestamp."""
    if manager is None:
        from .lazy_model_manager import ModelManager
        manager = ModelManager()

    info = manager.get_model(model_name)
    if not info:
        logger.warning(f"Model '{model_name}' not found in registry")
        return False

    if not hasattr(info, 'metadata') or info.metadata is None:
        info.metadata = {}

    if 'benchmarks' not in info.metadata:
        info.metadata['benchmarks'] = {}

    # Add timestamp if not present
    if 'timestamp' not in results:
        results['timestamp'] = time.time()

    if overwrite:
        info.metadata['benchmarks'].update(results)
    else:
        for k, v in results.items():
            if k not in info.metadata['benchmarks']:
                info.metadata['benchmarks'][k] = v

    try:
        if hasattr(manager, 'save_registry'):
            manager.save_registry()
        else:
            manager._save_registry()
    except AttributeError:
        if hasattr(manager, '_save_registry'):
            manager._save_registry()
        else:
            logger.error("Could not save registry: no save method found")
            return False

    logger.info(f"Benchmark results stored for model '{model_name}'")
    return True


# =============================================================================
# Long‑context helper (backward compatibility)
# =============================================================================
def benchmark_long_context(
    model_path: str,
    config: Config,
    context_lengths: Optional[List[int]] = None,
    num_trials: int = 3,
    needle: str = "The secret phrase is 'blue elephant'.",
    prompt_template: str = "Please extract the exact secret phrase from the text: \"\"\"{context}\"\"\"\nSecret phrase:",
    max_tokens_per_task: int = 50,
) -> Dict[str, Any]:
    """Needle‑in‑haystack retrieval benchmark (kept for backward compatibility)."""
    available_ram = get_available_ram_gb()
    if available_ram < 4.0:
        return {"error": "Low RAM; need at least 4 GB"}

    model_mem_estimate = estimate_memory_need(Path(model_path))
    if model_mem_estimate < 0.1:
        model_mem_estimate = 2.0
    if available_ram < model_mem_estimate * 1.5:
        return {"error": f"Insufficient RAM: need ~{model_mem_estimate*1.5:.1f} GB"}

    if context_lengths is None:
        context_lengths = [2048, 4096, 8192, 16384]
    max_safe = _get_max_context_length(available_ram, model_mem_estimate)
    filtered_lengths = [cl for cl in context_lengths if cl <= max_safe]
    if not filtered_lengths:
        return {"error": "No context length fits in available RAM"}

    if len(filtered_lengths) < len(context_lengths):
        logger.info(f"Reduced context lengths to {filtered_lengths} based on RAM")

    engine = None
    try:
        config_copy = copy.deepcopy(config)
        max_len = max(filtered_lengths) + 512
        if config_copy.max_seq_len < max_len:
            config_copy.max_seq_len = max_len
        engine = create_engine(model_path, config_copy)

        results = {}
        for context_len in filtered_lengths:
            successes = 0
            total_time = 0
            total_tokens = 0
            for trial in range(num_trials):
                context = _generate_needle_haystack_context_generator(needle, context_len)
                prompt = prompt_template.format(context=context)
                start = time.time()
                response_tokens = []
                for token in engine.lazy_generate_stream(prompt, max_tokens=max_tokens_per_task):
                    response_tokens.append(token)
                    if needle in ''.join(response_tokens):
                        break
                elapsed = time.time() - start
                response = ''.join(response_tokens)
                total_time += elapsed
                total_tokens += len(response_tokens)
                if needle.lower() in response.lower():
                    successes += 1
            success_rate = successes / num_trials if num_trials > 0 else 0
            tps = total_tokens / total_time if total_time > 0 else 0
            results[f"{context_len}"] = {
                "success_rate": success_rate,
                "tokens_per_second": tps,
                "num_trials": num_trials,
                "successes": successes,
                "avg_tokens": total_tokens / num_trials if num_trials > 0 else 0,
                "avg_time_seconds": total_time / num_trials if num_trials > 0 else 0,
            }
        return results
    except Exception as e:
        logger.error(f"Long-context benchmark failed: {e}")
        return {"error": str(e)}
    finally:
        if engine:
            engine.unload()
        gc.collect()


# =============================================================================
# Main benchmark_model with retry logic
# =============================================================================
def benchmark_model(
    model_path: str,
    prompt: str = "What is machine learning?",
    max_tokens: int = 100,
    config: Optional[Config] = None,
    model_name: Optional[str] = None,
    additional_metrics: Optional[List[str]] = None,
    val_texts: Optional[List[str]] = None,
    mc_questions: Optional[List[Dict[str, Any]]] = None,
    compute_repetition: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark a single model and return performance metrics.
    Includes retry logic for KV compression IndexError.
    """
    if config is None:
        config = load_config()

    original_compression = config.use_kv_cache_compression
    max_retries = 1
    last_exception = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning(f"Retry attempt {attempt}/{max_retries}: disabling KV cache compression.")
            config.use_kv_cache_compression = False
        else:
            config.use_kv_cache_compression = original_compression

        try:
            gc.collect()

            engine = None
            lazytorch_used = False
            path_obj = Path(model_path)

            # ---- vLLM ----
            if model_path.startswith("vllm://"):
                try:
                    engine = VLLMEngine(model_path.replace("vllm://", ""), config)
                except Exception as e:
                    raise RuntimeError(f"Failed to create vLLM engine: {e}")

            # ---- Ollama ----
            elif model_path.startswith("ollama://"):
                try:
                    engine = OllamaInferenceEngine(model_path.replace("ollama://", ""), config)
                except Exception as e:
                    raise RuntimeError(f"Failed to create Ollama engine: {e}")

            # ---- GGUF ----
            elif path_obj.suffix == ".gguf" or model_path.endswith(".gguf"):
                if not is_valid_gguf(path_obj):
                    if path_obj.is_dir():
                        logger.warning(f"Path '{path_obj}' is a directory but has .gguf suffix; attempting Transformers.")
                        if not validate_tokenizer_cached(path_obj):
                            raise ValueError(f"Model directory {path_obj} has a corrupt tokenizer.")
                        engine = TransformersInferenceEngine(str(path_obj), config)
                    else:
                        raise ValueError(f"File '{model_path}' is not a valid GGUF model.")
                else:
                    try:
                        engine = LazyGGUFEngine(model_path, config)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load GGUF engine: {e}")

            # ---- LazyTorch ----
            elif is_lazytorch_model(path_obj) or (config.use_lazytorch and (path_obj.is_dir() or path_obj.suffix == '.lazytorch')):
                try:
                    if path_obj.is_dir() and not is_lazytorch_model(path_obj):
                        candidate = path_obj.with_suffix('.lazytorch')
                        if candidate.exists() and is_lazytorch_model(candidate):
                            model_path = str(candidate)
                            path_obj = candidate
                    tokenizer_path = path_obj if path_obj.is_dir() else path_obj.parent
                    if not validate_tokenizer_cached(tokenizer_path):
                        raise ValueError(f"Tokenizer in LazyTorch model at {path_obj} is corrupt.")
                    engine = LazyTorchEngine(model_path, config)
                    lazytorch_used = True
                except Exception as e:
                    logger.warning(f"Failed to load LazyTorch engine: {e}. Falling back to Transformers.")
                    engine = None

            # ---- Fallback to Transformers ----
            if engine is None:
                if path_obj.is_dir():
                    if not validate_tokenizer_cached(path_obj):
                        raise ValueError(f"Model directory {path_obj} has a corrupt tokenizer.")
                    engine_config = copy.deepcopy(config)
                    if engine_config.use_e8_quantization:
                        estimated_mem = estimate_memory_need(path_obj)
                        available_ram = get_available_ram_gb()
                        if available_ram < estimated_mem * 1.5:
                            logger.warning(f"Disabling E8 quantization for this benchmark (low RAM).")
                            engine_config.use_e8_quantization = False
                        if engine_config.use_e8_quantization and get_available_ram_gb() < 8.0:
                            logger.warning("Disabling E8 quantization to prevent OOM.")
                            engine_config.use_e8_quantization = False
                    try:
                        engine = TransformersInferenceEngine(model_path, engine_config)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load Transformers engine from '{model_path}': {e}")
                else:
                    raise ValueError(f"Cannot benchmark '{model_path}'. Supported formats: Ollama, vLLM, GGUF, HF dir, LazyTorch.")

            if model_name is None:
                model_name = Path(model_path).stem

            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024**3)
            start = time.time()
            tokens = []
            full_text = ""
            peak_mem = mem_before

            try:
                for i, token in enumerate(engine.lazy_generate_stream(prompt, max_tokens=max_tokens)):
                    tokens.append(token)
                    full_text += token
                    if (i + 1) % 2 == 0:
                        current_mem = process.memory_info().rss / (1024**3)
                        if current_mem > peak_mem:
                            peak_mem = current_mem
            except Exception as e:
                engine.unload()
                raise

            elapsed = time.time() - start
            mem_after = process.memory_info().rss / (1024**3)

            results = {
                "model_name": model_name,
                "model_path": model_path,
                "tokens_generated": len(tokens),
                "time_seconds": elapsed,
                "tokens_per_second": len(tokens) / elapsed if elapsed > 0 else 0,
                "avg_latency_ms": (elapsed / len(tokens)) * 1000 if tokens else 0,
                "memory_usage_gb": mem_after,
                "peak_memory_gb": peak_mem,
                "memory_delta_gb": mem_after - mem_before,
                "success": True,
                "e8_quantized": engine_config.use_e8_quantization if 'engine_config' in locals() else getattr(config, 'use_e8_quantization', False),
                "kv_compressed": getattr(config, 'use_kv_cache_compression', False),
                "lazytorch_used": lazytorch_used,
                "engine_type": engine.get_model_type() if engine else "unknown",
            }
            # ---- NEW: repetition rate ----
            if compute_repetition and full_text:
                results['repetition_rate'] = compute_repetition_rate(full_text, n=4)
            else:
                results['repetition_rate'] = 0.0

            engine.unload()

            if additional_metrics:
                if 'perplexity' in additional_metrics and val_texts:
                    logger.info("Computing perplexity...")
                    ppl = benchmark_perplexity(model_path, val_texts, config, model_name=model_name)
                    results['perplexity'] = ppl.get('perplexity', float('nan'))
                    results['perplexity_avg_loss'] = ppl.get('avg_loss', float('nan'))
                if 'multiple_choice' in additional_metrics and mc_questions:
                    logger.info("Computing multiple-choice accuracy...")
                    mc = benchmark_multiple_choice(model_path, mc_questions, config, model_name=model_name)
                    results['multiple_choice_accuracy'] = mc.get('accuracy', 0.0)

            config.use_kv_cache_compression = original_compression
            return results

        except IndexError as e:
            if not original_compression or attempt == max_retries:
                config.use_kv_cache_compression = original_compression
                logger.error(f"IndexError during benchmark: {e}")
                raise
            logger.warning(f"IndexError: {e}. Retrying without KV compression (attempt {attempt+1}/{max_retries+1}).")
            last_exception = e
            continue

        except Exception as e:
            config.use_kv_cache_compression = original_compression
            raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Benchmark failed for unknown reason.")


# =============================================================================
# Student‑only benchmarking with rich settings and reporting
# =============================================================================
def benchmark_student_models(
    settings: BenchmarkSettings,
    config: Optional[Config] = None,
    progress_callback: Optional[Callable[[str, str, Optional[Dict]], None]] = None,
) -> Dict[str, Any]:
    """
    Benchmark all student models with the given settings.
    Returns a dict with:
        - 'results': list of per-model result dicts (including success, error_type, metrics)
        - 'summary': aggregate statistics (total, succeeded, failed, avg TPS, etc.)

    This function intelligently avoids double‑computing perplexity:
        - If both run_perplexity and check_viability are True and the same validation texts are used,
          the perplexity computed for the benchmark is reused for viability.
        - If run_perplexity is False but check_viability is True, a separate perplexity computation
          is performed for viability.

    Args:
        settings: BenchmarkSettings instance.
        config: Optional Config.
        progress_callback: Optional callback called with (model_name, status, result_dict).
            Status: 'starting', 'done', 'error'.
    """
    if config is None:
        config = load_config()

    # Get student models
    from .lazy_model_manager import ModelManager
    mm = ModelManager()
    mm.sync_ollama()
    mm.reload_registry()

    student_entries = []
    for info in mm.list_models():
        if '_distilled' in info.name or info.name.endswith('_pruned'):
            if info.path and not info.path.startswith("ollama://"):
                student_entries.append({
                    'name': info.name,
                    'path': info.path,
                    'info': info,
                })

    if not student_entries:
        logger.warning("No student models found")
        return {"results": [], "summary": {"total": 0, "succeeded": 0, "failed": 0}}

    # Determine if we can run long-context
    available_ram = get_available_ram_gb()
    can_run_long_context = settings.run_long_context and available_ram >= 6.0
    if settings.run_long_context and not can_run_long_context:
        logger.warning("Long-context requested but RAM < 6 GB; skipping.")

    # Determine if we need additional metrics
    additional_metrics = []
    if settings.run_perplexity and settings.val_texts:
        additional_metrics.append('perplexity')
    if settings.run_multiple_choice and settings.mc_questions:
        additional_metrics.append('multiple_choice')

    # For viability, we may need to compute perplexity separately.
    # Optimisation: if both run_perplexity and check_viability are True and the validation texts
    # are the same (settings.viability_val_texts is None or equal to settings.val_texts),
    # we can reuse the perplexity from the benchmark.
    check_viability = settings.check_viability
    viability_threshold = settings.perplexity_threshold
    viability_val_texts = settings.viability_val_texts
    if check_viability and not viability_val_texts:
        # Use config validation prompts as fallback
        viability_val_texts = getattr(config, 'validation_prompts', None)
        if not viability_val_texts:
            logger.warning("No validation texts for viability; using a small default set.")
            viability_val_texts = [
                "What is Python?",
                "Explain recursion.",
                "Write a loop summing 1 to 10.",
                "What is the capital of France?",
                "Define machine learning.",
            ]
    compute_repetition = settings.compute_repetition_rate

    results = []
    for entry in student_entries:
        if progress_callback:
            progress_callback(entry['name'], "starting", None)

        loadable, reason = _is_model_loadable(entry)
        if not loadable:
            logger.info(f"Skipping unloadable model {entry['name']}: {reason}")
            result = {
                "model_name": entry['name'],
                "model_path": entry['path'],
                "success": False,
                "error_type": "LoadError",
                "error_message": reason,
                "metrics": {},
                "viable": False,
                "perplexity": float('inf'),
                "repetition_rate": 0.0,
            }
            results.append(result)
            if progress_callback:
                progress_callback(entry['name'], "error", result)
            if settings.store_in_registry:
                store_benchmark_results(entry['name'], {
                    "benchmark_error": True,
                    "error_type": "LoadError",
                    "error_message": reason,
                })
            continue

        logger.info(f"Benchmarking student model: {entry['name']} ({entry['path']})")
        model_result = {
            "model_name": entry['name'],
            "model_path": entry['path'],
            "success": False,
            "error_type": None,
            "error_message": None,
            "metrics": {},
            "viable": False,
            "perplexity": float('inf'),
            "repetition_rate": 0.0,
        }

        try:
            # Determine if we can reuse perplexity from benchmark
            use_perplexity_for_viability = (
                check_viability
                and settings.run_perplexity
                and settings.val_texts is not None
                and (viability_val_texts is None or viability_val_texts == settings.val_texts)
            )

            # If we can reuse, we don't need to add 'perplexity' to additional_metrics? Actually we need it from the benchmark.
            # We'll add 'perplexity' to additional_metrics if run_perplexity is True, or if we need it for viability and we can't reuse.
            effective_additional_metrics = additional_metrics.copy()
            if check_viability and not use_perplexity_for_viability:
                # Need to compute perplexity separately for viability; but we can still add 'perplexity' to metrics if not already.
                if 'perplexity' not in effective_additional_metrics and viability_val_texts:
                    effective_additional_metrics.append('perplexity')

            res = benchmark_model(
                entry['path'],
                prompt=settings.prompt,
                max_tokens=settings.max_tokens,
                config=config,
                model_name=entry['name'],
                additional_metrics=effective_additional_metrics if effective_additional_metrics else None,
                val_texts=settings.val_texts if settings.run_perplexity else viability_val_texts if check_viability and not use_perplexity_for_viability else None,
                mc_questions=settings.mc_questions if settings.run_multiple_choice else None,
                compute_repetition=compute_repetition,
            )
            if res.get('success', False):
                model_result['success'] = True
                model_result['metrics'] = {
                    'tokens_per_second': res.get('tokens_per_second', 0.0),
                    'peak_memory_gb': res.get('peak_memory_gb', 0.0),
                    'e8_quantized': res.get('e8_quantized', False),
                    'kv_compressed': res.get('kv_compressed', False),
                    'lazytorch_used': res.get('lazytorch_used', False),
                }
                if 'perplexity' in res:
                    model_result['metrics']['perplexity'] = res['perplexity']
                    model_result['perplexity'] = res['perplexity']
                if 'multiple_choice_accuracy' in res:
                    model_result['metrics']['multiple_choice_accuracy'] = res['multiple_choice_accuracy']
                if 'repetition_rate' in res:
                    model_result['repetition_rate'] = res['repetition_rate']

                # ---- Viability check ----
                if check_viability:
                    # Determine perplexity to use
                    if use_perplexity_for_viability and 'perplexity' in res and not math.isnan(res['perplexity']):
                        ppl = res['perplexity']
                    else:
                        # Compute or use existing
                        if 'perplexity' in res and not math.isnan(res['perplexity']):
                            ppl = res['perplexity']
                        else:
                            logger.info(f"Computing perplexity for viability of {entry['name']}...")
                            ppl_res = benchmark_perplexity(
                                entry['path'],
                                viability_val_texts,
                                config=config,
                                model_name=entry['name']
                            )
                            ppl = ppl_res.get('perplexity', float('inf'))
                            model_result['perplexity'] = ppl
                    viable = (ppl < viability_threshold) and not math.isnan(ppl)
                    model_result['viable'] = viable
                else:
                    model_result['viable'] = True  # if not checking, consider viable

                # Long-context if enabled and RAM allows
                if can_run_long_context and settings.run_long_context:
                    model_mem = res.get('peak_memory_gb', 2.0)
                    if model_mem < 0.5:
                        model_mem = 1.0
                    max_safe = _get_max_context_length(available_ram, model_mem)
                    ctx_lengths = [cl for cl in (settings.context_lengths or [2048, 4096, 8192, 16384]) if cl <= max_safe]
                    if ctx_lengths:
                        logger.info(f"Running long-context for {entry['name']} with lengths {ctx_lengths}")
                        long_res = benchmark_long_context(
                            entry['path'],
                            config,
                            context_lengths=ctx_lengths,
                            num_trials=settings.num_trials,
                            max_tokens_per_task=settings.long_context_max_tokens,
                        )
                        if 'error' not in long_res:
                            model_result['metrics']['long_context'] = long_res
                        else:
                            model_result['metrics']['long_context_error'] = long_res['error']
                    else:
                        logger.warning(f"Skipping long-context for {entry['name']}: insufficient RAM for 4K context.")
            else:
                model_result['error_type'] = 'BenchmarkFailed'
                model_result['error_message'] = res.get('error', 'Unknown error')
                model_result['viable'] = False
        except IndexError as e:
            model_result['error_type'] = 'IndexError (KV cache)'
            model_result['error_message'] = str(e)
            model_result['viable'] = False
        except MemoryError as e:
            model_result['error_type'] = 'OutOfMemory'
            model_result['error_message'] = str(e)
            model_result['viable'] = False
        except Exception as e:
            model_result['error_type'] = type(e).__name__
            model_result['error_message'] = str(e)
            model_result['traceback'] = traceback.format_exc()[:500]  # Truncate
            model_result['viable'] = False

        # Store in registry if requested (both success and failure)
        if settings.store_in_registry:
            store_data = {
                'success': model_result['success'],
                'viable': model_result.get('viable', False),
                'perplexity': model_result.get('perplexity', float('inf')),
                'repetition_rate': model_result.get('repetition_rate', 0.0),
            }
            if model_result['success']:
                store_data.update({
                    'tokens_per_second': model_result['metrics'].get('tokens_per_second'),
                    'peak_memory_gb': model_result['metrics'].get('peak_memory_gb'),
                    'e8_quantized': model_result['metrics'].get('e8_quantized'),
                    'kv_compressed': model_result['metrics'].get('kv_compressed'),
                    'lazytorch_used': model_result['metrics'].get('lazytorch_used'),
                })
                if 'perplexity' in model_result['metrics']:
                    store_data['perplexity'] = model_result['metrics']['perplexity']
                if 'multiple_choice_accuracy' in model_result['metrics']:
                    store_data['multiple_choice_accuracy'] = model_result['metrics']['multiple_choice_accuracy']
                if 'long_context' in model_result['metrics']:
                    store_data['long_context'] = model_result['metrics']['long_context']
            else:
                store_data['error_type'] = model_result['error_type']
                store_data['error_message'] = model_result['error_message']
            store_benchmark_results(entry['name'], store_data)

        results.append(model_result)
        if progress_callback:
            status = "done" if model_result['success'] else "error"
            progress_callback(entry['name'], status, model_result)

    # Compute summary
    succeeded = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    viable = [r for r in results if r.get('viable', False)]
    summary = {
        "total": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "viable": len(viable),
        "avg_tps": 0.0,
        "avg_peak_mem_gb": 0.0,
    }
    if succeeded:
        avg_tps = sum(r['metrics'].get('tokens_per_second', 0.0) for r in succeeded) / len(succeeded)
        avg_mem = sum(r['metrics'].get('peak_memory_gb', 0.0) for r in succeeded) / len(succeeded)
        summary['avg_tps'] = avg_tps
        summary['avg_peak_mem_gb'] = avg_mem

    return {
        "results": results,
        "summary": summary,
    }


# =============================================================================
# Format benchmark summary as rich table
# =============================================================================
def format_benchmark_summary(results: List[Dict[str, Any]]) -> str:
    """
    Return a formatted string (Rich table) showing key metrics for each model.
    Includes success/failure indicators and error messages for failures.
    Adds a "Viable" column if viability was checked (perplexity threshold).
    """
    from rich.table import Table
    from rich.console import Console
    from rich.text import Text

    table = Table(title="Student Model Benchmark Report")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("TPS", justify="right")
    table.add_column("Peak Mem (GB)", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("MC Acc", justify="right")
    table.add_column("LC Score", justify="right")  # combined long-context score
    table.add_column("Repetition", justify="right")
    table.add_column("Viable", justify="center")
    table.add_column("Status")

    for r in results:
        name = r['model_name']
        if r['success']:
            m = r['metrics']
            tps = f"{m.get('tokens_per_second', 0.0):.2f}"
            mem = f"{m.get('peak_memory_gb', 0.0):.2f}"
            ppl = f"{m.get('perplexity', float('nan')):.3f}" if 'perplexity' in m else "—"
            mc = f"{m.get('multiple_choice_accuracy', 0.0):.2%}" if 'multiple_choice_accuracy' in m else "—"
            # Compute a combined long-context score: average success rate across all lengths
            lc_score = "—"
            if 'long_context' in m:
                lc_data = m['long_context']
                # Collect all success rates from the task results
                rates = []
                for task in lc_data.values():
                    if isinstance(task, dict):
                        for length, val in task.items():
                            if isinstance(val, dict) and 'success_rate' in val:
                                rates.append(val['success_rate'])
                if rates:
                    avg_lc = sum(rates) / len(rates)
                    lc_score = f"{avg_lc:.1%}"
            # Repetition rate
            rep = f"{r.get('repetition_rate', 0.0):.1%}" if 'repetition_rate' in r else "—"
            # Viability
            viable = r.get('viable', False)
            viable_text = "✅" if viable else "❌"
            status = Text("✅", style="green")
        else:
            tps = mem = ppl = mc = lc_score = rep = "—"
            viable_text = "❌"
            error_type = r.get('error_type', 'Unknown')
            error_msg = r.get('error_message', '')
            status = Text(f"❌ {error_type}", style="red")
            if error_msg:
                status.append(f"\n{error_msg[:60]}", style="dim")

        table.add_row(name, tps, mem, ppl, mc, lc_score, rep, viable_text, status)

    console = Console()
    with console.capture() as capture:
        console.print(table)
    return capture.get()


# =============================================================================
# Legacy function (for backward compatibility with older callers)
# =============================================================================
def benchmark_student_models_legacy(
    output_file: Optional[str] = None,
    prompt: str = "What is machine learning?",
    max_tokens: int = 100,
    config: Optional[Config] = None,
    run_long_context: Optional[bool] = None,
    run_perplexity: bool = False,
    val_texts: Optional[List[str]] = None,
    run_multiple_choice: bool = False,
    mc_questions: Optional[List[Dict[str, Any]]] = None,
    store_in_registry: bool = True,
) -> List[Dict]:
    """
    Legacy wrapper for benchmark_student_models that returns a simple list.
    Deprecated; prefer using benchmark_student_models(settings).
    """
    logger.warning("benchmark_student_models_legacy is deprecated; use benchmark_student_models(settings).")
    settings = BenchmarkSettings(
        prompt=prompt,
        max_tokens=max_tokens,
        run_perplexity=run_perplexity,
        val_texts=val_texts,
        run_multiple_choice=run_multiple_choice,
        mc_questions=mc_questions,
        run_long_context=run_long_context or False,
        store_in_registry=store_in_registry,
    )
    result = benchmark_student_models(settings, config)
    # Convert to old format: list of dicts with 'model_name', 'success', etc.
    legacy_results = []
    for r in result['results']:
        legacy_results.append({
            'model_name': r['model_name'],
            'model_path': r['model_path'],
            'success': r['success'],
            'error': r.get('error_message'),
            **r['metrics']
        })
    return legacy_results


# =============================================================================
# Endless benchmarking and decision helpers (v3.6)
# =============================================================================
def run_endless_benchmark(
    models: List[str],
    cycles: int = -1,
    sleep: int = 60,
    callback: Optional[Callable] = None
) -> Dict[str, List[float]]:
    """Endless benchmarking of a list of models, recording TPS history."""
    from .lazy_model_manager import ModelManager
    config = load_config()
    manager = ModelManager(config)
    history = {m: [] for m in models}

    cycle = 0
    while cycles == -1 or cycle < cycles:
        cycle += 1
        logger.info(f"Endless benchmark cycle {cycle}")
        if callback:
            callback(f"Benchmark cycle {cycle}")

        for model_name in models:
            if not manager.model_exists(model_name):
                logger.warning(f"Model {model_name} not found, skipping")
                continue
            info = manager.get_model(model_name)
            if not info or not info.path:
                continue
            loadable, _ = _is_model_loadable({'path': info.path})
            if not loadable:
                logger.warning(f"Model {model_name} is not loadable, skipping")
                history[model_name].append(None)
                continue
            try:
                res = benchmark_model(info.path, config=config, model_name=model_name)
                tps = res.get('tokens_per_second', 0.0)
                history[model_name].append(tps)
                logger.info(f"{model_name}: {tps:.2f} tok/s")
                if callback:
                    callback(f"{model_name}: {tps:.2f} tok/s")
            except Exception as e:
                logger.error(f"Benchmark {model_name} failed: {e}")
                history[model_name].append(None)

        if cycles != -1 and cycle >= cycles:
            break
        if sleep > 0:
            time.sleep(sleep)

    return history


def decide_action(
    history: Dict[str, List[float]],
    model_names: List[str],
    policy: str = "worst"
) -> Tuple[str, str]:
    """Decide which model to improve and which action to take."""
    avg_tps = {}
    for name in model_names:
        vals = [v for v in history.get(name, []) if v is not None]
        avg_tps[name] = sum(vals) / len(vals) if vals else 0.0

    if policy == "worst":
        model = min(avg_tps, key=avg_tps.get)
    elif policy == "best":
        model = max(avg_tps, key=avg_tps.get)
    else:
        import random
        model = random.choice(model_names)

    tps = avg_tps.get(model, 0.0)
    if tps < 2.0:
        action = "distill"
    elif tps < 5.0:
        action = "prune"
    else:
        action = "finetune"
    return model, action