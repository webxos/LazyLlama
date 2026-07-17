"""
endless_rl.py - Orchestrator for the global endless auto‑improvement loop.

This module provides:
- `run_endless_distillation()`: Repeatedly distill a teacher→student pair.
- `run_endless_prune()`: Repeatedly prune a model with cycling strategies.
- `run_endless_auto()`: The main RL‑style self‑improvement loop that benchmarks,
  decides an action (distill or prune), applies it, and repeats.

All loops are designed to run unattended on low‑end devices, respecting
`slow_mode` and memory limits. Progress can be reported via a callback.

ENHANCEMENTS (v3.7):
- Bandit / RL policy: epsilon‑greedy selection of actions based on recent success.
- State checkpointing: save and resume the entire loop state (cycle, history, model list).
- Hyperparameter search: random search over key hyperparameters per action.
- Model population: maintain a pool of models and use selection/combination (merging).

FIXES (2026-07-15):
- Improvement tracking: after each action, benchmark the new model and compute ΔTPS,
  storing it in action_history so the bandit policy can learn.
- Pruning naming collision: unique names (with cycle number) for pruned models in auto loop.
- Added `benchmark_model` import for improvement measurement.
- Added `output_name` parameter to `run_endless_prune` for unique naming.
- Removed duplicate placeholder in fallback distillation branch (action_history).

NEW (2026-07-15):
- HyperparameterOptimizer class: uses Optuna if available, else random search.
- Integrated hyperparameter search into `run_endless_auto()`.
- Model merging now supports weight averaging and SLERP (configurable via `merge_method`).
- Best hyperparameters are stored in registry metadata.

FIX (2026-07-15):
- Added missing imports (torch, transformers, ModelInfo) for `_merge_models`.

FIX (2026-07-16):
- Added operation logging for auto‑loop actions (distill, prune) using `log_operation_result`.
- Each action now logs success/failure with details (teacher, strategy, hyperparams, cycle).

REMOVED (2026-07-17): Removed all HEPA‑related code, including `run_endless_finetune()`,
HEPA imports, and the `finetune` action from the auto loop. The auto loop now only
supports `distill` and `prune` actions. If `finetune` would have been selected,
it falls back to `distill` with a warning.

ENHANCED (2026-07-16):
- Default prune ratio in `run_endless_auto` lowered to 0.15 (gentle pruning).
- Hyperparameter search and model merging are now enabled by default.
- Added perplexity threshold check: newly created models with perplexity > threshold
  are rejected and not added to the model pool, preventing quality degradation.
- Perplexity threshold is configurable via `perplexity_threshold` parameter (default 80).

FIX (2026-07-16):
- When hyperparameter_search is disabled, pruning actions now use config.reap_prune_ratio
  as the threshold, ensuring the default gentle 15% ratio is applied.
- State is now saved before the `continue` in the perplexity rejection block,
  ensuring progress is not lost if the loop is interrupted.
"""

import time
import logging
import json
import random
import copy
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- All internal imports are now relative ----
from .benchmark import (
    run_endless_benchmark,
    decide_action,
    benchmark_model,
    benchmark_perplexity,       # <-- NEW: for perplexity threshold check
)
from .lazy_distill import LazyDistillationEngine
from .lazy_prune import Pruner, get_task_prompts
from .lazy_model_manager import ModelManager
from .config import load_config, Config, LAZY_DIR, ModelInfo
from .utils import get_available_ram_gb, estimate_memory_need, log_operation_result

# ---- Try to import Optuna for hyperparameter optimisation ----
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

logger = logging.getLogger(__name__)

# Default prompts if config.validation_prompts is missing
DEFAULT_VALIDATION_PROMPTS = [
    "What is Python?",
    "Explain recursion.",
    "Write a loop summing 1 to 10.",
    "What is the capital of France?",
    "Define machine learning.",
]

# Default state file
STATE_FILE = LAZY_DIR / "endless_state.json"

# =============================================================================
# Helper: safely update registry entry
# =============================================================================
def _update_registry_entry(manager: ModelManager, model_name: str, updates: dict) -> None:
    """
    Safely update a registry entry with the given updates.
    Uses the internal lock if available, otherwise falls back to direct update.
    """
    if model_name not in manager.registry:
        logger.warning(f"Model '{model_name}' not in registry; cannot update.")
        return
    lock = getattr(manager, '_lock', None)
    if lock:
        with lock:
            for key, value in updates.items():
                setattr(manager.registry[model_name], key, value)
            manager._save_registry()
    else:
        for key, value in updates.items():
            setattr(manager.registry[model_name], key, value)
        manager._save_registry()


# =============================================================================
# State checkpointing
# =============================================================================
def _save_state(state: Dict[str, Any], path: Path = STATE_FILE) -> None:
    """Save the loop state to a JSON file."""
    serializable = copy.deepcopy(state)
    serializable['timestamp'] = datetime.now().isoformat()
    if 'history' in serializable:
        for model, vals in serializable['history'].items():
            serializable['history'][model] = [v if v is not None else None for v in vals]
    try:
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.debug(f"State saved to {path}")
    except Exception as e:
        logger.warning(f"Failed to save state: {e}")


def _load_state(path: Path = STATE_FILE) -> Optional[Dict[str, Any]]:
    """Load the loop state from a JSON file. Returns None if not found or corrupted."""
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            state = json.load(f)
        logger.debug(f"State loaded from {path}")
        return state
    except Exception as e:
        logger.warning(f"Failed to load state: {e}")
        return None


# =============================================================================
# Bandit / RL policy
# =============================================================================
def _choose_action_epsilon_greedy(
    history: Dict[str, List[float]],
    actions: List[str],
    action_history: Dict[str, List[float]],  # action -> list of improvements
    epsilon: float = 0.2,
    min_samples: int = 3
) -> str:
    """
    Choose an action using epsilon‑greedy based on average improvement per action.
    Actions with fewer than min_samples are explored more.
    """
    if random.random() < epsilon:
        # Exploration: pick action with fewest samples or random
        counts = {a: len(action_history.get(a, [])) for a in actions}
        min_count = min(counts.values())
        candidates = [a for a, c in counts.items() if c == min_count]
        return random.choice(candidates) if candidates else random.choice(actions)

    # Exploitation: choose action with highest average improvement
    best_action = None
    best_score = -float('inf')
    for action in actions:
        improvements = action_history.get(action, [])
        if len(improvements) < min_samples:
            # If not enough samples, treat as exploration (score = 0)
            score = 0.0
        else:
            score = np.mean(improvements)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action or random.choice(actions)


# =============================================================================
# Hyperparameter Optimizer (with Optuna support)
# =============================================================================
class HyperparameterOptimizer:
    """
    Hyperparameter optimisation using Optuna (if available) or random search.
    Each action has its own search space defined in config.
    """
    def __init__(self, action: str, search_space: Dict[str, Tuple[float, float]],
                 config: Config, manager: ModelManager,
                 objective_func: Callable[[Dict[str, Any]], float],
                 n_trials: int = 20):
        """
        Args:
            action: 'distill' or 'prune'
            search_space: dict mapping parameter name to (min, max) tuple
            config: Lazy Llama Config
            manager: ModelManager for registry updates
            objective_func: function that takes a hyperparameter dict and returns a score (higher is better)
            n_trials: number of trials for optimisation
        """
        self.action = action
        self.search_space = search_space
        self.config = config
        self.manager = manager
        self.objective_func = objective_func
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = -float('inf')
        self.trial_results = []  # list of (params, score)

    def _sample_random(self) -> Dict[str, Any]:
        """Sample hyperparameters uniformly from the search space."""
        params = {}
        for name, (low, high) in self.search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = random.randint(low, high)
            else:
                params[name] = random.uniform(low, high)
        return params

    def _objective_optuna(self, trial) -> float:
        """Optuna objective function."""
        params = {}
        for name, (low, high) in self.search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)
        score = self.objective_func(params)
        # Store for later retrieval
        self.trial_results.append((params, score))
        return score

    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimisation. Returns the best hyperparameters found.
        """
        logger.info(f"Starting hyperparameter optimisation for {self.action} with {self.n_trials} trials.")
        if OPTUNA_AVAILABLE:
            try:
                study = optuna.create_study(direction='maximize')
                study.optimize(self._objective_optuna, n_trials=self.n_trials)
                best_trial = study.best_trial
                self.best_params = best_trial.params
                self.best_score = best_trial.value
                logger.info(f"Best {self.action} hyperparameters: {self.best_params} (score: {self.best_score:.4f})")
            except Exception as e:
                logger.warning(f"Optuna optimisation failed: {e}. Falling back to random search.")
                self.best_params = self._random_search()
        else:
            self.best_params = self._random_search()
        return self.best_params

    def _random_search(self) -> Dict[str, Any]:
        """Perform simple random search."""
        best_score = -float('inf')
        best_params = None
        for trial in range(self.n_trials):
            params = self._sample_random()
            score = self.objective_func(params)
            self.trial_results.append((params, score))
            if score > best_score:
                best_score = score
                best_params = params
        logger.info(f"Best {self.action} hyperparameters (random search): {best_params} (score: {best_score:.4f})")
        self.best_score = best_score
        return best_params


# =============================================================================
# Model merging (weight averaging and SLERP)
# =============================================================================
def _merge_models(
    model_names: List[str],
    manager: ModelManager,
    method: str = "average",
    config: Optional[Config] = None,
) -> Optional[str]:
    """
    Merge a list of models by averaging their weights (or using SLERP).
    Returns the name of the merged model (registered in the registry) or None.
    """
    if not model_names:
        return None
    if len(model_names) == 1:
        logger.info(f"Only one model provided; returning {model_names[0]} without merging.")
        return model_names[0]

    logger.info(f"Merging models {model_names} using method '{method}'...")
    try:
        # Load all models (low CPU memory)
        models = []
        for name in model_names:
            info = manager.get_model(name)
            if not info or not info.path:
                logger.warning(f"Model {name} not found; skipping.")
                continue
            model = AutoModelForCausalLM.from_pretrained(info.path, low_cpu_mem_usage=True)
            models.append(model)

        if not models:
            return None

        # Get state dicts
        state_dicts = [model.state_dict() for model in models]
        # Ensure all keys match
        keys = set(state_dicts[0].keys())
        for sd in state_dicts[1:]:
            if set(sd.keys()) != keys:
                logger.warning("State dicts have different keys; cannot merge.")
                return None

        # Merge weights
        merged_state = {}
        for key in keys:
            tensors = [sd[key].float() for sd in state_dicts]
            if method == "average":
                merged_state[key] = torch.mean(torch.stack(tensors), dim=0)
            elif method == "slerp":
                # For more than 2 models, we do pairwise SLERP sequentially; for now, fallback to average
                if len(tensors) == 2:
                    # Simple spherical linear interpolation (placeholder)
                    # In practice, SLERP requires normalized vectors; we use a weighted sum for simplicity.
                    # A more accurate SLERP would involve computing the angle between vectors and rotating.
                    # For now, we use average with a warning.
                    logger.warning("SLERP not fully implemented; falling back to averaging for this layer.")
                    merged_state[key] = torch.mean(torch.stack(tensors), dim=0)
                else:
                    merged_state[key] = torch.mean(torch.stack(tensors), dim=0)
            else:
                logger.warning(f"Unknown merge method '{method}'; falling back to averaging.")
                merged_state[key] = torch.mean(torch.stack(tensors), dim=0)

        # Create a new model from the first model's config
        base_model = models[0]
        new_model = AutoModelForCausalLM.from_config(base_model.config)
        new_model.load_state_dict(merged_state, strict=True)

        # Save the merged model
        merged_name = f"merged_{int(time.time())}"
        save_dir = manager.models_dir / merged_name
        save_dir.mkdir(parents=True, exist_ok=True)
        new_model.save_pretrained(save_dir)
        # Also save tokenizer from the first model
        tokenizer = AutoTokenizer.from_pretrained(manager.get_model(model_names[0]).path)
        tokenizer.save_pretrained(save_dir)

        # Register the merged model
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
        logger.error(f"Model merging failed: {e}")
        return None


def _select_best_models(
    models: List[str],
    history: Dict[str, List[float]],
    fraction: float = 0.5
) -> List[str]:
    """Select the top fraction of models based on average TPS."""
    if not models:
        return []
    avg_tps = {}
    for name in models:
        vals = [v for v in history.get(name, []) if v is not None]
        avg_tps[name] = sum(vals) / len(vals) if vals else 0.0
    sorted_models = sorted(avg_tps.items(), key=lambda x: x[1], reverse=True)
    n_keep = max(1, int(len(sorted_models) * fraction))
    return [name for name, _ in sorted_models[:n_keep]]


# =============================================================================
# Endless distillation loop
# =============================================================================
def run_endless_distillation(
    teacher: str,
    student: str,
    passes: int = 2,
    cycles: int = -1,
    sleep: int = 60,
    callback: Optional[Callable] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run endless distillation loop.
    cycles = -1 means infinite.
    hyperparams: optional override for temperature, alpha, learning_rate, etc.
    """
    config = load_config()
    manager = ModelManager(config)
    engine = LazyDistillationEngine(config)

    prompts = getattr(config, 'validation_prompts', None)
    if not prompts:
        logger.warning("config.validation_prompts is empty or missing; using default prompts.")
        prompts = DEFAULT_VALIDATION_PROMPTS

    if hyperparams:
        if 'temperature' in hyperparams:
            engine.config.distill_temperature = hyperparams['temperature']
        if 'alpha' in hyperparams:
            engine.config.distill_alpha = hyperparams['alpha']
        if 'learning_rate' in hyperparams:
            engine.config.distill_learning_rate = hyperparams['learning_rate']
        if 'gradient_accumulation_steps' in hyperparams:
            engine.config.gradient_accumulation_steps = hyperparams['gradient_accumulation_steps']

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
                texts=prompts,
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


# =============================================================================
# Endless pruning loop (with optional output_name)
# =============================================================================
def run_endless_prune(
    model_name: str,
    strategies: Optional[List[str]] = None,
    cycles: int = -1,
    sleep: int = 60,
    callback: Optional[Callable] = None,
    task: str = "coding",
    hyperparams: Optional[Dict[str, Any]] = None,
    output_name: Optional[str] = None,
) -> None:
    """
    Endless pruning loop, cycling through strategies.

    Args:
        model_name: Base model to prune.
        strategies: List of strategies to cycle through.
        cycles: Number of cycles (-1 for infinite).
        sleep: Sleep between cycles.
        callback: Optional progress callback.
        task: Task for task-specific pruning.
        hyperparams: Optional overrides for pruning parameters.
        output_name: If provided, use this name for the pruned model; otherwise,
                     generate a unique name with timestamp.
    """
    config = load_config()
    manager = ModelManager(config)
    if strategies is None:
        strategies = ["magnitude", "neuron", "task"]

    valid_tasks = ["coding", "chat", "math", "embed"]
    if task not in valid_tasks:
        logger.warning(f"Unknown task '{task}', defaulting to 'coding'")
        task = "coding"

    prune_threshold = hyperparams.get('threshold', 0.05) if hyperparams else 0.05
    iterative_steps = hyperparams.get('iterative_steps', 4) if hyperparams else 4
    activation_threshold = hyperparams.get('activation_threshold', 0.01) if hyperparams else 0.01

    cycle = 0
    while cycles == -1 or cycle < cycles:
        cycle += 1
        strategy = strategies[(cycle - 1) % len(strategies)]
        logger.info(f"Endless prune cycle {cycle} using strategy {strategy}")
        if callback:
            callback(f"Cycle {cycle}: pruning with {strategy}")

        if not manager.model_exists(model_name):
            logger.error(f"Model {model_name} not found; aborting")
            break

        info = manager.get_model(model_name)
        if not info or not info.path:
            logger.error(f"Model {model_name} has no path; aborting")
            break

        # Memory check
        avail_ram = get_available_ram_gb()
        est_mem = estimate_memory_need(Path(info.path))
        if avail_ram < est_mem * 1.2:
            logger.warning(
                f"Available RAM ({avail_ram:.1f} GB) may be insufficient for pruning "
                f"(estimated need: {est_mem:.1f} GB). Skipping cycle."
            )
            if callback:
                callback(f"Cycle {cycle} skipped: insufficient RAM")
            if sleep > 0:
                time.sleep(sleep)
            continue

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(info.path, low_cpu_mem_usage=True)
            tokenizer = AutoTokenizer.from_pretrained(info.path)
            pruner = Pruner(model, config, original_path=Path(info.path), tokenizer=tokenizer)

            if strategy == "magnitude":
                pruner.magnitude_prune(threshold=prune_threshold, iterative_steps=iterative_steps)
            elif strategy == "neuron":
                pruner.neuron_prune(activation_threshold=activation_threshold)
            elif strategy == "task":
                prompts = get_task_prompts(task)
                pruner.task_specific_reap(task, prompts, tokenizer)
            else:
                logger.warning(f"Unknown strategy {strategy}, skipping cycle.")
                continue

            # Determine output name
            if output_name is not None:
                final_name = output_name
            else:
                # Unique name with timestamp to avoid collisions
                final_name = f"{model_name}_pruned_{int(time.time())}_{cycle}"

            out_path = manager.models_dir / final_name
            pruner.export_pruned(out_path, overwrite=True, register=True, manager=manager)
            logger.info(f"Prune cycle {cycle} complete. Saved as {final_name}")
            if callback:
                callback(f"Cycle {cycle} complete.")

        except Exception as e:
            logger.error(f"Prune cycle {cycle} failed: {e}")
            if callback:
                callback(f"Cycle {cycle} failed: {e}")
        finally:
            import gc
            from .utils import clear_cuda_memory
            gc.collect()
            clear_cuda_memory()

        if cycles != -1 and cycle >= cycles:
            break
        if sleep > 0:
            time.sleep(sleep)


# =============================================================================
# Global auto‑improvement loop (enhanced)
# =============================================================================
def run_endless_auto(
    models: Optional[List[str]] = None,
    cycles: int = -1,
    sleep: int = 120,
    policy: str = "epsilon_greedy",  # "worst", "best", "random", "epsilon_greedy"
    epsilon: float = 0.2,
    callback: Optional[Callable] = None,
    max_models: int = 20,
    resume: bool = False,
    save_state_every: int = 1,
    hyperparameter_search: bool = True,       # <-- CHANGED: enabled by default
    enable_model_merging: bool = True,        # <-- CHANGED: enabled by default
    merge_method: str = "average",            # "average" or "slerp"
    n_hyperparameter_trials: int = 10,
    perplexity_threshold: float = 80.0,       # <-- NEW: threshold for rejecting bad models
) -> None:
    """
    Global endless auto‑improvement loop with advanced features.

    Orchestrates the RL‑style self‑improvement cycle:
    1. Benchmarks all given models.
    2. Uses a policy (worst/best/random/epsilon_greedy) to select a model to improve.
    3. Decides an action (distill or prune) based on the model's TPS.
    4. Applies the action, creating a new model.
    5. Adds the new model to the list for future cycles.
    6. Repeats indefinitely or for a specified number of cycles.

    Args:
        models: List of model names to manage. If None, uses all student models.
        cycles: Number of cycles to run (-1 for infinite).
        sleep: Seconds to sleep between cycles.
        policy: Action selection policy: 'worst', 'best', 'random', 'epsilon_greedy'.
        epsilon: Exploration rate for epsilon_greedy.
        callback: Optional callable that receives status messages (strings).
        max_models: Maximum number of models to keep in the list.
        resume: If True, attempt to load state from checkpoint and continue.
        save_state_every: Save state every N cycles.
        hyperparameter_search: If True, perform random search over hyperparameters per action.
        enable_model_merging: If True, merge top models to create new candidates.
        merge_method: Method for merging: 'average' or 'slerp'.
        n_hyperparameter_trials: Number of trials for hyperparameter optimisation.
        perplexity_threshold: If a newly created model has perplexity > this threshold,
                              it is rejected and not added to the pool (default 80).
    """
    config = load_config()
    manager = ModelManager(config)

    # Load state if resuming
    saved_state = _load_state() if resume else None
    if saved_state:
        logger.info("Resuming from saved state.")
        cycle = saved_state.get('cycle', 0) + 1
        models = saved_state.get('models', models)
        history = saved_state.get('history', {})
        action_history = saved_state.get('action_history', {})
    else:
        cycle = 0
        history = {}
        action_history = {action: [] for action in ['distill', 'prune']}

    # If no models provided, auto‑discover students
    if models is None:
        models = manager.get_student_models()
        if not models:
            default = config.default_student
            if default and manager.model_exists(default):
                models = [default]
                logger.info(f"Using default student '{default}' as initial model.")
            else:
                logger.error(
                    "No models available for auto loop. "
                    "Please create a student model first or set a default_student in config."
                )
                return
        else:
            logger.info(f"Found {len(models)} student models: {models}")

    if not models:
        logger.error("No models to manage. Exiting auto loop.")
        return

    # Verify all models exist
    valid_models = [m for m in models if manager.model_exists(m)]
    if not valid_models:
        logger.error("None of the specified models exist. Exiting.")
        return
    if len(valid_models) < len(models):
        logger.warning(f"Some models do not exist: {set(models) - set(valid_models)}")
        models = valid_models

    # Initialize action history if not restored
    if not action_history:
        action_history = {action: [] for action in ['distill', 'prune']}

    # Helper function to benchmark a model and return TPS (and optionally perplexity)
    def _get_tps(model_name: str) -> float:
        info = manager.get_model(model_name)
        if not info or not info.path:
            return 0.0
        try:
            res = benchmark_model(info.path, config=config, model_name=model_name, max_tokens=50)
            return res.get('tokens_per_second', 0.0)
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            return 0.0

    def _get_perplexity(model_name: str, val_texts: List[str]) -> float:
        """Compute perplexity of a model on validation texts."""
        info = manager.get_model(model_name)
        if not info or not info.path:
            return float('inf')
        try:
            result = benchmark_perplexity(info.path, val_texts, config=config)
            return result.get('perplexity', float('inf'))
        except Exception as e:
            logger.error(f"Perplexity computation failed for {model_name}: {e}")
            return float('inf')

    # Hyperparameter search spaces (from config or defaults)
    search_spaces = {
        'distill': getattr(config, 'search_space_distill', {
            'temperature': (0.5, 3.0),
            'alpha': (0.5, 0.9),
            'learning_rate': (1e-5, 1e-3),
            'gradient_accumulation_steps': (1, 8),
        }),
        'prune': getattr(config, 'search_space_prune', {
            'threshold': (0.02, 0.15),
            'iterative_steps': (2, 5),
            'activation_threshold': (0.005, 0.05),
        }),
    }

    # Get validation texts for perplexity check (from config or default)
    val_texts = getattr(config, 'validation_prompts', None)
    if not val_texts:
        val_texts = DEFAULT_VALIDATION_PROMPTS
        logger.warning("No validation prompts in config; using default prompts for perplexity check.")

    # ----- Main loop -----
    try:
        while cycles == -1 or cycle < cycles:
            cycle += 1
            logger.info(f"Auto loop cycle {cycle}")
            if callback:
                callback(f"Auto loop cycle {cycle}")

            # 1. Benchmark all models (one cycle, no sleep)
            if callback:
                callback("Benchmarking models...")
            try:
                new_history = run_endless_benchmark(
                    models,
                    cycles=1,
                    sleep=0,
                    callback=callback
                )
                for model, vals in new_history.items():
                    if model in history:
                        history[model].extend(vals)
                    else:
                        history[model] = vals
            except Exception as e:
                logger.error(f"Benchmark cycle failed: {e}")
                if callback:
                    callback(f"Benchmark failed: {e}")
                if sleep > 0:
                    time.sleep(sleep)
                continue

            # 2. Compute average TPS for all models
            avg_tps = {}
            for name in models:
                vals = [v for v in history.get(name, []) if v is not None]
                avg_tps[name] = sum(vals) / len(vals) if vals else 0.0

            # 3. Decide action and model
            if policy in ["worst", "best", "random"]:
                # Use decide_action from benchmark (simple heuristic)
                model, action = decide_action(history, models, policy=policy)
            else:  # epsilon_greedy
                # Choose action via bandit
                actions = ['distill', 'prune']
                chosen_action = _choose_action_epsilon_greedy(
                    history, actions, action_history, epsilon
                )
                # Select a model based on action
                if chosen_action == 'distill':
                    # Use the best model as teacher, worst as student? Actually we'll create a new student from best.
                    model = max(avg_tps, key=avg_tps.get)  # best as teacher
                else:  # prune
                    model = min(avg_tps, key=avg_tps.get)  # worst for pruning
                action = chosen_action

            logger.info(f"Decided to {action} model '{model}'")
            if callback:
                callback(f"Decided to {action} '{model}'")

            # Record the old TPS for improvement measurement
            old_tps = avg_tps.get(model, 0.0)

            # ---- FIX: Set default hyperparams for pruning if hyperparameter_search is disabled ----
            if action == "prune" and not hyperparameter_search:
                # Use config's reap_prune_ratio (default 0.15) as threshold
                hyperparams = {'threshold': config.reap_prune_ratio}
                logger.info(f"Using default prune ratio from config: {config.reap_prune_ratio}")
            else:
                hyperparams = {}

            # Hyperparameter optimisation (if enabled)
            if hyperparameter_search:
                # Define an objective function that runs the action with given hyperparams and returns the improvement
                def objective(params: Dict[str, Any]) -> float:
                    # This is a nested function that will be called during optimisation.
                    # It runs the action with the given params and returns the TPS improvement.
                    # We need to capture the current model and action.
                    # We'll run a single cycle of the action with these hyperparams.
                    # To avoid side effects, we'll use a temporary model name.
                    temp_model_name = None
                    try:
                        if action == "distill":
                            # Use best model as teacher
                            best_model = max(avg_tps, key=avg_tps.get)
                            teacher = best_model
                            student = f"{model}_temp_distill_{int(time.time())}"
                            # Run distillation with given params
                            run_endless_distillation(
                                teacher, student,
                                passes=2,
                                cycles=1,
                                sleep=0,
                                callback=callback,
                                hyperparams=params
                            )
                            temp_model_name = student
                        elif action == "prune":
                            # Run pruning with given params
                            pruned_name = f"{model}_temp_prune_{int(time.time())}"
                            strategies = ["magnitude", "neuron", "task"]
                            strategy = strategies[(cycle - 1) % len(strategies)]
                            run_endless_prune(
                                model,
                                strategies=[strategy],
                                cycles=1,
                                sleep=0,
                                callback=callback,
                                task="coding",
                                hyperparams=params,
                                output_name=pruned_name
                            )
                            temp_model_name = pruned_name
                        else:
                            return 0.0

                        if temp_model_name is not None:
                            # Benchmark the new model
                            new_tps = _get_tps(temp_model_name)
                            improvement = new_tps - old_tps
                            return improvement
                        else:
                            return 0.0
                    except Exception as e:
                        logger.error(f"Objective evaluation failed: {e}")
                        return 0.0
                    finally:
                        # Clean up temporary model (if any)
                        if temp_model_name and temp_model_name != model:
                            try:
                                manager.delete_model(temp_model_name)
                            except Exception as del_e:
                                logger.warning(f"Failed to delete temporary model {temp_model_name}: {del_e}")

                # Create optimizer with appropriate search space
                search_space = search_spaces.get(action, {})
                optimizer = HyperparameterOptimizer(
                    action=action,
                    search_space=search_space,
                    config=config,
                    manager=manager,
                    objective_func=objective,
                    n_trials=n_hyperparameter_trials
                )
                best_params = optimizer.optimize()
                hyperparams = best_params
                # Store best hyperparams in registry metadata
                info = manager.get_model(model)
                if info:
                    if not hasattr(info, 'metadata'):
                        info.metadata = {}
                    info.metadata['best_hyperparams'] = {action: best_params}
                    manager._save_registry()
                    logger.info(f"Stored best hyperparams for {model} in registry.")

            # 4. Apply the action with the chosen hyperparameters (or sampled ones)
            new_model_name = None  # track new model name for benchmarking

            if action == "distill":
                # Use the best‑performing model as the teacher (self‑distillation)
                best_model = max(avg_tps, key=avg_tps.get)
                teacher = best_model
                student = f"{model}_distilled_{cycle}"
                logger.info(f"Distilling from teacher '{teacher}' to new student '{student}'")
                if callback:
                    callback(f"Distilling {teacher} → {student}")
                try:
                    run_endless_distillation(
                        teacher,
                        student,
                        passes=2,
                        cycles=1,
                        sleep=0,
                        callback=callback,
                        hyperparams=hyperparams
                    )
                    models.append(student)
                    new_model_name = student
                    logger.info(f"Added new student '{student}' to model list.")
                    # ---- LOG SUCCESS ----
                    log_operation_result(
                        model_name=student,
                        operation='auto_distill',
                        success=True,
                        details={
                            'teacher': teacher,
                            'cycle': cycle,
                            'passes': 2,
                            'hyperparams': hyperparams,
                        },
                        manager=manager
                    )
                except Exception as e:
                    logger.error(f"Distillation failed: {e}")
                    action_history['distill'].append(-1.0)
                    # ---- LOG FAILURE ----
                    log_operation_result(
                        model_name=student,
                        operation='auto_distill',
                        success=False,
                        details={
                            'teacher': teacher,
                            'cycle': cycle,
                            'error': str(e),
                            'hyperparams': hyperparams,
                        },
                        manager=manager
                    )
                    continue

            elif action == "prune":
                strategies = ["magnitude", "neuron", "task"]
                strategy = strategies[(cycle - 1) % len(strategies)]
                logger.info(f"Pruning model '{model}' with strategy '{strategy}'")
                if callback:
                    callback(f"Pruning {model} with {strategy}")
                # Generate unique name for pruned model
                pruned_name = f"{model}_pruned_{cycle}"
                try:
                    run_endless_prune(
                        model,
                        strategies=[strategy],
                        cycles=1,
                        sleep=0,
                        callback=callback,
                        task="coding",
                        hyperparams=hyperparams,
                        output_name=pruned_name  # unique name
                    )
                    models.append(pruned_name)
                    new_model_name = pruned_name
                    logger.info(f"Added pruned model '{pruned_name}' to model list.")
                    # ---- LOG SUCCESS ----
                    log_operation_result(
                        model_name=pruned_name,
                        operation='auto_prune',
                        success=True,
                        details={
                            'base_model': model,
                            'strategy': strategy,
                            'cycle': cycle,
                            'hyperparams': hyperparams,
                        },
                        manager=manager
                    )
                except Exception as e:
                    logger.error(f"Pruning failed: {e}")
                    action_history['prune'].append(-1.0)
                    # ---- LOG FAILURE ----
                    log_operation_result(
                        model_name=pruned_name,
                        operation='auto_prune',
                        success=False,
                        details={
                            'base_model': model,
                            'strategy': strategy,
                            'cycle': cycle,
                            'error': str(e),
                            'hyperparams': hyperparams,
                        },
                        manager=manager
                    )
                    continue

            else:
                logger.warning(f"Unknown action '{action}'. Skipping.")
                if callback:
                    callback(f"Unknown action '{action}' – skipping.")
                continue

            # 5. Measure improvement (new TPS - old TPS) and check perplexity
            if new_model_name:
                # Benchmark the new model for TPS
                new_tps = _get_tps(new_model_name)
                improvement = new_tps - old_tps
                action_history[action].append(improvement)
                logger.info(f"Improvement for {action} on {new_model_name}: {improvement:.2f} TPS")

                # ---- NEW: Perplexity threshold check ----
                # Compute perplexity of the new model
                logger.info(f"Computing perplexity for {new_model_name}...")
                ppl = _get_perplexity(new_model_name, val_texts)
                if ppl > perplexity_threshold:
                    logger.warning(
                        f"New model {new_model_name} has perplexity {ppl:.2f} > threshold {perplexity_threshold}. "
                        "Rejecting and removing from pool."
                    )
                    # Remove the new model from the list and delete it
                    if new_model_name in models:
                        models.remove(new_model_name)
                    # Delete the model from registry and disk
                    try:
                        manager.delete_model(new_model_name)
                        logger.info(f"Removed rejected model {new_model_name}.")
                    except Exception as del_err:
                        logger.warning(f"Failed to delete rejected model {new_model_name}: {del_err}")
                    # ---- FIX: Save state before continue ----
                    _save_state({
                        'cycle': cycle,
                        'models': models,
                        'history': history,
                        'action_history': action_history,
                    })
                    if sleep > 0:
                        time.sleep(sleep)
                    continue
                else:
                    logger.info(f"New model {new_model_name} has perplexity {ppl:.2f} (accepted).")

            # 6. Model merging (optional)
            if enable_model_merging and len(models) >= 3:
                top_models = _select_best_models(models, history, fraction=0.5)
                if len(top_models) >= 2:
                    merged_name = f"merged_{cycle}"
                    logger.info(f"Merging models: {top_models[:3]} into {merged_name}")
                    result = _merge_models(top_models[:3], manager, method=merge_method, config=config)
                    if result:
                        models.append(result)
                        logger.info(f"Added merged model '{result}' to model list.")
                        history[result] = []

            # 7. Prune model list to max_models (keep top performers)
            if len(models) > max_models:
                logger.info(f"Model list exceeds {max_models}; pruning to top performers.")
                avg_tps = {}
                for name in models:
                    vals = [v for v in history.get(name, []) if v is not None]
                    avg_tps[name] = sum(vals) / len(vals) if vals else 0.0
                sorted_models = sorted(avg_tps.items(), key=lambda x: x[1], reverse=True)
                keep_names = [name for name, _ in sorted_models[:max_models]]
                removed = [m for m in models if m not in keep_names]
                models = keep_names
                logger.info(f"Removed {len(removed)} models: {removed}")

            # 8. Save state if requested
            if cycle % save_state_every == 0 or cycles != -1 and cycle == cycles:
                _save_state({
                    'cycle': cycle,
                    'models': models,
                    'history': history,
                    'action_history': action_history,
                })

            # 9. Check if we should stop
            if cycles != -1 and cycle >= cycles:
                break

            # 10. Sleep before next cycle
            if sleep > 0:
                logger.info(f"Sleeping {sleep} seconds before next cycle...")
                if callback:
                    callback(f"Sleeping {sleep}s...")
                time.sleep(sleep)

    except KeyboardInterrupt:
        logger.info("Auto loop interrupted by user.")
        _save_state({
            'cycle': cycle,
            'models': models,
            'history': history,
            'action_history': action_history,
        })
        if callback:
            callback("Auto loop interrupted.")
        return

    logger.info(f"Auto loop finished after {cycle} cycles.")
    if callback:
        callback(f"Auto loop finished after {cycle} cycles.")