# Wishlist of Contributions and Upcomming updates to Lazy Llama:

## Overview of Additions

| Feature | Description |
|---------|-------------|
| **Full Training Pipeline** | Orchestrates distill → prune → finetune → benchmark → export in an endless loop with user‑supplied dataset. |
| **Dashboard Redesign** | Central status terminal with live logs, global progress bar; Full Training tab with sliders for all stage parameters. |
| **Optuna Auto‑tuning** | Before each stage, run an Optuna study to find optimal hyperparameters (e.g., temperature, alpha, prune ratio, learning rate) using a small internal evaluation (e.g., validation perplexity or TPS). |

---

## File‑by‑File Modification Plan

### 1. `config.py` – Add Pipeline, Dashboard, and Optuna Settings

- **New fields in `Config` dataclass**:
  - Pipeline: `full_training_enabled`, `full_training_cycles`, `full_training_sleep`, `full_training_dataset_path`, `full_training_max_samples`, `full_training_export_format`.
  - Stage overrides (for dashboard sliders): `distill_passes`, `distill_temperature`, `distill_alpha`, `prune_strategy`, `prune_ratio`, `prune_iterative_steps`, `finetune_learning_rate`, `finetune_epochs`, `finetune_batch_size`, `finetune_use_qlora`, `finetune_qlora_r`, `finetune_qlora_alpha`, `benchmark_prompt`, `benchmark_max_tokens`, `benchmark_run_perplexity`, `benchmark_run_multiple_choice`, `benchmark_run_long_context`.
  - **Optuna settings**:
    - `optuna_enabled` (bool) – global toggle.
    - `optuna_n_trials` (int, default 20).
    - `optuna_search_space_distill` – dict of parameter ranges (e.g., temperature: (0.5, 3.0), alpha: (0.5, 0.9), learning_rate: (1e-5, 1e-3)).
    - `optuna_search_space_prune` – e.g., prune_ratio: (0.05, 0.3), iterative_steps: (2, 6).
    - `optuna_search_space_finetune` – e.g., learning_rate: (1e-5, 1e-3), epochs: (1, 5), qlora_r: (4, 16).
  - `optuna_objective_metric` – e.g., "perplexity" or "tps" (used to score each trial).

- **Update `migrate()`** to add these fields with defaults when loading old configs.
- **Update `validate_all()`** to warn if Optuna is enabled but the package is not installed.

---

### 2. `utils.py` – Dataset Helpers & Optuna Helper (Optional)

- Add `load_user_dataset()` as described previously (supports Hugging Face, JSONL, CSV, plain text).
- Add a helper `get_optuna_study(storage=None, study_name=None)` that creates or loads an Optuna study (with optional persistence to SQLite) – used by the pipeline to share results across runs.

Extended Dataset & Validation Helpers:
What to Add
a) Dataset Loading Function

    Location: near the bottom, after existing utility functions.

    Purpose: Load user‑supplied datasets from various sources (Hugging Face Hub, local JSONL, CSV, plain text) into a list of text strings.

    Signature: load_user_dataset(path_or_id, split="train", max_samples=10000, text_column="text")

    Behavior:

        If path_or_id contains a / (like "user/dataset"), treat it as a Hugging Face dataset ID and use datasets.load_dataset (with graceful fallback if datasets is not installed).

        If it’s a local file path:

            If .jsonl, parse each line as JSON and extract text_column.

            If .csv, use pandas to read and extract the column.

            Otherwise, treat as plain text (one example per line).

        Truncate to max_samples and return a list of non‑empty strings.

        Log errors and return an empty list on failure.

b) Dataset Splitting Function

    Purpose: Split the loaded dataset into train and validation sets for Optuna objective evaluation.

    Signature: split_dataset(data, validation_ratio=0.1, shuffle=True)

    Behavior:

        Shuffle the list (if requested) and split into two lists: train and val.

        Return (train_texts, val_texts).

c) Helper to Store/Retrieve Best Hyperparameters (Optional)

    Purpose: Persist the best hyperparameters found by Optuna into a model’s registry metadata or a global cache.

    Signature: store_best_params(model_name, stage, params, manager=None) and get_best_params(model_name, stage, manager=None)

    These can be implemented directly in the pipeline, but adding them to utils.py keeps things organized.

What to Modify

    None – we only add new functions.


---

### 3. New Module: `hyperparameter_tuning.py` (or extend `endless_rl.py`)

- Create a module with functions to run Optuna studies for each stage:
  - `tune_distillation(teacher, student, base_texts, search_space, n_trials, objective_metric)` – runs a study where each trial runs a distillation with different hyperparams and returns the evaluation metric (e.g., perplexity on a held‑out validation set).
  - `tune_pruning(model_name, search_space, n_trials, objective_metric)` – similar for pruning.
  - `tune_finetuning(model_path, dataset, search_space, n_trials, objective_metric)` – similar for fine‑tuning.
- Each function should:
  - Define an objective function that runs the stage with given hyperparams, measures performance (e.g., validation loss, perplexity, or benchmark TPS), and returns a float (higher is better).
  - Use `optuna.create_study(direction='maximize' or 'minimize')` and `study.optimize(objective, n_trials=...)`.
  - Return the best parameters found.
- If Optuna is not installed, these functions should raise an ImportError with a clear message.

---

### 4. `endless_rl.py` – Integrate Optuna into the Full Training Pipeline

- Modify `run_full_training_pipeline` to check `config.optuna_enabled` before each stage:
  - If enabled, call the appropriate tuning function (e.g., `tune_distillation`) using the current model and the dataset/validation texts.
  - Use the best hyperparameters returned for that stage, then run the stage normally (or optionally run it with those best params).
  - Log the best parameters and store them in the registry metadata for traceability.
- The pipeline should still accept manual overrides (from dashboard sliders) – if Optuna is enabled, the sliders might be ignored or used as bounds for the search space.
- Ensure that the tuning process itself reports progress to the callback (so dashboard shows "Tuning distillation..." etc.).

---

### 5. `bootstrap.py` – CLI Flags for Auto‑tuning

- Add arguments to the `full-train` subparser:
  - `--tune` (store_true) to enable Optuna.
  - `--trials` (int, default 20) to set number of trials.
  - `--objective` (choices: 'perplexity', 'tps') to choose the metric.
- Also add per‑stage override flags (e.g., `--distill-passes`) as before.
- In dispatch, pass these to `run_full_training_pipeline`.

---

### 6. `lazy_tui.py` – TUI Menu for Auto‑tuning

- Extend the Full Training menu to ask:
  - "Enable hyperparameter tuning? (y/n)"
  - "Number of trials?"
  - "Objective metric (perplexity/tps)?"
- Pass these options to the background pipeline.

---

### 7. `dashboard_server.py` – UI Controls for Optuna & Status Terminal

#### Dashboard Redesign (Replaces Graph)
- **Remove** the Chart.js graph and canvas.
- **Add** a central **Status Terminal** panel:
  - A scrollable `<div>` that shows live log messages (from the pipeline callback).
  - A **global progress bar** (0‑100%) at the top of this panel (or in the header) that reflects overall pipeline progress (e.g., 20% for distillation, 40% for pruning, etc.).
  - The terminal auto‑scrolls to the bottom.
- **Keep** the metrics header (RAM, CPU, E8, LazyTorch) – update via `/api/metrics`.

#### Full Training Tab (New or re‑purposed)
- Add a tab/panel called **"Full Training"** with:
  - Dataset upload (file input + HF ID text field).
  - **Optuna section**:
    - Checkbox "Enable auto‑hyperparameter tuning".
    - Number input for "Trials".
    - Dropdown for "Objective metric".
  - **Stage sliders** (enabled/disabled based on tuning):
    - Distillation: passes, temperature, alpha.
    - Pruning: strategy dropdown, prune ratio, iterative steps.
    - Finetuning: epochs, learning rate, batch size, QLoRA toggles.
    - Benchmark: prompt, max tokens, checkboxes.
    - Export: format dropdown.
    - Loop: cycles (number or infinite), sleep seconds.
  - **Start / Stop** buttons.
  - The status terminal and progress bar (already present) update in real time.

#### New API Endpoints
- `POST /api/full-training/start` – accepts a full settings payload (including optuna flags) and starts the pipeline.
- `GET /api/full-training/status` – returns progress, current stage, logs, and the best hyperparameters found so far.
- `POST /api/full-training/stop` – gracefully stops the loop.
- `POST /api/full-training/upload-dataset` – handles file upload.

#### JavaScript Changes
- On page load, fetch current config to populate sliders.
- When Start is clicked, package all settings (including optuna) into a JSON payload and POST to `/api/full-training/start`.
- Poll `/api/full-training/status` every 1‑2 seconds:
  - Update progress bar.
  - Append new log entries to the terminal.
  - If best params are available, display them in a small info box.
- Handle Stop button to send a POST to `/api/full-training/stop`.

---

### 8. Background Thread in `dashboard_server.py` – Pipeline Runner

- Add a global state dict `_full_training_state` (protected by a lock) with fields:
  - `running`, `progress` (0‑100), `stage` (string), `cycle`, `logs` (list of recent messages), `best_params` (dict), `error`.
- Create a thread function `_run_full_training_thread` that:
  - Parses the request settings.
  - Calls `run_full_training_pipeline` with a callback that updates the state (stage, progress, logs, best_params).
  - On completion, sets `running = False`.
- The `/api/full-training/status` endpoint returns this state.

---

### 9. `finetune.py` – Dedicated Fine‑Tuning Module (New)

- Create a function `finetune_model(model_path, dataset, output_path, **kwargs)` that:
  - Loads model and tokenizer.
  - Applies QLoRA if requested.
  - Prepares dataset (tokenization, formatting).
  - Runs training loop (or uses Hugging Face Trainer).
  - Saves model and updates registry.
- This function is used by the pipeline after pruning.
- It should also accept an optional `validation_dataset` to compute validation loss (used by Optuna objective for fine‑tuning).

---


## Summary of Ideas by File

| File | Actions |
|------|---------|
| `config.py` | Add pipeline, stage overrides, and Optuna settings. |
| `utils.py` | Add dataset loader; optionally add Optuna helper. |
| `hyperparameter_tuning.py` | New file with tuning functions for each stage. |
| `endless_rl.py` | Integrate Optuna calls into `run_full_training_pipeline`; accept tuning flags. |
| `bootstrap.py` | Add CLI args for tuning. |
| `lazy_tui.py` | Add tuning questions to Full Training menu. |
| `dashboard_server.py` | Redesign UI (remove graph, add terminal + progress bar); add Full Training tab with sliders and optuna controls; add API endpoints for start/stop/status/upload. |
| `finetune.py` | New file: fine‑tuning logic with validation split support. |
| `dataset_utils.py` | (Optional) separate dataset loading and splitting. |
| `lazy_distill.py` | (Optional) expose more parameters. |
| `lazy_prune.py` | (Optional) expose more parameters. |

---

# UPDATED 7/18/2026

``` markdown
# Lazy Llama v4.0 Enhancement Guide: Full Training Pipeline + Dashboard Redesign

This document provides a comprehensive, file-by-file implementation plan to upgrade **Lazy Llama v3.6** to **v4.0**. It adds:

- A **full training pipeline** (`distill → prune → finetune → benchmark → export`) with **endless looping** and a user-provided dataset.
- **Optuna hyperparameter tuning** for each pipeline stage.
- **GRPO-style fine-tuning** (reward modeling) and **Micro MoE** integration.
- A **redesigned dashboard** with a live status terminal, global progress bar, and a new **Full Training** tab with sliders and Optuna controls.
- Dataset loading from Hugging Face Hub, JSONL, CSV, and plain text.
- CLI support for the pipeline with tuning flags.

---

## Prerequisites

Install additional Python packages:

```bash
pip install optuna datasets trl
```

If you want GRPO, `trl` is required; for MoE, the existing `micro_moe.py` is used.

---

## Overview of Changes

| File | Actions |
|------|---------|
| `config.py` | Add pipeline, stage override, and Optuna settings. |
| `utils.py` | Add dataset loading and splitting helpers. |
| `hyperparameter_tuning.py` | New file: Optuna objective functions for each stage. |
| `finetune.py` | New file: QLoRA/GRPO/MoE fine-tuning logic. |
| `endless_rl.py` | Add `run_full_training_pipeline` with Optuna integration. |
| `lazy_tui.py` | Add Full Training menu with tuning options. |
| `dashboard_server.py` | Redesign UI (remove graph, add terminal + progress bar), add Full Training tab, API endpoints for start/stop/status/upload. |
| `bootstrap.py` | Add `full-train` CLI command with flags. |

---

## File-by-File Implementation Guide

### 1. `config.py`

**Add** the following fields to the `Config` dataclass (after existing fields):

```python
# ---- Full Training Pipeline ----
full_training_enabled: bool = False
full_training_cycles: int = -1          # -1 = infinite
full_training_sleep: int = 60
full_training_dataset_path: Optional[str] = None
full_training_max_samples: int = 10000
full_training_export_format: str = "lazytorch"  # "lazytorch", "gguf", "hf"

# ---- Stage Overrides (used by sliders) ----
distill_passes: int = 3
distill_temperature: float = 2.0
distill_alpha: float = 0.8
prune_strategy: str = "magnitude"       # "magnitude", "neuron", "task"
prune_ratio: float = 0.15
prune_iterative_steps: int = 6
finetune_learning_rate: float = 1e-4
finetune_epochs: int = 3
finetune_batch_size: int = 4
finetune_use_qlora: bool = True
finetune_qlora_r: int = 16
finetune_qlora_alpha: int = 32
benchmark_prompt: str = "What is machine learning?"
benchmark_max_tokens: int = 100
benchmark_run_perplexity: bool = True
benchmark_run_multiple_choice: bool = True
benchmark_run_long_context: bool = False

# ---- Optuna Settings ----
optuna_enabled: bool = False
optuna_n_trials: int = 20
optuna_search_space_distill: Dict[str, Tuple] = field(default_factory=lambda: {
    "temperature": (0.5, 3.0),
    "alpha": (0.5, 0.9),
    "learning_rate": (1e-5, 1e-3),
})
optuna_search_space_prune: Dict[str, Tuple] = field(default_factory=lambda: {
    "prune_ratio": (0.05, 0.3),
    "iterative_steps": (2, 6),
})
optuna_search_space_finetune: Dict[str, Tuple] = field(default_factory=lambda: {
    "learning_rate": (1e-5, 1e-3),
    "epochs": (1, 5),
    "qlora_r": (4, 16),
})
optuna_objective_metric: str = "perplexity"  # "perplexity" or "tps"
```

**Update `migrate()`** to add these fields with defaults when loading old configs.

**Update `validate_all()`** to warn if `optuna_enabled` is True but `optuna` is not installed, and to validate range of pipeline settings.

---

### 2. `utils.py`

**Add** the following functions at the end of the file:

- `load_user_dataset(path_or_id, split="train", max_samples=10000, text_column="text")`:
  - If path_or_id contains `/` and is not a local file, use `datasets.load_dataset`.
  - If it ends with `.jsonl`, parse each line as JSON and extract `text_column`.
  - If `.csv`, use pandas to read the column.
  - Otherwise, treat as plain text (one example per line).
  - Return a list of non-empty strings, truncated to `max_samples`.

- `split_dataset(data, validation_ratio=0.1, shuffle=True)`:
  - Shuffle (optional), split into train and validation lists.

**Pseudo‑code** (already provided in the original plan).

---

### 3. New File: `hyperparameter_tuning.py`

Create this file with three main functions:

- `tune_distillation(teacher, student, texts, search_space, n_trials, objective_metric, config, manager)`:
  - Uses Optuna to sample hyperparameters (temperature, alpha, learning_rate).
  - For each trial, copies the student model to a temporary name, runs distillation with the sampled params on training split, evaluates on validation split (perplexity or TPS), returns the score.
  - Cleans up the temporary model.
  - Returns the best params found.

- `tune_pruning(model_name, search_space, n_trials, objective_metric, config, manager)`:
  - Similar, but runs pruning with sampled `prune_ratio`, `iterative_steps`, etc.
  - Uses the existing `Pruner` class and exports to a temporary model.

- `tune_finetuning(model_name, dataset, search_space, n_trials, objective_metric, config, manager)`:
  - Uses the new `finetune_model` function (see below) to train with sampled hyperparameters, evaluates on validation set.

All functions should use `optuna.create_study(direction='minimize' or 'maximize')` and `study.optimize(objective, n_trials)`.

---

### 4. New File: `finetune.py`

Implement `finetune_model` with the following signature:

```python
def finetune_model(
    model_path: Union[str, Path],
    train_texts: List[str],
    output_path: Optional[Union[str, Path]] = None,
    val_texts: Optional[List[str]] = None,
    learning_rate: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 4,
    use_qlora: bool = True,
    qlora_r: int = 16,
    qlora_alpha: int = 32,
    use_grpo: bool = False,
    grpo_epochs: int = 1,
    use_moe: bool = False,
    moe_num_experts: int = 4,
    moe_top_k: int = 1,
    config: Optional[Config] = None,
) -> Path
```

**Steps:**

1. Load tokenizer and model (with QLoRA if `use_qlora` is True).
2. Prepare dataset (tokenize, create `Dataset` objects).
3. If `use_grpo` and `trl` is available, use `GRPOTrainer` with reward modeling (you'll need to define a reward function based on validation set or a simple metric).
4. Otherwise, use standard `Trainer` with `DataCollatorForLanguageModeling`.
5. Train for the specified number of epochs.
6. If `use_moe` is True, call `convert_dense_to_micro_moe` on the model before saving.
7. Save the model and tokenizer to `output_path`.
8. Register the model in the registry (optionally).

**Note:** GRPO implementation requires a reward model or a reward function. For simplicity, you can use a simple heuristic (e.g., perplexity improvement) or skip GRPO in the first version.

---

### 5. `endless_rl.py`

**Add** a new function `run_full_training_pipeline`:

```python
def run_full_training_pipeline(
    config: Config,
    dataset_path: str,
    teacher: Optional[str] = None,
    student: Optional[str] = None,
    callback: Optional[Callable] = None
) -> None:
    """
    Orchestrates distill -> prune -> finetune -> benchmark -> export in a loop.
    Uses Optuna if config.optuna_enabled is True.
    """
```

**Implementation outline:**

1. Load dataset using `load_user_dataset`.
2. Split into train and validation sets.
3. Use global teacher/student from config if not provided.
4. Loop for `config.full_training_cycles` (or infinite if -1):
   - **Distillation stage**:
     - If `optuna_enabled`, call `tune_distillation` with the current student and teacher, get best params.
     - Run distillation using those params (or manual overrides) via `LazyDistillationEngine`.
     - The new student becomes the current student for subsequent stages.
   - **Pruning stage**:
     - If `optuna_enabled`, call `tune_pruning` on the current student.
     - Run pruning with best params (or manual overrides).
     - The pruned model becomes the new current student.
   - **Fine-tuning stage**:
     - If `optuna_enabled`, call `tune_finetuning` on the current student using the training dataset.
     - Run `finetune_model` with best params (or manual overrides).
     - The finetuned model becomes the new current student.
   - **Benchmark stage**:
     - Run `benchmark_model` or `benchmark_student_models` on the current student.
     - Store results in registry.
   - **Export stage**:
     - Export the current student to the specified format (`config.full_training_export_format`).
   - Update progress via callback.
   - Sleep for `config.full_training_sleep` seconds before next cycle.

**Callback** should receive `stage`, `progress`, `message`, `best_params` (if any) so the dashboard can update the terminal and progress bar.

---

### 6. `lazy_tui.py`

**Add** a new menu option (e.g., "[T] Full Training") that:

- Asks for dataset path (or uses config default).
- Asks for teacher/student if not set.
- Asks for number of cycles (or infinite).
- Asks whether to enable Optuna tuning and number of trials.
- Starts the pipeline in a background thread (or uses `run_full_training_pipeline` with a callback that updates the UI).
- Displays a live progress log.

---

### 7. `dashboard_server.py` (Major Redesign)

#### a. HTML/UI Changes

- **Remove** the Chart.js graph canvas (`<canvas id="tpsChart">`) and related code.
- **Add** a new **Status Terminal** panel:
  - A scrollable `<div id="terminal">` that displays log messages.
  - A **global progress bar** (`<div id="global-progress">`) at the top of the panel.
- **Modify** the tabs: add a new tab called **"Full Training"** (or rename an existing one).
- **Full Training Tab** contains:
  - Dataset input (file upload + HF ID text field).
  - **Optuna section**: checkbox "Enable auto‑hyperparameter tuning", number input for "Trials", dropdown for "Objective metric".
  - **Stage sliders** (all the overrides from config):
    - Distillation: passes, temperature, alpha.
    - Pruning: strategy dropdown, prune ratio, iterative steps.
    - Finetuning: learning rate, epochs, batch size, QLoRA toggles (r, alpha).
    - Benchmark: prompt, max tokens, checkboxes for perplexity, MC, long-context.
    - Export: format dropdown.
    - Loop: cycles (number or infinite), sleep seconds.
  - **Start / Stop** buttons.
  - The status terminal (already present) updates in real time.

#### b. New API Endpoints

- `POST /api/full-training/start` – accepts a JSON payload with all settings (dataset path, optuna flags, stage overrides, loop settings). Starts the pipeline in a background thread.
- `GET /api/full-training/status` – returns current state: `running`, `progress` (0-100), `stage` (string), `cycle`, `logs` (list of recent messages), `best_params` (dict), `error`.
- `POST /api/full-training/stop` – gracefully stops the loop (sets a stop flag).
- `POST /api/full-training/upload-dataset` – handles file upload (multipart/form-data) and saves the file; returns the path.

#### c. Background Thread State

Create a global dict `_full_training_state` with a lock:

```python
_full_training_state = {
    "running": False,
    "progress": 0,
    "stage": "",
    "cycle": 0,
    "logs": [],
    "best_params": {},
    "error": None,
}
```

In the thread function `_run_full_training_thread`, call `run_full_training_pipeline` with a callback that updates this state. The callback should:

- Update `stage` and `progress`.
- Append log messages to `logs` (keep last 500 lines).
- Update `best_params` when new ones are found.

#### d. JavaScript Updates

- On page load, fetch current config (or use default values) to pre‑fill sliders.
- When Start is clicked, gather all settings (including optuna flags) and send a POST to `/api/full-training/start`.
- Poll `/api/full-training/status` every 1‑2 seconds to update the progress bar, terminal, and best params display.
- Handle Stop button to send a POST to `/api/full-training/stop`.
- Upload dataset via a separate POST with `FormData`.

---

### 8. `bootstrap.py`

**Add** a new subparser for `full-train`:

```bash
python -m lazy_llama.bootstrap full-train --dataset /path/to/data --teacher llama2 --student my_student --cycles 5 --tune --trials 10 --objective tps
```

Arguments:

- `--dataset` (required): path or HF ID.
- `--teacher`, `--student`: optional, uses global defaults if not set.
- `--cycles`: number of cycles (default -1 = infinite).
- `--sleep`: sleep between cycles.
- `--tune`: enable Optuna.
- `--trials`: number of trials (default 20).
- `--objective`: 'perplexity' or 'tps'.
- Plus all stage override flags (e.g., `--distill-passes`, `--prune-ratio`, `--finetune-epochs`, etc.).

In the dispatch, call `run_full_training_pipeline` with the parsed arguments.

---

## Integration Checklist

1. **Install dependencies**: `optuna`, `datasets`, `trl` (optional).
2. **Update config** with new fields.
3. **Add dataset helpers** to `utils.py`.
4. **Create `hyperparameter_tuning.py`** with tuning functions.
5. **Create `finetune.py`** with fine‑tuning logic (QLoRA, GRPO, MoE).
6. **Update `endless_rl.py`** with `run_full_training_pipeline`.
7. **Update `lazy_tui.py`** with Full Training menu.
8. **Overhaul `dashboard_server.py`**: redesign UI, add Full Training tab, new API endpoints, and background thread.
9. **Update `bootstrap.py`** with `full-train` command.
10. **Test end‑to‑end**:
    - Load a dataset (e.g., `"wikitext-2"`).
    - Run pipeline with Optuna enabled.
    - Monitor dashboard terminal and progress bar.
    - Verify that models are saved and registered correctly.

---

## Notes & Caveats

- **Optuna Trials**: Each trial creates and deletes a temporary model. Ensure you have enough disk space and that the manager's copy/hardlink is efficient.
- **GRPO**: The `trl` library is still experimental; you may want to implement a simple reward function (e.g., based on validation perplexity improvement) rather than a full GRPO trainer.
- **MoE**: The existing `micro_moe.py` works, but fine‑tuning with MoE may require additional gradient checkpointing to avoid OOM.
- **Dashboard terminal**: Keep logs in memory (limit to ~500 lines) and poll efficiently to avoid performance issues.
- **Dataset upload**: For security, validate file types and size (e.g., limit to 100 MB).

---

This guide provides a complete roadmap for adding powerful self‑improvement capabilities to Lazy Llama v4.0. Proceed file by file, and test each component as you go. Good luck!```
