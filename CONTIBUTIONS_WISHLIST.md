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

