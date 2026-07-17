# lazy_tui.py
"""Rich TUI with full menu, resource monitoring, dashboard launcher, export, system status, and LazyTorch integration.
   Unified model selection: global teacher/student from dashboard state are primary.
   All operations default to global selections; manual override allowed.
   Added: Select Models menu to set global teacher/student from TUI.
   Added: Refresh globals option.
   Fixed: Distill, Prune, Benchmark, Export now use global selections as defaults.
   Styling: Black/white (monochrome) theme via Rich theme mapping.
   FIXED: Model lists and global selectors now force registry/Ollama refresh before displaying.
   FIXED: Global teacher/student are reloaded from disk before each menu render.
   FIXED: Model dropdowns in select_global_models include both Ollama and local models with type labels.
   FIXED: After distill, prune, import, rename, model lists are refreshed.
   FIXED: _refresh_global_state() only prints when global values change, reducing console noise.
   FIXED: _get_engine() handles missing model_info and missing resolve_model gracefully.
   FIXED: Added time-based cache for Ollama sync to reduce frequent calls.

   NEW: _rename_model now updates global teacher/student if the renamed model was selected.
   FIX: _rename_model now explicitly calls _refresh_global_state(force_sync=True) to reload registry.
   FIX: _refresh_global_state always calls reload_registry() to keep in-memory state fresh.
   FIX: _import_model_zip now uses ModelManager.validate_model_directory() for consistent validation.

   NEW: Added `ollama_timeout` setting in the Settings menu and status display.
   NEW: In Models menu, added options to validate and delete models directly.

   NEW: Model list now shows validation status (✓/✗) and includes invalid models.
   NEW: Added "validate-all" action to re‑validate all local models and update their invalid flag.

   ============================================================================
   ADDITIONAL IMPROVEMENTS (2026-07-11):
   - Startup model selection now filters out the "use existing" option when no local models exist.
   - When creating a student, checks for duplicate student name and prompts to overwrite or choose a new name.
   - Enhanced error handling for download failures with actionable messages.
   - Added explicit validation for vLLM engine availability.
   - Wrapped default student download in try/except to avoid aborting on single failure.

   ============================================================================
   FIXES (2026-07-12):
   - Added early tokenizer validation using `_validate_tokenizer_deep` before any operation
     that loads a model (distill, prune, student creation). This prevents corrupt models
     from being used and provides clear error messages.
   - In `_distill`, validate the student's tokenizer before starting distillation.
   - In `_select_models_at_startup`, after downloading a base model, validate its tokenizer
     before creating a student.
   - In `_prune`, tokenizer validation already existed; kept as is.
   - Ensured `_refresh_global_state` is called after all registry-modifying operations
     (create, delete, rename, import) to keep global selectors consistent.
   - Added `_validate_tokenizer_for_model()` helper to reduce duplication.

   ============================================================================
   NEW (2026-07-14): REAP Pipeline Status banner and slash commands.
   - Added REAP Pipeline Checklist display in main menu header.
   - Added slash command support in main menu (e.g., /chat, /status, /export).
   - Integrated quick chat command for single-turn interaction.
   - Cleaned up runtime warnings by using `python -m lazy_llama.bootstrap` as recommended.

   ============================================================================
   FIX (2026-07-14): Resolved circular import with bootstrap.py by moving the
   import of `run_reap_pipeline_checklist` inside the `_main_menu()` method.
   ============================================================================
   FIX (2026-07-15): TUI input handling - strip whitespace, log invalid choices.

   ============================================================================
   NEW (2026-07-16): Benchmark Settings and Reports.
   - Added Benchmark Settings menu (`[B]`) to configure prompt, max_tokens, perplexity, MC, long-context.
   - Added View Reports menu (`[V]`) to show historical benchmark results from registry.
   - `_benchmark_students` now uses the current settings and displays a progress bar.
   - Benchmark settings are persisted to `~/.lazy_llama/benchmark_settings.json`.
   - Compatible with updated benchmark.py (progress_callback, long_context_max_tokens).

   REMOVED (2026-07-17): Removed all HEPA-related code, including HEPA menu items
   and imports. HEPA has been removed from the project.
"""

import time
import webbrowser
import torch
import requests
import gc
import shutil
import json
import logging
import numpy as np
import socket
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.theme import Theme

# ---- All internal imports are now RELATIVE ----
from .config import load_config, Config, recommend_enhancements, auto_optimize_config, save_config, LAZY_DIR
from .utils import (
    get_available_ram_gb, get_total_ram_gb, clear_cuda_memory, export_to_ollama,
    estimate_memory_need, is_lazytorch_model, get_lazytorch_model_size,
    _validate_tokenizer_deep, detect_platform
)
from .lazy_model_manager import ModelManager
from .lazy_infer import (
    LazyGGUFEngine,
    OllamaInferenceEngine,
    TransformersInferenceEngine,
    LazyTorchEngine,
    VLLMEngine,
    create_engine,
)
from .metrics_store import MetricsStore
from .benchmark import (
    benchmark_model,
    BenchmarkSettings,
    benchmark_student_models,
    format_benchmark_summary,
)
from .lazy_prune import get_task_prompts

# ---- REMOVED: from .bootstrap import run_reap_pipeline_checklist (causes circular import) ----

# ---- LOGGER DEFINED EARLY ----
logger = logging.getLogger(__name__)

# ---- Endless RL (optional, relative imports) ----
ENDLESS_AVAILABLE = False
try:
    from .endless_rl import (
        run_endless_distillation,
        run_endless_prune,
        run_endless_auto,
    )
    ENDLESS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Endless RL modules not available: {e}")

# Full ASCII logo with updated version (unchanged)
LAZY_LOGO = r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║     ██╗      █████╗ ███████╗██╗   ██╗    ██╗     ██╗      █████╗ ███╗   ███╗ █████╗  ║
║     ██║     ██╔══██╗╚══███╔╝╚██╗ ██╔╝    ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗ ║
║     ██║     ███████║  ███╔╝  ╚████╔╝     ██║     ██║     ███████║██╔████╔██║███████║ ║
║     ██║     ██╔══██║ ███╔╝    ╚██╔╝      ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║ ║
║     ███████╗██║  ██║███████╗   ██║       ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║ ║
║     ╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝       ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ║
║                                                                                      ║
║              L      A      Z      Y       L      L      A      M      A              ║
║         ╔══════════════════════════════════════════════════════════════╗             ║
║         ║  Low-End Inference Engine - Trade Time for Accuracy          ║             ║
║         ║                    v3.6 - LazyTorch + E8 + KV Compression    ║             ║
║         ║          + True Lazy Loading + Checkpoint Resume             ║             ║
║         ║                  + Multi-OS Support (Linux/macOS/Windows)    ║             ║
║         ║                    + Endless RL Self‑Improvement             ║             ║
║         ║              + Benchmark Settings & Reports                 ║             ║
║         ╚══════════════════════════════════════════════════════════════╝             ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

# Path to global state file (same as dashboard)
GLOBAL_STATE_FILE = LAZY_DIR / "global_state.json"
BENCHMARK_SETTINGS_FILE = LAZY_DIR / "benchmark_settings.json"
_LAST_SYNC_TIME = 0.0
_SYNC_INTERVAL = 5.0  # seconds between forced syncs

# ---- Endless loop state ----
_endless_threads = {}  # name -> (thread, stop_flag)
_endless_lock = threading.Lock()

# ---- Curated list of lightweight models suitable for low-end devices ----
# Each entry: (Hugging Face ID, description/estimated size)
# This can be overridden by config.suggested_student_models if present.
DEFAULT_SUGGESTED_STUDENT_MODELS = [
    ("distilgpt2", "DistilGPT2, 82M parameters, ~300MB"),
    ("gpt2", "GPT-2 small, 124M parameters, ~500MB"),
    ("facebook/opt-125m", "OPT-125M, 125M parameters, ~500MB"),
    ("EleutherAI/gpt-neo-125m", "GPT-Neo 125M, ~500MB"),
    ("microsoft/phi-2", "Phi-2, 2.7B parameters, ~1.6GB (larger but efficient)"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B, ~2.2GB (good for chat)"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B, ~1GB"),
]


def _load_global_state():
    """Read global teacher/student selections saved by dashboard."""
    if GLOBAL_STATE_FILE.exists():
        try:
            with open(GLOBAL_STATE_FILE) as f:
                data = json.load(f)
                return data.get("teacher", ""), data.get("student", "")
        except Exception:
            pass
    return "", ""


def _save_global_state(teacher, student):
    """Persist global selections to disk (same format as dashboard)."""
    with open(GLOBAL_STATE_FILE, "w") as f:
        json.dump({"teacher": teacher, "student": student}, f)


class LazyTUI:
    def __init__(self):
        # ---- MONOCHROME THEME ----
        monochrome_theme = Theme({
            "red": "white",
            "green": "white",
            "yellow": "white",
            "blue": "white",
            "magenta": "white",
            "cyan": "white",
            "dim": "dim white",
            "bold": "bold white",
            "default": "white",
            "bold cyan": "bold white",
            "bold green": "bold white",
            "bold magenta": "bold white",
            "bold yellow": "bold white",
            "bold blue": "bold white",
            "bold red": "bold white",
            "progress.description": "white",
            "progress.percentage": "white",
            "progress.download": "white",
        })
        self.console = Console(theme=monochrome_theme)
        # ---------------------------------

        self.config = load_config()
        self.model_manager = ModelManager()
        # Ensure default student is downloaded if configured
        self.model_manager.ensure_default_student()
        self.current_model = None
        self.running = True
        self.metrics_store = MetricsStore()
        self.dashboard_server = None
        self.command_aliases = {
            "/exit": "/exit",
            "/q": "/exit",
            "/quit": "/exit",
            "/clear": "/clear",
            "/dashboard": "/dashboard",
            "/help": "/help",
            "/status": "/status",
            "/export": "/export",
        }
        # Load global student/teacher from dashboard state
        self.global_student = None
        self.global_teacher = None
        self._prev_global_student = None
        self._prev_global_teacher = None
        self._refresh_global_state()

        # ---- Load benchmark settings ----
        self.benchmark_settings = self._load_benchmark_settings()

        # Cache for student engine (to avoid reloading)
        self._student_engine = None

    # ------------------------------------------------------------------
    # Helper for "Press Enter" waits – non‑blocking if no TTY
    # ------------------------------------------------------------------
    def _wait_for_enter(self, prompt: str = "Press Enter to continue..."):
        """Wait for user input, but if not in a terminal, skip immediately."""
        if sys.stdin.isatty():
            input(prompt)
        else:
            # Non-interactive: just print and continue
            self.console.print(f"[dim]{prompt}[/dim]")

    # ------------------------------------------------------------------
    # Helper for Prompt.ask with non-interactive fallback
    # ------------------------------------------------------------------
    def _safe_prompt(self, prompt: str, default: str = "", choices: Optional[List[str]] = None, **kwargs):
        """Wrap Prompt.ask to handle EOFError/KeyboardInterrupt gracefully."""
        try:
            if choices is not None:
                result = Prompt.ask(prompt, choices=choices, default=default, **kwargs)
            else:
                result = Prompt.ask(prompt, default=default, **kwargs)
            return result.strip()  # <-- FIX: strip whitespace
        except EOFError:
            # Non-interactive: use default
            if default:
                self.console.print(f"[dim]Using default: {default}[/dim]")
                return default.strip() if default else ""
            else:
                self.console.print("[red]No input available and no default. Exiting.[/red]")
                sys.exit(1)
        except KeyboardInterrupt:
            self.console.print("\n[red]Interrupted.[/red]")
            sys.exit(0)

    def _safe_confirm(self, prompt: str, default: bool = False) -> bool:
        """Wrap Confirm.ask to handle non-interactive."""
        try:
            return Confirm.ask(prompt, default=default)
        except EOFError:
            return default
        except KeyboardInterrupt:
            self.console.print("\n[red]Interrupted.[/red]")
            sys.exit(0)

    # ------------------------------------------------------------------
    # Benchmark settings persistence
    # ------------------------------------------------------------------
    def _load_benchmark_settings(self) -> BenchmarkSettings:
        """Load benchmark settings from disk, or return defaults."""
        if BENCHMARK_SETTINGS_FILE.exists():
            try:
                with open(BENCHMARK_SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                # Ensure the settings object has the new field if missing
                if "long_context_max_tokens" not in data:
                    data["long_context_max_tokens"] = 20
                return BenchmarkSettings(**data)
            except Exception as e:
                logger.warning(f"Failed to load benchmark settings: {e}")
        return BenchmarkSettings()

    def _save_benchmark_settings(self):
        """Save current benchmark settings to disk."""
        try:
            with open(BENCHMARK_SETTINGS_FILE, "w") as f:
                json.dump(asdict(self.benchmark_settings), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save benchmark settings: {e}")

    # ------------------------------------------------------------------
    # Global state refresh with validation
    # ------------------------------------------------------------------
    def _refresh_global_state(self, force_sync: bool = False):
        """Update global student/teacher from persisted state, and sync Ollama.
           Only prints when values change to reduce console noise.
           If force_sync is False, uses a time‑based cache to avoid too‑frequent syncs.
           Always reloads the registry to keep in‑memory state fresh.
           Additionally, validate that the loaded global models exist and are valid;
           if not, clear them and save the corrected state.
        """
        global _LAST_SYNC_TIME
        # Only sync if forced or if enough time has passed
        if force_sync or (time.time() - _LAST_SYNC_TIME) > _SYNC_INTERVAL:
            self.model_manager.sync_ollama()
            self.model_manager.reload_registry(sync_ollama=False)  # avoid extra sync
            _LAST_SYNC_TIME = time.time()
        else:
            # Still reload registry from disk (fast) without syncing Ollama again
            self.model_manager.reload_registry(sync_ollama=False)

        # Load from file
        teacher, student = _load_global_state()

        # Validate: check if they exist in registry and are valid
        if teacher and not self.model_manager.model_exists(teacher):
            logger.debug(f"Global teacher '{teacher}' is invalid; clearing.")
            teacher = ""
        if student and not self.model_manager.model_exists(student):
            logger.debug(f"Global student '{student}' is invalid; clearing.")
            student = ""

        # If we cleared any, save the corrected state
        original_teacher, original_student = _load_global_state()
        if teacher != original_teacher or student != original_student:
            _save_global_state(teacher, student)

        # Set instance variables
        self.global_teacher = teacher
        self.global_student = student

        # Check if values changed (for printing)
        changed = False
        if self.global_teacher != self._prev_global_teacher:
            changed = True
            self._prev_global_teacher = self.global_teacher
        if self.global_student != self._prev_global_student:
            changed = True
            self._prev_global_student = self.global_student

        # Only print if changed
        if changed:
            if self.global_student:
                self.console.print(f"[dim]Global student: {self.global_student}[/dim]")
            if self.global_teacher:
                self.console.print(f"[dim]Global teacher: {self.global_teacher}[/dim]")

        # Invalidate cached engine if student changed
        if changed and self.global_student:
            self._student_engine = None

    def _save_global_state(self):
        """Save current global selections to file."""
        _save_global_state(self.global_teacher, self.global_student)

    def _cleanup(self):
        # Only stop the server if we actually started it (i.e., it's not the sentinel True)
        if self.dashboard_server is not None and self.dashboard_server is not True:
            try:
                self.dashboard_server.stop()
            except Exception:
                pass
        if torch.cuda.is_available():
            clear_cuda_memory()
        # Stop any endless loops
        self._stop_all_endless_loops()

    # ------------------------------------------------------------------
    # Endless loops management (improved cancellation)
    # ------------------------------------------------------------------
    def _stop_all_endless_loops(self):
        """Signal all running endless loops to stop and wait for them."""
        with _endless_lock:
            for name, (thread, stop_flag) in list(_endless_threads.items()):
                stop_flag.set()
            # Wait a bit for threads to finish
            time.sleep(0.5)
            # Clear the dict
            _endless_threads.clear()

    def _start_endless_loop(self, name: str, target, args=()):
        """Start an endless loop thread with a stop flag."""
        stop_flag = threading.Event()

        def wrapper():
            try:
                target(*args, stop_flag=stop_flag)
            except KeyboardInterrupt:
                logger.info(f"Endless loop '{name}' interrupted by user.")
                self.console.print(f"[yellow]Endless loop '{name}' stopped.[/yellow]")
            except Exception as e:
                logger.exception(f"Endless loop '{name}' failed: {e}")
                self.console.print(f"[red]Endless loop '{name}' failed: {e}[/red]")
            finally:
                # Clean up the thread from the dict
                with _endless_lock:
                    _endless_threads.pop(name, None)

        thread = threading.Thread(target=wrapper, daemon=True)
        with _endless_lock:
            _endless_threads[name] = (thread, stop_flag)
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Platform selection at startup
    # ------------------------------------------------------------------
    def _select_platform(self):
        """Prompt user to select platform (one-time setup)."""
        self.console.print("[bold]Platform Selection[/bold]")
        self.console.print("Please select your operating system to optimize Lazy Llama for your environment.")
        self.console.print("[1] Linux")
        self.console.print("[2] macOS")
        self.console.print("[3] Windows (native or WSL2)")

        # Detect current platform and show as suggestion
        detected = detect_platform()
        self.console.print(f"[dim]Detected platform: {detected}[/dim]")

        choice = self._safe_prompt("Choice", choices=["1", "2", "3"], default="1")
        plat_map = {"1": "linux", "2": "darwin", "3": "windows"}
        self.config.platform = plat_map[choice]
        self.config.save()
        self.console.print(f"[green]Platform set to: {self.config.platform}[/green]")

    # ------------------------------------------------------------------
    # Helper: Validate tokenizer for a model by name/path
    # ------------------------------------------------------------------
    def _validate_tokenizer_for_model(self, model_name: str, model_info=None) -> bool:
        """
        Validate the tokenizer of a model. Returns True if valid, False if invalid.
        If model_info is None, look it up from manager.
        If invalid, prints an error message and returns False.
        """
        if model_info is None:
            model_info = self.model_manager.get_model(model_name)
        if not model_info or not model_info.path:
            self.console.print(f"[red]Model '{model_name}' not found or has no path.[/red]")
            return False
        path = Path(model_info.path)
        if not path.exists():
            self.console.print(f"[red]Model path does not exist: {path}[/red]")
            return False
        if path.is_dir():
            if not _validate_tokenizer_deep(path):
                self.console.print(f"[red]Model '{model_name}' has a corrupt tokenizer. Please delete and re-download it.[/red]")
                return False
            return True
        # For non-directory models (GGUF, etc.), we may not have a tokenizer, so consider valid.
        return True

    # ------------------------------------------------------------------
    # Get student engine (cached)
    # ------------------------------------------------------------------
    def _get_student_engine(self):
        """Load and cache the engine for the global student model."""
        if self._student_engine is not None:
            return self._student_engine
        if not self.global_student:
            self.console.print("[red]No global student set.[/red]")
            return None
        info = self.model_manager.get_model(self.global_student)
        if not info:
            self.console.print(f"[red]Student '{self.global_student}' not found.[/red]")
            return None
        try:
            engine = self._get_engine(info)
            self._student_engine = engine
            return engine
        except Exception as e:
            self.console.print(f"[red]Failed to load student engine: {e}[/red]")
            return None

    # ------------------------------------------------------------------
    # Startup model selection (with curated list of student models and skip option)
    # ------------------------------------------------------------------
    def _select_models_at_startup(self):
        """Prompt user to select teacher (Ollama) and choose/create a student model.
           If Ollama is not available, offer to skip teacher selection.
        """
        self.console.print("[bold]Welcome! Please select your models for this session.[/bold]")

        # ----- Teacher selection (Ollama) ----
        teacher = None
        try:
            ollama_bin = self.config.get_ollama_binary()
            result = subprocess.check_output([ollama_bin, "list"], text=True)
            lines = result.strip().splitlines()
            models = []
            for line in lines:
                if line.startswith("NAME") or not line.strip():
                    continue
                parts = line.split()
                if parts:
                    models.append(parts[0])
            if not models:
                self.console.print("[red]No Ollama models found. You can pull a model later.[/red]")
                teacher = None
            else:
                self.console.print("[yellow]Available Ollama models:[/yellow]")
                for i, m in enumerate(models, 1):
                    self.console.print(f"  {i}. {m}")

                while True:
                    choice = self._safe_prompt("Select teacher (number or name, or 'skip' to continue without teacher)", default=str(models[0]))
                    if choice.lower() == "skip":
                        teacher = None
                        break
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(models):
                            teacher = models[idx]
                            break
                    else:
                        if choice in models:
                            teacher = choice
                            break
                    self.console.print("[red]Invalid selection. Try again.[/red]")
        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
            logger.warning(f"Ollama listing failed: {e}")
            self.console.print("[yellow]Could not list Ollama models. You can skip teacher selection or pull a model later.[/yellow]")
            if self._safe_confirm("Continue without a teacher?", default=True):
                teacher = None
            else:
                # User wants to fix Ollama; exit
                self.console.print("[red]Please install and start Ollama, then restart the TUI.[/red]")
                sys.exit(1)

        # ----- Student selection -----
        # Check if config has a default student that exists
        default_student = self.config.default_student
        if default_student and self.model_manager.model_exists(default_student):
            student = default_student
            self.console.print(f"[green]Using default student '{student}' from config.[/green]")
        else:
            # If default doesn't exist, offer options
            self.console.print("\n[yellow]Student model selection:[/yellow]")
            self.console.print("You can use an existing local model, download a lightweight model from Hugging Face, or enter a custom name.")

            # Show existing local models (non-Ollama, non-vLLM) as options
            local_models = [m.name for m in self.model_manager.list_models()
                            if m.path and not m.path.startswith(("ollama://", "vllm://"))]

            options = []
            if local_models:
                options.append(("use_existing", "Use an existing local model"))
            options.append(("suggested", "Download one of the suggested lightweight models"))
            options.append(("custom", "Enter a custom Hugging Face model name"))

            choice_map = {str(i + 1): opt for i, (opt, _) in enumerate(options)}
            choice_labels = [f"[{i + 1}] {label}" for i, (_, label) in enumerate(options)]
            self.console.print("\n".join(choice_labels))
            sel = self._safe_prompt("Choose an option", choices=list(choice_map.keys()), default="1")

            if choice_map[sel] == "use_existing":
                # Let user pick from local models
                self.console.print("[yellow]Available local models:[/yellow]")
                for i, m in enumerate(local_models, 1):
                    self.console.print(f"  {i}. {m}")
                while True:
                    pick = self._safe_prompt("Select a model (number or name)", default=str(local_models[0]))
                    if pick.isdigit():
                        idx = int(pick) - 1
                        if 0 <= idx < len(local_models):
                            base_model = local_models[idx]
                            break
                    else:
                        if pick in local_models:
                            base_model = pick
                            break
                    self.console.print("[red]Invalid selection. Try again.[/red]")
                # Use the existing model directly as the student (it's already registered)
                student = base_model
                self.console.print(f"[green]Using existing model '{student}' as student.[/green]")

            elif choice_map[sel] == "suggested":
                # Get suggested models from config or use default
                suggested = getattr(self.config, 'suggested_student_models', None)
                if suggested is None:
                    suggested = DEFAULT_SUGGESTED_STUDENT_MODELS
                # Show curated list
                self.console.print("[yellow]Suggested lightweight models:[/yellow]")
                table = Table(show_header=True, header_style="bold white")
                table.add_column("#", style="dim")
                table.add_column("Model ID", style="cyan")
                table.add_column("Description", style="green")
                for i, (model_id, desc) in enumerate(suggested, 1):
                    table.add_row(str(i), model_id, desc)
                self.console.print(table)

                while True:
                    pick = self._safe_prompt("Select a model (number or model ID)", default="1")
                    if pick.isdigit():
                        idx = int(pick) - 1
                        if 0 <= idx < len(suggested):
                            base_model = suggested[idx][0]
                            break
                    else:
                        # Check if the input matches any model ID
                        matches = [mid for mid, _ in suggested if mid == pick]
                        if matches:
                            base_model = matches[0]
                            break
                    self.console.print("[red]Invalid selection. Try again.[/red]")

                # Ask for a student name (default: student_from_{base_model_short})
                default_student_name = f"student_from_{base_model.split('/')[-1]}"
                # Check if student name already exists
                while True:
                    student = self._safe_prompt("Enter a name for the student model", default=default_student_name)
                    if self.model_manager.model_exists(student):
                        overwrite = self._safe_confirm(f"Student '{student}' already exists. Overwrite?", default=False)
                        if overwrite:
                            # Delete existing model to allow recreation
                            self.model_manager.delete_model(student)
                            break
                        else:
                            self.console.print("[yellow]Please choose a different name.[/yellow]")
                            continue
                    else:
                        break

                # Check if the base model is already downloaded; if not, download it
                if not self.model_manager.model_exists(base_model):
                    self.console.print(f"[yellow]Model '{base_model}' not found locally. Downloading...[/yellow]")
                    try:
                        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeRemainingColumn()) as prog:
                            task = prog.add_task(f"[yellow]Downloading {base_model}...", total=None)

                            def cb(pct, msg):
                                prog.update(task, description=f"[yellow]{msg} ({pct}%)")
                                prog.update(task, completed=pct)

                            self.model_manager.download_from_hf(base_model, progress_callback=cb)
                            self.console.print(f"[green]Downloaded {base_model} successfully.[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Failed to download {base_model}: {e}[/red]")
                        self.console.print("[yellow]You may need to check your internet connection or try a different model.[/yellow]")
                        sys.exit(1)

                # ---- Validate tokenizer of the downloaded base model ----
                base_info = self.model_manager.get_model(base_model)
                if not self._validate_tokenizer_for_model(base_model, base_info):
                    self.console.print(f"[red]Base model '{base_model}' has a corrupt tokenizer. Please delete and re-download it.[/red]")
                    sys.exit(1)

                # Create the student from the base model
                self.console.print(f"[yellow]Creating student '{student}' from base '{base_model}'...[/yellow]")
                success = self.model_manager.create_student(base_model, student, auto_download=False)
                if not success:
                    self.console.print("[red]Failed to create student model. Please check logs.[/red]")
                    sys.exit(1)
                self.console.print(f"[green]Student '{student}' created successfully.[/green]")

            else:  # custom
                base_model = self._safe_prompt("Enter Hugging Face model name (e.g., 'distilgpt2')", default="distilgpt2")
                default_student_name = f"student_from_{base_model.split('/')[-1]}"
                # Check if student name already exists
                while True:
                    student = self._safe_prompt("Enter a name for the student model", default=default_student_name)
                    if self.model_manager.model_exists(student):
                        overwrite = self._safe_confirm(f"Student '{student}' already exists. Overwrite?", default=False)
                        if overwrite:
                            # Delete existing model to allow recreation
                            self.model_manager.delete_model(student)
                            break
                        else:
                            self.console.print("[yellow]Please choose a different name.[/yellow]")
                            continue
                    else:
                        break

                # Check if exists; if not, download
                if not self.model_manager.model_exists(base_model):
                    self.console.print(f"[yellow]Model '{base_model}' not found locally. Downloading...[/yellow]")
                    try:
                        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeRemainingColumn()) as prog:
                            task = prog.add_task(f"[yellow]Downloading {base_model}...", total=None)

                            def cb(pct, msg):
                                prog.update(task, description=f"[yellow]{msg} ({pct}%)")
                                prog.update(task, completed=pct)

                            self.model_manager.download_from_hf(base_model, progress_callback=cb)
                            self.console.print(f"[green]Downloaded {base_model} successfully.[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Failed to download {base_model}: {e}[/red]")
                        self.console.print("[yellow]You may need to check your internet connection or try a different model.[/yellow]")
                        sys.exit(1)

                # ---- Validate tokenizer of the downloaded base model ----
                base_info = self.model_manager.get_model(base_model)
                if not self._validate_tokenizer_for_model(base_model, base_info):
                    self.console.print(f"[red]Base model '{base_model}' has a corrupt tokenizer. Please delete and re-download it.[/red]")
                    sys.exit(1)

                # Create student
                self.console.print(f"[yellow]Creating student '{student}' from base '{base_model}'...[/yellow]")
                success = self.model_manager.create_student(base_model, student, auto_download=False)
                if not success:
                    self.console.print("[red]Failed to create student model. Please check logs.[/red]")
                    sys.exit(1)
                self.console.print(f"[green]Student '{student}' created successfully.[/green]")

        # Set global state (teacher may be None)
        self.global_teacher = teacher
        self.global_student = student
        self._save_global_state()
        self._refresh_global_state(force_sync=True)
        if teacher:
            self.console.print(f"[green]Models set: Teacher={teacher}, Student={student}[/green]")
        else:
            self.console.print(f"[green]Student set: {student} (no teacher selected)[/green]")

    # ------------------------------------------------------------------
    # Ensure default student models are installed (NEW)
    # ------------------------------------------------------------------
    def _ensure_default_students(self):
        """Download and register default student models if not already present."""
        if getattr(self.config, 'default_students_installed', False):
            return  # already done

        default_models = [
            "distilgpt2",
            "gpt2",
            "facebook/opt-125m"
        ]
        self.console.print("[yellow]Checking for default student models...[/yellow]")
        for model_name in default_models:
            if not self.model_manager.model_exists(model_name):
                self.console.print(f"[yellow]Downloading default student '{model_name}'...[/yellow]")
                try:
                    # Use the existing download method; it will register automatically.
                    self.model_manager.download_from_hf(model_name)
                    self.console.print(f"[green]Downloaded and registered '{model_name}'.[/green]")
                except Exception as e:
                    self.console.print(f"[red]Failed to download {model_name}: {e}[/red]")
                    # Continue with other models
            else:
                self.console.print(f"[dim]'{model_name}' already exists.[/dim]")

        # Mark as installed in config and save
        self.config.default_students_installed = True
        save_config(self.config)
        self.console.print("[green]Default student models are ready.[/green]")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self):
        # ---- Platform selection if not already set ----
        if self.config.platform == "auto":
            self._select_platform()

        # ---- Ensure default student models are installed ----
        self._ensure_default_students()

        # ---- Startup model selection if not already set ----
        if not self.global_teacher or not self.global_student:
            self._select_models_at_startup()

        try:
            self._main_menu()
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Slash command handler
    # ------------------------------------------------------------------
    def _handle_slash_command(self, command: str):
        """Handle slash commands like /chat, /status, /export, etc."""
        command = command.strip()
        if not command.startswith('/'):
            return False  # not a slash command

        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/chat":
            if not arg:
                self.console.print("[red]Usage: /chat <prompt>[/red]")
                return True
            # Quick chat with student model
            engine = self._get_student_engine()
            if not engine:
                self.console.print("[red]No student engine available.[/red]")
                return True
            self.console.print("[yellow]Generating response...[/yellow]")
            try:
                response = ""
                for token in engine.lazy_generate_stream(arg, max_tokens=128):
                    response += token
                self.console.print(f"[bold green]Student:[/bold green] {response}")
            except Exception as e:
                self.console.print(f"[red]Chat error: {e}[/red]")
            return True

        elif cmd == "/status":
            self._show_status()
            return True

        elif cmd == "/export":
            self._export_current_model()
            return True

        elif cmd == "/dashboard":
            self._dashboard()
            return True

        elif cmd == "/help":
            self._show_help()
            return True

        elif cmd == "/clear":
            self.console.clear()
            return True

        elif cmd == "/exit" or cmd == "/quit":
            self.running = False
            return True

        else:
            self.console.print(f"[yellow]Unknown command: {cmd}. Try /help[/yellow]")
            return True

    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------
    def _main_menu(self):
        while self.running:
            # Refresh model registry and global selections at start of each loop
            self._refresh_global_state(force_sync=False)  # uses cache

            self.console.clear()
            # Logo – bold white (theme maps "bold cyan" to "bold white")
            self.console.print(LAZY_LOGO, style="bold cyan")
            ram_used = get_total_ram_gb() - get_available_ram_gb()
            lazytorch_status = "ON" if self.config.use_lazytorch else "OFF"
            header_lines = [
                f"RAM: {ram_used:.1f}/{get_total_ram_gb():.1f} GB",
                f"E8: {'ON' if self.config.use_e8_quantization else 'OFF'} | KV bits: {self.config.kv_cache_bits if self.config.use_kv_cache_compression else 0}",
                f"LazyTorch: {lazytorch_status} | Unload after forward: {self.config.lazytorch_unload_after_forward}",
                f"Platform: {self.config.get_platform()}"
            ]
            header = Panel(Align.center("\n".join(header_lines)), style="bold blue")
            self.console.print(header)

            # ---- REAP Pipeline Status Banner ----
            if self.global_student:
                # Lazy import to avoid circular dependency
                from .bootstrap import run_reap_pipeline_checklist
                passed = run_reap_pipeline_checklist(self.global_student, self.model_manager)
                status_text = "✅ FULLY VERIFIED" if passed else "⚠️ PARTIAL"
                status_color = "green" if passed else "yellow"
                self.console.print(Panel(
                    f"REAP Pipeline Status for '{self.global_student}': [{status_color}]{status_text}[/{status_color}]",
                    style="dim"
                ))
            else:
                self.console.print(Panel("REAP Pipeline Status: No student selected", style="dim"))

            # Show global selections
            teacher = self.global_teacher or "None"
            student = self.global_student or "None"
            self.console.print(f"[dim]Global Teacher: {teacher}  |  Global Student: {student}[/dim]")

            table = Table(show_header=False, box=None)
            table.add_row("[1] Chat", "[2] Distill", "[3] Prune")
            table.add_row("[4] Models", "[5] Dashboard", "[6] Benchmark Single")
            table.add_row("[7] Settings", "[8] Export to Ollama", "[9] Import Model from Zip")
            table.add_row("[0] LazyTorch Convert", "[e] Export Student as Zip", "[r] Rename Model")
            table.add_row("[b] Benchmark Students", "[g] Select Global Models", "")
            table.add_row("[B] Benchmark Settings", "[V] View Reports", "[E] Endless RL Loop")
            table.add_row("[x] Exit", "", "")
            table.add_row("[cmd] Slash commands: /chat, /status, /export, /dashboard, /help", "")
            self.console.print(Panel(table, title="Main Menu"))

            curr = "None"
            if self.current_model and hasattr(self.current_model, 'model_path'):
                curr = self.current_model.model_path
            self.console.print(Panel(f"Active Model: {curr}", style="dim"))

            # ---- Get user input ----
            choice = self._safe_prompt("Choice (or /command)", default="1")
            choice = choice.strip()  # <-- FIX: strip whitespace

            # Check if it's a slash command
            if choice.startswith('/'):
                self._handle_slash_command(choice)
                self._wait_for_enter()
                continue

            # Otherwise, process numeric choice
            try:
                if choice == "1":
                    self._chat()
                elif choice == "2":
                    self._distill()
                elif choice == "3":
                    self._prune()
                elif choice == "4":
                    self._models()
                elif choice == "5":
                    self._dashboard()
                elif choice == "6":
                    self._benchmark()
                elif choice == "7":
                    self._settings()
                elif choice == "8":
                    self._export_current_model()
                elif choice == "9":
                    self._import_model_zip()
                elif choice == "0":
                    self._lazytorch_convert_menu()
                elif choice == "e":
                    self._export_student_zip()
                elif choice == "r":
                    self._rename_model()
                elif choice == "b":
                    self._benchmark_students()
                elif choice == "g":
                    self._select_global_models()
                elif choice in ("B", "b_settings"):
                    self._benchmark_settings_menu()
                elif choice in ("V", "v_reports"):
                    self._view_reports()
                elif choice in ("E", "e_endless"):
                    self._endless_menu()
                elif choice == "x":
                    self.running = False
                else:
                    logger.warning(f"Unrecognised menu choice: '{choice}'")
                    self.console.print(f"[red]Invalid choice: '{choice}'. Try again.[/red]")
                    self._wait_for_enter()
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {e}[/red]")
                logger.exception("Unhandled exception in main menu")
                self.console.print("[yellow]Please check the logs for more details.[/yellow]")
                self._wait_for_enter()

    # ------------------------------------------------------------------
    # Benchmark Settings Menu
    # ------------------------------------------------------------------
    def _benchmark_settings_menu(self):
        """Interactive menu to configure benchmark settings."""
        self.console.print(Panel("Benchmark Settings", style="bold cyan"))
        self.console.print("Configure parameters for student benchmarking.\n")

        settings = self.benchmark_settings

        # Prompt
        prompt = self._safe_prompt(
            "Prompt for generation",
            default=settings.prompt
        )
        settings.prompt = prompt

        # Max tokens
        max_tokens = int(self._safe_prompt(
            "Max tokens to generate",
            default=str(settings.max_tokens)
        ))
        settings.max_tokens = max_tokens

        # Perplexity
        run_perplexity = self._safe_confirm(
            "Run perplexity benchmark?",
            default=settings.run_perplexity
        )
        settings.run_perplexity = run_perplexity
        if run_perplexity:
            # val_texts: we can use default validation prompts from config or ask for file
            self.console.print("[yellow]Perplexity requires a list of validation texts.[/yellow]")
            use_default = self._safe_confirm(
                "Use default validation prompts from config?",
                default=True
            )
            if use_default:
                settings.val_texts = self.config.validation_prompts
            else:
                # Ask for a file path (one text per line)
                file_path = self._safe_prompt(
                    "Path to text file (one text per line)",
                    default=""
                )
                if file_path and Path(file_path).exists():
                    with open(file_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    settings.val_texts = lines
                    self.console.print(f"[green]Loaded {len(lines)} texts.[/green]")
                else:
                    self.console.print("[red]Invalid file path; using default prompts.[/red]")
                    settings.val_texts = self.config.validation_prompts

        # Multiple choice
        run_mc = self._safe_confirm(
            "Run multiple-choice accuracy benchmark?",
            default=settings.run_multiple_choice
        )
        settings.run_multiple_choice = run_mc
        if run_mc:
            # mc_questions: we can provide a sample set or ask for a JSON file
            self.console.print("[yellow]Multiple-choice requires a list of questions.[/yellow]")
            use_default_mc = self._safe_confirm(
                "Use a small default set of MC questions?",
                default=True
            )
            if use_default_mc:
                # Provide a few sample questions (hardcoded)
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
                self.console.print("[green]Using default MC questions.[/green]")
            else:
                file_path = self._safe_prompt(
                    "Path to JSON file with MC questions (list of dicts)",
                    default=""
                )
                if file_path and Path(file_path).exists():
                    try:
                        with open(file_path, "r") as f:
                            questions = json.load(f)
                        settings.mc_questions = questions
                        self.console.print(f"[green]Loaded {len(questions)} questions.[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Failed to load MC questions: {e}[/red]")
                        settings.mc_questions = None
                else:
                    self.console.print("[red]Invalid file path; skipping MC benchmark.[/red]")
                    settings.run_multiple_choice = False

        # Long context
        run_lc = self._safe_confirm(
            "Run long-context (RULER-style) benchmarks?",
            default=settings.run_long_context
        )
        settings.run_long_context = run_lc
        if run_lc:
            # Context lengths
            ctx_str = self._safe_prompt(
                "Context lengths (comma-separated, e.g., 2048,4096,8192)",
                default=",".join(str(x) for x in settings.context_lengths)
            )
            try:
                ctx_list = [int(x.strip()) for x in ctx_str.split(",") if x.strip()]
                if ctx_list:
                    settings.context_lengths = ctx_list
            except ValueError:
                self.console.print("[red]Invalid context lengths; using defaults.[/red]")
                settings.context_lengths = [2048, 4096, 8192, 16384]

            # Num trials
            num_trials = int(self._safe_prompt(
                "Number of trials per length",
                default=str(settings.num_trials)
            ))
            settings.num_trials = num_trials

            # Max tokens per task (new field, fallback to 20 if missing)
            default_lc_tokens = getattr(settings, 'long_context_max_tokens', 20)
            lc_max_tokens = int(self._safe_prompt(
                "Max tokens to generate per long-context task",
                default=str(default_lc_tokens)
            ))
            settings.long_context_max_tokens = lc_max_tokens

        # Store in registry
        store = self._safe_confirm(
            "Store results in registry?",
            default=settings.store_in_registry
        )
        settings.store_in_registry = store

        # Save settings
        self._save_benchmark_settings()
        self.benchmark_settings = settings
        self.console.print("[green]Benchmark settings updated and saved.[/green]")
        self._wait_for_enter()

    # ------------------------------------------------------------------
    # View Reports
    # ------------------------------------------------------------------
    def _view_reports(self):
        """Display historical benchmark results stored in registry."""
        self.console.print(Panel("Benchmark Reports (from Registry)", style="bold cyan"))

        # Get all student models
        student_models = []
        for info in self.model_manager.list_models(include_invalid=False):
            if '_distilled' in info.name or info.name.endswith('_pruned'):
                student_models.append(info)

        if not student_models:
            self.console.print("[yellow]No student models found in registry.[/yellow]")
            self._wait_for_enter()
            return

        # Build a table
        table = Table(title="Stored Benchmark Results")
        table.add_column("Model", style="cyan")
        table.add_column("TPS", justify="right")
        table.add_column("Peak Mem (GB)", justify="right")
        table.add_column("Perplexity", justify="right")
        table.add_column("MC Acc", justify="right")
        table.add_column("LC Score", justify="right")
        table.add_column("Timestamp")

        for info in student_models:
            metadata = getattr(info, 'metadata', {})
            benchmarks = metadata.get('benchmarks', {})
            if not benchmarks:
                table.add_row(info.name, "—", "—", "—", "—", "—", "No data")
                continue

            # Extract latest run (or average)
            tps = benchmarks.get('tokens_per_second', None)
            mem = benchmarks.get('peak_memory_gb', None)
            ppl = benchmarks.get('perplexity', None)
            mc = benchmarks.get('multiple_choice_accuracy', None)
            lc_data = benchmarks.get('long_context', {})
            lc_score = "—"
            if lc_data:
                rates = []
                for task in lc_data.values():
                    if isinstance(task, dict):
                        for length, val in task.items():
                            if isinstance(val, dict) and 'success_rate' in val:
                                rates.append(val['success_rate'])
                if rates:
                    lc_score = f"{sum(rates)/len(rates):.1%}"
            timestamp = benchmarks.get('timestamp', '')
            if timestamp:
                ts_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
            else:
                ts_str = ""

            tps_str = f"{tps:.2f}" if tps is not None else "—"
            mem_str = f"{mem:.2f}" if mem is not None else "—"
            ppl_str = f"{ppl:.3f}" if ppl is not None else "—"
            mc_str = f"{mc:.2%}" if mc is not None else "—"

            table.add_row(info.name, tps_str, mem_str, ppl_str, mc_str, lc_score, ts_str)

        self.console.print(table)
        self._wait_for_enter()

    # ------------------------------------------------------------------
    # Endless RL Loop Menu (with stop option)
    # ------------------------------------------------------------------
    def _endless_menu(self):
        """Interactive menu for endless RL loops."""
        if not ENDLESS_AVAILABLE:
            self.console.print("[red]Endless RL modules not available. Please ensure endless_rl.py is present.[/red]")
            self._wait_for_enter()
            return

        self.console.clear()
        self.console.print(Panel("Endless RL Loop", style="bold magenta"))
        self.console.print("Choose an endless self‑improvement mode:")
        self.console.print("  [1] Endless Distillation")
        self.console.print("  [2] Endless Pruning")
        self.console.print("  [3] Global Auto (benchmark + decide + improve)")
        self.console.print("  [4] Stop all running endless loops")
        self.console.print("  [b] Back to main menu")

        choice = self._safe_prompt("Choice", choices=["1", "2", "3", "4", "b"], default="b")
        if choice == "b":
            return

        if choice == "4":
            self._stop_all_endless_loops()
            self.console.print("[green]All endless loops stopped (if any were running).[/green]")
            self._wait_for_enter()
            return

        # Gather common parameters
        cycles = int(self._safe_prompt("Number of cycles (-1 for infinite)", default="-1"))
        sleep = int(self._safe_prompt("Sleep between cycles (seconds)", default="60"))

        if choice == "1":
            # Endless distillation
            self._refresh_global_state(force_sync=True)
            default_teacher = self.global_teacher or ""
            default_student = self.global_student or ""
            teacher = self._safe_prompt("Teacher model", default=default_teacher)
            student = self._safe_prompt("Student model", default=default_student)
            passes = int(self._safe_prompt("Distillation passes per cycle", default="2"))

            # ---- Validate student tokenizer before starting ----
            if not self._validate_tokenizer_for_model(student):
                self.console.print("[red]Student model has a corrupt tokenizer. Cannot distill.[/red]")
                self._wait_for_enter()
                return

            self.console.print(f"[yellow]Starting endless distillation: {teacher} -> {student}[/yellow]")
            # Launch in background thread with stop flag
            def run_loop(stop_flag):
                def cb(msg):
                    if stop_flag.is_set():
                        raise KeyboardInterrupt("Stopped by user")
                    self.console.print(f"[dim]{msg}[/dim]")

                run_endless_distillation(teacher, student, passes, cycles, sleep, callback=cb)

            self._start_endless_loop("distillation", run_loop)
            self.console.print("[green]Endless distillation started in background. Check logs for progress.[/green]")
            self._wait_for_enter()

        elif choice == "2":
            # Endless pruning
            self._refresh_global_state(force_sync=True)
            default_model = self.global_student if self.global_student and self.model_manager.model_exists(self.global_student) else ""
            model = self._safe_prompt("Model to prune", default=default_model)
            strategies_str = self._safe_prompt("Strategies (space-separated, e.g., 'magnitude neuron task')", default="magnitude neuron task")
            strategies = strategies_str.split()
            if not strategies:
                strategies = ["magnitude", "neuron", "task"]

            # ---- Validate tokenizer before pruning ----
            if not self._validate_tokenizer_for_model(model):
                self.console.print("[red]Model has a corrupt tokenizer. Cannot prune.[/red]")
                self._wait_for_enter()
                return

            self.console.print(f"[yellow]Starting endless pruning on {model} with strategies: {strategies}[/yellow]")

            def run_loop(stop_flag):
                def cb(msg):
                    if stop_flag.is_set():
                        raise KeyboardInterrupt("Stopped by user")
                    self.console.print(f"[dim]{msg}[/dim]")

                run_endless_prune(model, strategies, cycles, sleep, callback=cb)

            self._start_endless_loop("prune", run_loop)
            self.console.print("[green]Endless pruning started in background. Check logs for progress.[/green]")
            self._wait_for_enter()

        elif choice == "3":
            # Global auto loop
            self._refresh_global_state(force_sync=True)
            student_models = self.model_manager.get_student_models()
            if not student_models:
                self.console.print("[yellow]No student models found. Using global student if set.[/yellow]")
                default_models = [self.global_student] if self.global_student else []
            else:
                default_models = student_models

            models_str = self._safe_prompt("Models to manage (space-separated, leave empty for all students)", default=" ".join(default_models))
            if models_str.strip():
                models = models_str.split()
            else:
                models = default_models

            policy = self._safe_prompt("Policy", choices=["worst", "best", "random"], default="worst")

            self.console.print(f"[yellow]Starting global auto loop on models: {models} with policy {policy}[/yellow]")

            def run_loop(stop_flag):
                def cb(msg):
                    if stop_flag.is_set():
                        raise KeyboardInterrupt("Stopped by user")
                    self.console.print(f"[dim]{msg}[/dim]")

                run_endless_auto(
                    models=models,
                    cycles=cycles,
                    sleep=sleep,
                    policy=policy,
                    callback=cb
                )

            self._start_endless_loop("auto", run_loop)
            self.console.print("[green]Global auto loop started in background. Check logs for progress.[/green]")
            self._wait_for_enter()

        else:
            self.console.print("[red]Invalid choice.[/red]")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Select Global Models (updated to show combined list)
    # ------------------------------------------------------------------
    def _select_global_models(self):
        """Set global teacher and student from available models (both local and Ollama)."""
        try:
            self.console.print(Panel("Select Global Models", style="bold cyan"))

            # Force sync and reload
            self._refresh_global_state(force_sync=True)

            # List all models with type (only valid ones)
            models = self.model_manager.list_models()  # filters invalid by default
            if not models:
                self.console.print("[red]No models available. Download some first.[/red]")
                self._wait_for_enter()
                return

            # Show current
            self.console.print(f"Current teacher: {self.global_teacher or 'None'}")
            self.console.print(f"Current student: {self.global_student or 'None'}\n")

            # Build choices: show type
            choices = {}
            for m in models:
                if m.path and m.path.startswith("ollama://"):
                    label = f"{m.name} (Ollama)"
                elif m.path and m.path.endswith(".gguf"):
                    label = f"{m.name} (GGUF)"
                elif m.path and is_lazytorch_model(Path(m.path)):
                    label = f"{m.name} (LazyTorch)"
                elif m.path and m.path.startswith("vllm://"):
                    label = f"{m.name} (vLLM)"
                else:
                    label = f"{m.name} (Local)"
                choices[label] = m.name

            # Select teacher
            self.console.print("[yellow]Select Teacher (or leave blank to keep current):[/yellow]")
            teacher_choice = self._safe_prompt("Teacher", choices=[""] + list(choices.keys()), default="")
            if teacher_choice:
                self.global_teacher = choices[teacher_choice]
            else:
                # Keep current
                pass

            # Select student
            self.console.print("[yellow]Select Student (or leave blank to keep current):[/yellow]")
            student_choice = self._safe_prompt("Student", choices=[""] + list(choices.keys()), default="")
            if student_choice:
                self.global_student = choices[student_choice]
            else:
                # Keep current
                pass

            # Persist
            self._save_global_state()
            # Force refresh to update UI immediately with new global selections
            self._refresh_global_state(force_sync=True)
            self.console.print(f"[green]Global models set: Teacher={self.global_teacher or 'None'}, Student={self.global_student or 'None'}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to select global models: {e}[/red]")
            logger.exception("Error in _select_global_models")
        finally:
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Engine creation with validation
    # ------------------------------------------------------------------
    def _get_engine(self, model_info):
        """Validate and create engine; respects global LazyTorch settings.
           Handles missing model_info gracefully.
           Falls back to Transformers if LazyTorch fails due to corruption.
        """
        if model_info is None or not model_info.path:
            raise ValueError("Invalid model info: missing path")

        # If the manager has resolve_model, use it to get fresh info
        if hasattr(self.model_manager, 'resolve_model'):
            resolved = self.model_manager.resolve_model(model_info.name)
            if resolved:
                model_info = resolved
            else:
                self.console.print(f"[yellow]Could not resolve model '{model_info.name}' via resolve_model, using as-is.[/yellow]")

        path = model_info.path
        name = model_info.name

        if path is None:
            raise ValueError(f"Model {name} has no path associated")

        if not path.startswith("ollama://") and not Path(path).exists() and not path.startswith("vllm://"):
            self.console.print(f"[yellow]Warning: Model path may not exist: {path}[/yellow]")

        # ---- vLLM ----
        if path.startswith("vllm://"):
            try:
                # Check if openai is installed; if not, provide a clear error.
                try:
                    import openai  # noqa
                except ImportError:
                    raise RuntimeError(
                        "The openai package is required for vLLM engine. "
                        "Please install it with: pip install openai"
                    )
                self.console.print("[cyan]Using vLLM engine (remote inference via OpenAI API)[/cyan]")
                model_name = path.replace("vllm://", "")
                return VLLMEngine(model_name, self.config)
            except Exception as e:
                self.console.print(f"[red]Failed to create vLLM engine: {e}[/red]")
                raise

        # Try LazyTorch if enabled
        if self.config.use_lazytorch:
            lazytorch_path = self.model_manager.get_lazytorch_path(path)
            if lazytorch_path:
                try:
                    # Validate tokenizer before loading
                    tokenizer_path = lazytorch_path if lazytorch_path.is_dir() else lazytorch_path.with_suffix('')
                    if not _validate_tokenizer_deep(tokenizer_path):
                        raise ValueError("Tokenizer in LazyTorch model is corrupt.")
                    self.console.print("[cyan]Using LazyTorch engine (memory-mapped loading)[/cyan]")
                    return LazyTorchEngine(str(lazytorch_path), self.config)
                except Exception as e:
                    self.console.print(f"[yellow]Failed to load LazyTorch engine: {e}. Falling back to Transformers.[/yellow]")
                    # Fall through to Transformers

        # Validate HF directory if not Ollama/GGUF
        if not path.startswith("ollama://") and not path.endswith(".gguf") and not path.startswith("vllm://"):
            path_obj = Path(path)
            if path_obj.is_dir():
                # Deep tokenizer validation
                if not _validate_tokenizer_deep(path_obj):
                    raise ValueError(f"Model directory {path} has a corrupt tokenizer. Please delete and re-download.")
                if not (path_obj / "config.json").exists():
                    raise ValueError(f"Model path {path} is not a valid Hugging Face directory (missing config.json).")

        try:
            if path.startswith("ollama://"):
                try:
                    requests.get("http://localhost:11434/api/tags", timeout=2)
                except Exception:
                    raise RuntimeError("Ollama service not reachable. Is 'ollama serve' running?")
                return OllamaInferenceEngine(name, self.config)
            elif path.endswith(".gguf"):
                return LazyGGUFEngine(path, self.config)
            else:
                return TransformersInferenceEngine(path, self.config)
        except Exception as e:
            self.console.print(f"[red]Failed to create engine: {e}[/red]")
            raise

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def _chat(self):
        try:
            if not self.current_model:
                # If global student is set, try to load that first
                if self.global_student and self.model_manager.model_exists(self.global_student):
                    info = self.model_manager.get_model(self.global_student)
                    try:
                        self.current_model = self._get_engine(info)
                        self.metrics_store.register_inference_engine(self.current_model)
                        self.metrics_store.set_active_model(self.global_student)
                        self.console.print(f"[green]Loaded global student: {self.global_student}[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Failed to load global student: {e}[/red]")
                        # fall through to manual selection
                if not self.current_model:
                    # Manual selection
                    models = self.model_manager.list_models()
                    if not models:
                        self.console.print("[red]No models. Download first.[/red]")
                        return
                    chosen = self._safe_prompt("Select model", choices=[m.name for m in models])
                    info = self.model_manager.get_model(chosen)
                    if not info:
                        self.console.print("[red]Model not found[/red]")
                        return
                    try:
                        self.current_model = self._get_engine(info)
                        self.metrics_store.register_inference_engine(self.current_model)
                        self.metrics_store.set_active_model(chosen)
                    except Exception as e:
                        self.console.print(f"[red]Failed to load model: {e}[/red]")
                        return

            self.console.clear()
            self.console.print(Panel("Chat - /exit, /dashboard, /help, /status, /export, /chat <prompt> for quick chat", style="green"))
            try:
                while True:
                    user = self._safe_prompt("[bold cyan]You[/bold cyan]")
                    cmd = self.command_aliases.get(user, user)
                    if cmd == "/exit":
                        break
                    if cmd == "/clear":
                        self.console.clear()
                        continue
                    if cmd == "/dashboard":
                        self._launch_dashboard()
                        continue
                    if cmd == "/help":
                        self._show_help()
                        continue
                    if cmd == "/status":
                        self._show_status()
                        continue
                    if cmd == "/export":
                        self._export_current_model()
                        continue
                    # Quick chat using /chat prompt
                    if user.startswith("/chat "):
                        prompt = user[6:].strip()
                        if not prompt:
                            self.console.print("[red]Usage: /chat <prompt>[/red]")
                            continue
                        self.console.print("[yellow]Generating response...[/yellow]")
                        try:
                            response = ""
                            for token in self.current_model.lazy_generate_stream(prompt, max_tokens=128):
                                response += token
                            self.console.print(f"[bold green]Assistant:[/bold green] {response}")
                        except Exception as e:
                            self.console.print(f"[red]Chat error: {e}[/red]")
                        continue
                    # Regular interactive chat
                    self.console.print("[bold green]Assistant:[/bold green] ", end="")
                    for token in self.current_model.lazy_generate_stream(user, max_tokens=150):
                        self.console.print(token, end="")
                    self.console.print()
            finally:
                if torch.cuda.is_available():
                    clear_cuda_memory()
        except Exception as e:
            self.console.print(f"[red]Chat error: {e}[/red]")
            logger.exception("Chat error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Dashboard launch
    # ------------------------------------------------------------------
    def _launch_dashboard(self):
        """
        Launch the dashboard web server. If the configured port is already in use,
        we automatically find a free port and start the server there.
        """
        # If we already have a real server object (not the sentinel), just open browser
        if self.dashboard_server is not None and self.dashboard_server is not True:
            webbrowser.open(f"http://127.0.0.1:{self.config.dashboard_port}/dashboard")
            return

        configured_port = self.config.dashboard_port
        # Check if the configured port is already in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', configured_port))
        sock.close()

        port_to_use = configured_port
        if result == 0:
            # Port is in use – find a free port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', 0))
            port_to_use = sock.getsockname()[1]
            sock.close()
            self.console.print(f"[yellow]Port {configured_port} is already in use. Using alternative port {port_to_use} instead.[/yellow]")
        else:
            self.console.print(f"[dim]Port {configured_port} is free. Starting dashboard there.[/dim]")

        try:
            from .dashboard_server import start_dashboard
            self.dashboard_server = start_dashboard(port_to_use, self.config.dashboard_auto_open)
            self.console.print(f"[green]Dashboard started at http://127.0.0.1:{port_to_use}/dashboard[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to start dashboard: {e}[/red]")
            self.console.print("[yellow]Check that the port is free and you have the required permissions.[/yellow]")
            logger.exception("Dashboard start failed")
            # Don't set dashboard_server to anything

    def _dashboard(self):
        self._launch_dashboard()
        self.console.print("[yellow]Press Enter to return[/yellow]")
        self._wait_for_enter()

    # ------------------------------------------------------------------
    # Distill
    # ------------------------------------------------------------------
    def _distill(self):
        try:
            self.console.print(Panel("Distillation", style="bold magenta"))
            # Refresh before showing (force sync to get latest)
            self._refresh_global_state(force_sync=True)

            default_teacher = self.global_teacher or ""
            default_student = self.global_student or ""
            teacher = self._safe_prompt("Teacher (Ollama)", default=default_teacher)
            student = self._safe_prompt("Student model (local)", default=default_student)

            # ---- FIX: Validate that student is a local model (not Ollama/vLLM) ----
            student_info = self.model_manager.get_model(student)
            if student_info and student_info.path and (student_info.path.startswith("ollama://") or student_info.path.startswith("vllm://")):
                self.console.print("[red]Student model must be a local Hugging Face model, not Ollama or vLLM.[/red]")
                self.console.print("[yellow]Please select a local model or create one first.[/yellow]")
                self._wait_for_enter()
                return

            # ---- Validate tokenizer of the student model ----
            if not self._validate_tokenizer_for_model(student, student_info):
                self.console.print("[red]Student model has a corrupt tokenizer. Cannot distill.[/red]")
                self.console.print("[yellow]Please delete the student model and re-create it from a valid base.[/yellow]")
                self._wait_for_enter()
                return

            passes = int(self._safe_prompt("Passes", default="3"))
            resume = self._safe_confirm("Resume from checkpoint?", default=True)

            # ---- µMoE options ----
            use_moe = self._safe_confirm("Use Micro Mixture of Experts (µMoE)?", default=False)
            num_experts = 4
            top_k = 1
            moe_reduction = 2
            aux_weight = 0.01
            if use_moe:
                num_experts = int(self._safe_prompt("Number of experts", default="4"))
                top_k = int(self._safe_prompt("Top-k routing", default="1"))
                moe_reduction = int(self._safe_prompt("Intermediate dimension reduction factor", default="2"))
                aux_weight = float(self._safe_prompt("Auxiliary loss weight", default="0.01"))

            if self._safe_confirm("Start?"):
                try:
                    from .lazy_distill import LazyDistillationEngine
                    engine = LazyDistillationEngine(
                        self.config,
                        use_moe=use_moe,
                        num_experts=num_experts,
                        top_k=top_k,
                        moe_reduction_factor=moe_reduction,
                        aux_loss_weight=aux_weight
                    )
                    val = self.config.validation_prompts
                    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeRemainingColumn()) as prog:
                        task = prog.add_task("[yellow]Distilling...", total=passes * len(val))

                        def cb(p, total_p, b, total_b):
                            completed = (p - 1) * total_b + b
                            prog.update(task, completed=completed)
                            pct = int(completed / (total_p * total_b) * 100)
                            self.metrics_store.set_distillation_progress(pct, f"pass {p}/{total_p}")

                        engine.set_progress_callback(cb)
                        engine.run_distillation(
                            teacher, student, val, passes, resume=resume,
                            use_moe=use_moe,
                            num_experts=num_experts,
                            top_k=top_k,
                            moe_reduction_factor=moe_reduction,
                            aux_loss_weight=aux_weight
                        )
                    self.metrics_store.set_distillation_progress(100, "complete")
                    self.console.print("[green]Done![/green]")
                except ValueError as e:
                    self.console.print(f"[red]Distillation failed: {e}[/red]")
                    self.console.print("[yellow]Check that the student model is a valid PyTorch model and not a GGUF file. "
                                       "If the tokenizer is corrupt, delete the student and recreate it.[/yellow]")
                    logger.exception("Distillation ValueError")
                except RuntimeError as e:
                    self.console.print(f"[red]Distillation runtime error: {e}[/red]")
                    self.console.print("[yellow]Ensure Ollama is running if using an Ollama teacher, and that the teacher model exists.[/yellow]")
                    logger.exception("Distillation RuntimeError")
                except Exception as e:
                    self.console.print(f"[red]Unexpected error during distillation: {e}[/red]")
                    logger.exception("Distillation unexpected error")
                finally:
                    # Refresh global state (new distilled model may appear)
                    self._refresh_global_state(force_sync=True)
        except Exception as e:
            self.console.print(f"[red]Distillation menu error: {e}[/red]")
            logger.exception("Distillation menu error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------
    def _prune(self):
        try:
            self.console.print(Panel("Pruning", style="bold yellow"))
            self._refresh_global_state(force_sync=True)
            models = self.model_manager.list_models()
            if not models:
                self.console.print("[red]No models[/red]")
                return
            # Default to global student if available and local
            default_model = self.global_student if self.global_student and self.model_manager.model_exists(self.global_student) else models[0].name
            chosen = self._safe_prompt("Model", choices=[m.name for m in models], default=default_model)

            # ---- FIX: Validate that the chosen model is local (not Ollama/vLLM) ----
            info = self.model_manager.get_model(chosen)
            if info and info.path and (info.path.startswith("ollama://") or info.path.startswith("vllm://")):
                self.console.print("[red]Model must be a local Hugging Face model, not Ollama or vLLM.[/red]")
                self.console.print("[yellow]Pruning only works with local models.[/yellow]")
                self._wait_for_enter()
                return

            # ---- Validate tokenizer before pruning ----
            if not self._validate_tokenizer_for_model(chosen, info):
                self.console.print("[red]Model has a corrupt tokenizer. Cannot prune.[/red]")
                self.console.print("[yellow]Please delete and re-download the model, or repair the tokenizer.[/yellow]")
                self._wait_for_enter()
                return

            strategy = self._safe_prompt("Strategy", choices=["magnitude", "neuron", "task"])

            if strategy == "task":
                task = self._safe_prompt("Task", choices=["coding", "chat", "embed", "math"])
                samples = get_task_prompts(task)
            else:
                samples = ["def hello():", "print('hi')", "for i in range(10):"]

            # ---- µMoE export options ----
            export_as_moe = self._safe_confirm("Export pruned model as Micro MoE (µMoE)?", default=False)
            num_experts = 4
            top_k = 1
            moe_reduction = 2
            if export_as_moe:
                num_experts = int(self._safe_prompt("Number of experts", default="4"))
                top_k = int(self._safe_prompt("Top-k routing", default="1"))
                moe_reduction = int(self._safe_prompt("Intermediate dimension reduction factor", default="2"))

            if self._safe_confirm("Proceed?"):
                info = self.model_manager.get_model(chosen)
                if not info or not info.path or info.path.startswith("ollama://"):
                    self.console.print("[red]Invalid model[/red]")
                    return

                required_gb = estimate_memory_need(Path(info.path))
                available_gb = get_available_ram_gb()
                if required_gb > available_gb * 0.9:
                    self.console.print(f"[red]Insufficient RAM: need ~{required_gb:.1f} GB, only {available_gb:.1f} GB free.[/red]")
                    self.console.print("[yellow]Try pruning with a smaller model or free up memory.[/yellow]")
                    return

                from .lazy_prune import Pruner
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.console.print("[yellow]Loading model (may take a moment)...[/yellow]")
                model = None
                tokenizer = None
                try:
                    # Validate tokenizer before loading (already done, but double-check)
                    if not _validate_tokenizer_deep(Path(info.path)):
                        raise ValueError(f"Tokenizer in model {chosen} is corrupt. Please delete and re-download.")
                    model = AutoModelForCausalLM.from_pretrained(info.path, low_cpu_mem_usage=True)
                    tokenizer = AutoTokenizer.from_pretrained(info.path)
                    pruner = Pruner(model, self.config)
                    if strategy == "magnitude":
                        pruner.magnitude_prune()
                    elif strategy == "neuron":
                        pruner.neuron_prune()
                    else:
                        pruner.task_specific_reap(task, samples, tokenizer)

                    # Determine output path
                    out = self.model_manager.models_dir / f"{chosen}_pruned"
                    # Export with µMoE options
                    pruner.export_pruned(
                        str(out),
                        overwrite=True,
                        export_as_moe=export_as_moe,
                        num_experts=num_experts,
                        top_k=top_k,
                        moe_reduction_factor=moe_reduction
                    )
                    self.console.print(f"[green]Saved to {out}[/green]")
                    # Refresh registry
                    self.model_manager.reload_registry(sync_ollama=False)
                    self._refresh_global_state(force_sync=True)
                except ValueError as e:
                    self.console.print(f"[red]Pruning failed: {e}[/red]")
                    self.console.print("[yellow]Ensure the model is not a GGUF file and that the tokenizer is valid. "
                                       "Try re-downloading the model if corruption is suspected.[/yellow]")
                    logger.exception("Prune ValueError")
                except MemoryError as e:
                    self.console.print(f"[red]Pruning failed due to insufficient memory: {e}[/red]")
                    self.console.print("[yellow]Try pruning with a smaller model or disable E8 quantization.[/yellow]")
                    logger.exception("Prune MemoryError")
                except Exception as e:
                    self.console.print(f"[red]Pruning failed: {e}[/red]")
                    logger.exception("Prune unexpected error")
                finally:
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
                    gc.collect()
                    clear_cuda_memory()
                    self.console.print("[dim]Memory cleaned up.[/dim]")
        except Exception as e:
            self.console.print(f"[red]Prune menu error: {e}[/red]")
            logger.exception("Prune menu error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def _models(self):
        try:
            self.console.clear()
            self._refresh_global_state(force_sync=True)
            table = Table(title="Models (✓ = valid, ✗ = invalid)")
            table.add_column("Name")
            table.add_column("Type")
            table.add_column("Size MB")
            table.add_column("E8")
            table.add_column("LazyTorch")
            table.add_column("Status", justify="center")

            # Include invalid models to show status
            for m in self.model_manager.list_models(include_invalid=True):
                if m.path and m.path.startswith("ollama://"):
                    mtype = "Ollama"
                elif m.path and m.path.endswith(".gguf"):
                    mtype = "GGUF"
                elif m.path and is_lazytorch_model(Path(m.path)):
                    mtype = "LazyTorch"
                elif m.path and m.path.startswith("vllm://"):
                    mtype = "vLLM"
                else:
                    mtype = "Local"
                e8_flag = "✓" if getattr(m, 'e8_quantized', False) else ""
                lt_flag = "✓" if getattr(m, 'lazytorch_format', False) else ""
                status = "✓" if not m.invalid else "✗"
                table.add_row(m.name, mtype, f"{m.original_size_mb:.1f}", e8_flag, lt_flag, status)

            self.console.print(table)

            # Ask what to do next
            action = self._safe_prompt(
                "What would you like to do?",
                choices=["download", "validate", "validate-all", "delete", "cancel"],
                default="cancel"
            )
            if action == "download":
                self._download_model()
            elif action == "validate":
                self._validate_model_prompt()
            elif action == "validate-all":
                self._validate_all_models_prompt()
            elif action == "delete":
                self._delete_model_prompt()
            # else cancel – do nothing

            self._refresh_global_state(force_sync=True)
        except Exception as e:
            self.console.print(f"[red]Models menu error: {e}[/red]")
            logger.exception("Models menu error")
            self._wait_for_enter()

    def _download_model(self):
        """Helper to download a new model (original code)."""
        try:
            if self._safe_confirm("Download?"):
                src = self._safe_prompt("Source", choices=["ollama", "huggingface"])
                name = self._safe_prompt("Name")
                try:
                    if src == "ollama":
                        self.model_manager.download_from_ollama(name)
                    else:
                        convert = self._safe_confirm("Convert to LazyTorch after download?", default=self.config.use_lazytorch)
                        gguf = self._safe_prompt("GGUF filename (optional, press Enter to skip)", default="")
                        self.model_manager.download_from_hf(name, gguf if gguf else None, convert_to_lazytorch_after=convert)
                    self.model_manager._save_registry()
                    self.console.print("[green]Download initiated and registry saved[/green]")
                except requests.exceptions.ConnectionError:
                    self.console.print("[red]Network error: Could not connect to Hugging Face or Ollama.[/red]")
                    self.console.print("[yellow]Check your internet connection and try again.[/yellow]")
                    logger.exception("Download connection error")
                except FileNotFoundError as e:
                    self.console.print(f"[red]Download failed: {e}[/red]")
                    self.console.print("[yellow]The specified model may not exist or the path is invalid.[/yellow]")
                    logger.exception("Download file not found")
                except Exception as e:
                    self.console.print(f"[red]Download failed: {e}[/red]")
                    self.console.print("[yellow]Please check the logs for more details.[/yellow]")
                    logger.exception("Download error")
                finally:
                    self._refresh_global_state(force_sync=True)
        except Exception as e:
            self.console.print(f"[red]Download menu error: {e}[/red]")
            logger.exception("Download menu error")
            self._wait_for_enter()

    def _validate_model_prompt(self):
        """Prompt user to select a model and validate it."""
        try:
            models = self.model_manager.list_models(include_invalid=True)
            if not models:
                self.console.print("[red]No models available to validate.[/red]")
                return
            choices = [m.name for m in models]
            name = self._safe_prompt("Select model to validate", choices=choices)
            info = self.model_manager.get_model(name)
            if not info:
                self.console.print("[red]Model not found.[/red]")
                return
            valid = self.model_manager.validate_model(name)
            if valid:
                self.console.print(f"[green]Model '{name}' is valid.[/green]")
            else:
                self.console.print(f"[red]Model '{name}' is invalid. Reason: {self._get_invalid_reason(info)}[/red]")
                self.console.print("[yellow]You may delete and re-download this model.[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Validation error: {e}[/red]")
            logger.exception("Validation error")
            self._wait_for_enter()

    def _validate_all_models_prompt(self):
        """Re-validate all local models and update invalid flags."""
        try:
            self.console.print("[yellow]Re-validating all local models...[/yellow]")
            self.model_manager.validate_all_models()
            self.console.print("[green]Validation complete. Invalid models have been marked in the registry.[/green]")
            self._refresh_global_state(force_sync=True)
        except Exception as e:
            self.console.print(f"[red]Validation error: {e}[/red]")
            logger.exception("Validate-all error")
        finally:
            self._wait_for_enter()

    def _get_invalid_reason(self, info) -> str:
        """Return a human-readable reason why a model is invalid."""
        if not info.path:
            return "No path associated with this model."
        if info.path.startswith("ollama://") or info.path.startswith("vllm://"):
            return f"{info.model_type.capitalize()} models are considered valid remotely."
        path_obj = Path(info.path)
        if not path_obj.exists():
            return f"Path does not exist: {path_obj}"
        if path_obj.is_dir():
            if not (path_obj / "config.json").exists():
                return "Directory missing config.json"
            if not ((path_obj / "pytorch_model.bin").exists() or (path_obj / "model.safetensors").exists()):
                return "No weight files found (pytorch_model.bin or model.safetensors)"
            tokenizer_files = ["tokenizer.json", "tokenizer.model", "vocab.json"]
            if not any((path_obj / f).exists() for f in tokenizer_files):
                return "Missing tokenizer files (tokenizer.json, tokenizer.model, or vocab.json)"
            return "Directory does not appear to be a valid Hugging Face model (tokenizer corrupt)."
        if path_obj.is_file() and path_obj.suffix != ".gguf":
            return "File is not a .gguf"
        return "Unknown reason."

    def _delete_model_prompt(self):
        """Prompt user to select a model and delete it."""
        try:
            models = self.model_manager.list_models(include_invalid=True)
            if not models:
                self.console.print("[red]No models available to delete.[/red]")
                return
            choices = [m.name for m in models]
            name = self._safe_prompt("Select model to delete", choices=choices)
            if not self._safe_confirm(f"Are you sure you want to delete '{name}'? This action is irreversible."):
                return
            try:
                if self.model_manager.delete_model(name):
                    self.console.print(f"[green]Model '{name}' deleted successfully.[/green]")
                    # If the deleted model was global student/teacher, clear them
                    if self.global_student == name:
                        self.global_student = None
                        self._save_global_state()
                    if self.global_teacher == name:
                        self.global_teacher = None
                        self._save_global_state()
                else:
                    self.console.print(f"[red]Failed to delete model '{name}'.[/red]")
            except PermissionError as e:
                self.console.print(f"[red]Permission error: {e}[/red]")
                self.console.print("[yellow]Ensure you have write permissions to the models directory.[/yellow]")
                logger.exception("Delete PermissionError")
            except Exception as e:
                self.console.print(f"[red]Deletion failed: {e}[/red]")
                logger.exception("Delete error")
            finally:
                self._refresh_global_state(force_sync=True)
        except Exception as e:
            self.console.print(f"[red]Delete menu error: {e}[/red]")
            logger.exception("Delete menu error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # LazyTorch Conversion
    # ------------------------------------------------------------------
    def _lazytorch_convert_menu(self):
        try:
            self.console.print(Panel("LazyTorch Conversion", style="bold cyan"))
            self._refresh_global_state(force_sync=True)
            models = self.model_manager.list_models(include_invalid=True)
            # Only valid local models not already LazyTorch
            local_models = [m for m in models if m.path and not m.path.startswith("ollama://") and not m.path.startswith("vllm://") and not m.invalid and not getattr(m, 'lazytorch_format', False)]
            if not local_models:
                self.console.print("[red]No valid local models available for conversion (already LazyTorch or none).[/red]")
                self._wait_for_enter()
                return
            choices = [m.name for m in local_models]
            chosen = self._safe_prompt("Select model to convert", choices=choices)
            if self._safe_confirm(f"Convert {chosen} to LazyTorch? This may take several minutes."):
                try:
                    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeRemainingColumn()) as prog:
                        task = prog.add_task("[yellow]Converting to LazyTorch...", total=None)

                        def callback(msg):
                            prog.update(task, description=f"[yellow]{msg}")

                        result = self.model_manager.convert_to_lazytorch(chosen, progress_callback=callback)
                        if result:
                            self.console.print(f"[green]Conversion successful: {result}[/green]")
                        else:
                            self.console.print("[red]Conversion failed.[/red]")
                            self.console.print("[yellow]Check logs for details. Ensure the model is valid and you have enough disk space.[/yellow]")
                except MemoryError as e:
                    self.console.print(f"[red]Conversion failed due to insufficient memory: {e}[/red]")
                    self.console.print("[yellow]Try converting a smaller model or free up memory.[/yellow]")
                    logger.exception("Convert MemoryError")
                except Exception as e:
                    self.console.print(f"[red]Conversion error: {e}[/red]")
                    logger.exception("Conversion error")
                finally:
                    self._refresh_global_state(force_sync=True)
                self._wait_for_enter()
        except Exception as e:
            self.console.print(f"[red]Conversion menu error: {e}[/red]")
            logger.exception("Conversion menu error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Benchmark Single Model
    # ------------------------------------------------------------------
    def _benchmark(self):
        try:
            self.console.print(Panel("Benchmark Single Model", style="bold cyan"))
            self._refresh_global_state(force_sync=True)
            models = self.model_manager.list_models()  # only valid
            if not models:
                self.console.print("[red]No valid models[/red]")
                return
            default_model = self.global_student if self.global_student and self.model_manager.model_exists(self.global_student) else models[0].name
            chosen = self._safe_prompt("Model to benchmark", choices=[m.name for m in models], default=default_model)
            info = self.model_manager.get_model(chosen)
            if not info:
                self.console.print("[red]Model not found in registry[/red]")
                return
            if not info.path or (info.path.startswith("ollama://") and not info.path.startswith("vllm://")):
                self.console.print("[red]Cannot benchmark Ollama model directly (use 'ollama' engine)[/red]")
                return
            path_obj = Path(info.path)
            if not path_obj.exists() and not info.path.startswith("vllm://"):
                self.console.print(f"[red]Model path does not exist: {path_obj}[/red]")
                return
            # Validate tokenizer for local models
            if not info.path.startswith("ollama://") and not info.path.startswith("vllm://"):
                if not _validate_tokenizer_deep(path_obj):
                    self.console.print(f"[red]Model tokenizer is corrupt. Cannot benchmark.[/red]")
                    return
            self.console.print("[yellow]Running benchmark (may take a minute)...[/yellow]")
            try:
                res = benchmark_model(str(info.path), prompt="What is machine learning?",
                                      max_tokens=100, config=self.config, model_name=chosen)
                self.console.print("")
                self.console.print(f"[green]Results for {chosen}:[/green]")
                self.console.print(f"  Tokens/sec: {res['tokens_per_second']:.2f}")
                self.console.print(f"  Avg latency: {res['avg_latency_ms']:.1f} ms")
                self.console.print(f"  Memory usage: {res['memory_usage_gb']:.2f} GB")
                self.console.print(f"  E8: {res.get('e8_quantized', False)} | KV compressed: {res.get('kv_compressed', False)}")
                if info.lazytorch_format:
                    lt_size = get_lazytorch_model_size(path_obj)
                    self.console.print(f"  LazyTorch model size on disk: {lt_size / (1024**3):.2f} GB")
            except ValueError as e:
                self.console.print(f"[red]Benchmark failed: {e}[/red]")
                self.console.print("[yellow]Check that the model is a valid Hugging Face, LazyTorch, or vLLM model.[/yellow]")
                logger.exception("Benchmark ValueError")
            except RuntimeError as e:
                self.console.print(f"[red]Benchmark runtime error: {e}[/red]")
                self.console.print("[yellow]Ensure you have enough free RAM and the model is not corrupted.[/yellow]")
                logger.exception("Benchmark RuntimeError")
            except Exception as e:
                self.console.print(f"[red]Benchmark failed: {e}[/red]")
                logger.exception("Benchmark error")
        except Exception as e:
            self.console.print(f"[red]Benchmark menu error: {e}[/red]")
            logger.exception("Benchmark menu error")
        finally:
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Benchmark Students (using settings and progress)
    # ------------------------------------------------------------------
    def _benchmark_students(self):
        try:
            self.console.print(Panel("Benchmark Student Models", style="bold cyan"))

            # Get current settings
            settings = self.benchmark_settings
            self.console.print(f"Using settings: prompt='{settings.prompt}', max_tokens={settings.max_tokens}, "
                               f"perplexity={settings.run_perplexity}, MC={settings.run_multiple_choice}, "
                               f"long-context={settings.run_long_context}")

            # Get student list
            student_entries = []
            for info in self.model_manager.list_models():
                if '_distilled' in info.name or info.name.endswith('_pruned'):
                    if info.path and not info.path.startswith("ollama://"):
                        student_entries.append(info)

            if not student_entries:
                self.console.print("[yellow]No student models found.[/yellow]")
                self._wait_for_enter()
                return

            # Confirm
            if not self._safe_confirm(f"Run benchmark on {len(student_entries)} student models? This may take some time.", default=True):
                return

            # Run with progress
            with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeRemainingColumn()) as prog:
                task = prog.add_task("[yellow]Benchmarking students...", total=len(student_entries))

                # Progress callback
                def progress_callback(model_name: str, status: str, result: Optional[Dict]):
                    if status == "done":
                        prog.update(task, advance=1)
                        self.console.print(f"[dim]{model_name}: completed[/dim]")
                    elif status == "error":
                        prog.update(task, advance=1)
                        self.console.print(f"[red]{model_name}: failed - {result.get('error_message', 'unknown')}[/red]")
                    elif status == "starting":
                        prog.update(task, description=f"[yellow]Benchmarking {model_name}...[/yellow]")

                # Run benchmark
                results = benchmark_student_models(settings, config=self.config, progress_callback=progress_callback)

            # Display summary
            self.console.print("")
            self.console.print(Panel("Benchmark Complete", style="green"))
            summary = results['summary']
            self.console.print(f"Total: {summary['total']}, Succeeded: {summary['succeeded']}, Failed: {summary['failed']}")
            if summary['succeeded'] > 0:
                self.console.print(f"Average TPS: {summary['avg_tps']:.2f}, Avg peak memory: {summary['avg_peak_mem_gb']:.2f} GB")

            # Show detailed table
            self.console.print("")
            self.console.print(format_benchmark_summary(results['results']))

            self._wait_for_enter()

        except TypeError as e:
            # Fallback in case benchmark_student_models doesn't accept progress_callback
            self.console.print("[yellow]Progress callback not supported by benchmark_student_models; running without progress bar.[/yellow]")
            try:
                results = benchmark_student_models(settings, config=self.config)
                self.console.print("")
                self.console.print(Panel("Benchmark Complete", style="green"))
                summary = results['summary']
                self.console.print(f"Total: {summary['total']}, Succeeded: {summary['succeeded']}, Failed: {summary['failed']}")
                if summary['succeeded'] > 0:
                    self.console.print(f"Average TPS: {summary['avg_tps']:.2f}, Avg peak memory: {summary['avg_peak_mem_gb']:.2f} GB")
                self.console.print("")
                self.console.print(format_benchmark_summary(results['results']))
            except Exception as e2:
                self.console.print(f"[red]Benchmark failed: {e2}[/red]")
                logger.exception("Benchmark students fallback error")
            self._wait_for_enter()

        except Exception as e:
            self.console.print(f"[red]Benchmark students error: {e}[/red]")
            logger.exception("Benchmark students error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _settings(self):
        try:
            self.console.print(Panel("Settings", style="bold"))

            while True:
                try:
                    new_ram = float(self._safe_prompt("RAM limit GB", default=str(self.config.ram_limit_gb)))
                    if new_ram <= 0:
                        raise ValueError("RAM limit must be positive")
                    self.config.ram_limit_gb = new_ram
                    break
                except ValueError as e:
                    self.console.print(f"[red]Invalid input: {e}. Please enter a positive number.[/red]")

            slow = self._safe_confirm("Slow mode?", default=self.config.slow_mode)
            self.config.slow_mode = slow

            enable_e8 = self._safe_confirm("Enable E8 quantization (extreme compression)?", default=self.config.use_e8_quantization)
            self.config.use_e8_quantization = enable_e8
            if enable_e8:
                while True:
                    try:
                        bpw = float(self._safe_prompt("E8 bits per weight (2-5)", default=str(self.config.e8_bits_per_weight)))
                        bpw = max(2.0, min(5.0, bpw))
                        self.config.e8_bits_per_weight = bpw
                        break
                    except ValueError:
                        self.console.print("[red]Please enter a number between 2 and 5.[/red]")

            enable_kv = self._safe_confirm("Enable KV cache compression?", default=self.config.use_kv_cache_compression)
            self.config.use_kv_cache_compression = enable_kv
            if enable_kv:
                while True:
                    try:
                        kv_bits = int(self._safe_prompt("KV compression bits (2-8)", default=str(self.config.kv_cache_bits)))
                        kv_bits = max(2, min(8, kv_bits))
                        self.config.kv_cache_bits = kv_bits
                        break
                    except ValueError:
                        self.console.print("[red]Please enter an integer between 2 and 8.[/red]")

            # ---- NEW: Ollama timeout setting ----
            while True:
                try:
                    new_timeout = int(self._safe_prompt("Ollama timeout (seconds)", default=str(self.config.ollama_timeout)))
                    if new_timeout <= 0:
                        raise ValueError("Timeout must be positive")
                    self.config.ollama_timeout = new_timeout
                    break
                except ValueError as e:
                    self.console.print(f"[red]Invalid input: {e}. Please enter a positive integer.[/red]")

            # ---- NEW: Platform setting ----
            self.console.print("[yellow]Change platform?[/yellow]")
            self.console.print(f"Current platform: {self.config.get_platform()}")
            if self._safe_confirm("Change platform?", default=False):
                self.console.print("Select platform:")
                self.console.print("[1] Linux")
                self.console.print("[2] macOS")
                self.console.print("[3] Windows")
                plat_choice = self._safe_prompt("Choice", choices=["1", "2", "3"])
                plat_map = {"1": "linux", "2": "darwin", "3": "windows"}
                self.config.platform = plat_map[plat_choice]
                self.console.print(f"[green]Platform set to: {self.config.platform}[/green]")

            current_lazytorch = self.config.use_lazytorch
            new_lazytorch = self._safe_confirm("Enable LazyTorch memory-mapped loading (extreme RAM savings)?", default=current_lazytorch)
            if new_lazytorch != current_lazytorch:
                self.console.print("[yellow]LazyTorch mode changed. Model reload required.[/yellow]")
                self.config.use_lazytorch = new_lazytorch
                if self.current_model:
                    self.console.print("[yellow]Current model will be reloaded if needed.[/yellow]")
            unload_after = self._safe_confirm("Unload parameters after each forward pass (maximum memory savings)?", default=self.config.lazytorch_unload_after_forward)
            self.config.lazytorch_unload_after_forward = unload_after

            zs = self._safe_confirm("Enable zero-shot compensation?", default=self.config.use_zero_shot_compensation)
            self.config.use_zero_shot_compensation = zs

            if self._safe_confirm("Auto-optimize config based on current system?", default=False):
                self.config = auto_optimize_config(self.config)
                self.console.print("[green]Auto-optimization applied.[/green]")

            self.config.save()
            self.console.print("[green]Settings updated and saved to disk.[/green]")
        except Exception as e:
            self.console.print(f"[red]Settings error: {e}[/red]")
            logger.exception("Settings error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _show_status(self):
        try:
            self.console.print(Panel("System Status", style="bold cyan"))
            recommendations = recommend_enhancements()
            self.console.print(recommendations)

            status_table = Table(title="Current Configuration")
            status_table.add_column("Parameter", style="cyan")
            status_table.add_column("Value", style="green")
            status_table.add_row("RAM limit", f"{self.config.ram_limit_gb} GB")
            status_table.add_row("Slow mode", str(self.config.slow_mode))
            status_table.add_row("E8 quantization", f"{self.config.use_e8_quantization} ({self.config.e8_bits_per_weight} bpw)")
            status_table.add_row("KV compression", f"{self.config.use_kv_cache_compression} ({self.config.kv_cache_bits} bits)")
            status_table.add_row("Ollama timeout", f"{self.config.ollama_timeout} s")
            status_table.add_row("LazyTorch", f"{self.config.use_lazytorch} (unload after forward: {self.config.lazytorch_unload_after_forward})")
            status_table.add_row("Zero-shot compensation", str(self.config.use_zero_shot_compensation))
            status_table.add_row("Device", self.config.device)
            status_table.add_row("Platform", self.config.get_platform())
            status_table.add_row("Global Teacher", self.global_teacher or "None")
            status_table.add_row("Global Student", self.global_student or "None")
            self.console.print(status_table)
        except Exception as e:
            self.console.print(f"[red]Status error: {e}[/red]")
            logger.exception("Status error")
        finally:
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Export to Ollama
    # ------------------------------------------------------------------
    def _export_current_model(self):
        """Export model to Ollama; if no current engine, use global student."""
        try:
            self._refresh_global_state(force_sync=True)
            model_path = None
            model_name = None

            if self.current_model and hasattr(self.current_model, 'model_path'):
                model_path = self.current_model.model_path
                model_name = self.current_model.model_name if hasattr(self.current_model, 'model_name') else Path(model_path).stem
            else:
                # Try global student
                if self.global_student and self.model_manager.model_exists(self.global_student):
                    info = self.model_manager.get_model(self.global_student)
                    if info and info.path and not info.path.startswith("ollama://") and not info.path.startswith("vllm://"):
                        model_path = info.path
                        model_name = self.global_student
                if not model_path:
                    # Fallback: ask user
                    models = self.model_manager.list_models()
                    if not models:
                        self.console.print("[red]No models available for export.[/red]")
                        return
                    valid_models = [m for m in models if m.path and not m.path.startswith("ollama://") and not m.path.startswith("vllm://")]
                    if not valid_models:
                        self.console.print("[red]No local models found to export.[/red]")
                        return
                    default = self.global_student if self.global_student and self.model_manager.model_exists(self.global_student) else valid_models[0].name
                    chosen = self._safe_prompt("Select model to export", choices=[m.name for m in valid_models], default=default)
                    info = self.model_manager.get_model(chosen)
                    if not info or not info.path:
                        self.console.print("[red]Model path not found.[/red]")
                        return
                    model_path = info.path
                    model_name = chosen

            # Validate tokenizer before export
            if model_path and not model_path.startswith("ollama://") and not model_path.startswith("vllm://"):
                p = Path(model_path)
                if p.is_dir() and not _validate_tokenizer_deep(p):
                    self.console.print(f"[red]Model tokenizer is corrupt. Cannot export.[/red]")
                    return

            self.console.print(f"[yellow]Exporting {model_name} to Ollama...[/yellow]")
            try:
                success = export_to_ollama(model_path, model_name)
                if success:
                    self.console.print(f"[green]Successfully exported {model_name} to Ollama.[/green]")
                else:
                    self.console.print("[red]Export failed. Check that Ollama is installed and running.[/red]")
                    self.console.print("[yellow]Ensure the model is a valid Hugging Face directory or GGUF file.[/yellow]")
            except subprocess.TimeoutExpired:
                self.console.print("[red]Export timed out. Ollama may be slow or unresponsive.[/red]")
                self.console.print("[yellow]Consider increasing the timeout in settings.[/yellow]")
                logger.exception("Export TimeoutExpired")
            except Exception as e:
                self.console.print(f"[red]Export error: {e}[/red]")
                logger.exception("Export error")
            finally:
                # Refresh global state to include the newly exported Ollama model in the selector
                self._refresh_global_state(force_sync=True)
        except Exception as e:
            self.console.print(f"[red]Export menu error: {e}[/red]")
            logger.exception("Export menu error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Export Student as Zip
    # ------------------------------------------------------------------
    def _export_student_zip(self):
        try:
            export_dir = Path.home() / ".lazy_llama/exports"
            export_dir.mkdir(parents=True, exist_ok=True)

            self.console.print(Panel("Export Student Model as Zip", style="bold cyan"))
            self._refresh_global_state(force_sync=True)
            models = self.model_manager.list_models()
            local_models = [m for m in models if m.path and not m.path.startswith("ollama://") and not m.path.startswith("vllm://")]
            if not local_models:
                self.console.print("[red]No local models available for export.[/red]")
                return

            default = self.global_student if self.global_student and self.model_manager.model_exists(self.global_student) else local_models[0].name
            choices = [m.name for m in local_models]
            chosen = self._safe_prompt("Select model to export", choices=choices, default=default)
            format_type = self._safe_prompt("Export format", choices=["pytorch", "vllm"], default="pytorch")

            info = self.model_manager.get_model(chosen)
            if not info or not info.path:
                self.console.print("[red]Model not found[/red]")
                return

            model_path = Path(info.path)
            if not model_path.exists():
                self.console.print(f"[red]Model path does not exist: {model_path}[/red]")
                return

            # Validate tokenizer before export
            if model_path.is_dir() and not _validate_tokenizer_deep(model_path):
                self.console.print(f"[red]Model tokenizer is corrupt. Cannot export.[/red]")
                return

            zip_name = f"{chosen}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = export_dir / zip_name

            self.console.print(f"[yellow]Exporting {chosen} as {format_type} zip...[/yellow]")
            try:
                if format_type == "vllm":
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    self.console.print("[yellow]Loading model for vLLM export...[/yellow]")
                    model = AutoModelForCausalLM.from_pretrained(str(model_path), low_cpu_mem_usage=True)
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

                    temp_dir = export_dir / f"temp_{chosen}"
                    temp_dir.mkdir(exist_ok=True)
                    model.save_pretrained(temp_dir, safe_serialization=True)
                    tokenizer.save_pretrained(temp_dir)

                    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', temp_dir)
                    shutil.rmtree(temp_dir)
                    del model
                    del tokenizer
                    gc.collect()
                else:
                    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', model_path)

                self.console.print(f"[green]Export successful: {zip_path}[/green]")
            except ValueError as e:
                self.console.print(f"[red]Export failed: {e}[/red]")
                self.console.print("[yellow]The model may be corrupt or incompatible with the chosen format.[/yellow]")
                logger.exception("Export zip ValueError")
            except Exception as e:
                self.console.print(f"[red]Export failed: {e}[/red]")
                logger.exception("Export zip error")
            finally:
                clear_cuda_memory()
        except Exception as e:
            self.console.print(f"[red]Export zip menu error: {e}[/red]")
            logger.exception("Export zip menu error")
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Import Model from Zip
    # ------------------------------------------------------------------
    def _import_model_zip(self):
        try:
            self.console.print(Panel("Import Model from Zip", style="bold cyan"))
            zip_path_str = self._safe_prompt("Path to zip file")
            zip_path = Path(zip_path_str).expanduser().resolve()
            if not zip_path.exists():
                self.console.print(f"[red]Zip file not found: {zip_path}[/red]")
                return

            # Determine model name
            model_name = self._safe_prompt("Model name", default=zip_path.stem)
            dest_dir = self.model_manager.models_dir / model_name
            if dest_dir.exists():
                if not self._safe_confirm(f"Model directory {dest_dir} already exists. Overwrite?"):
                    return
                shutil.rmtree(dest_dir)

            # Extract the zip
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.unpack_archive(str(zip_path), str(dest_dir))
            except Exception as e:
                self.console.print(f"[red]Failed to extract zip: {e}[/red]")
                self.console.print("[yellow]Ensure the zip file is not corrupted.[/yellow]")
                shutil.rmtree(dest_dir, ignore_errors=True)
                return

            # Use centralized validation
            if not self.model_manager.validate_model_directory(dest_dir):
                self.console.print("[red]Invalid model: missing config.json, tokenizer files, or weights.[/red]")
                shutil.rmtree(dest_dir)
                return

            # Register model
            size_mb = sum(f.stat().st_size for f in dest_dir.glob("*") if f.is_file()) / (1024 * 1024)
            with self.model_manager._lock:
                self.model_manager.registry[model_name] = self.model_manager._create_model_info(
                    name=model_name,
                    path=str(dest_dir),
                    size_mb=size_mb
                )
                self.model_manager._save_registry()
            # Sync after import
            self.model_manager.sync_ollama()
            self.model_manager.reload_registry(sync_ollama=False)
            self.console.print(f"[green]Model {model_name} imported successfully from {zip_path}[/green]")
            self._refresh_global_state(force_sync=True)
        except PermissionError as e:
            self.console.print(f"[red]Permission error: {e}[/red]")
            self.console.print("[yellow]Ensure you have write permissions to the models directory.[/yellow]")
            logger.exception("Import zip PermissionError")
        except Exception as e:
            self.console.print(f"[red]Import failed: {e}[/red]")
            logger.exception("Import zip error")
        finally:
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Rename Model
    # ------------------------------------------------------------------
    def _rename_model(self):
        try:
            self.console.print(Panel("Rename Model", style="bold yellow"))
            self._refresh_global_state(force_sync=True)
            models = self.model_manager.list_models(include_invalid=True)
            if not models:
                self.console.print("[red]No models available[/red]")
                return

            choices = [m.name for m in models]
            old_name = self._safe_prompt("Current model name", choices=choices)
            new_name = self._safe_prompt("New model name")
            if not new_name:
                self.console.print("[red]New name cannot be empty[/red]")
                return
            if new_name in self.model_manager.registry:
                self.console.print(f"[red]Model {new_name} already exists[/red]")
                return

            try:
                if self.model_manager.rename_model(old_name, new_name):
                    self.console.print(f"[green]Renamed {old_name} → {new_name}[/green]")
                    # ---- Update global state if the renamed model was selected ----
                    if self.global_student == old_name:
                        self.global_student = new_name
                        self.console.print(f"[dim]Updated global student to: {new_name}[/dim]")
                    if self.global_teacher == old_name:
                        self.global_teacher = new_name
                        self.console.print(f"[dim]Updated global teacher to: {new_name}[/dim]")
                    # Persist the updated global state
                    self._save_global_state()
                    # Refresh registry and UI (force sync to reload registry and update displays)
                    self._refresh_global_state(force_sync=True)
                else:
                    self.console.print("[red]Rename failed[/red]")
                    self.console.print("[yellow]Check that the old model exists and the new name is valid.[/yellow]")
            except PermissionError as e:
                self.console.print(f"[red]Permission error: {e}[/red]")
                self.console.print("[yellow]Ensure you have write permissions to the models directory.[/yellow]")
                logger.exception("Rename PermissionError")
            except Exception as e:
                self.console.print(f"[red]Rename error: {e}[/red]")
                logger.exception("Rename error")
        except Exception as e:
            self.console.print(f"[red]Rename menu error: {e}[/red]")
            logger.exception("Rename menu error")
        finally:
            self._wait_for_enter()

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------
    def _show_help(self):
        help_text = """
╔══════════════════════════════════════════════════════════════════╗
║ HELP                                                             ║
╠══════════════════════════════════════════════════════════════════╣
║ /exit      - quit chat                                           ║
║ /clear     - clear screen                                        ║
║ /dashboard - open web dashboard                                  ║
║ /status    - show system status and recommendations              ║
║ /export    - export current model to Ollama                      ║
║ /help      - this                                                ║
║ /e         - enter Endless RL Loop menu                          ║
║ /chat <prompt> - quick chat with student model                  ║
╚══════════════════════════════════════════════════════════════════╝
        """
        self.console.print(help_text, style="cyan")
        self._wait_for_enter()