"""
metrics_store.py - Singleton metrics store with real‑time updates for TUI and dashboard.
Includes E8/KV flags, LazyTorch status, and active task progress.

FIXED: Changed import from `from lazy_llama.config import Config, ModelInfo` to
       `from .config import Config, ModelInfo` to use relative imports.
"""

import time
import threading
import psutil
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Optional, Any

# ---- Relative imports (fixed) ----
from .config import Config, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of system and inference metrics at a point in time."""
    timestamp: float
    tokens_per_second: float
    inference_latency_ms: int
    ram_used_gb: float
    ram_total_gb: float
    cpu_percent: float
    distillation_progress: int
    distillation_status: str
    active_model: str
    queue_length: int
    # Flags
    e8_enabled: bool = False
    kv_compression_bits: int = 0
    lazytorch_enabled: bool = False
    lazytorch_savings_percent: float = 0.0
    # Active task
    active_task: str = ""
    task_progress: int = 0


class MetricsStore:
    """
    Singleton metrics store that collects system metrics and inference stats.
    Subscribers receive periodic snapshots.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Internal state
        self._data_lock = threading.Lock()
        self._history: List[MetricsSnapshot] = []
        self._listeners: List[Callable] = []
        self._running = True

        # Current values
        self._active_model = "None"
        self._distill_progress = 0
        self._distill_status = "idle"
        self._tps = 0.0
        self._latency = 0
        self._queue = 0
        self._inference_engine = None

        # Flags
        self._e8_enabled = False
        self._kv_bits = 0
        self._lazytorch_enabled = False
        self._lazytorch_savings = 0.0

        # Active task
        self._active_task = ""
        self._task_progress = 0

        # Model manager (lazy loaded)
        self._model_manager = None

        # Start background updater
        threading.Thread(target=self._update_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Registration and flag updates
    # ------------------------------------------------------------------
    def register_inference_engine(self, engine: Any) -> None:
        """Register an inference engine to pull metrics from its config."""
        with self._data_lock:
            self._inference_engine = engine
            if engine is not None:
                config = getattr(engine, 'config', None)
                if config is not None:
                    self._e8_enabled = getattr(config, 'use_e8_quantization', False)
                    self._kv_bits = getattr(config, 'kv_cache_bits', 0)
                    self._lazytorch_enabled = getattr(config, 'use_lazytorch', False)

    def update_e8_kv_flags(self, e8_enabled: bool, kv_bits: int) -> None:
        """Manually update E8/KV flags."""
        with self._data_lock:
            self._e8_enabled = e8_enabled
            self._kv_bits = kv_bits

    def set_lazytorch_status(self, enabled: bool, savings_percent: float = 95.0) -> None:
        """Manually set LazyTorch status."""
        with self._data_lock:
            self._lazytorch_enabled = enabled
            self._lazytorch_savings = savings_percent

    # ------------------------------------------------------------------
    # Active task management
    # ------------------------------------------------------------------
    def set_active_task(self, task: str, progress: int) -> None:
        """Set the current background task and its progress (0-100)."""
        with self._data_lock:
            self._active_task = task
            self._task_progress = max(0, min(100, progress))
            # Also update legacy distillation fields
            if task == "distillation":
                self._distill_progress = progress
                self._distill_status = f"running ({progress}%)"
            elif task == "prune":
                self._distill_progress = progress
                self._distill_status = f"pruning ({progress}%)"
            elif task == "student_creation":
                self._distill_progress = progress
                self._distill_status = f"creating student ({progress}%)"
            elif task == "benchmark":
                self._distill_progress = progress
                self._distill_status = f"benchmarking ({progress}%)"
            elif task == "export":
                self._distill_progress = progress
                self._distill_status = f"exporting ({progress}%)"
            else:
                self._distill_progress = progress
                self._distill_status = f"{task} ({progress}%)"

    def clear_task(self) -> None:
        """Clear the active task (no background task running)."""
        with self._data_lock:
            self._active_task = ""
            self._task_progress = 0
            if self._distill_status.startswith(("running", "pruning", "creating", "benchmarking", "exporting")):
                self._distill_status = "idle"
                self._distill_progress = 0

    # ------------------------------------------------------------------
    # Background update loop
    # ------------------------------------------------------------------
    def _update_loop(self) -> None:
        """Periodically collect system metrics and create snapshots."""
        while self._running:
            time.sleep(1)

            # Copy state under lock
            with self._data_lock:
                engine = self._inference_engine
                tps = self._tps
                latency = self._latency
                e8_enabled = self._e8_enabled
                kv_bits = self._kv_bits
                lazytorch_enabled = self._lazytorch_enabled
                lazytorch_savings = self._lazytorch_savings
                active_task = self._active_task
                task_progress = self._task_progress
                distill_progress = self._distill_progress
                distill_status = self._distill_status
                active_model = self._active_model
                queue_length = self._queue

                # ---- Read engine metrics ----
                if engine is not None:
                    engine_tps = getattr(engine, '_last_tps', 0.0)
                    if engine_tps == 0.0:
                        engine_tps = getattr(engine, 'last_tps', 0.0)
                    engine_latency = getattr(engine, '_last_latency', 0)
                    if engine_latency == 0:
                        engine_latency = getattr(engine, 'last_latency', 0)
                    tps = engine_tps if engine_tps != 0 else tps
                    latency = engine_latency if engine_latency != 0 else latency

                    # Update flags from engine config
                    config = getattr(engine, 'config', None)
                    if config is not None:
                        e8_enabled = getattr(config, 'use_e8_quantization', False)
                        kv_bits = getattr(config, 'kv_cache_bits', 0)
                        lazytorch_enabled = getattr(config, 'use_lazytorch', False)

                    # Update active model if engine has get_model_name
                    if hasattr(engine, 'get_model_name'):
                        try:
                            engine_model = engine.get_model_name()
                            if engine_model and engine_model != active_model:
                                active_model = engine_model
                        except Exception:
                            pass
                else:
                    # No engine: reload global config for flags
                    try:
                        from .config import load_config
                        config = load_config()
                        lazytorch_enabled = getattr(config, 'use_lazytorch', False)
                        e8_enabled = getattr(config, 'use_e8_quantization', False)
                        kv_bits = getattr(config, 'kv_cache_bits', 0)
                    except Exception:
                        pass

                lazytorch_savings = 95.0 if lazytorch_enabled else 0.0

                # ---- Pseudo-TPS mapping for background tasks ----
                if tps == 0.0 and active_task:
                    if task_progress > 0:
                        tps = max(0.5, task_progress / 10.0)
                    else:
                        tps = 0.2

                # Update local state
                self._tps = tps
                self._latency = latency
                self._e8_enabled = e8_enabled
                self._kv_bits = kv_bits
                self._lazytorch_enabled = lazytorch_enabled
                self._lazytorch_savings = lazytorch_savings
                self._active_model = active_model
                self._queue = queue_length

            # Build snapshot (no lock needed for read-only)
            snapshot = MetricsSnapshot(
                timestamp=time.time(),
                tokens_per_second=tps,
                inference_latency_ms=latency,
                ram_used_gb=psutil.virtual_memory().used / (1024**3),
                ram_total_gb=psutil.virtual_memory().total / (1024**3),
                cpu_percent=psutil.cpu_percent(),
                distillation_progress=distill_progress,
                distillation_status=distill_status,
                active_model=active_model,
                queue_length=queue_length,
                e8_enabled=e8_enabled,
                kv_compression_bits=kv_bits,
                lazytorch_enabled=lazytorch_enabled,
                lazytorch_savings_percent=lazytorch_savings,
                active_task=active_task,
                task_progress=task_progress,
            )
            self._history.append(snapshot)
            if len(self._history) > 3600:
                self._history = self._history[-3600:]

            # Notify subscribers
            for cb in self._listeners:
                try:
                    cb(snapshot)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_history(self, seconds: int = 60) -> List[Dict]:
        """Return snapshots from the last `seconds` seconds as dicts."""
        cutoff = time.time() - seconds
        with self._data_lock:
            return [asdict(m) for m in self._history if m.timestamp > cutoff]

    def subscribe(self, callback: Callable) -> None:
        """Register a callback to receive snapshots."""
        self._listeners.append(callback)

    # ------------------------------------------------------------------
    # Active model management
    # ------------------------------------------------------------------
    def set_active_model(self, name: str) -> None:
        with self._data_lock:
            self._active_model = name

    def set_active_model_from_info(self, model_info: ModelInfo) -> None:
        """Set active model from ModelInfo; skip if invalid."""
        with self._data_lock:
            if model_info is None:
                self._active_model = "None"
                return
            if getattr(model_info, 'invalid', False):
                logger.warning(f"Attempted to set active model to invalid model '{model_info.name}'. Skipping.")
                return
            self._active_model = model_info.name

    def get_active_model(self) -> str:
        with self._data_lock:
            return self._active_model

    def validate_active_model(self, model_manager=None) -> bool:
        """Re‑validate the tokenizer of the currently active model."""
        with self._data_lock:
            name = self._active_model
        if not name or name == "None":
            return False

        if model_manager is not None:
            mgr = model_manager
        else:
            if self._model_manager is None:
                from .lazy_model_manager import ModelManager
                self._model_manager = ModelManager()
            mgr = self._model_manager

        info = mgr.get_model(name)
        if info is None:
            logger.warning(f"Active model '{name}' not found in registry.")
            return False
        valid = mgr.validate_model(name)
        if not valid:
            logger.warning(f"Active model '{name}' failed validation and is marked invalid.")
        return valid

    # ------------------------------------------------------------------
    # Legacy distillation progress (used by dashboard/TUI)
    # ------------------------------------------------------------------
    def set_distillation_progress(self, progress: int, status: str) -> None:
        with self._data_lock:
            self._distill_progress = progress
            self._distill_status = status
            # Map status to active task
            if "running" in status.lower() or "pruning" in status.lower() or "creating" in status.lower() or "benchmarking" in status.lower() or "exporting" in status.lower():
                task_map = {
                    "distillation": "distillation",
                    "pruning": "prune",
                    "creating": "student_creation",
                    "benchmarking": "benchmark",
                    "exporting": "export"
                }
                for keyword, task_name in task_map.items():
                    if keyword in status.lower():
                        self._active_task = task_name
                        self._task_progress = progress
                        break
            else:
                if "idle" in status.lower() or "completed" in status.lower() or "failed" in status.lower():
                    self._active_task = ""
                    self._task_progress = 0

    # ------------------------------------------------------------------
    # Setters for direct metric updates
    # ------------------------------------------------------------------
    def set_tps(self, tps: float) -> None:
        with self._data_lock:
            self._tps = tps

    def set_latency(self, ms: int) -> None:
        with self._data_lock:
            self._latency = ms

    def set_queue_length(self, length: int) -> None:
        with self._data_lock:
            self._queue = length

    def stop(self) -> None:
        """Stop the background update loop."""
        self._running = False