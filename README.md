# Lazy Llama v3.6 (Under Development + Testing)

```
    ██╗      █████╗ ███████╗██╗   ██╗    ██╗     ██╗      █████╗ ███╗   ███╗ █████╗  
    ██║     ██╔══██╗╚══███╔╝╚██╗ ██╔╝    ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗ 
    ██║     ███████║  ███╔╝  ╚████╔╝     ██║     ██║     ███████║██╔████╔██║███████║ 
    ██║     ██╔══██║ ███╔╝    ╚██╔╝      ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║ 
    ███████╗██║  ██║███████╗   ██║       ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║ 
    ╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝       ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ 
```

*Low‑End Inference Engine – Trade Time for Accuracy*

Lazy inference engine for distilling, pruning, and bench-marking large language models on resource‑constrained devices. It combines memory‑mapped loading, extreme quantization, KV cache compression, and a Mixture of Experts student architecture to achieve sub‑500MB peak RAM for 7B models, while still delivering usable inference speeds.

---

## Features

| Category | Feature | Description |
|----------|---------|-------------|
| Loading | LazyTorch | Memory‑mapped weights (on‑disk, loaded on demand) – peak RAM under 500MB for 7B models. |
| Quantization | E8 Lattice | 2‑5 bits per weight, 4‑bit default, with optional GGUF export. |
| Compression | KV Cache | TurboQuant and mixed‑dimension compression (4‑8 bits), residual tokens for accuracy. |
| Students | Micro MoE | Replaces dense FFN with sparse Mixture of Experts (up to 16+ experts, hierarchical routing). |
| Distillation | KL / fine‑tuning | HF teacher (KL) or Ollama/GGUF (fine‑tuning) with LoRA/QLoRA, progressive layers, combined loss. |
| Pruning | Magnitude / Neuron / Task | Structured pruning of heads/FFN, gradient‑based Fisher pruning, embedding pruning. |
| Self‑improvement | REAP + Endless RL | Automated pipeline (Distill -> Prune -> Recover -> Eval) with bandit policy, hyperparameter search, model merging. |
| Inference Engines | GGUF, Ollama, Transformers, vLLM, LazyTorch | Unified API for all formats, streaming, batch, speculative decoding. |
| UI | TUI + Web Dashboard | Real‑time metrics, model management, full control via Rich TUI or Flask dashboard. |
| Recovery Pipeline | One‑click prune + distill | Prune a model (15% gentle ratio) and immediately recover via QLoRA distillation. |

---

## Installation

### Prerequisites

- Python 3.10 – 3.13
- pip
- Ollama
- Other packages auto installed

# 1. Download and put all files into this file structure on your system:

 File Tree

```
/lazyllama/ (Main folder)
├── README.md                   # Manual
├── start.sh                    # Linux/macOS/WSL startup script
├── start.bat                   # Windows CMD startup script
├── start.ps1                   # Windows PowerShell startup script
├── requirements.txt            # Python dependencies
├── Dockerfile                  
├── docker-compose-yml
├── setup.py                    # Package installation
├── pyproject.toml              # Build configuration
├── __init__.py                 # Package exports
 ----/lazyllama/lazy_llama/ (Sub folder)
      ├── lazy_prune.py               # Pruning engine (magnitude, neuron, task, structured, Fisher)
      ├── bootstrap.py                # CLI entry point with all commands
      ├── config.py                   # Configuration management
      ├── utils.py                    # Shared utilities (memory, checkpoints, validation)
      ├── lazy_model_manager.py       # Registry and student creation
      ├── lazy_infer.py               # Inference engines (GGUF, Ollama, Transformers, LazyTorch, vLLM)
      ├── lazy_distill.py             # Distillation engine (KL and fine‑tuning)
      ├── lazy_tui.py                 # Rich TUI interface
      ├── dashboard_server.py         # Web dashboard
      ├── benchmark.py                # Benchmarking (TPS, perplexity, MC, long‑context)
      ├── endless_rl.py               # Endless self‑improvement loop
      ├── micro_moe.py                # Mixture of Experts implementation
      ├── lazy_speculative.py         # Speculative decoding (DraftHead, MedusaHead)
      ├── zero_shot_compensation.py   # Adapter‑based compensation
      ├── e8_quantize.py              # E8 lattice quantization
      ├── kv_compressor.py            # KV cache compression
      ├── model_merging.py            # Model merging strategies
      ├── metrics_store.py            # Metrics collection for UI
      ├── lazytorch_core.py           # LazyTorch memory‑mapped core

```

# 2. Run this unified startup script:

```bash
cd ~/lazyllama/ (The folder you have the files in)

sudo rm -rf /tmp/* /tmp/.* 2>/dev/null; export TMPDIR=~/tmp; mkdir -p ~/tmp; (for tmp storage cleanup issues)

chmod +x ~/lazyllama/start.sh
cd ~/lazyllama/ && ./start.sh
```

### Settings

| Field | Default | Description |
|-------|---------|-------------|
| `use_lazytorch` | `true` | Memory‑mapped loading (extreme RAM savings). |
| `use_e8_quantization` | `false` | Apply E8 quantization (2‑5 bits per weight). |
| `e8_bits_per_weight` | `4.0` | Bits per weight (2.0‑5.0). |
| `use_kv_cache_compression` | `true` | Compress KV cache with TurboQuant. |
| `kv_cache_bits` | `4` | Bits for KV cache (4‑8). |
| `reap_prune_ratio` | `0.15` | Fraction of weights to prune (gentle default). |
| `reap_force_cpu` | `true` | Force CPU for REAP (avoids GPU memory issues). |
| `endless_max_cycles` | `-1` | Number of auto‑improvement cycles (-1 = infinite). |
| `endless_policy` | `epsilon_greedy` | Action selection policy (worst, best, random, epsilon_greedy). |
| `hyperparameter_search_enabled` | `false` | Use hyperparameter search in endless auto loop. |
| `use_qlora` | `true` | Enable QLoRA (4‑bit + LoRA) for distillation/finetuning. |
| `qlora_r` | `16` | LoRA rank (effective rank) – higher for recovery. |
| `use_zero_shot_compensation` | `true` | Apply adapters after pruning to recover performance. |
| `distill_alpha` | `0.8` | Teacher signal strength in distillation (higher = more teacher influence). |


Or edit the JSON file directly.

---

## CLI Command Reference

All commands are invoked via `python -m lazy_llama.bootstrap <command> [options]`.

| Command | Description |
|---------|-------------|
| `chat` | Start interactive chat with a model. |
| `download` | Download a model (Hugging Face or Ollama). |
| `create-student` | Create a student from a base model (with LazyTorch conversion option). |
| `export-zip` | Export a model as a zip archive (with/without Hugging Face metadata). |
| `import-zip` | Import a model from a zip archive. |
| `rename` | Rename a registered model. |
| `remove` | Delete a model and its files. |
| `benchmark-students` | Benchmark all student models (TPS, memory, long‑context, perplexity, MC accuracy). |
| `recover` | One‑click recovery: prune (15%) + distill with QLoRA to restore performance. |
| `endless distill` | Run endless distillation (teacher -> student). |
| `endless prune` | Run endless pruning (cycling strategies). |
| `endless auto` | Run the global self‑improvement loop (benchmark -> decide -> act -> repeat). |
| `health-check` | Check all models for issues (corrupt tokenizers, missing files, orphaned checkpoints). |
| `dashboard` | Launch the web dashboard (same as TUI menu). |

Use `--help` on any command for detailed options.

---

## More Features

### LazyTorch – Extreme Memory Savings

- Weights are stored as memory‑mapped NumPy arrays.
- Only the current forward pass loads the needed parameters.
- Peak RAM for a 7B model under 500MB (compared to over 14GB for PyTorch).
- Supports all Hugging Face architectures (except those with custom dynamic layers).

### E8 Quantization

- Uses an 8‑dimensional lattice (E8) to compress weights to 2‑5 bits.
- Lossy but preserves most performance (accuracy degradation under 2% for 4‑bit).
- Also supports external GGUF quantization via llama.cpp.

### KV Cache Compression

- TurboQuant: rotate + uniform codebook, 4‑8 bits per element.
- Mixed‑dimension: reduce dimension for older tokens (experimental).
- Residual tokens (default 128) keep recent tokens uncompressed for high accuracy.

### Micro MoE (Mixture of Experts)

- Replaces dense FFN layers with sparse MoE (4‑16+ experts, top‑1/4 routing).
- Hierarchical routing reduces overhead for many experts.
- Static routing (KMeans) makes inference deterministic and fuses well with llama.cpp/Ollama.

### Speculative Decoding (Medusa‑style)

- Draft head predicts multiple tokens in parallel.
- Tree‑based verification with greedy prefix acceptance.
- Confidence‑based early stopping.
- Multiple draft heads can be trained (distillation) for longer drafts.

### Zero‑Shot Compensation

- Low‑rank adapters compensate for errors from pruning or quantization.
- Trained on calibration data without labelled outputs.

### Model Merging

- Supports SLERP, TIES, DARE, and simple averaging.
- Used in the auto‑loop to combine top‑performing models.

### Recovery Pipeline

- One‑click `recover` command: prunes the model (15% gentle ratio) and immediately runs QLoRA distillation to recover performance.
- Integrated into `create-student --with-recovery` for new students.
- All parameters (prune ratio, distillation passes) are configurable.

### REAP & Endless Self‑Improvement

- REAP Pipeline: Refine -> Evaluate -> Apply -> Persist.
- Stages: Distillation -> Pruning -> Recovery -> Evaluation.
- Endless RL Loop:
  - Benchmarks all models (TPS, perplexity).
  - Chooses worst/best/random/epsilon‑greedy policy.
  - Applies distillation or pruning (with hyperparameter search).
  - Checks perplexity threshold (default 80) – rejects bad models.
  - Merges top models and prunes the pool to max size.
  - Saves state for resuming.

### Health Check Command

The `health-check` command scans all models in the registry and reports issues:

```bash
python -m lazy_llama.bootstrap health-check
```

It checks for:
- Missing model paths
- Corrupt tokenizers
- Missing weight files
- Invalid LazyTorch manifests
- Orphaned checkpoints

---

## Troubleshooting

### Out‑of‑Memory (OOM)

- Enable LazyTorch (`use_lazytorch: true`).
- Reduce `max_seq_len` (default 512).
- Use E8 quantization with lower bits (2‑3).
- Use KV cache compression with lower bits (2‑4).
- Use QLoRA (4‑bit) for distillation/finetuning.

### Illegal Instruction Error

This error occurs on older CPUs without AVX2 support. The startup scripts automatically install the CPU‑only version of PyTorch. If you need to fix it manually:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Slow Inference

- Enable speculative decoding (train a draft head).
- Use static routing (Micro MoE) to reduce routing overhead.
- Reduce `top_k` in generation (default 1).

### Tokenizer Errors

- Run `python -m lazy_llama.bootstrap remove --model <model_name>` and re‑download.
- Ensure the model is a valid Hugging Face directory (not just a GGUF file).
- If you see `Padding_idx must be within num_embeddings`, update LazyTorch to the latest version (it auto‑fixes this).

### Ollama Connection Issues

- Ensure Ollama is running (`ollama serve`).
- Check `ollama_timeout` in config (increase if needed).
- Verify the teacher model is pulled (`ollama pull <model>`).

### Pruned Models Are Too Degraded

- Use the `recover` command immediately after pruning.
- Enable `use_zero_shot_compensation` in config.
- Reduce `reap_prune_ratio` to 0.10‑0.15.
- Increase `qlora_r` to 16 or 32 during recovery.
- Run `benchmark-students` to check perplexity – if over 80, reject the model.

### Tmp is Full

- Run this bash for Linux Debian to clear tmp:
  ```bash
  sudo rm -rf /tmp/* /tmp/.* 2>/dev/null; export TMPDIR=~/tmp; mkdir -p ~/tmp; (for tmp storage cleanup issues)
  ```
  
---

## License

MIT

# SCREENSHOTS:

![Alt Text](https://github.com/webxos/LazyLlama/blob/main/assets/screen1.png)

![Alt Text](https://github.com/webxos/LazyLlama/blob/main/assets/screen2.png)

![Alt Text](https://github.com/webxos/LazyLlama/blob/main/assets/screen3.png)

![Alt Text](https://github.com/webxos/LazyLlama/blob/main/assets/screen4.png)

![Alt Text](https://github.com/webxos/LazyLlama/blob/main/assets/screen5.png)

![Alt Text](https://github.com/webxos/LazyLlama/blob/main/assets/screen6.png)

![Alt Text](https://github.com/webxos/LazyLlama/blob/main/assets/screen7.png)

![Alt Text](https://github.com/webxos/LazyLlama/blob/main/assets/screen8.png)

