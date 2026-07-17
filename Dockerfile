# ============================================================
# LAZY LLAMA v3.6 - Docker Image
# ============================================================
# Multi‑stage build for minimal size; uses Python 3.10 slim.
# CPU‑only PyTorch is installed; for GPU support, override
# the pip index with --index-url https://download.pytorch.org/whl/cu118
# ============================================================

# ---- Stage 1: builder ----
FROM python:3.10-slim AS builder

# Install system build dependencies (required for some packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install PyTorch CPU version and all dependencies into a temporary directory
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install \
    -r requirements.txt

# ---- Stage 2: final image ----
FROM python:3.10-slim

# Install only runtime libraries (no build tools)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy the entire project (excluding .dockerignore)
COPY . .

# Create required directories (for models, checkpoints, etc.)
RUN mkdir -p /root/.lazy_llama/{models,checkpoints,logs,cache,lazytorch_cache,exports}

# ---- Ensure pandas and scikit-learn are installed (fallback) ----
# Even though they are in requirements.txt, this guarantees they are present
RUN pip install --no-cache-dir pandas scikit-learn

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LAZY_PLATFORM=linux
ENV OLLAMA_HOST=http://ollama:11434

# Expose dashboard port
EXPOSE 8080

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import lazy_llama" || exit 1

# Entry point: launch the TUI (or CLI) via bootstrap
ENTRYPOINT ["python", "-m", "lazy_llama.bootstrap"]
# Default command: no arguments -> TUI
CMD []
