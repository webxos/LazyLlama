#!/usr/bin/env python3
"""
setup.py - Setup for Lazy Llama with CPU/GPU extras.
Reads configuration from pyproject.toml when available.
"""

import setuptools

# Try to read version and description from pyproject.toml
try:
    import tomli
except ImportError:
    tomli = None

version = "3.6.0"
description = "Low‑end inference engine with distillation, pruning, E8 quantization, KV compression, LazyTorch memory-mapped loading, and Endless RL self‑improvement"

if tomli:
    try:
        with open("pyproject.toml", "rb") as f:
            data = tomli.load(f)
            version = data["project"]["version"]
            description = data["project"]["description"]
    except Exception:
        pass

# Base dependencies (CPU‑friendly, matches requirements.txt)
install_requires = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",
    "transformers>=4.35.0",
    "accelerate>=0.25.0",
    "peft>=0.7.0",
    "flask>=2.3.0",
    "flask-cors>=4.0.0",
    "huggingface-hub>=0.22.0",
    "psutil>=5.9.0",
    "rich>=13.7.0",
    "textual>=0.41.0",
    "markdown>=3.5.0",
    "requests>=2.31.0",
    "aiohttp>=3.9.0",
    "tqdm>=4.66.0",
    "numpy>=1.26.0,<3.0.0",
    "sentencepiece>=0.2.0",
    "protobuf>=3.20.0",
    "tokenizers>=0.19.0",
    "safetensors>=0.3.0",
    "nltk>=3.8.1",
    "packaging>=24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.5.0",
    "ollama-python>=0.1.0",
    "llama-cpp-python>=0.2.27",
]

# Optional extras
extras_require = {
    "cpu": [],  # default, no extra needed
    "gpu": [
        "bitsandbytes>=0.41.0",
        "cuda-python>=12.0",
    ],
    "dev": [
        "pytest>=7.0",
        "black>=23.0",
        "flake8>=6.0",
        "mypy>=1.0",
    ],
    "all": [
        "bitsandbytes>=0.41.0",
        "cuda-python>=12.0",
        "pytest>=7.0",
        "black>=23.0",
        "flake8>=6.0",
        "mypy>=1.0",
    ],
}

setuptools.setup(
    name="lazy_llama",
    version=version,
    description=description,
    author="Lazy Llama Team",
    author_email="team@lazy-llama.ai",
    license="MIT",
    # The source code is inside the 'lazy_llama' directory, which is the top-level package.
    # We set package_dir to map the root package to that directory.
    package_dir={"": "lazy_llama"},
    # find_packages() will discover all packages under the root, including the top-level 'lazy_llama'
    # because we set where="." (the current directory) and package_dir points to the source.
    packages=setuptools.find_packages(where="."),
    package_data={
        "lazy_llama": [
            "templates/*.html",
            "static/*",
            "py.typed",
        ],
    },
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "lazyllama = lazy_llama.bootstrap:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
)