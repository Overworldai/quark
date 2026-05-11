"""pytest configuration: ensure CUDA_PATH is set so cupy can find nvcc."""

import os

os.environ.setdefault("CUDA_PATH", "/usr/local/cuda-12.9")
