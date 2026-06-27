#!/usr/bin/env bash
# Idempotent environment setup for the gDTR ICML 2026 reproduction stack.
# Requires Ubuntu 22.04 + CUDA 12.8 driver (or compatible).
set -euo pipefail

# Base deps
apt-get install -y libcudnn9-dev-cuda-12 libnccl-dev || sudo apt-get install -y libcudnn9-dev-cuda-12 libnccl-dev

# PyTorch matched to system CUDA
pip install --quiet torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Scientific stack
pip install --quiet numpy==1.24.4 pandas==2.2.2 scipy==1.14.0 scikit-learn==1.5.1 \
    seaborn==0.13.2 matplotlib==3.9.2 tqdm==4.66.5 pyarrow biopython==1.87 \
    pyfaidx==0.9.0.4 pyBigWig==0.3.25 gffutils==0.14 statannotations==0.7.2 \
    transformers==4.49.0 huggingface_hub==0.36.2

# transformer_engine from source (needs cudnn + nccl headers)
export CPATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/include:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/include:/usr/include/x86_64-linux-gnu
pip install --no-build-isolation --no-binary transformer_engine_torch \
    'transformer_engine[pytorch]==2.16.0' --extra-index-url https://pypi.nvidia.com

# evo2 (pulls vortex, flash-attn)
pip install --quiet evo2

# Rebuild flash-attn against installed torch
pip uninstall -y flash_attn flash-attn
FLASH_ATTENTION_FORCE_BUILD=1 pip install --no-build-isolation flash-attn

echo "[env_setup] DONE"
