#!/usr/bin/env bash
# Single-node Piper joint20 fine-tuning. --prepare-only never starts training.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv/bin/python; install the OpenWAM environment first." >&2
    exit 1
fi
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# This recipe uses PyTorch AdamW and needs no compiled DeepSpeed CUDA ops.
# DeepSpeed otherwise probes optional quantization kernels even at import time.
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
    export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
fi

if [[ "${1:-}" == "--prepare-only" ]]; then
    shift
    exec python scripts/prepare_piper_joint20.py "$@"
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Usage: bash scripts/train_piper_joint20.sh [Hydra overrides...]
       bash scripts/train_piper_joint20.sh --prepare-only [--force] [Hydra overrides...]

Defaults: eight GPUs, local Foundation weights, offline W&B, joint20 interface.
Preparation checks data/statistics before starting torchrun.
To inspect config: .venv/bin/python scripts/train.py --config-name train_piper_joint20 --cfg job --resolve
USAGE
    exit 0
fi

python scripts/prepare_piper_joint20.py "$@"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29510}"
exec bash scripts/train.sh --config-name train_piper_joint20 "$@"
