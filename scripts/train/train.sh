#!/usr/bin/env bash
# Linux/CUDA entry point. Preserve the author's training configuration.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src/transformers/src:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="${1:-config/ad_bev_v4.yaml}"
if (( $# > 0 )); then shift; fi
exec torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" src/train.py "$CONFIG" "$@"
