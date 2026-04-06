#!/usr/bin/env bash
set -euo pipefail

python -m feathervit_emotion.benchmark \
  --checkpoint "${1:-}" \
  --img-size "${2:-256}" \
  --batch-size "${3:-1}" \
  --warmup "${4:-20}" \
  --iters "${5:-200}" \
  --device "${6:-auto}"
