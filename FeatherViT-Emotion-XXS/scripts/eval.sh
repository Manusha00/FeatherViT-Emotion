#!/usr/bin/env bash
set -euo pipefail

python -m feathervit_emotion.evaluate \
  --val-dir "${1:?val_dir required}" \
  --checkpoint "${2:?checkpoint required}" \
  --img-size "${3:-256}" \
  --batch-size "${4:-64}" \
  --device "${5:-auto}"
