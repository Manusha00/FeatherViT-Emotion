#!/usr/bin/env bash
set -euo pipefail

python -m feathervit_emotion.export \
  --checkpoint "${1:?checkpoint required}" \
  --output-dir "${2:-./exports}" \
  --img-size "${3:-256}"
