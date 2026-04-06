#!/usr/bin/env bash
set -euo pipefail

python -m feathervit_emotion.evaluate \
  --val-dir "datasets/dog_emotion/valid" \
  --checkpoint "${1:-runs/feathervit_emotion_xxs_dog_m3_mps_scratch/best.pt}" \
  --img-size "${2:-224}" \
  --batch-size "${3:-64}" \
  --device "${4:-mps}"
