#!/usr/bin/env bash
set -euo pipefail

python -m feathervit_emotion.train \
  --train-dir "${1:?train_dir required}" \
  --val-dir "${2:?val_dir required}" \
  --output-dir "${3:-./runs/feathervit_emotion_xxs}" \
  --epochs "${4:-100}" \
  --batch-size "${5:-32}" \
  --img-size "${6:-256}" \
  --device "${7:-auto}" \
  --val-every "${8:-1}"
