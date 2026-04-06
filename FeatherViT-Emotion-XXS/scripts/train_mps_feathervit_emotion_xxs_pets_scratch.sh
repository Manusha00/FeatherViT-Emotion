#!/usr/bin/env bash
set -euo pipefail

# Preset for MacBook M3 Pro + MPS, training from scratch on pets_emotion dataset.
python -m feathervit_emotion.train \
  --train-dir "datasets/pets_emotion/train" \
  --val-dir "datasets/pets_emotion/valid" \
  --output-dir "runs/feathervit_emotion_xxs_pets_m3_mps_scratch" \
  --epochs 120 \
  --batch-size 32 \
  --num-workers 4 \
  --img-size 224 \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --dropout 0.2 \
  --seed 42 \
  --save-every 1 \
  --val-every 1 \
  --device mps
