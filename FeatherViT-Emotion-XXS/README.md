# FeatherViT-Emotion-XXS

A standalone, lightweight training and inference package for the **FeatherViT-Emotion-XXS** model.
This repository is designed for reproducible emotion classification workflows with a compact model
that can be trained, evaluated, exported, and integrated into downstream applications.

## Overview

`FeatherViT-Emotion-XXS` provides:

- a self-contained model implementation (no external CVNets dependency),
- an end-to-end `ImageFolder` training pipeline,
- export support for deployment (`TorchScript` + `ONNX`),
- utility scripts for common research and production handoff tasks.

### Model size

- `1.273M` parameters (`--num-classes 1000`)
- `0.953M` parameters (`--num-classes 4`)

## Architecture diagram

![FeatherViT-Emotion-XXS Architecture](./images/feathervit_emotion_architecture_diagram.png)

<p align="center">
  <img src="./images/feathervit_emotion_architecture_diagram.png" alt="FeatherViT-Emotion-XXS Architecture" width="900" />
</p>

The diagram above illustrates the FeatherViT-Emotion-XXS pipeline from input preprocessing to final
emotion classification output.

## Repository layout

```text
FeatherViT-Emotion-XXS/
  feathervit_emotion/
    model.py
    data.py
    utils.py
    train.py
    evaluate.py
    predict.py
    benchmark.py
    export.py
    count_params.py
  configs/
  scripts/
  datasets/
  images/
  runs/
  weights/
  requirements.txt
  README.md
  COLAB_TRAINING_FEATHERVIT_EMOTION_XXS.md
```

## Requirements

- Python `3.10+` recommended
- macOS/Linux/Windows
- optional GPU acceleration (`cuda` or `mps`)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quick start

### 1) Dataset structure

The training and evaluation scripts expect `torchvision.datasets.ImageFolder` format:

```text
dataset_root/
  train/
    class_a/
    class_b/
  valid/   # or val
    class_a/
    class_b/
  test/    # optional
```

Example classes for dog emotion use-case:

```text
angry, happy, relaxed, sad
```

### 2) Train

```bash
python3 -m feathervit_emotion.train \
  --train-dir "datasets/dog_emotion/train" \
  --val-dir "datasets/dog_emotion/valid" \
  --output-dir "runs/feathervit_emotion_xxs_dog" \
  --epochs 80 \
  --batch-size 64 \
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
```

### 3) Evaluate

```bash
python3 -m feathervit_emotion.evaluate \
  --val-dir "datasets/dog_emotion/valid" \
  --checkpoint "runs/feathervit_emotion_xxs_dog/best.pt" \
  --batch-size 64 \
  --img-size 224 \
  --device mps
```

### 4) Predict a single image

```bash
python3 -m feathervit_emotion.predict \
  --image "/path/to/image.jpg" \
  --checkpoint "runs/feathervit_emotion_xxs_dog/best.pt" \
  --img-size 224 \
  --topk 4 \
  --device mps
```

### 5) Export for deployment

```bash
python3 -m feathervit_emotion.export \
  --checkpoint "runs/feathervit_emotion_xxs_dog/best.pt" \
  --output-dir "weights/exports" \
  --img-size 224
```

Generated artifacts:

- `weights/exports/feathervit_emotion_xxs.ts`
- `weights/exports/feathervit_emotion_xxs.onnx`

## CLI modules

| Module | Purpose |
|---|---|
| `feathervit_emotion.train` | Train from scratch or resume from checkpoint |
| `feathervit_emotion.evaluate` | Evaluate checkpoint on validation split |
| `feathervit_emotion.predict` | Run single-image top-k inference |
| `feathervit_emotion.export` | Export TorchScript and ONNX artifacts |
| `feathervit_emotion.count_params` | Report parameter count for class setup |
| `feathervit_emotion.benchmark` | Synthetic throughput/latency benchmark |

### Parameter count examples

```bash
python3 -m feathervit_emotion.count_params --num-classes 1000
python3 -m feathervit_emotion.count_params --num-classes 4
```

## Preset scripts

Convenience scripts are provided in `scripts/`:

- `train_mps_feathervit_emotion_xxs_dog_scratch.sh`
- `train_mps_feathervit_emotion_xxs_pets_scratch.sh`
- `train_mps_feathervit_emotion_xxs_pets_scratch_val5.sh`
- `eval_feathervit_emotion_xxs_dog.sh`
- generic wrappers: `train.sh`, `eval.sh`, `export.sh`, `benchmark.sh`

Run examples:

```bash
bash scripts/train_mps_feathervit_emotion_xxs_dog_scratch.sh
bash scripts/eval_feathervit_emotion_xxs_dog.sh
```

## Config files

The YAML files in `configs/` are experiment presets/documentation snapshots for reproducibility:

- `configs/feathervit_emotion_xxs_dog_m3_mps_scratch.yaml`
- `configs/feathervit_emotion_xxs_pets_m3_mps_scratch.yaml`
- `configs/feathervit_emotion_xxs_pets_m3_mps_scratch_val_every_5.yaml`

These are not auto-loaded by the CLI directly; pass values through command-line flags or shell scripts.

## Colab workflow

For a full Google Colab setup (Drive mount, training, export, and sync), refer to:

- `COLAB_TRAINING_FEATHERVIT_EMOTION_XXS.md`

## Benchmark note

`feathervit_emotion.benchmark` is intended for synthetic inference profiling.
For CUDA, use `--device cuda --amp` for AMP profiling.

```bash
python3 -m feathervit_emotion.benchmark \
  --checkpoint "runs/feathervit_emotion_xxs_dog/best.pt" \
  --num-classes 4 \
  --img-size 224 \
  --batch-size 1 \
  --warmup 20 \
  --iters 200 \
  --device cuda \
  --amp
```

## Integration

The exported ONNX model is used directly by the application backend in:

- `../FeatherViT-Emotion-XXS-App/backend`

Recommended deployment artifact path:

- `weights/exports/feathervit_emotion_xxs.onnx`
