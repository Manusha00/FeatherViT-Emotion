from __future__ import annotations

import argparse
import os

import torch

from .model import build_feathervit_emotion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export FeatherViT-Emotion-XXS")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--onnx-opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = payload.get("class_to_idx", {})
    num_classes = len(class_to_idx) if class_to_idx else 1000

    model = build_feathervit_emotion(num_classes=num_classes).eval()
    model.load_state_dict(payload["model"])
    example = torch.randn(1, 3, args.img_size, args.img_size)

    ts_path = os.path.join(args.output_dir, "feathervit_emotion_xxs.ts")
    onnx_path = os.path.join(args.output_dir, "feathervit_emotion_xxs.onnx")

    traced = torch.jit.trace(model, example)
    traced.save(ts_path)

    torch.onnx.export(
        model,
        example,
        onnx_path,
        export_params=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.onnx_opset,
    )

    print(f"TorchScript: {ts_path}")
    print(f"ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
