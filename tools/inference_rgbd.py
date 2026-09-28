"""Run PIDNet RGB-D inference on a paired RGB image and uint16 depth map."""

from __future__ import annotations

import argparse
from pathlib import Path

import _init_paths  # noqa: F401
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import models
from export_indoor_onnx import load_checkpoint_weights


DEPTH_SCALE_MM = 8000.0
DEPTH_MEAN = 0.29934640639806365
DEPTH_STD = 0.158218660916179
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIDNet RGB-D inference")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rgbd(rgb_path: Path, depth_path: Path, width: int, height: int) -> torch.Tensor:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"Unable to read RGB image: {rgb_path}")
    if depth_mm is None:
        raise FileNotFoundError(f"Unable to read depth map: {depth_path}")
    if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
        raise ValueError(
            f"Depth must be single-channel uint16; got {depth_mm.dtype}, {depth_mm.shape}"
        )
    if bgr.shape[:2] != depth_mm.shape:
        raise ValueError(
            f"RGB/depth size mismatch: RGB={bgr.shape[:2]}, depth={depth_mm.shape}"
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)

    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - RGB_MEAN) / RGB_STD
    depth = np.clip(depth_mm.astype(np.float32), 0.0, DEPTH_SCALE_MM) / DEPTH_SCALE_MM
    depth = (depth - DEPTH_MEAN) / DEPTH_STD
    rgbd = np.concatenate([rgb, depth[..., None]], axis=2)
    return torch.from_numpy(rgbd.transpose(2, 0, 1).copy()).unsqueeze(0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = models.pidnet.get_pred_model(
        "pidnet-s", args.num_classes, in_channels=4
    )
    matched, skipped, _ = load_checkpoint_weights(model, args.checkpoint.resolve())
    model.to(device).eval()

    inputs = load_rgbd(
        args.rgb.resolve(), args.depth.resolve(), args.width, args.height
    ).to(device)
    with torch.no_grad():
        logits = model(inputs)
        logits = F.interpolate(
            logits,
            size=(args.height, args.width),
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.argmax(logits, dim=1)[0].to(torch.uint8).cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask):
        raise RuntimeError(f"Unable to write prediction: {args.output}")
    print(f"loaded tensors: {matched}; skipped tensors: {skipped}")
    print(f"input shape: {tuple(inputs.shape)}")
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
