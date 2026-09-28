import argparse
from pathlib import Path

import _init_paths  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from export_indoor_onnx import load_checkpoint_weights


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "output" / "datamix" / "2026-07-08-22-02-17" / "best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "deploy" / "best0708_mask.onnx"


class PidnetMaskWrapper(nn.Module):
    def __init__(self, model, output_size, fast_mask=False):
        super().__init__()
        self.model = model
        self.output_size = output_size
        self.fast_mask = fast_mask

    def forward(self, x):
        logits = self.model(x)
        if self.fast_mask:
            mask = torch.argmax(logits, dim=1, keepdim=True).to(torch.float32)
            return F.interpolate(mask, size=self.output_size, mode="nearest")

        logits = F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
        return torch.argmax(logits, dim=1, keepdim=True).to(torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Export PIDNet with GPU argmax and mask resize in ONNX.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to the trained best.pt file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output ONNX file path.")
    parser.add_argument("--arch", default="pidnet-s", choices=["pidnet-s", "pidnet-m", "pidnet-l"], help="PIDNet variant.")
    parser.add_argument("--num-classes", default=15, type=int, help="Number of segmentation classes.")
    parser.add_argument("--in-channels", default=4, type=int, choices=[3, 4], help="Input channels; RGB-D uses 4.")
    parser.add_argument("--height", default=480, type=int, help="Input tensor height.")
    parser.add_argument("--width", default=640, type=int, help="Input tensor width.")
    parser.add_argument("--opset", default=11, type=int, help="ONNX opset version.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--fast-mask", action="store_true", help="Argmax at 60x80 then resize mask; faster but blockier.")
    parser.add_argument("--skip-onnx-check", action="store_true", help="Skip onnx.checker validation after export.")
    parser.add_argument("--dad-enabled", action="store_true", help="Enable stage-4 DAD; must match checkpoint.")
    parser.add_argument("--dad-channels", type=int, default=16, help="DAD projection channels; must match checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    # RGB-only construction kept for reference:
    # model = models.pidnet.get_pred_model(args.arch, args.num_classes)
    model = models.pidnet.get_pred_model(
        # Before DAD: args.arch, args.num_classes, in_channels=args.in_channels
        args.arch, args.num_classes, in_channels=args.in_channels,
        dad_enabled=args.dad_enabled, dad_channels=args.dad_channels
    )
    matched, skipped, head_shape = load_checkpoint_weights(model, checkpoint_path)
    wrapped = PidnetMaskWrapper(model, output_size=(args.height, args.width), fast_mask=args.fast_mask)
    wrapped.to(device).eval()

    # RGB-only dummy input kept for reference:
    # dummy_input = torch.randn(1, 3, args.height, args.width, device=device)
    dummy_input = torch.randn(
        1, args.in_channels, args.height, args.width, device=device
    )
    with torch.no_grad():
        output = wrapped(dummy_input)

    print(f"checkpoint: {checkpoint_path}")
    print(f"loaded tensors: {matched}, skipped tensors: {skipped}")
    print(f"final head shape: {head_shape}")
    print(f"input shape: {tuple(dummy_input.shape)}")
    print(f"output mask shape: {tuple(output.shape)} dtype={output.dtype}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["mask"],
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
    )

    if not args.skip_onnx_check:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    print(f"exported: {output_path}")


if __name__ == "__main__":
    main()
