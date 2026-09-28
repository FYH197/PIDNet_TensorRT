import argparse
from pathlib import Path

import _init_paths  # noqa: F401
import torch

import models


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "output" / "datamix" / "2026-07-07-21-48-51" / "best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "deploy" / "pidnet_s_indoor_15_640x480.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Export the 15-class indoor PIDNet-S checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to the trained best.pt file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output ONNX file path.")
    parser.add_argument("--arch", default="pidnet-s", choices=["pidnet-s", "pidnet-m", "pidnet-l"], help="PIDNet variant.")
    parser.add_argument("--num-classes", default=15, type=int, help="Number of segmentation classes.")
    parser.add_argument("--in-channels", default=4, type=int, choices=[3, 4], help="Input channels; RGB-D uses 4.")
    parser.add_argument("--height", default=480, type=int, help="Input tensor height.")
    parser.add_argument("--width", default=640, type=int, help="Input tensor width.")
    parser.add_argument("--opset", default=11, type=int, help="ONNX opset version.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--check-only", action="store_true", help="Load weights and run one forward pass without exporting.")
    parser.add_argument("--skip-onnx-check", action="store_true", help="Skip onnx.checker validation after export.")
    return parser.parse_args()


def normalize_key(key):
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_checkpoint_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()

    matched = {}
    skipped = []
    for key, value in state_dict.items():
        normalized = normalize_key(key)
        if normalized in model_state and hasattr(value, "shape") and value.shape == model_state[normalized].shape:
            matched[normalized] = value
        else:
            skipped.append(key)

    if not matched:
        raise RuntimeError(f"No checkpoint tensors matched model '{type(model).__name__}'.")

    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)

    final_head = "final_layer.conv2.weight"
    if final_head not in matched:
        raise RuntimeError(
            f"Checkpoint did not load {final_head}; check --num-classes and checkpoint path."
        )

    return len(matched), len(skipped), tuple(matched[final_head].shape)


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
        args.arch, args.num_classes, in_channels=args.in_channels
    )
    matched, skipped, head_shape = load_checkpoint_weights(model, checkpoint_path)
    model.to(device).eval()

    # RGB-only dummy input kept for reference:
    # dummy_input = torch.randn(1, 3, args.height, args.width, device=device)
    dummy_input = torch.randn(
        1, args.in_channels, args.height, args.width, device=device
    )
    with torch.no_grad():
        output = model(dummy_input)

    print(f"checkpoint: {checkpoint_path}")
    print(f"loaded tensors: {matched}, skipped tensors: {skipped}")
    print(f"final head shape: {head_shape}")
    print(f"input shape: {tuple(dummy_input.shape)}")
    print(f"output shape: {tuple(output.shape)}")

    if args.check_only:
        print("check-only complete; ONNX export skipped.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
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
