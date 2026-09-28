#!/usr/bin/env python
"""Build aligned 16-bit depth maps and RGB-D split files for SUNRGBD-15.

The existing RGB images, labels, and two-column split files are kept intact.
New outputs are written to ``depths/<split>`` and ``*_rgbd.txt``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
# Previous project-local dataset root kept for reference:
# DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "indoor"
DEFAULT_DATASET_ROOT = Path(r"F:\SUNRGBD_Data\processed")
# Previous project-local raw root kept for reference:
# DEFAULT_RAW_ROOT = Path(
#     r"F:\S2-net\S2Net-master\S2Net-master\data\SUNRGBD\SUNRGBD"
# )
DEFAULT_RAW_ROOT = Path(
    r"F:\SUNRGBD_Data\raw\SUNRGBD"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export decoded SUNRGBD depth maps for the active RGB-D split."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite an existing depth PNG. Existing RGB/label files are never changed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N samples for a smoke test; 0 processes all samples.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def find_single_png(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.png"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one PNG in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def decode_sunrgbd_depth(raw_depth: np.ndarray) -> np.ndarray:
    """Decode SUNRGBD's rotated uint16 depth into clipped millimetres."""
    if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise ValueError(
            f"Expected a single-channel uint16 depth map, got "
            f"dtype={raw_depth.dtype}, shape={raw_depth.shape}"
        )
    decoded = np.bitwise_or(
        np.right_shift(raw_depth, 3),
        np.left_shift(raw_depth, 13),
    ).astype(np.uint16)
    decoded[decoded > 8000] = 8000
    return decoded


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    text = "".join(
        f"{row['image_path']} {row['depth_path']} {row['label_path']}\n"
        for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    dataset_root = args.dataset_root.resolve()
    source_manifest_path = dataset_root / "split_manifest.csv"
    active_manifest_path = dataset_root / "active_split_manifest.csv"

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw SUNRGBD root not found: {raw_root}")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest_path}")
    if not active_manifest_path.is_file():
        raise FileNotFoundError(f"Active manifest not found: {active_manifest_path}")

    source_by_id = {
        row["sample_id"]: row for row in read_csv(source_manifest_path)
    }
    active_rows = read_csv(active_manifest_path)
    if args.limit:
        active_rows = active_rows[: args.limit]

    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    audit_rows: list[dict[str, object]] = []
    failures: list[str] = []
    split_counts: Counter[str] = Counter()
    train_valid_count = 0
    train_sum = 0.0
    train_sum_sq = 0.0

    for position, active in enumerate(active_rows, start=1):
        sample_id = active["sample_id"]
        split = active["active_split"]
        try:
            if split not in split_rows:
                raise ValueError(f"Unsupported active split: {split}")
            source = source_by_id[sample_id]
            rgb_path = dataset_root / active["image_path"]
            label_path = dataset_root / active["label_path"]
            scene_dir = raw_root / source["raw_relative_path"]
            raw_depth_path = find_single_png(scene_dir / "depth_bfx")

            if not rgb_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing RGB or label: {rgb_path}, {label_path}"
                )

            with Image.open(rgb_path) as rgb_image:
                rgb_width, rgb_height = rgb_image.size
            with Image.open(label_path) as label_image:
                label_width, label_height = label_image.size
            if (rgb_width, rgb_height) != (label_width, label_height):
                raise ValueError(
                    f"RGB/label size mismatch: RGB={(rgb_width, rgb_height)}, "
                    f"label={(label_width, label_height)}"
                )

            raw_depth = cv2.imread(str(raw_depth_path), cv2.IMREAD_UNCHANGED)
            if raw_depth is None:
                raise RuntimeError(f"OpenCV could not read {raw_depth_path}")
            depth_mm = decode_sunrgbd_depth(raw_depth)
            original_depth_height, original_depth_width = depth_mm.shape
            resized = (original_depth_width, original_depth_height) != (
                rgb_width,
                rgb_height,
            )
            if resized:
                depth_mm = cv2.resize(
                    depth_mm,
                    (rgb_width, rgb_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            relative_depth = Path("depths") / split / f"{sample_id}.png"
            output_depth = dataset_root / relative_depth
            output_depth.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite or not output_depth.exists():
                temporary_depth = output_depth.with_name(
                    output_depth.stem + ".tmp.png"
                )
                if not cv2.imwrite(str(temporary_depth), depth_mm):
                    raise RuntimeError(f"Failed to write {temporary_depth}")
                temporary_depth.replace(output_depth)

            check_depth = cv2.imread(str(output_depth), cv2.IMREAD_UNCHANGED)
            if (
                check_depth is None
                or check_depth.dtype != np.uint16
                or check_depth.shape != (rgb_height, rgb_width)
            ):
                raise RuntimeError(f"Invalid exported depth map: {output_depth}")

            valid = depth_mm > 0
            valid_values = depth_mm[valid].astype(np.float64) / 8000.0
            valid_count = int(valid.sum())
            if split == "train" and valid_count:
                train_valid_count += valid_count
                train_sum += float(valid_values.sum())
                train_sum_sq += float(np.square(valid_values).sum())

            manifest_row = {
                "image_path": active["image_path"].replace("\\", "/"),
                "depth_path": relative_depth.as_posix(),
                "label_path": active["label_path"].replace("\\", "/"),
            }
            split_rows[split].append(manifest_row)
            split_counts[split] += 1
            audit_rows.append(
                {
                    "sample_id": sample_id,
                    "active_split": split,
                    "raw_depth": raw_depth_path.relative_to(raw_root).as_posix(),
                    "depth_path": relative_depth.as_posix(),
                    "rgb_width": rgb_width,
                    "rgb_height": rgb_height,
                    "source_depth_width": original_depth_width,
                    "source_depth_height": original_depth_height,
                    "resized_to_rgb": int(resized),
                    "valid_depth_ratio": valid_count / depth_mm.size,
                    "min_valid_depth_mm": int(depth_mm[valid].min()) if valid_count else 0,
                    "max_valid_depth_mm": int(depth_mm[valid].max()) if valid_count else 0,
                }
            )
        except Exception as error:
            failures.append(f"{sample_id}: {error}")

        if position % 500 == 0 or position == len(active_rows):
            print(f"Processed {position}/{len(active_rows)} samples")

    audit_path = dataset_root / "rgbd_depth_audit.csv"
    if audit_rows:
        with audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
            writer.writeheader()
            writer.writerows(audit_rows)

    if failures:
        for failure in failures[:20]:
            print(f"ERROR: {failure}")
        failure_path = dataset_root / "rgbd_depth_failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Depth export failed for {len(failures)} samples. See {failure_path}"
        )

    if args.limit:
        print("Smoke-test limit used; split manifests and statistics were not written.")
        return

    expected_counts = {"train": 4227, "val": 1057, "test": 5050}
    if dict(split_counts) != expected_counts:
        raise RuntimeError(
            f"Unexpected split counts: {dict(split_counts)}; expected {expected_counts}"
        )

    for split, rows in split_rows.items():
        write_manifest(dataset_root / f"{split}_rgbd.txt", rows)

    train_mean = train_sum / train_valid_count
    train_variance = max(train_sum_sq / train_valid_count - train_mean**2, 0.0)
    stats = {
        "encoding": "uint16 millimetres",
        "source": "SUNRGBD depth_bfx with official uint16 bit rotation",
        "clip_depth_mm": 8000,
        "network_scale": "depth_mm / 8000",
        "invalid_value": 0,
        "train_valid_pixel_count": train_valid_count,
        "train_valid_mean_0_1": train_mean,
        "train_valid_std_0_1": train_variance**0.5,
        "split_counts": expected_counts,
    }
    (dataset_root / "rgbd_depth_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failure_path = dataset_root / "rgbd_depth_failures.txt"
    if failure_path.exists():
        # Keep prior failure history rather than deleting it.
        failure_path.rename(failure_path.with_suffix(".resolved.txt"))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
