#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from indoor_15_label_map import (
    IGNORE_LABEL,
    TARGET_CLASSES,
    build_target_mapping,
    write_mapping_metadata,
)


SUNRGBD_37_CLASSES = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor_mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "fridge",
    25: "tv",
    26: "paper",
    27: "towel",
    28: "shower_curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "night_stand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap SUNRGBD labels into the indoor 15-class id space."
    )
    parser.add_argument(
        "--input-root",
        # Previous project-local default kept for reference:
        # default=r"F:\PIDnet_RT\PIDNet_TensorRT\data\sunrgbd",
        required=True,
        help="Legacy flat SUNRGBD-37 root containing images/ and labels/",
    )
    parser.add_argument(
        "--output-root",
        # Previous project-local default kept for reference:
        # default=r"F:\PIDnet_RT\PIDNet_TensorRT\data\sunrgbd_15class_aligned",
        required=True,
        help="Explicit output directory for remapped labels, lists and metadata",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=IGNORE_LABEL,
        help="Ignore label for classes outside the target set",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=5050,
        help="Number of leading samples to place into val.lst",
    )
    return parser.parse_args()


def decode_label(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.dtype == np.uint16:
        return (raw // 257).astype(np.uint8)
    return raw.astype(np.uint8)


def build_remap_table(ignore_label: int) -> tuple[np.ndarray, list[dict], list[dict], list[dict]]:
    name_to_sunrgbd_id = {name: raw_id for raw_id, name in SUNRGBD_37_CLASSES.items()}
    raw_to_target, class_info, flat_mapping, ignored_info = build_target_mapping(name_to_sunrgbd_id)

    table = np.full(256, ignore_label, dtype=np.uint8)
    for raw_id, target_id in raw_to_target.items():
        table[int(raw_id)] = int(target_id)
    return table, class_info, flat_mapping, ignored_info


def write_list(list_path: Path, names: list[str], image_prefix: str) -> None:
    lines = [f"{image_prefix}/{name} labels/{name}" for name in names]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    labels_in = input_root / "labels"
    labels_out = output_root / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    if not (input_root / "images").exists() or not labels_in.exists():
        raise FileNotFoundError("input-root must contain images/ and labels/")

    label_paths = sorted(labels_in.glob("*.png"))
    if not label_paths:
        raise RuntimeError(f"No label files found in {labels_in}")

    remap_table, class_info, flat_mapping, ignored_info = build_remap_table(args.ignore_label)
    pixel_counter = {class_name: 0 for class_name in TARGET_CLASSES}

    for index, label_path in enumerate(label_paths, start=1):
        decoded = decode_label(np.array(Image.open(label_path)))
        remapped = remap_table[decoded]
        Image.fromarray(remapped, mode="L").save(labels_out / label_path.name)

        for target_id, class_name in enumerate(TARGET_CLASSES):
            pixel_counter[class_name] += int((remapped == target_id).sum())

        if index % 500 == 0 or index == len(label_paths):
            print(f"Processed {index}/{len(label_paths)}")

    names = [path.name for path in label_paths]
    val_names = names[: args.val_count]
    train_names = names[args.val_count :]
    image_prefix = "../sunrgbd/images"

    write_list(output_root / "all.lst", names, image_prefix)
    write_list(output_root / "train.lst", train_names, image_prefix)
    write_list(output_root / "val.lst", val_names, image_prefix)
    write_mapping_metadata(output_root, class_info, flat_mapping, ignored_info, args.ignore_label)

    pixel_lines = [
        f"{target_id}\t{name}\t{pixel_counter[name]}"
        for target_id, name in enumerate(TARGET_CLASSES)
    ]
    (output_root / "pixel_counts.txt").write_text("\n".join(pixel_lines) + "\n", encoding="utf-8")

    print(f"Output root: {output_root}")
    print(f"Train samples: {len(train_names)}")
    print(f"Val samples: {len(val_names)}")
    print("Active source labels:")
    for item in flat_mapping:
        print(
            f"  SUNRGBD {item['raw_id']:2d} {item['source_name']:<14} "
            f"-> target {item['target_id']:2d} {item['target_name']}"
        )


if __name__ == "__main__":
    main()
