#!/usr/bin/env python3
"""Create the paper split for the custom SUNRGBD-15 dataset.

The official development pool (the exported train.txt and val.txt) is
repartitioned deterministically into 4,227 training samples and 1,057
validation samples. The official 5,050-sample test set is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = [
    "floor",
    "wall_boundary",
    "door",
    "stairs",
    "chair",
    "table",
    "sofa",
    "bed",
    "storage",
    "small_obstacle",
    "person",
    "sanitary",
    "household_item",
    "other_obstacle",
    "cup",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "indoor",
    )
    parser.add_argument("--train-count", type=int, default=4227)
    parser.add_argument("--val-count", type=int, default=1057)
    parser.add_argument("--seed", type=int, default=304)
    return parser.parse_args()


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_lines(path: Path, lines: list[str]) -> None:
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def sample_id(line: str) -> str:
    image_path, label_path = line.split()
    image_id = Path(image_path).stem
    label_id = Path(label_path).stem
    if image_id != label_id:
        raise ValueError(f"Image/label ID mismatch: {line}")
    return image_id


def ensure_backup(data_root: Path) -> Path:
    backup_dir = data_root / "splits" / "official_train2665_val2619"
    backup_dir.mkdir(parents=True, exist_ok=True)
    expected_counts = {"train.txt": 2665, "val.txt": 2619, "test.txt": 5050}

    for name, expected_count in expected_counts.items():
        source = data_root / name
        target = backup_dir / name
        if not target.exists():
            lines = read_nonempty_lines(source)
            if len(lines) != expected_count:
                raise ValueError(
                    f"Refusing to create original-split backup: {source} has "
                    f"{len(lines)} lines, expected {expected_count}."
                )
            shutil.copy2(source, target)
        if len(read_nonempty_lines(target)) != expected_count:
            raise ValueError(f"Invalid backup manifest: {target}")

    metadata = {
        "description": "Original exported SUNRGBD trainvalsplit.train/val and alltest manifests",
        "counts": expected_counts,
        "sha256": {name: sha256(backup_dir / name) for name in expected_counts},
    }
    atomic_write_text(
        backup_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    return backup_dir


def split_development_pool(
    backup_dir: Path, train_count: int, val_count: int, seed: int
) -> tuple[list[str], list[str], list[str]]:
    original_train = read_nonempty_lines(backup_dir / "train.txt")
    original_val = read_nonempty_lines(backup_dir / "val.txt")
    test_lines = read_nonempty_lines(backup_dir / "test.txt")
    development = original_train + original_val

    if train_count + val_count != len(development):
        raise ValueError(
            f"Requested {train_count}+{val_count} samples, but the development pool "
            f"contains {len(development)}."
        )
    if len(set(development)) != len(development):
        raise ValueError("Duplicate entries exist in the development pool.")

    development_ids = {sample_id(line) for line in development}
    test_ids = {sample_id(line) for line in test_lines}
    if development_ids & test_ids:
        raise ValueError("Development/test overlap detected before repartitioning.")

    order = list(range(len(development)))
    random.Random(seed).shuffle(order)
    val_indices = set(order[:val_count])
    train_lines = [line for index, line in enumerate(development) if index not in val_indices]
    val_lines = [line for index, line in enumerate(development) if index in val_indices]
    return train_lines, val_lines, test_lines


def validate_paths(data_root: Path, splits: dict[str, list[str]]) -> None:
    missing: list[Path] = []
    for lines in splits.values():
        for line in lines:
            image_rel, label_rel = line.split()
            for relative in (image_rel, label_rel):
                path = data_root / relative
                if not path.is_file():
                    missing.append(path)
                    if len(missing) >= 10:
                        break
            if len(missing) >= 10:
                break
        if len(missing) >= 10:
            break
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(map(str, missing)))


def source_split_by_id(backup_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for line in read_nonempty_lines(backup_dir / f"{split}.txt"):
            result[sample_id(line)] = split
    return result


def write_active_manifest(
    data_root: Path,
    backup_dir: Path,
    splits: dict[str, list[str]],
) -> None:
    source_splits = source_split_by_id(backup_dir)
    path = data_root / "active_split_manifest.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["sample_id", "source_official_split", "active_split", "image_path", "label_path"]
        )
        for active_split, lines in splits.items():
            for line in lines:
                image_path, label_path = line.split()
                identifier = sample_id(line)
                writer.writerow(
                    [identifier, source_splits[identifier], active_split, image_path, label_path]
                )
    temporary.replace(path)


def count_pixels(data_root: Path, lines: list[str]) -> tuple[np.ndarray, int, int]:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    invalid_files = 0
    size_mismatches = 0
    for line in lines:
        image_rel, label_rel = line.split()
        image_path = data_root / image_rel
        label_path = data_root / label_rel
        with Image.open(image_path) as image, Image.open(label_path) as label:
            if image.size != label.size:
                size_mismatches += 1
            values = np.asarray(label, dtype=np.int64)
        invalid = values[(values != 255) & ((values < 0) | (values >= len(CLASS_NAMES)))]
        if invalid.size:
            invalid_files += 1
        valid = values[(values >= 0) & (values < len(CLASS_NAMES))]
        counts += np.bincount(valid, minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]
    return counts, invalid_files, size_mismatches


def write_validation_report(
    data_root: Path,
    splits: dict[str, list[str]],
    seed: int,
) -> None:
    split_ids = {name: {sample_id(line) for line in lines} for name, lines in splits.items()}
    overlap_count = sum(
        len(split_ids[left] & split_ids[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )

    pixel_counts: dict[str, np.ndarray] = {}
    invalid_files = 0
    size_mismatches = 0
    for split, lines in splits.items():
        counts, invalid, mismatch = count_pixels(data_root, lines)
        pixel_counts[split] = counts
        invalid_files += invalid
        size_mismatches += mismatch

    total_counts = sum(pixel_counts.values(), start=np.zeros(len(CLASS_NAMES), dtype=np.int64))
    all_classes_present = all(bool(np.all(pixel_counts[split] > 0)) for split in splits)
    report_lines = [
        "dataset: custom SUNRGBD-15 home-guidance 80/20 development split",
        "source: raw per-sample seg.mat + names",
        "intermediate_37class_labels_used: false",
        "development_pool_source: allsplit.mat trainvalsplit.train + trainvalsplit.val",
        "test_source: allsplit.mat alltest",
        "active_split_protocol: fixed-seed random 80/20 partition of exported development pool",
        f"active_split_seed: {seed}",
        "source_development_count: 5284",
        "source_test_count: 5050",
        f"exported_counts: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}",
        f"exported_total: {sum(map(len, splits.values()))}",
        "skipped_samples: 1",
        "allowed_values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255]",
        f"invalid_label_files: {invalid_files}",
        f"size_mismatches: {size_mismatches}",
        f"split_overlap_count: {overlap_count}",
        f"all_15_classes_present_in_each_split: {str(all_classes_present).lower()}",
    ]
    for heading, counts in [("total", total_counts), *pixel_counts.items()]:
        report_lines.extend(["", f"pixel_counts_{heading}:"])
        report_lines.extend(
            f"  {class_name}: {int(count)}"
            for class_name, count in zip(CLASS_NAMES, counts, strict=True)
        )
    atomic_write_text(data_root / "validation_report.txt", "\n".join(report_lines) + "\n")


def update_provenance(data_root: Path, seed: int, splits: dict[str, list[str]]) -> None:
    path = data_root / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if "split_fields" in provenance:
        provenance["source_split_fields"] = provenance.pop("split_fields")
    if "exported_counts" in provenance:
        provenance["source_exported_counts"] = provenance.pop("exported_counts")
    provenance["active_split"] = {
        "protocol": "fixed-seed random 80/20 partition of the exported development pool",
        "development_pool": "trainvalsplit.train + trainvalsplit.val",
        "test": "alltest (unchanged)",
        "seed": seed,
        "counts": {name: len(lines) for name, lines in splits.items()},
        "original_manifests": "splits/official_train2665_val2619",
    }
    atomic_write_text(path, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    backup_dir = ensure_backup(data_root)
    train_lines, val_lines, test_lines = split_development_pool(
        backup_dir, args.train_count, args.val_count, args.seed
    )
    splits = {"train": train_lines, "val": val_lines, "test": test_lines}
    validate_paths(data_root, splits)

    if len(train_lines) != args.train_count or len(val_lines) != args.val_count:
        raise AssertionError("Generated split counts do not match the request.")
    if read_nonempty_lines(data_root / "test.txt") != test_lines:
        raise ValueError("Active test.txt differs from the backed-up official test manifest.")

    write_lines(data_root / "train.txt", train_lines)
    write_lines(data_root / "val.txt", val_lines)
    write_active_manifest(data_root, backup_dir, splits)
    update_provenance(data_root, args.seed, splits)
    write_validation_report(data_root, splits, args.seed)

    print(f"train={len(train_lines)} val={len(val_lines)} test={len(test_lines)} seed={args.seed}")
    print(f"original manifests: {backup_dir}")
    print(f"active manifests: {data_root}")


if __name__ == "__main__":
    main()
