from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = [
    "floor",
    "wall",
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
    "window",
    "other_obstacle",
    "cup",
]

RARE_CLASS_REPEAT = {
    3: 4,  # stairs
    10: 3,  # person
    11: 2,  # sanitary
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a rare-class boosted train list for the 15-class indoor dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        # Previous project-local default:
        # default=Path("F:/PIDnet_RT/PIDNet_TensorRT/data/indoor"),
        default=Path("F:/SUNRGBD_Data/processed"),
        help="Central RGB-D dataset root containing train_rgbd.txt, images/, depths/, and labels/.",
    )
    # Previous RGB-only list default kept for reference:
    # parser.add_argument("--input", default="train.lst", help="Source train list filename.")
    parser.add_argument("--input", default="train_rgbd.txt", help="Source RGB-D train list filename.")
    parser.add_argument(
        "--output",
        # Previous RGB-only output kept for reference:
        # default="train_rare_boost.lst",
        default="train_rgbd_rare_boost.txt",
        help="Output train list filename with repeated rare-class samples.",
    )
    parser.add_argument(
        "--report",
        default="train_rare_boost_report.txt",
        help="Output report filename for the generated sampling policy and counts.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_values(label_path: Path) -> set[int]:
    label = np.asarray(Image.open(label_path))
    return {int(value) for value in np.unique(label) if int(value) != 255}


def repeat_for(values: set[int]) -> int:
    repeat = 1
    for class_id, class_repeat in RARE_CLASS_REPEAT.items():
        if class_id in values:
            repeat = max(repeat, class_repeat)
    return repeat


def main() -> None:
    args = parse_args()
    root = args.root
    input_path = root / args.input
    output_path = root / args.output
    report_path = root / args.report

    rows = read_rows(input_path)
    boosted_rows: list[str] = []
    original_presence: Counter[int] = Counter()
    boosted_presence: Counter[int] = Counter()
    repeat_distribution: Counter[int] = Counter()

    for row in rows:
        parts = row.split()
        # RGB-only parsing kept for reference:
        # if len(parts) != 2:
        #     raise ValueError(f"Expected two columns in {input_path}: {row!r}")
        # image_rel, label_rel = parts
        if len(parts) != 3:
            raise ValueError(f"Expected three RGB-D columns in {input_path}: {row!r}")

        image_rel, depth_rel, label_rel = parts
        image_path = root / image_rel
        depth_path = root / depth_rel
        label_path = root / label_rel
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)
        if not label_path.exists():
            raise FileNotFoundError(label_path)

        values = label_values(label_path)
        repeat = repeat_for(values)
        repeat_distribution[repeat] += 1
        boosted_rows.extend([row] * repeat)

        for value in values:
            if 0 <= value < len(CLASS_NAMES):
                original_presence[value] += 1
                boosted_presence[value] += repeat

    output_path.write_text("\n".join(boosted_rows) + "\n", encoding="utf-8")

    report_lines = [
        "rare-class boosted training list report",
        f"source_list: {args.input}",
        f"output_list: {args.output}",
        f"original_rows: {len(rows)}",
        f"boosted_rows: {len(boosted_rows)}",
        f"added_rows: {len(boosted_rows) - len(rows)}",
        "policy: total repeat count is max matched rare-class repeat per image; cup is not repeated",
        "rare_class_repeat:",
    ]
    for class_id, repeat in sorted(RARE_CLASS_REPEAT.items()):
        report_lines.append(f"  {class_id},{CLASS_NAMES[class_id]},{repeat}")

    report_lines.append("repeat_distribution:")
    for repeat, count in sorted(repeat_distribution.items()):
        report_lines.append(f"  repeat_{repeat}: {count}")

    report_lines.append("class_presence_rows:")
    report_lines.append("class_id,class_name,original_rows,boosted_rows")
    for class_id, class_name in enumerate(CLASS_NAMES):
        report_lines.append(
            f"{class_id},{class_name},{original_presence[class_id]},{boosted_presence[class_id]}"
        )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
