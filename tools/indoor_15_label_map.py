from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


IGNORE_LABEL = 255

EXCLUDED_SOURCE_NAMES = [
    "door knob",
    "door lock",
    "faucet",
]

TARGET_CLASSES = [
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

TARGET_CLASS_TO_ID = {name: index for index, name in enumerate(TARGET_CLASSES)}

TARGET_SOURCE_NAMES = {
    "floor": ["floor", "carpet", "rug", "mat", "floor mat", "carpets"],
    "wall_boundary": [
        "wall",
        "window",
        "curtain",
        "blinds",
        "blind",
        "window frame",
        "window screen",
        "window shade",
        "window sill",
    ],
    "door": [
        "door",
        "garage door",
        "shower door",
        "doorway",
        "door frame",
        "entrance",
        "door way",
        "door opening",
        "opening",
        "entryway",
    ],
    "stairs": ["stairs", "steps", "stair", "staircase"],
    "chair": ["chair", "stool", "bench"],
    "table": ["table", "desk", "counter"],
    "sofa": ["sofa", "sofas"],
    "bed": ["bed", "mattress"],
    "storage": [
        "cabinet",
        "shelf",
        "shelves",
        "bookshelf",
        "cupboard",
        "dresser",
        "drawer",
        "rack",
        "bookcase",
        "book case",
        "night stand",
        "wardrobe",
        "chest of drawers",
    ],
    "small_obstacle": [
        "box",
        "bin",
        "dustbin",
        "garbage bin",
        "basket",
        "clothes",
        "bag",
        "shoe",
        "shoes",
        "toy",
        "cord",
        "cable",
        "recycling bin",
        "recycle bin",
        "dust bin",
        "waste bin",
        "laundry basket",
    ],
    "person": ["person", "people", "human"],
    "sanitary": ["toilet", "sink", "bathtub", "shower curtain"],
    "household_item": [
        "pillow",
        "book",
        "books",
        "paper",
        "towel",
        "keyboard",
        "mouse",
        "bottle",
        "plate",
        "bowl",
        "telephone",
        "notebook",
        "newspaper",
        "tissue box",
        "remote control",
        "stapler",
        "headphones",
        "dish",
        "knife",
        "spoon",
        "fork",
        "utensil",
        "cell phone",
    ],
    "other_obstacle": [
        "fridge",
        "refrigerator",
        "refridgerator",
        "lamp",
        "tv",
        "monitor",
        "computer",
        "printer",
        "machine",
        "plant",
        "pillar",
    ],
    "cup": [
        "cup",
        "cups",
        "cup and saucer",
        "coffee cup",
        "coca cola cup",
        "cup with cloth",
        "cup with silverware",
        "disposable coffee cup",
        "disposable cup",
        "foam cup",
        "measuring cup",
        "paper cup",
        "pen cup",
        "pink plastic cup",
        "plastic cup",
        "plastic cup of coffee",
        "porcelain cup",
        "sideways cup",
        "sippy cup",
        "stack of cups",
        "strofoam cup",
        "styrofoam coffee cup",
        "styrofoam cup",
        "tea cup",
        "tea cup and saucer",
        "toothbrush cup",
        "travel cup",
    ],
}

IGNORE_SOURCE_NAMES = [
    "background",
    "unknown",
    "unlabeled",
    "unlabeled / unknown",
    "ceiling",
    "picture",
    "mirror",
    "whiteboard",
    "poster",
    "tag",
    "outlet",
    *EXCLUDED_SOURCE_NAMES,
]


def normalize_label_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", " ")
    return " ".join(normalized.split())


def build_name_lookup(name_to_raw_id: Mapping[str, int]) -> Dict[str, List[dict]]:
    lookup: Dict[str, List[dict]] = {}
    for source_name, raw_id in name_to_raw_id.items():
        key = normalize_label_name(source_name)
        item = {"raw_id": int(raw_id), "source_name": str(source_name)}
        if item not in lookup.setdefault(key, []):
            lookup[key].append(item)

    for items in lookup.values():
        items.sort(key=lambda item: (item["raw_id"], item["source_name"]))
    return lookup


def _resolve_names(
    source_names: Iterable[str],
    lookup: Mapping[str, List[dict]],
) -> Tuple[List[dict], List[str]]:
    resolved: List[dict] = []
    missing: List[str] = []

    for source_name in source_names:
        matches = lookup.get(normalize_label_name(source_name), [])
        if not matches:
            missing.append(source_name)
            continue
        for match in matches:
            item = {
                "raw_id": int(match["raw_id"]),
                "source_name": match["source_name"],
                "matched_source_name": source_name,
            }
            if item not in resolved:
                resolved.append(item)

    resolved.sort(key=lambda item: (item["raw_id"], item["source_name"], item["matched_source_name"]))
    return resolved, missing


def build_target_mapping(name_to_raw_id: Mapping[str, int]):
    lookup = build_name_lookup(name_to_raw_id)
    raw_to_target: Dict[int, int] = {}
    class_info: List[dict] = []
    flat_mapping: List[dict] = []

    for target_id, target_name in enumerate(TARGET_CLASSES):
        source_names = TARGET_SOURCE_NAMES[target_name]
        raw_ids, missing_names = _resolve_names(source_names, lookup)

        for item in raw_ids:
            raw_id = int(item["raw_id"])
            previous_target = raw_to_target.get(raw_id)
            if previous_target is not None and previous_target != target_id:
                previous_name = TARGET_CLASSES[previous_target]
                raise ValueError(
                    f"Raw label {raw_id} ({item['source_name']}) maps to both "
                    f"{previous_name} and {target_name}"
                )
            raw_to_target[raw_id] = target_id
            flat_mapping.append(
                {
                    "raw_id": raw_id,
                    "source_name": item["source_name"],
                    "matched_source_name": item["matched_source_name"],
                    "target_id": target_id,
                    "target_name": target_name,
                }
            )

        class_info.append(
            {
                "target_id": target_id,
                "target_name": target_name,
                "source_names": list(source_names),
                "raw_ids": raw_ids,
                "missing_source_names": missing_names,
            }
        )

    ignored_info = []
    for source_name in IGNORE_SOURCE_NAMES:
        raw_ids, missing_names = _resolve_names([source_name], lookup)
        ignored_info.append(
            {
                "source_name": source_name,
                "raw_ids": raw_ids,
                "missing": bool(missing_names),
            }
        )

    flat_mapping.sort(key=lambda item: (item["target_id"], item["raw_id"], item["source_name"]))
    return raw_to_target, class_info, flat_mapping, ignored_info


def class_names_text(ignore_label: int = IGNORE_LABEL) -> str:
    lines = [f"{target_id}\t{name}" for target_id, name in enumerate(TARGET_CLASSES)]
    lines.append(f"{ignore_label}\tignore")
    return "\n".join(lines) + "\n"


def ignored_source_names_text(ignored_info: Iterable[dict]) -> str:
    lines = []
    for item in ignored_info:
        raw_ids = item["raw_ids"]
        if raw_ids:
            raw_text = ", ".join(f"{entry['raw_id']}:{entry['source_name']}" for entry in raw_ids)
        else:
            raw_text = "missing"
        lines.append(f"{item['source_name']}\t{raw_text}")
    return "\n".join(lines) + "\n"


def write_mapping_metadata(
    output_root: Path,
    class_info: List[dict],
    flat_mapping: List[dict],
    ignored_info: List[dict],
    ignore_label: int = IGNORE_LABEL,
) -> None:
    metadata = {
        "ignore_label": int(ignore_label),
        "excluded_source_names": list(EXCLUDED_SOURCE_NAMES),
        "target_classes": [
            {"target_id": target_id, "target_name": name}
            for target_id, name in enumerate(TARGET_CLASSES)
        ],
        "classes": class_info,
        "ignored_source_names": ignored_info,
    }
    (output_root / "target_label_mapping.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "selected_name_to_raw_id.json").write_text(
        json.dumps(class_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "source_name_to_target_id.json").write_text(
        json.dumps(flat_mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "class_names.txt").write_text(
        class_names_text(ignore_label),
        encoding="utf-8",
    )
    (output_root / "ignored_source_names.txt").write_text(
        ignored_source_names_text(ignored_info),
        encoding="utf-8",
    )
