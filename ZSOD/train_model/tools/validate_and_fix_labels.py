#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
import sys
import yaml


def load_num_classes(dataset_yaml_path: Path) -> int:
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {dataset_yaml_path}")
    with dataset_yaml_path.open("r") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    if isinstance(names, str):
        try:
            names = eval(names)
        except Exception:
            pass
    if not isinstance(names, list):
        raise ValueError("dataset.yaml has invalid 'names' format")
    return len(names)


def validate_and_fix(labels_dir: Path, num_classes: int) -> dict:
    stats = {
        "files_scanned": 0,
        "lines_seen": 0,
        "kept_lines": 0,
        "removed_bad_fields": 0,
        "removed_bad_parse": 0,
        "removed_bad_class": 0,
        "removed_bad_range": 0,
        "empty_files": 0,
    }

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    for label_path in sorted(labels_dir.glob("*.txt")):
        stats["files_scanned"] += 1
        out_lines = []

        with label_path.open("r") as f:
            for line in f:
                s = line.strip().split()
                if not s:
                    continue
                stats["lines_seen"] += 1

                if len(s) != 5:
                    stats["removed_bad_fields"] += 1
                    continue

                try:
                    class_id = int(s[0])
                    cx, cy, w, h = map(float, s[1:])
                except Exception:
                    stats["removed_bad_parse"] += 1
                    continue

                if class_id < 0 or class_id >= num_classes:
                    stats["removed_bad_class"] += 1
                    continue

                if any(
                    (math.isnan(v) or math.isinf(v) or v < 0.0 or v > 1.0)
                    for v in (cx, cy, w, h)
                ):
                    stats["removed_bad_range"] += 1
                    continue

                out_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with label_path.open("w") as f:
            if out_lines:
                f.write("\n".join(out_lines) + "\n")
                stats["kept_lines"] += len(out_lines)
            else:
                stats["empty_files"] += 1
                f.write("")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate and fix YOLO labels")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Dataset root dir containing images/ and labels/ and dataset.yaml",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    labels_dir = dataset_dir / "labels"
    dataset_yaml_path = dataset_dir / "dataset.yaml"

    try:
        num_classes = load_num_classes(dataset_yaml_path)
    except Exception as e:
        print(f"Error reading dataset.yaml: {e}")
        sys.exit(1)

    stats = validate_and_fix(labels_dir, num_classes)

    print("=== Label Validation Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


