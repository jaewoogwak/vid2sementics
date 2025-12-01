#!/usr/bin/env python3
"""
Utility to reassign the split field for the first N entries in MSRVTT_untrimmed.json.

By default this script rewrites the first 250 entries to use split="val" and keeps the
remaining entries unchanged (usually "train"). Run it like:

python scripts/set_msrvtt_val_split.py \
    --annotation /dev/ssd1/gjw/prvr/dataset/data/MSRVTT/annotation/MSRVTT_untrimmed.json \
    --count 250
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite MSRVTT untrimmed splits.")
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Path to MSRVTT_untrimmed.json",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=250,
        help="Number of entries from the start to mark as val.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, overwrites the input file in-place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = args.annotation.expanduser()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    entries: List[Dict[str, Any]] = data.get("videos", [])
    count = max(0, min(int(args.count), len(entries)))
    for idx, entry in enumerate(entries):
        if idx < count:
            entry["split"] = "val"
        else:
            # leave existing split if already something else; default to train
            entry.setdefault("split", "train")
    output_path = args.output.expanduser() if args.output else annotation_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {count} entries to split='val' in {output_path}")


if __name__ == "__main__":
    main()
