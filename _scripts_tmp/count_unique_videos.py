#!/usr/bin/env python3
"""
Count unique source videos per split for QVHighlights.

Inputs are JSONL files where each line is a JSON object containing a
`vid` field like "<video_id>_<start>_<end>", e.g. "L9cUEhaXnK4_210.0_360.0".

We strip the trailing "_<start>_<end>" to get the base video id and
report unique counts for train/val and their overlap/union.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Set


def extract_base_vid(vid: str) -> str:
    """Return the base video id by removing the trailing "_<start>_<end>".

    Uses rsplit with maxsplit=2 so that any underscores inside the
    base id (if present) are preserved.
    """
    parts = vid.rsplit("_", 2)
    return parts[0] if len(parts) >= 3 else vid


def load_base_ids(jsonl_path: Path) -> Set[str]:
    base_ids: Set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = obj.get("vid")
            if not isinstance(vid, str):
                continue
            base_ids.add(extract_base_vid(vid))
    return base_ids


def load_segment_ids(jsonl_path: Path) -> Set[str]:
    """Load unique segment ids (full vid string including start/end)."""
    seg_ids: Set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = obj.get("vid")
            if isinstance(vid, str) and vid:
                seg_ids.add(vid)
    return seg_ids


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_train = repo_root / "dataset" / "qvhighlights" / "highlight_train_release.jsonl"
    default_val = repo_root / "dataset" / "qvhighlights" / "highlight_val_release.jsonl"

    ap = argparse.ArgumentParser(description="Count unique base videos per QVHighlights split")
    ap.add_argument("--train-jsonl", type=Path, default=default_train, help="Path to train JSONL")
    ap.add_argument("--val-jsonl", type=Path, default=default_val, help="Path to val JSONL")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.train_jsonl.expanduser()
    val_path = args.val_jsonl.expanduser()

    if not train_path.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Val JSONL not found: {val_path}")

    train_ids = load_base_ids(train_path)
    val_ids = load_base_ids(val_path)

    only_train = train_ids - val_ids
    only_val = val_ids - train_ids
    overlap = train_ids & val_ids
    union = train_ids | val_ids

    print("QVHighlights unique base video counts")
    print(f"  train : {len(train_ids)}")
    print(f"  val   : {len(val_ids)}")
    print(f"  union : {len(union)}")
    print(f"  overlap(train∩val): {len(overlap)}")
    print(f"  only_train       : {len(only_train)}")
    print(f"  only_val         : {len(only_val)}")

    # Segment-level (base + start/end)
    train_segs = load_segment_ids(train_path)
    val_segs = load_segment_ids(val_path)
    seg_overlap = train_segs & val_segs
    seg_union = train_segs | val_segs

    print()
    print("QVHighlights unique segment (vid_start_end) counts")
    print(f"  train : {len(train_segs)}")
    print(f"  val   : {len(val_segs)}")
    print(f"  union : {len(seg_union)}")
    print(f"  overlap(train∩val): {len(seg_overlap)}")


if __name__ == "__main__":
    main()
