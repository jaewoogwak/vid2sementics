#!/usr/bin/env python3
"""
Emit unique base video_ids from QVHighlights split JSONLs to stdout.

Base id = vid with trailing _<start>_<end> removed.

Usage examples:
  python scripts/list_split_video_ids.py                  # both splits
  python scripts/list_split_video_ids.py --which train    # only train
  python scripts/list_split_video_ids.py --which val      # only val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Set


def base_video_id(vid: str) -> str:
    parts = vid.rsplit("_", 2)
    return parts[0] if len(parts) >= 3 else vid


def load_base_ids(jsonl_path: Path) -> Set[str]:
    ids: Set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = obj.get("vid") or obj.get("video_id")
            if isinstance(vid, str) and vid:
                ids.add(base_video_id(vid))
    return ids


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_train = repo_root / "dataset" / "qvhighlights" / "highlight_train_release.jsonl"
    default_val = repo_root / "dataset" / "qvhighlights" / "highlight_val_release.jsonl"

    ap = argparse.ArgumentParser(description="List unique base video ids from split JSONLs")
    ap.add_argument("--train-jsonl", type=Path, default=default_train)
    ap.add_argument("--val-jsonl", type=Path, default=default_val)
    ap.add_argument("--which", choices=["both", "train", "val"], default="both")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ids: Set[str] = set()
    if args.which in ("both", "train"):
        p = args.train_jsonl.expanduser()
        if p.is_file():
            ids |= load_base_ids(p)
    if args.which in ("both", "val"):
        p = args.val_jsonl.expanduser()
        if p.is_file():
            ids |= load_base_ids(p)
    for vid in sorted(ids):
        print(vid)


if __name__ == "__main__":
    main()

