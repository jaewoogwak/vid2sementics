#!/usr/bin/env python3
"""
Verify that concatenated videos map to grouped query JSONLs and vice versa.

Checks that every MP4 in the concat directory corresponds to a video_id in
queries_by_video_{train,val}.jsonl, and that every video_id in those JSONLs
has a corresponding MP4 in the concat directory.

Outputs a concise summary and lists a few examples for each mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Set, Tuple


def read_grouped_ids(jsonl_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not jsonl_path.is_file():
        return ids
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] != "{":
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = obj.get("video_id")
            if isinstance(vid, str) and vid:
                ids.add(vid)
    return ids


def list_concat_ids(concat_dir: Path) -> Set[str]:
    ids: Set[str] = set()
    if not concat_dir.is_dir():
        return ids
    for p in concat_dir.glob("*.mp4"):
        ids.add(p.stem)
    return ids


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_concat = repo_root / "dataset" / "qvhighlights" / "concat"
    # Prefer JSONLs in dataset/qvhighlights (as per user), else fallback to concat
    default_json_train = repo_root / "dataset" / "qvhighlights" / "queries_by_video_train.jsonl"
    default_json_val = repo_root / "dataset" / "qvhighlights" / "queries_by_video_val.jsonl"
    alt_json_train = default_concat / "queries_by_video_train.jsonl"
    alt_json_val = default_concat / "queries_by_video_val.jsonl"

    ap = argparse.ArgumentParser(description="Verify concat videos vs grouped JSONLs")
    ap.add_argument("--concat-dir", type=Path, default=default_concat)
    ap.add_argument("--train-jsonl", type=Path, default=default_json_train)
    ap.add_argument("--val-jsonl", type=Path, default=default_json_val)
    ap.add_argument("--fallback-to-concat-jsonl", action="store_true", help="If default JSONLs are missing, try files in concat dir")
    args = ap.parse_args()

    # Apply fallback if requested and defaults don't exist
    if args.fallback_to_concat_jsonl:
        if not args.train_jsonl.is_file() and alt_json_train.is_file():
            args.train_jsonl = alt_json_train
        if not args.val_jsonl.is_file() and alt_json_val.is_file():
            args.val_jsonl = alt_json_val
    return args


def main() -> None:
    args = parse_args()
    concat_dir = args.concat_dir.expanduser()
    train_p = args.train_jsonl.expanduser()
    val_p = args.val_jsonl.expanduser()

    file_ids = list_concat_ids(concat_dir)
    train_ids = read_grouped_ids(train_p)
    val_ids = read_grouped_ids(val_p)
    jsonl_union = train_ids | val_ids

    files_not_in_jsonl = sorted(file_ids - jsonl_union)
    jsonl_not_in_files = sorted(jsonl_union - file_ids)

    print("Verify concat videos vs grouped JSONLs")
    print(f"  concat_dir : {concat_dir}")
    print(f"  train_jsonl: {train_p} ({len(train_ids)} ids)")
    print(f"  val_jsonl  : {val_p} ({len(val_ids)} ids)")
    print(f"  files(.mp4): {len(file_ids)}")
    print()
    print(f"Missing in JSONLs (files minus JSONLs): {len(files_not_in_jsonl)}")
    if files_not_in_jsonl:
        print("  e.g.,", ", ".join(files_not_in_jsonl[:10]))
    print(f"Missing MP4 files (JSONLs minus files): {len(jsonl_not_in_files)}")
    if jsonl_not_in_files:
        print("  e.g.,", ", ".join(jsonl_not_in_files[:10]))


if __name__ == "__main__":
    main()

