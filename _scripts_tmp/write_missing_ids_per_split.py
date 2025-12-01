#!/usr/bin/env python3
"""
Write video_ids present in queries_by_video_{train,val}.jsonl but missing
from the concat directory (no corresponding MP4).

Outputs two text files (one id per line):
  - missing_train_ids.txt
  - missing_val_ids.txt

Defaults assume repository layout:
  - concat dir: dataset/qvhighlights/concat
  - JSONLs    : dataset/qvhighlights/queries_by_video_train.jsonl,
                dataset/qvhighlights/queries_by_video_val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set


def read_ids(jsonl_path: Path) -> Set[str]:
    ids: Set[str] = set()
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
    return {p.stem for p in concat_dir.glob("*.mp4")}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_concat = repo_root / "dataset" / "qvhighlights" / "concat"
    default_train = repo_root / "dataset" / "qvhighlights" / "queries_by_video_train.jsonl"
    default_val = repo_root / "dataset" / "qvhighlights" / "queries_by_video_val.jsonl"

    ap = argparse.ArgumentParser(description="Write missing concat MP4 ids per split")
    ap.add_argument("--concat-dir", type=Path, default=default_concat)
    ap.add_argument("--train-jsonl", type=Path, default=default_train)
    ap.add_argument("--val-jsonl", type=Path, default=default_val)
    ap.add_argument("--outdir", type=Path, default=default_concat, help="Directory to write missing_* files")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    concat_dir = args.concat_dir.expanduser()
    train_p = args.train_jsonl.expanduser()
    val_p = args.val_jsonl.expanduser()
    outdir = args.outdir.expanduser()

    if not concat_dir.is_dir():
        raise FileNotFoundError(f"Concat directory not found: {concat_dir}")
    if not train_p.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {train_p}")
    if not val_p.is_file():
        raise FileNotFoundError(f"Val JSONL not found: {val_p}")

    file_ids = list_concat_ids(concat_dir)
    train_ids = read_ids(train_p)
    val_ids = read_ids(val_p)

    missing_train = sorted(train_ids - file_ids)
    missing_val = sorted(val_ids - file_ids)

    outdir.mkdir(parents=True, exist_ok=True)
    train_out = outdir / "missing_train_ids.txt"
    val_out = outdir / "missing_val_ids.txt"

    with train_out.open("w", encoding="utf-8") as f:
        for vid in missing_train:
            f.write(vid + "\n")
    with val_out.open("w", encoding="utf-8") as f:
        for vid in missing_val:
            f.write(vid + "\n")

    print(f"Wrote {len(missing_train)} missing train ids -> {train_out}")
    print(f"Wrote {len(missing_val)} missing val ids   -> {val_out}")


if __name__ == "__main__":
    main()

