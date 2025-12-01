#!/usr/bin/env python3
"""
Verify that every video_id in queries_by_video_{train,val}.jsonl
has a corresponding MP4 in the concat directory.

Notes
- Extra MP4s in concat that don't appear in the JSONLs are allowed
  (e.g., test split not included) and are only reported as info.

Defaults
- concat dir: dataset/qvhighlights/concat
- jsonl     : dataset/qvhighlights/queries_by_video_train.jsonl,
              dataset/qvhighlights/queries_by_video_val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set


def read_video_ids(jsonl_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
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
    if not concat_dir.is_dir():
        raise FileNotFoundError(f"Concat directory not found: {concat_dir}")
    return {p.stem for p in concat_dir.glob("*.mp4")}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_concat = repo_root / "dataset" / "qvhighlights" / "concat"
    default_train = repo_root / "dataset" / "qvhighlights" / "queries_by_video_train.jsonl"
    default_val = repo_root / "dataset" / "qvhighlights" / "queries_by_video_val.jsonl"

    ap = argparse.ArgumentParser(description="Verify JSONL video_ids exist in concat MP4s")
    ap.add_argument("--concat-dir", type=Path, default=default_concat)
    ap.add_argument("--train-jsonl", type=Path, default=default_train)
    ap.add_argument("--val-jsonl", type=Path, default=default_val)
    ap.add_argument("--list-limit", type=int, default=20, help="Print at most this many examples per list")
    ap.add_argument("--write-missing", type=Path, default=None, help="Optional path to write all missing ids (one per line)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    concat_dir = args.concat_dir.expanduser()
    train_p = args.train_jsonl.expanduser()
    val_p = args.val_jsonl.expanduser()

    file_ids = list_concat_ids(concat_dir)
    train_ids = read_video_ids(train_p)
    val_ids = read_video_ids(val_p)
    json_ids = train_ids | val_ids

    # Per-split and overall missing
    train_missing = sorted(train_ids - file_ids)
    val_missing = sorted(val_ids - file_ids)
    missing = sorted(json_ids - file_ids)
    extras = sorted(file_ids - json_ids)

    print("Verify JSONL ids exist in concat")
    print(f"  concat_dir : {concat_dir}")
    print(f"  train_jsonl: {train_p} ({len(train_ids)} ids)")
    print(f"  val_jsonl  : {val_p} ({len(val_ids)} ids)")
    print(f"  files(.mp4): {len(file_ids)}")
    print()
    print(f"Missing MP4s in train        : {len(train_missing)}")
    if train_missing:
        print("  e.g.,", ", ".join(train_missing[: max(0, args.list_limit)]))
    print(f"Missing MP4s in val          : {len(val_missing)}")
    if val_missing:
        print("  e.g.,", ", ".join(val_missing[: max(0, args.list_limit)]))
    print(f"Missing MP4s in union        : {len(missing)}")
    if missing:
        print("  e.g.,", ", ".join(missing[: max(0, args.list_limit)]))
    print(f"Extra MP4s not in JSONLs (ok): {len(extras)}")
    if extras:
        print("  e.g.,", ", ".join(extras[: max(0, args.list_limit)]))

    if args.write_missing:
        outp = args.write_missing.expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for vid in missing:
                f.write(vid + "\n")
        print(f"Wrote missing ids to: {outp}")


if __name__ == "__main__":
    main()
