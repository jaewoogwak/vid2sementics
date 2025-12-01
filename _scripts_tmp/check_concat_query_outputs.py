#!/usr/bin/env python3
"""
Check whether every video_id referenced in queries_by_video_{train,val}.jsonl
has a concatenated MP4 in dataset/qvhighlights/concat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Set


def iter_video_ids(jsonl_path: Path) -> Iterable[str]:
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse {jsonl_path}:{idx}")
                continue
            vid = obj.get("video_id")
            if isinstance(vid, str) and vid:
                yield vid


def load_ids(paths: Iterable[Path]) -> Set[str]:
    ids: Set[str] = set()
    for path in paths:
        if not path.is_file():
            print(f"[WARN] queries JSONL not found: {path}")
            continue
        ids.update(iter_video_ids(path))
    return ids


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_concat = repo_root / "dataset" / "qvhighlights" / "concat"
    default_train = repo_root / "dataset" / "qvhighlights" / "queries_by_video_train.jsonl"
    default_val = repo_root / "dataset" / "qvhighlights" / "queries_by_video_val.jsonl"

    ap = argparse.ArgumentParser(description="Verify concat outputs cover all query-listed video IDs")
    ap.add_argument("--concat-dir", type=Path, default=default_concat, help="Directory containing concatenated MP4s")
    ap.add_argument(
        "--queries",
        type=Path,
        nargs="*",
        default=None,
        help="queries_by_video JSONL files to read (default: train & val)",
    )
    ap.add_argument("--limit", type=int, default=20, help="Max missing IDs to display")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    concat_dir = args.concat_dir.expanduser()
    query_paths = args.queries or [
        Path(__file__).resolve().parents[1] / "dataset" / "qvhighlights" / "queries_by_video_train.jsonl",
        Path(__file__).resolve().parents[1] / "dataset" / "qvhighlights" / "queries_by_video_val.jsonl",
    ]
    query_paths = [p.expanduser() for p in query_paths]

    if not concat_dir.is_dir():
        raise FileNotFoundError(f"Concat directory not found: {concat_dir}")

    ids = load_ids(query_paths)
    if not ids:
        print("[INFO] No video IDs found in provided queries files.")
        return

    missing = sorted(vid for vid in ids if not (concat_dir / f"{vid}.mp4").is_file())
    present = len(ids) - len(missing)

    print(f"[INFO] Total unique video IDs in queries: {len(ids)}")
    print(f"[INFO] Videos present in concat dir: {present}")
    print(f"[INFO] Missing videos: {len(missing)}")

    if missing:
        display = missing[: args.limit] if args.limit else missing
        print("[MISSING] " + ", ".join(display))
        if args.limit and len(missing) > args.limit:
            print(f"... {len(missing) - args.limit} more")


if __name__ == "__main__":
    main()
