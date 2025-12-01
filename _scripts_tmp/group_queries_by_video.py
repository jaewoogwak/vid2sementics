#!/usr/bin/env python3
"""
Group QVHighlights queries by base video_id (strip _<start>_<end>).

Reads train/val JSONL files and writes grouped JSONL files under
dataset/qvhighlights/concat:
  - queries_by_video_train.jsonl
  - queries_by_video_val.jsonl

Each output line: {"video_id": str, "queries": [str, ...]}
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def base_video_id(vid: str) -> str:
    parts = vid.rsplit("_", 2)
    return parts[0] if len(parts) >= 3 else vid


def ordered_unique(seq: Iterable[str]) -> List[str]:
    seen = OrderedDict()
    for s in seq:
        if s is None:
            continue
        if not isinstance(s, str):
            continue
        if s not in seen:
            seen[s] = None
    return list(seen.keys())


def group_file(jsonl_path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
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
            query = obj.get("query")
            if not isinstance(vid, str):
                continue
            bid = base_video_id(vid)
            if isinstance(query, str) and query:
                mapping[bid].append(query)
    # de-duplicate, preserve order
    return {k: ordered_unique(vs) for k, vs in mapping.items()}


def write_grouped(mapping: Dict[str, List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for vid, queries in mapping.items():
            rec = {"video_id": vid, "queries": queries}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_train = repo_root / "dataset" / "qvhighlights" / "highlight_train_release.jsonl"
    default_val = repo_root / "dataset" / "qvhighlights" / "highlight_val_release.jsonl"
    default_outdir = repo_root / "dataset" / "qvhighlights" / "concat"

    ap = argparse.ArgumentParser(description="Group QVHighlights queries by base video id")
    ap.add_argument("--train-jsonl", type=Path, default=default_train)
    ap.add_argument("--val-jsonl", type=Path, default=default_val)
    ap.add_argument("--outdir", type=Path, default=default_outdir)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_p = args.train_jsonl.expanduser()
    val_p = args.val_jsonl.expanduser()
    outdir = args.outdir.expanduser()

    if not train_p.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {train_p}")
    if not val_p.is_file():
        raise FileNotFoundError(f"Val JSONL not found: {val_p}")

    train_map = group_file(train_p)
    val_map = group_file(val_p)

    write_grouped(train_map, outdir / "queries_by_video_train.jsonl")
    write_grouped(val_map, outdir / "queries_by_video_val.jsonl")

    print(f"Wrote: {outdir / 'queries_by_video_train.jsonl'}  (videos: {len(train_map)})")
    print(f"Wrote: {outdir / 'queries_by_video_val.jsonl'}    (videos: {len(val_map)})")


if __name__ == "__main__":
    main()

