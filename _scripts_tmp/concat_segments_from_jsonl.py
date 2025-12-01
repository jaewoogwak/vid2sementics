#!/usr/bin/env python3
"""
Batch-concatenate QVHighlights segments listed in the release JSONLs.

For every base video id that appears in the provided JSONL files and has
corresponding raw MP4 segment files under dataset/qvhighlights/raw/videos,
this script concatenates the segments in chronological order and writes the
merged MP4 into dataset/qvhighlights/concat (one MP4 per base video id).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# Reuse helpers from scripts/concat_segments.py
try:
    from concat_segments import Segment, build_concat_list_file, run_ffmpeg_concat
except ImportError:  # pragma: no cover - fallback when executed from elsewhere
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from concat_segments import Segment, build_concat_list_file, run_ffmpeg_concat


def parse_vid_identifier(vid: str) -> Tuple[str, float, float] | None:
    parts = vid.rsplit("_", 2)
    if len(parts) != 3:
        return None
    base, s, e = parts
    try:
        start = float(s)
        end = float(e)
    except ValueError:
        return None
    return base, start, end


def iter_jsonl_objects(jsonl_path: Path) -> Iterable[Dict[str, object]]:
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse JSON object in {jsonl_path} line {idx}", file=sys.stderr)
                continue
            yield obj


def collect_segments(
    jsonl_paths: Sequence[Path],
    raw_dir: Path,
    allowed_ids: Set[str] | None = None,
) -> Tuple[Dict[str, List[Segment]], int, List[str]]:
    segments: Dict[str, List[Segment]] = defaultdict(list)
    seen_keys: Set[Tuple[str, float, float]] = set()
    missing_count = 0
    missing_examples: List[str] = []

    for jsonl_path in jsonl_paths:
        if not jsonl_path.is_file():
            print(f"[WARN] JSONL not found: {jsonl_path}", file=sys.stderr)
            continue
        for obj in iter_jsonl_objects(jsonl_path):
            vid_value = obj.get("vid") or obj.get("video_id")
            if not isinstance(vid_value, str):
                continue
            parsed = parse_vid_identifier(vid_value)
            if parsed is None:
                continue
            base, start, end = parsed
            if allowed_ids is not None and base not in allowed_ids:
                continue
            key = (base, start, end)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            seg_path = raw_dir / f"{vid_value}.mp4"
            if not seg_path.is_file():
                missing_count += 1
                if len(missing_examples) < 8:
                    missing_examples.append(seg_path.name)
                continue
            segments[base].append(Segment(path=seg_path, start=start, end=end))

    for segs in segments.values():
        segs.sort(key=lambda s: s.start)

    return segments, missing_count, missing_examples


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_raw = repo_root / "dataset" / "qvhighlights" / "raw" / "videos"
    default_out = repo_root / "dataset" / "qvhighlights" / "concat"
    default_train = repo_root / "dataset" / "qvhighlights" / "highlight_train_release.jsonl"
    default_val = repo_root / "dataset" / "qvhighlights" / "highlight_val_release.jsonl"

    ap = argparse.ArgumentParser(description="Concatenate QVHighlights segments listed in JSONLs")
    ap.add_argument(
        "--jsonl",
        type=Path,
        nargs="*",
        default=None,
        help="JSONL files to read. Defaults to highlight_{train,val}_release.jsonl",
    )
    ap.add_argument("--raw-dir", type=Path, default=default_raw, help="Directory containing raw segment MP4 files")
    ap.add_argument("--outdir", type=Path, default=default_out, help="Directory to write concatenated MP4 files")
    ap.add_argument("--only", nargs="*", default=None, help="Optional subset of base video ids to process")
    ap.add_argument("--limit", type=int, default=None, help="Process at most this many video ids")
    ap.add_argument("--reencode", action="store_true", help="Re-encode instead of stream copy when concatenating")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs instead of skipping them")
    ap.set_defaults(_default_jsonls=[default_train, default_val])
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = args.raw_dir.expanduser()
    outdir = args.outdir.expanduser()
    jsonl_paths = args.jsonl or args._default_jsonls
    jsonl_paths = [p.expanduser() for p in jsonl_paths]

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw segment directory not found: {raw_dir}")
    outdir.mkdir(parents=True, exist_ok=True)

    allowed_ids = set(args.only) if args.only else None
    segments, missing_count, missing_examples = collect_segments(jsonl_paths, raw_dir, allowed_ids)

    if missing_count:
        sample = ", ".join(missing_examples)
        if sample:
            print(f"[WARN] {missing_count} listed segments missing in raw dir (e.g., {sample})")
        else:
            print(f"[WARN] {missing_count} listed segments missing in raw dir")

    if not segments:
        print("[INFO] No matching segments found. Nothing to concatenate.")
        return

    video_ids = sorted(segments.keys())
    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    total = len(video_ids)
    processed = 0
    skipped = 0

    for idx, vid in enumerate(video_ids, 1):
        segs = segments[vid]
        if not segs:
            continue
        output_path = outdir / f"{vid}.mp4"
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] ({idx}/{total}) {vid} -> {output_path.name} already exists")
            skipped += 1
            continue
        list_file = build_concat_list_file(segs)
        try:
            print(f"[INFO] ({idx}/{total}) Concatenating {len(segs)} segments for {vid}")
            run_ffmpeg_concat(list_file, output_path, reencode=args.reencode)
            processed += 1
        finally:
            try:
                os.unlink(list_file)
            except OSError:
                pass

    print(f"[DONE] Processed {processed} video ids, skipped {skipped}. Outputs in {outdir}")
    print(f"[INFO] Total concatenated videos: {processed}")


if __name__ == "__main__":
    main()
