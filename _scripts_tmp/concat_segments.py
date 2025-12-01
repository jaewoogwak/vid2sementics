#!/usr/bin/env python3
"""
Concatenate QVHighlights segments of the same video id into one MP4.

Assumes segment filenames are like: <video_id>_<start>_<end>.mp4
and are stored under dataset/qvhighlights/raw/videos by default.

By default uses ffmpeg concat demuxer with stream copy (no re-encode).
If streams are not exactly matching across segments, use --reencode.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
import shutil
import json
from pathlib import Path
from typing import List, Tuple


@dataclass
class Segment:
    path: Path
    start: float
    end: float


def parse_segment_filename(path: Path, expected_id: str) -> Segment | None:
    stem = path.stem  # without extension
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    vid, s, e = parts
    if vid != expected_id:
        return None
    try:
        start = float(s)
        end = float(e)
    except ValueError:
        return None
    return Segment(path=path, start=start, end=end)


def find_segments(raw_dir: Path, video_id: str) -> List[Segment]:
    segments: List[Segment] = []
    for p in sorted(raw_dir.glob(f"{video_id}_*.mp4")):
        seg = parse_segment_filename(p, video_id)
        if seg is not None:
            segments.append(seg)
    segments.sort(key=lambda x: x.start)
    return segments


def load_allowed_segments(jsonl_paths: List[Path], video_id: str) -> List[Tuple[float, float]]:
    allowed: List[Tuple[float, float]] = []
    base = video_id
    for jp in jsonl_paths:
        if not jp.is_file():
            continue
        with jp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid = obj.get("vid") or obj.get("video_id")
                if not isinstance(vid, str):
                    continue
                parts = vid.rsplit("_", 2)
                if len(parts) != 3:
                    continue
                b, s, e = parts
                if b != base:
                    continue
                try:
                    start = float(s)
                    end = float(e)
                except ValueError:
                    continue
                allowed.append((start, end))
    # unique and sorted by start
    allowed = sorted(list({(s, e) for (s, e) in allowed}), key=lambda x: x[0])
    return allowed


def build_concat_list_file(segments: List[Segment]) -> Path:
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    try:
        for seg in segments:
            # Use absolute paths and quote safely for ffmpeg concat demuxer
            tf.write(f"file {shlex.quote(str(seg.path.resolve()))}\n")
        tf.flush()
        return Path(tf.name)
    finally:
        tf.close()


def run_ffmpeg_concat(list_file: Path, output: Path, reencode: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError(
            "ffmpeg not found. Install ffmpeg or set $FFMPEG_BIN to its path.\n"
            "Examples:\n"
            "  - Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
            "  - Conda : conda install -c conda-forge ffmpeg\n"
        )
    if reencode:
        cmd = [
            ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            str(output),
        ]
    else:
        cmd = [
            ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output),
        ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_raw = repo_root / "dataset" / "qvhighlights" / "raw" / "videos"
    default_outdir = repo_root / "dataset" / "qvhighlights" / "concat"

    ap = argparse.ArgumentParser(description="Concatenate segments for a given video id")
    ap.add_argument("video_id", type=str, help="Base video id (without start/end)")
    ap.add_argument("--raw-dir", type=Path, default=default_raw, help="Directory containing segment MP4s")
    ap.add_argument("--output", type=Path, default=None, help="Output MP4 path. Defaults to <outdir>/<video_id>.mp4")
    ap.add_argument("--outdir", type=Path, default=default_outdir, help="Directory to place output when --output not set")
    ap.add_argument("--segments-jsonl", type=Path, nargs="*", default=None, help="If provided, only concatenate segments listed in these JSONLs for this video id")
    ap.add_argument("--reencode", action="store_true", help="Re-encode if stream copy fails or parameters mismatch")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir.expanduser()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw video directory not found: {raw_dir}")

    # Determine candidate segments
    if args.segments_jsonl:
        jsonl_paths = [p.expanduser() for p in args.segments_jsonl]
        allowed = load_allowed_segments(jsonl_paths, args.video_id)
        allowed_set = set(allowed)
        # Scan all existing segments for this id and filter by allowed (float-safe)
        all_segments = find_segments(raw_dir, args.video_id)
        segments = [seg for seg in all_segments if (seg.start, seg.end) in allowed_set]
    else:
        segments = find_segments(raw_dir, args.video_id)
    if not segments:
        print(f"No segments found for video id '{args.video_id}' in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(segments)} segments for '{args.video_id}':")
    for seg in segments:
        print(f"  {seg.path.name}  [start={seg.start:.3f}, end={seg.end:.3f}]")

    output: Path
    if args.output is not None:
        output = args.output.expanduser()
        if output.suffix.lower() != ".mp4":
            output = output.with_suffix(".mp4")
    else:
        output = args.outdir.expanduser() / f"{args.video_id}.mp4"

    lst = build_concat_list_file(segments)
    try:
        print(f"Concatenating into: {output}")
        run_ffmpeg_concat(lst, output, reencode=args.reencode)
        print("Done.")
    finally:
        try:
            os.unlink(lst)
        except OSError:
            pass


if __name__ == "__main__":
    main()
