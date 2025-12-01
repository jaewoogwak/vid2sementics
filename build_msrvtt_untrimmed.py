#!/usr/bin/env python3
"""Build JSON metadata describing untrimmed MSRVTT videos.

The script groups trimmed MSRVTT clips in 3-4-5 cycles (wrapping as needed)
and records the concatenated ordering, representative captions, and temporal
windows that would result from feature-level concatenation. No mp4 files are
written; only JSON metadata is produced.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

DEFAULT_DATASET_ROOT = Path("/dev/ssd1/gjw/prvr/dataset/data/MSRVTT")
DEFAULT_LEN_PATTERN: Tuple[int, ...] = (3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create metadata for untrimmed MSRVTT videos."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of the MSRVTT dataset.",
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=None,
        help="Path to MSRVTT_data.json. Defaults to {dataset_root}/annotation/MSRVTT_data.json.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to write the untrimmed metadata JSON. Defaults to {dataset_root}/annotation/MSRVTT_untrimmed.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_first_caption_map(sentences: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    first_caption: Dict[str, str] = {}
    for entry in sentences:
        video_id = entry.get("video_id")
        if not video_id or video_id in first_caption:
            continue
        caption = entry.get("caption")
        if caption is None:
            continue
        first_caption[str(video_id)] = str(caption)
    return first_caption


def extract_float(entry: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key not in entry:
            continue
        value = entry[key]
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def compute_duration(entry: Mapping[str, Any]) -> float:
    start = extract_float(entry, ("start time", "start_time", "start", "starttime"))
    end = extract_float(entry, ("end time", "end_time", "end", "endtime"))
    if start is None or end is None or end <= start:
        raise ValueError(f"Invalid start/end time for video {entry.get('video_id')}")
    return end - start


def video_numeric_index(video_id: str) -> int:
    suffix = ""
    for char in reversed(video_id):
        if char.isdigit():
            suffix = char + suffix
        else:
            break
    return int(suffix) if suffix else sys.maxsize


def sort_videos(videos: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    def key_fn(item: Mapping[str, Any]) -> Tuple[int, str]:
        video_id = str(item.get("video_id", ""))
        return video_numeric_index(video_id), video_id

    return sorted(videos, key=key_fn)


def gather_split_videos(
    videos: Sequence[Mapping[str, Any]]
) -> Dict[str, List[Mapping[str, Any]]]:
    by_split: Dict[str, List[Mapping[str, Any]]] = {"train": [], "test": []}
    for entry in videos:
        split_raw = entry.get("split")
        if not split_raw:
            continue
        split = str(split_raw).strip().lower()
        if split not in by_split:
            continue
        by_split[split].append(entry)
    for split in by_split:
        by_split[split] = sort_videos(by_split[split])
    return by_split


def group_videos_in_pattern(
    videos: Sequence[Mapping[str, Any]], pattern: Sequence[int]
) -> List[List[Mapping[str, Any]]]:
    if not pattern:
        raise ValueError("Grouping pattern must not be empty.")
    if any(size < 1 for size in pattern):
        raise ValueError("Pattern contains invalid chunk size.")
    min_size = min(pattern)
    groups: List[List[Mapping[str, Any]]] = []
    consumed = 0
    idx = 0
    total = len(videos)
    while consumed < total:
        remaining = total - consumed
        desired = pattern[idx % len(pattern)]
        if remaining < min_size:
            size = remaining
        else:
            size = min(desired, remaining)
        groups.append(list(videos[consumed : consumed + size]))
        consumed += size
        idx += 1
    return groups


def build_relevant_windows(durations: Sequence[float]) -> List[List[float]]:
    windows: List[List[float]] = []
    start = 0.0
    for duration in durations:
        end = start + max(duration, 0.0)
        windows.append([start, end])
        start = end
    return windows


def build_untrimmed_entries(
    split: str,
    videos: Sequence[Mapping[str, Any]],
    first_caption: Mapping[str, str],
    pattern: Sequence[int],
) -> List[Dict[str, Any]]:
    grouped = group_videos_in_pattern(videos, pattern)
    untrimmed: List[Dict[str, Any]] = []
    for group in grouped:
        video_ids = []
        durations = []
        for entry in group:
            video_id = entry.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise ValueError("Missing video_id in group.")
            video_ids.append(video_id)
            durations.append(compute_duration(entry))
        queries: List[str] = []
        for video_id in video_ids:
            caption = first_caption.get(video_id)
            if caption is None:
                raise KeyError(f"No caption found for video_id '{video_id}'")
            queries.append(caption)
        untrimmed.append(
            {
                "video_id": "_".join(video_ids),
                "split": split,
                "video_list": video_ids,
                "queries": queries,
                "relevant_windows": build_relevant_windows(durations),
            }
        )
    return untrimmed


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_root = args.dataset_root.resolve()
    annotation_path = (
        args.annotation_path
        or dataset_root / "annotation" / "MSRVTT_data.json"
    ).resolve()
    output_path = (
        args.output_path
        or dataset_root / "annotation" / "MSRVTT_untrimmed.json"
    ).resolve()

    logging.info("Reading annotations from %s", annotation_path)
    annotation = load_json(annotation_path)
    videos = annotation.get("videos")
    sentences = annotation.get("sentences")
    if not isinstance(videos, list):
        raise ValueError("Annotation JSON is missing a 'videos' list.")
    if not isinstance(sentences, list):
        raise ValueError("Annotation JSON is missing a 'sentences' list.")

    first_caption = build_first_caption_map(sentences)
    logging.info(
        "Loaded %d videos and %d sentences; %d video_ids have first captions.",
        len(videos),
        len(sentences),
        len(first_caption),
    )

    split_videos = gather_split_videos(videos)

    all_entries: List[Dict[str, Any]] = []
    for split in ("train", "test"):
        split_entries = split_videos.get(split, [])
        if not split_entries:
            logging.warning("Split '%s' has no videos; skipping.", split)
            continue
        untrimmed_entries = build_untrimmed_entries(
            split, split_entries, first_caption, DEFAULT_LEN_PATTERN
        )
        logging.info(
            "Split '%s': %d clips -> %d untrimmed entries.",
            split,
            len(split_entries),
            len(untrimmed_entries),
        )
        all_entries.extend(untrimmed_entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"videos": all_entries}, handle, indent=2)

    logging.info(
        "Wrote %d untrimmed metadata entries to %s",
        len(all_entries),
        output_path,
    )


if __name__ == "__main__":
    main()
