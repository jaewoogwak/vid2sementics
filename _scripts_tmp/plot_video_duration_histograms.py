#!/usr/bin/env python3
"""
Plot video duration distributions for QVHighlights concatenated videos and
MSRVTT untrimmed splits used by train_scene_autoregressive_qvh.py.

The script scans the QVHighlights highlight JSONL to find the base video ids,
reads the corresponding concatenated MP4 files to estimate their durations,
and parses the MSRVTT_untrimmed.json annotation to derive per-split video
durations from the relevant window endpoints. A single figure with two
histograms is saved to the path specified by --output.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - import guard
    cv2 = None  # type: ignore[assignment]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_qvh_root = repo_root / "dataset" / "qvhighlights"
    default_qvh_jsonl = default_qvh_root / "highlight_train_release.jsonl"
    default_concat_root = default_qvh_root / "concat"
    default_msrvtt_ann = (
        repo_root / "dataset" / "data" / "MSRVTT" / "annotation" / "MSRVTT_untrimmed.json"
    )
    default_output = repo_root / "logs" / "video_duration_hist.png"

    parser = argparse.ArgumentParser(
        description=(
            "Plot video duration histograms for QVHighlights concatenated videos "
            "and MSRVTT untrimmed splits."
        )
    )
    parser.add_argument(
        "--qvh-jsonl",
        type=Path,
        default=default_qvh_jsonl,
        help="QVHighlights JSONL file (e.g., highlight_train_release.jsonl).",
    )
    parser.add_argument(
        "--qvh-concat-root",
        type=Path,
        default=default_concat_root,
        help="Directory containing concatenated QVHighlights videos.",
    )
    parser.add_argument(
        "--msrvtt-annotation",
        type=Path,
        default=default_msrvtt_ann,
        help="MSRVTT untrimmed annotation JSON file.",
    )
    parser.add_argument(
        "--msrvtt-splits",
        type=str,
        default="train,val",
        help="Comma-separated list of MSRVTT splits to include (e.g., train,val,test).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=40,
        help="Histogram bin count for both datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to save the resulting histogram figure (PNG).",
    )
    return parser.parse_args()


def extract_base_video_id(vid: str) -> str:
    """Strip the trailing '_<start>_<end>' portion from a QVHighlights vid."""
    parts = vid.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else vid


def load_qvh_video_ids(jsonl_path: Path) -> List[str]:
    video_ids = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON line in %s", jsonl_path)
                continue
            vid = row.get("vid")
            if not isinstance(vid, str) or not vid:
                continue
            base_id = extract_base_video_id(vid)
            if base_id:
                video_ids.add(base_id)
    return sorted(video_ids)


def read_video_duration(video_path: Path) -> Optional[float]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read video durations.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not fps or math.isnan(fps) or fps <= 1e-6:
        fps = 30.0
    if not frame_count or math.isnan(frame_count) or frame_count <= 0:
        frame_count = None
    duration = None
    if frame_count is not None:
        duration = float(frame_count / fps)
    else:
        # Fall back to seeking to the end when frame count is unavailable.
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if duration_ms and not math.isnan(duration_ms) and duration_ms > 0:
            duration = duration_ms / 1000.0
    cap.release()
    return duration


def gather_qvh_durations(jsonl_path: Path, concat_root: Path) -> List[float]:
    base_ids = load_qvh_video_ids(jsonl_path)
    if not base_ids:
        logging.warning("No QVHighlights video ids found in %s", jsonl_path)
        return []
    durations: List[float] = []
    missing: List[Path] = []
    for video_id in base_ids:
        video_path = concat_root / f"{video_id}.mp4"
        if not video_path.is_file():
            missing.append(video_path)
            continue
        duration = read_video_duration(video_path)
        if duration is None:
            logging.warning("Failed to read duration for %s", video_path)
            continue
        durations.append(duration)
    if missing:
        logging.warning(
            "Missing %d concatenated videos referenced in %s (e.g., %s)",
            len(missing),
            jsonl_path,
            missing[0],
        )
    logging.info(
        "Loaded durations for %d/%d QVHighlights concatenated videos",
        len(durations),
        len(base_ids),
    )
    return durations


def gather_msrvtt_durations(annotation_path: Path, splits: Sequence[str]) -> Dict[str, List[float]]:
    with annotation_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("videos") or []
    normalized_splits = [split.strip().lower() for split in splits if split.strip()]
    if not normalized_splits:
        raise ValueError("At least one MSRVTT split must be specified.")
    durations: Dict[str, List[float]] = {split: [] for split in normalized_splits}
    split_set = set(normalized_splits)
    for entry in entries:
        split = str(entry.get("split", "")).strip().lower()
        if split not in split_set:
            continue
        windows = entry.get("relevant_windows") or []
        max_end: Optional[float] = None
        for window in windows:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                continue
            start, end = float(window[0]), float(window[1])
            if end <= start:
                continue
            max_end = end if max_end is None else max(max_end, end)
        if max_end is None:
            logging.warning(
                "Skipping video %s in split %s: no valid relevant windows",
                entry.get("video_id"),
                split or "(unknown)",
            )
            continue
        durations[split].append(max_end)
    for split in normalized_splits:
        logging.info(
            "Collected %d MSRVTT durations for split '%s'",
            len(durations.get(split, [])),
            split,
        )
    return durations


def describe_durations(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def histogram_bins(values: Sequence[float], num_bins: int) -> np.ndarray:
    if not values:
        return np.linspace(0.0, 1.0, max(2, num_bins + 1))
    vmax = max(values)
    if not math.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    upper = vmax * 1.05
    return np.linspace(0.0, upper, max(2, num_bins + 1))


def plot_histograms(
    qvh_durations: Sequence[float],
    msrvtt_durations: Dict[str, Sequence[float]],
    output_path: Path,
    num_bins: int,
) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # QVHighlights subplot
    ax_qvh = axes[0]
    if qvh_durations:
        bins = histogram_bins(qvh_durations, num_bins)
        ax_qvh.hist(
            qvh_durations,
            bins=bins,
            color="#1f77b4",
            edgecolor="black",
            alpha=0.75,
        )
        stats = describe_durations(qvh_durations)
        ax_qvh.set_title(
            f"QVHighlights concat (n={stats['count']})\n"
            f"mean={stats['mean']:.1f}s median={stats['median']:.1f}s",
            fontsize=11,
        )
    else:
        ax_qvh.text(0.5, 0.5, "No QVHighlights data", ha="center", va="center", fontsize=12)
        ax_qvh.set_title("QVHighlights concat")
    ax_qvh.set_xlabel("Duration (seconds)")
    ax_qvh.set_ylabel("Video count")
    ax_qvh.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # MSRVTT subplot
    ax_msrvtt = axes[1]
    all_msrvtt = [dur for durations in msrvtt_durations.values() for dur in durations]
    plotted = False
    if all_msrvtt:
        bins = histogram_bins(all_msrvtt, num_bins)
        split_order = list(msrvtt_durations.keys())
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(split_order))))
        for color, split in zip(colors, split_order):
            durations = msrvtt_durations.get(split, [])
            if not durations:
                continue
            ax_msrvtt.hist(
                durations,
                bins=bins,
                alpha=0.5,
                label=f"{split} (n={len(durations)})",
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            plotted = True
        ax_msrvtt.set_title("MSRVTT untrimmed")
        if plotted:
            ax_msrvtt.legend()
    if not plotted:
        ax_msrvtt.text(0.5, 0.5, "No MSRVTT data", ha="center", va="center", fontsize=12)
        ax_msrvtt.set_title("MSRVTT untrimmed")
    ax_msrvtt.set_xlabel("Duration (seconds)")
    ax_msrvtt.set_ylabel("Video count")
    ax_msrvtt.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    fig.suptitle("Video duration distributions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved histogram figure to %s", output_path)
    return output_path


def print_stats(label: str, durations: Sequence[float]) -> None:
    stats = describe_durations(durations)
    if stats["count"] == 0:
        print(f"{label}: no data")
        return
    print(
        f"{label}: n={stats['count']} "
        f"mean={stats['mean']:.2f}s median={stats['median']:.2f}s "
        f"min={stats['min']:.2f}s max={stats['max']:.2f}s"
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(message)s",
        datefmt="%H:%M:%S",
    )

    qvh_jsonl = args.qvh_jsonl.expanduser()
    concat_root = args.qvh_concat_root.expanduser()
    if not qvh_jsonl.is_file():
        raise FileNotFoundError(f"QVHighlights JSONL not found: {qvh_jsonl}")
    if not concat_root.is_dir():
        raise FileNotFoundError(f"QVHighlights concat directory not found: {concat_root}")

    msrvtt_annotation = args.msrvtt_annotation.expanduser()
    if not msrvtt_annotation.is_file():
        raise FileNotFoundError(f"MSRVTT annotation not found: {msrvtt_annotation}")

    split_tokens = [token.strip() for token in args.msrvtt_splits.split(",") if token.strip()]
    if not split_tokens:
        raise ValueError("Provide at least one MSRVTT split via --msrvtt-splits.")

    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required to inspect QVHighlights video durations. "
            "Install opencv-python in your environment."
        )

    qvh_durations = gather_qvh_durations(qvh_jsonl, concat_root)
    msrvtt_durations = gather_msrvtt_durations(msrvtt_annotation, split_tokens)

    output_path = args.output.expanduser()
    plot_histograms(qvh_durations, msrvtt_durations, output_path, args.num_bins)

    print("Duration summary statistics")
    print_stats("QVHighlights concat", qvh_durations)
    for split in split_tokens:
        normalized = split.strip().lower()
        print_stats(f"MSRVTT[{split}]", msrvtt_durations.get(normalized, []))


if __name__ == "__main__":
    main()
