#!/usr/bin/env python3
"""
Quick sanity check for InternVideo2 on QVHighlights clips.

The script loads the Stage2 multi-modality model, encodes a target video
segment plus two text queries (matching + mismatching), and prints their
cosine similarities so you can verify that the correct query is ranked higher.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import cv2
import torch


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_multi_root = repo_root / "InternVideo" / "InternVideo2" / "multi_modality"
    default_config = default_multi_root / "demo" / "internvideo2_stage2_config.py"
    default_video = (
        repo_root / "dataset" / "qvhighlights" / "raw" / "videos" / "L9cUEhaXnK4_210.0_360.0.mp4"
    )
    default_query_files = [
        repo_root / "dataset" / "qvhighlights" / "highlight_val_release.jsonl",
        repo_root / "dataset" / "qvhighlights" / "highlight_train_release.jsonl",
    ]

    parser = argparse.ArgumentParser(
        description="Encode a QVHighlights clip and compare positive/negative queries with InternVideo2."
    )
    parser.add_argument(
        "--multi-modality-root",
        type=Path,
        default=default_multi_root,
        help="Path to InternVideo2/multi_modality (default: %(default)s)",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=default_config,
        help="Configuration file to load (default: %(default)s)",
    )
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default="/dev/ssd1/gjw/prvr/InternVideo/InternVideo2/ckpt/InternVideo2-stage2_1b-224p-f4.pt",
        help="Checkpoint to load. If omitted, uses $INTERNVIDEO2_STAGE2_CKPT or config value.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=default_video,
        help="Video segment to encode (default: %(default)s)",
    )
    parser.add_argument(
        "--positive-hint",
        type=str,
        default=None,
        help="Substring to locate the positive query (defaults to the video file stem).",
    )
    parser.add_argument(
        "--negative-hint",
        type=str,
        default="_i9qWLsZToY",
        help="Substring to locate the negative query (default: %(default)s).",
    )
    parser.add_argument(
        "--query-files",
        type=Path,
        nargs="+",
        default=None,
        help="One or more JSONL files to search for queries.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="How many top queries to display (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to run inference on (default: %(default)s).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Override number of frames sampled from the video.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=None,
        help="Override square crop size used when sampling frames.",
    )

    args = parser.parse_args()
    if args.query_files is None:
        args.query_files = default_query_files
    return args


def ensure_syspath(path: Path) -> None:
    abs_path = path.resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"multi_modality root not found: {abs_path}")
    if str(abs_path) not in sys.path:
        sys.path.insert(0, str(abs_path))


def load_config(config_path: Path, pretrained_path: Path | None, device: str):
    import importlib

    demo_config = importlib.import_module("demo_config")
    cfg = demo_config.Config.from_file(str(config_path.resolve()))
    cfg = demo_config.eval_dict_leaf(cfg)

    resolved_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    if resolved_device != device:
        print("CUDA unavailable, falling back to CPU.")
    cfg.device = resolved_device

    ckpt = pretrained_path or os.environ.get("INTERNVIDEO2_STAGE2_CKPT")
    if ckpt:
        cfg.pretrained_path = str(Path(ckpt).expanduser())
    if not cfg.pretrained_path:
        raise ValueError("Set --pretrained-path or $INTERNVIDEO2_STAGE2_CKPT to a valid checkpoint.")
    if not Path(cfg.pretrained_path).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.pretrained_path}")

    return cfg


def maybe_override_inputs(cfg, num_frames: int | None, frame_size: int | None):
    if num_frames is not None:
        cfg.num_frames = num_frames
        cfg.num_frames_test = num_frames
        cfg.origin_num_frames = num_frames
        cfg.model.vision_encoder.num_frames = num_frames
    if frame_size is not None:
        cfg.size_t = frame_size
        cfg.model.vision_encoder.img_size = frame_size


def load_queries_for_video(video_hint: str, jsonl_paths: Sequence[Path]) -> List[Tuple[str, str, Path]]:
    """Collect all (vid, query, source_path) whose vid contains the hint.

    Returns an empty list if none found.
    """
    results: List[Tuple[str, str, Path]] = []
    for jsonl_path in jsonl_paths:
        jsonl_path = jsonl_path.expanduser()
        if not jsonl_path.is_file():
            continue
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                data = json.loads(line)
                vid = data.get("vid", "")
                if video_hint in vid:
                    results.append((vid, data.get("query", ""), jsonl_path))
    return results


def load_video_frames(video_path: Path) -> List:
    video_path = video_path.expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    from demo.utils import _frame_from_video

    frames = [frame for frame in _frame_from_video(capture)]
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


def encode_video(model, frames, frame_count: int, frame_size: int, device: torch.device):
    from demo.utils import frames2tensor

    if len(frames) < frame_count:
        raise ValueError(f"Need at least {frame_count} frames, got {len(frames)}")
    tensor = frames2tensor(frames, fnum=frame_count, target_size=(frame_size, frame_size), device=device)
    with torch.no_grad():
        vid_feat = model.get_vid_feat(tensor)
    return vid_feat


def encode_text(model, text: str):
    with torch.no_grad():
        return model.get_txt_feat(text)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b.T).squeeze().cpu())


def main() -> None:
    args = parse_args()
    ensure_syspath(args.multi_modality_root)

    cfg = load_config(args.config_path, args.pretrained_path, args.device)
    maybe_override_inputs(cfg, args.num_frames, args.frame_size)

    from demo.utils import setup_internvideo2

    model, _ = setup_internvideo2(cfg)
    device = torch.device(cfg.device)
    # Align sampling with the model/demo config
    frame_count = int(getattr(cfg.model.vision_encoder, "num_frames", getattr(cfg, "num_frames", 4)))
    frame_size = int(getattr(cfg.model.vision_encoder, "img_size", getattr(cfg, "size_t", 224)))

    pos_hint = args.positive_hint or Path(args.video_path).stem
    neg_hint = args.negative_hint
    query_files = [path if isinstance(path, Path) else Path(path) for path in args.query_files]
    pos_matches = load_queries_for_video(pos_hint, query_files)
    neg_matches = load_queries_for_video(neg_hint, query_files)
    if not pos_matches:
        raise ValueError(f"No queries found for positive hint '{pos_hint}' in files: {query_files}")
    if not neg_matches:
        raise ValueError(f"No queries found for negative hint '{neg_hint}' in files: {query_files}")

    frames = load_video_frames(args.video_path)
    vid_feat = encode_video(model, frames, frame_count=frame_count, frame_size=frame_size, device=device)

    # Batch-encode all queries for this video: positives + negatives
    texts = [q for (_vid, q, _src) in pos_matches] + [q for (_vid, q, _src) in neg_matches]
    labels = (["pos"] * len(pos_matches)) + (["neg"] * len(neg_matches))
    if not texts:
        raise ValueError("Matched entries contain no 'query' text.")
    text_feats = [encode_text(model, t) for t in texts]
    txt_feat = torch.cat(text_feats, dim=0)  # [N, D]

    # Similarities and optional softmax probabilities (scaled like demo)
    sims = (vid_feat @ txt_feat.T).squeeze(0)  # [N]
    probs = torch.softmax(100.0 * sims, dim=0)

    # Sort by similarity desc and show top-k
    topk = max(1, min(args.topk, sims.numel()))
    values, indices = torch.topk(sims, k=topk, dim=0)

    print("InternVideo2 query ranking (target vs other video queries)")
    print(f"  video           : {args.video_path}")
    print(f"  checkpoint      : {cfg.pretrained_path}")
    print(f"  device          : {cfg.device}")
    print(f"  target hint     : {pos_hint}  (queries: {len(pos_matches)})")
    print(f"  other hint      : {neg_hint}  (queries: {len(neg_matches)})")
    print(f"  total queries   : {len(texts)} from {[p.name for p in query_files]}")
    print()

    for rank, idx in enumerate(indices.tolist(), start=1):
        # Map back to pos/neg space
        set_label = labels[idx]
        if set_label == "pos":
            vid, q, src = pos_matches[idx]
        else:
            neg_idx = idx - len(pos_matches)
            vid, q, src = neg_matches[neg_idx]
        sim = float(values[rank - 1].cpu())
        pr = float(probs[idx].cpu())
        print(f"#{rank:2d} [{set_label}] sim={sim:.4f} prob={pr:.4f} | {q}")


if __name__ == "__main__":
    main()
