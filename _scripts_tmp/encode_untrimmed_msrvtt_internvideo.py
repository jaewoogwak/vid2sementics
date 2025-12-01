#!/usr/bin/env python3
"""Encode untrimmed MSRVTT clips with InternVideo2 vision/text encoders."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for --help usage
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from encode_internvideo_embeddings import (
    clips_to_tensor,
    ensure_syspath,
    iter_sampled_frames,
    load_demo_config,
    maybe_override_model_inputs,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset = repo_root / "dataset" / "data" / "MSRVTT"
    default_multi_root = repo_root / "InternVideo" / "InternVideo2" / "multi_modality"
    default_config = default_multi_root / "demo" / "internvideo2_stage2_config.py"
    default_ckpt = (
        repo_root
        / "InternVideo"
        / "InternVideo2"
        / "ckpt"
        / "InternVideo2-stage2_1b-224p-f4.pt"
    )

    parser = argparse.ArgumentParser(
        description="Encode untrimmed MSRVTT videos and queries with InternVideo2."
    )
    parser.add_argument(
        "--untrimmed-json",
        type=Path,
        default=default_dataset / "annotation" / "MSRVTT_untrimmed.json",
        help="Path to JSON describing untrimmed MSRVTT videos.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=default_dataset / "videos" / "all",
        help="Directory containing original MSRVTT clips (videoXXXX.mp4).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_dataset / "features" / "internvideo_untrimmed",
        help="Directory used to store encoded features (vision/ + text/).",
    )
    parser.add_argument(
        "--internvideo-config",
        type=Path,
        default=default_config,
        help="Config file for InternVideo2.",
    )
    parser.add_argument(
        "--internvideo-ckpt",
        type=Path,
        default=default_ckpt,
        help="Checkpoint to load for InternVideo2.",
    )
    parser.add_argument(
        "--multi-modality-root",
        type=Path,
        default=default_multi_root,
        help="Path to InternVideo2/multi_modality (where demo_config.py lives).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device used for inference.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        help="Square resolution frames are resized to before encoding.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="FPS used while uniformly sampling frames from each raw clip.",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=8,
        help="Number of evenly spaced sampled frames passed to the model per clip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if output files already exist.",
    )
    args = parser.parse_args()
    return args


def load_untrimmed_entries(json_path: Path) -> List[Dict[str, object]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("videos")
    if not isinstance(entries, list):
        raise ValueError(f"Expected 'videos' list in {json_path}")
    return entries


def evenly_select_frames(frames: Sequence[np.ndarray], target: int) -> List[np.ndarray]:
    if not frames:
        return []
    if target <= 0:
        raise ValueError("Target number of frames must be > 0.")
    indices = np.linspace(0, len(frames) - 1, num=target)
    indices = np.clip(np.round(indices).astype(int), 0, len(frames) - 1)
    return [frames[idx] for idx in indices]


def sample_clip_frames(
    video_path: Path,
    *,
    sample_fps: float,
    frame_size: int,
    frames_per_clip: int,
) -> np.ndarray | None:
    sampled = [
        frame
        for _, frame in iter_sampled_frames(
            video_path,
            sample_fps=sample_fps,
            frame_size=frame_size,
            log_interval=0,
        )
    ]
    if not sampled:
        logging.warning("No frames sampled from %s", video_path)
        return None
    selected = evenly_select_frames(sampled, frames_per_clip)
    if not selected:
        return None
    clip = np.stack(selected, axis=0).astype(np.float32, copy=False)
    return clip


def encode_clip_feature(model, clip: np.ndarray) -> torch.Tensor:
    device = torch.device(model.config.device)
    tensor = clips_to_tensor([clip], device=device, dtype=model.dtype)
    with torch.no_grad():
        feat = model.get_vid_feat(tensor)
    feat = feat.squeeze(1).squeeze(0).detach().cpu().float()
    return feat


def encode_query_features(model, queries: Sequence[str]) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    for text in queries:
        with torch.no_grad():
            embedding = model.get_txt_feat(text)
        feats.append(embedding.squeeze(0).detach().cpu().float())
    if not feats:
        return torch.empty((0, model.embed_dim), dtype=torch.float32)
    return torch.stack(feats, dim=0)


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required to run this script. Please install torch before continuing."
        ) from TORCH_IMPORT_ERROR
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ensure_syspath(args.multi_modality_root)
    cfg = load_demo_config(args.internvideo_config, args.internvideo_ckpt, args.device)
    maybe_override_model_inputs(
        cfg,
        num_frames=args.frames_per_clip,
        frame_size=args.frame_size,
        origin_num_frames_override=None,
    )

    from demo.utils import setup_internvideo2

    model, _ = setup_internvideo2(cfg)
    model.eval()

    entries = load_untrimmed_entries(args.untrimmed_json)
    vision_dir = args.output_root / "vision"
    text_dir = args.output_root / "text"
    vision_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    split_counts: Dict[str, int] = defaultdict(int)
    for idx, entry in enumerate(entries, start=1):
        video_id = entry.get("video_id")
        split = entry.get("split", "unknown")
        video_list = entry.get("video_list", [])
        queries = entry.get("queries", [])
        if not isinstance(video_id, str) or not video_id:
            logging.warning("Entry %d missing video_id; skipping.", idx)
            continue
        if not isinstance(video_list, list) or not video_list:
            logging.warning("Entry %s has empty video_list; skipping.", video_id)
            continue
        if not isinstance(queries, list) or len(queries) != len(video_list):
            logging.warning(
                "Entry %s queries mismatch (expected %d, got %d); skipping.",
                video_id,
                len(video_list),
                len(queries) if isinstance(queries, list) else -1,
            )
            continue

        vision_out = vision_dir / f"{video_id}.pt"
        text_out = text_dir / f"{video_id}.pt"
        if vision_out.exists() and text_out.exists() and not args.overwrite:
            logging.info("Features for %s exist; skipping.", video_id)
            split_counts[str(split)] += 1
            continue

        clip_features: List[torch.Tensor] = []
        missing_clip = False
        for clip_id in video_list:
            clip_path = args.video_root / f"{clip_id}.mp4"
            if not clip_path.is_file():
                logging.warning("Missing raw clip %s for entry %s", clip_path, video_id)
                missing_clip = True
                break
            clip_frames = sample_clip_frames(
                clip_path,
                sample_fps=args.sample_fps,
                frame_size=args.frame_size,
                frames_per_clip=args.frames_per_clip,
            )
            if clip_frames is None:
                missing_clip = True
                logging.warning("Could not sample frames from %s; skipping entry %s", clip_id, video_id)
                break
            clip_feat = encode_clip_feature(model, clip_frames)
            clip_features.append(clip_feat)

        if missing_clip or not clip_features:
            continue

        stacked_clip_feat = torch.stack(clip_features, dim=0)
        vision_payload = {
            "video_id": video_id,
            "split": split,
            "video_list": video_list,
            "relevant_windows": entry.get("relevant_windows", []),
            "feat": stacked_clip_feat,
        }
        torch.save(vision_payload, vision_out)

        text_features = encode_query_features(model, queries)
        text_payload = {
            "video_id": video_id,
            "split": split,
            "queries": queries,
            "text_feat": text_features,
        }
        torch.save(text_payload, text_out)
        split_counts[str(split)] += 1
        logging.info(
            "[%d/%d] Encoded %s: %d clips -> feat shape %s; %d queries.",
            idx,
            len(entries),
            video_id,
            len(video_list),
            tuple(stacked_clip_feat.shape),
            len(queries),
        )

    logging.info("Finished encoding untrimmed MSRVTT videos.")
    for split, count in sorted(split_counts.items()):
        logging.info("  Split %s: %d entries", split, count)
    logging.info("Vision features dir: %s", vision_dir)
    logging.info("Text features dir: %s", text_dir)


if __name__ == "__main__":
    main()
