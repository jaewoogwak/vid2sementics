#!/usr/bin/env python3
"""
Encode QVHighlights videos and text queries with InternVideo2 and store the
results in HDF5 files keyed by video_id.

Each video under dataset/qvhighlights/concat is sampled at 1 FPS. Sliding
windows of 8 frames with stride 4 are passed through the InternVideo2 vision
encoder to obtain clip embeddings. All clip embeddings for a video are stored
contiguously so downstream tasks can aggregate them in any way they like.

Text queries from queries_by_video_{train,val}.jsonl are encoded with the
matching InternVideo2 text encoder. The script saves both the embeddings and
the original query strings so they can be aligned later.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import os
import sys
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch


V_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
V_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset = repo_root / "dataset" / "qvhighlights"
    default_multi_root = repo_root / "InternVideo" / "InternVideo2" / "multi_modality"
    default_config = default_multi_root / "demo" / "internvideo2_stage2_config.py"

    parser = argparse.ArgumentParser(
        description="Encode QVHighlights videos and queries with InternVideo2."
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
        help="Configuration file used to instantiate InternVideo2 (default: %(default)s)",
    )
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=repo_root
        / "InternVideo"
        / "InternVideo2"
        / "ckpt"
        / "InternVideo2-stage2_1b-224p-f4.pt",
        help="Checkpoint to load (default: %(default)s)",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=default_dataset / "concat",
        help="Directory containing concatenated QVHighlights videos (default: %(default)s)",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=default_dataset / "queries_by_video_train.jsonl",
        help="JSONL with train queries grouped by video_id (default: %(default)s)",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=default_dataset / "queries_by_video_val.jsonl",
        help="JSONL with val queries grouped by video_id (default: %(default)s)",
    )
    parser.add_argument(
        "--video-hdf5",
        type=Path,
        default=default_dataset / "internvideo_video_embeddings.h5",
        help="Output HDF5 file that stores clip embeddings per video (default: %(default)s)",
    )
    parser.add_argument(
        "--text-hdf5",
        type=Path,
        default=default_dataset / "internvideo_text_embeddings.h5",
        help="Output HDF5 file that stores query embeddings per video (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Temporal sampling rate applied to every video before clipping (default: %(default)s)",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=8,
        help="Number of consecutive frames per clip passed to the model (default: %(default)s)",
    )
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=4,
        help="Stride between successive clips measured in sampled frames (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        help="Square spatial resolution to resize frames to (default: %(default)s)",
    )
    parser.add_argument(
        "--origin-num-frames",
        type=int,
        default=None,
        help="Number of frames the checkpoint was trained with (used for positional interpolation).",
    )
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=16,
        help="How many clips to encode per forward pass (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to run inference on (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if they already exist in the HDF5 files.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="How often to log progress while sampling long videos (in seconds of source video).",
    )
    args = parser.parse_args()
    return args


def ensure_syspath(path: Path) -> None:
    abs_path = path.expanduser().resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"multi_modality root not found: {abs_path}")
    if str(abs_path) not in sys.path:
        sys.path.insert(0, str(abs_path))


def load_demo_config(config_path: Path, pretrained_path: Path | None, device: str):
    import importlib

    demo_config = importlib.import_module("demo_config")
    cfg = demo_config.Config.from_file(str(config_path.resolve()))
    cfg = demo_config.eval_dict_leaf(cfg)

    resolved_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    if resolved_device != device:
        logging.warning("CUDA unavailable, falling back to CPU.")
    cfg.device = resolved_device

    ckpt = pretrained_path or Path(os.environ.get("INTERNVIDEO2_STAGE2_CKPT", ""))
    if ckpt:
        cfg.pretrained_path = str(Path(ckpt).expanduser())
    if not cfg.pretrained_path:
        raise ValueError("Set --pretrained-path or $INTERNVIDEO2_STAGE2_CKPT to a valid checkpoint.")
    if not Path(cfg.pretrained_path).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.pretrained_path}")

    return cfg


def maybe_override_model_inputs(
    cfg,
    num_frames: int | None,
    frame_size: int | None,
    origin_num_frames_override: int | None = None,
):
    original_training_frames = getattr(cfg, "origin_num_frames", cfg.model.vision_encoder.num_frames)
    if origin_num_frames_override is not None:
        original_training_frames = origin_num_frames_override

    if num_frames is not None:
        cfg.num_frames = num_frames
        cfg.num_frames_test = num_frames
        cfg.model.vision_encoder.num_frames = num_frames
        cfg.origin_num_frames = original_training_frames

    if frame_size is not None:
        cfg.size_t = frame_size
        cfg.model.vision_encoder.img_size = frame_size


def load_query_jsonl(json_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_video_metadata(train_json: Path, val_json: Path) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (ordered_video_ids, split_to_ids, video_to_queries)."""
    split_paths = {"train": train_json, "val": val_json}
    video_to_queries: Dict[str, List[str]] = {}
    split_to_ids: Dict[str, List[str]] = {split: [] for split in split_paths}

    for split, path in split_paths.items():
        rows = load_query_jsonl(path)
        for row in rows:
            vid = row["video_id"]
            queries = row.get("queries", [])
            if not isinstance(queries, list):
                raise ValueError(f"Expected 'queries' list for video_id={vid} in {path}")
            video_to_queries.setdefault(vid, [])
            video_to_queries[vid].extend(str(q) for q in queries)
            split_to_ids[split].append(vid)

    ordered_ids = sorted(video_to_queries)
    return ordered_ids, split_to_ids, video_to_queries


def preprocess_frame(frame: np.ndarray, frame_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - V_MEAN) / V_STD
    return np.transpose(normalized, (2, 0, 1))


def iter_sampled_frames(
    video_path: Path,
    *,
    sample_fps: float,
    frame_size: int,
    log_interval: int,
) -> Iterator[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or np.isnan(native_fps) or native_fps <= 0:
        native_fps = 30.0
    period = 1.0 / sample_fps
    next_time = 0.0
    sampled = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / native_fps
        if timestamp + (0.5 / native_fps) >= next_time:
            yield timestamp, preprocess_frame(frame, frame_size)
            next_time += period
            sampled += 1
            if log_interval and sampled % log_interval == 0:
                logging.info("Sampled %d seconds from %s", sampled, video_path.name)
        frame_idx += 1

    cap.release()


def clips_to_tensor(
    clips: Sequence[np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    data = np.stack(clips, axis=0)
    tensor = torch.from_numpy(data)
    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
    return tensor


def encode_video(
    model,
    video_path: Path,
    *,
    sample_fps: float,
    frames_per_clip: int,
    clip_stride: int,
    frame_size: int,
    batch_size: int,
    log_interval: int,
) -> Tuple[np.ndarray, np.ndarray]:
    frame_iterator = iter_sampled_frames(
        video_path,
        sample_fps=sample_fps,
        frame_size=frame_size,
        log_interval=log_interval,
    )
    buffer: Deque[np.ndarray] = deque(maxlen=frames_per_clip)
    time_buffer: Deque[float] = deque(maxlen=frames_per_clip)

    features: List[np.ndarray] = []
    starts_sec: List[float] = []
    device = torch.device(model.config.device)
    dtype = model.dtype

    batch_clips: List[np.ndarray] = []
    batch_starts: List[float] = []
    sample_idx = -1

    for sample_idx, (timestamp, processed_frame) in enumerate(frame_iterator):
        buffer.append(processed_frame)
        time_buffer.append(timestamp)
        if len(buffer) < frames_per_clip:
            continue
        clip_start_idx = sample_idx - frames_per_clip + 1
        if clip_start_idx % clip_stride != 0:
            continue

        batch_clips.append(np.stack(list(buffer), axis=0))
        batch_starts.append(time_buffer[0])

        if len(batch_clips) == batch_size:
            clip_tensor = clips_to_tensor(
                batch_clips,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                vid_feat = model.get_vid_feat(clip_tensor)
            vid_feat = vid_feat.squeeze(1).cpu().numpy().astype(np.float32, copy=False)
            features.append(vid_feat)
            starts_sec.extend(batch_starts)
            batch_clips.clear()
            batch_starts.clear()

    if batch_clips:
        clip_tensor = clips_to_tensor(batch_clips, device=device, dtype=dtype)
        with torch.no_grad():
            vid_feat = model.get_vid_feat(clip_tensor)
        vid_feat = vid_feat.squeeze(1).cpu().numpy().astype(np.float32, copy=False)
        features.append(vid_feat)
        starts_sec.extend(batch_starts)

    if not features:
        logging.warning(
            "Video %s has < %d sampled frames; skipping.",
            video_path.name,
            frames_per_clip,
        )
        return np.empty((0, model.embed_dim), dtype=np.float32), np.empty((0,), dtype=np.float32)

    all_features = np.concatenate(features, axis=0) if features else np.empty((0, model.embed_dim), dtype=np.float32)
    return all_features, np.array(starts_sec, dtype=np.float32)


def encode_queries(model, queries: Sequence[str]) -> np.ndarray:
    feats: List[np.ndarray] = []
    for text in queries:
        with torch.no_grad():
            feat = model.get_txt_feat(text)
        feats.append(feat.cpu().numpy()[0].astype(np.float32, copy=False))
    if not feats:
        return np.empty((0, model.embed_dim), dtype=np.float32)
    return np.stack(feats, axis=0)


def prepare_hdf5(path: Path) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(path, "a")


def write_splits(h5_file: h5py.File, split_to_ids: Dict[str, List[str]]) -> None:
    grp = h5_file.require_group("splits")
    str_dtype = h5py.string_dtype(encoding="utf-8")
    for split_name, ids in split_to_ids.items():
        sorted_ids = np.array(sorted(set(ids)), dtype=str_dtype)
        if split_name in grp:
            del grp[split_name]
        grp.create_dataset(split_name, data=sorted_ids)


def store_video_embeddings(
    h5_file: h5py.File,
    video_id: str,
    features: np.ndarray,
    starts_sec: np.ndarray,
    *,
    overwrite: bool,
) -> None:
    grp = h5_file.require_group("videos")
    if video_id in grp:
        if overwrite:
            del grp[video_id]
        else:
            logging.info("Video %s already exists in %s; skipping.", video_id, h5_file.filename)
            return
    vid_group = grp.create_group(video_id)
    vid_group.create_dataset("clip_embeddings", data=features, compression="gzip")
    vid_group.create_dataset("clip_start_seconds", data=starts_sec, compression="gzip")


def store_text_embeddings(
    h5_file: h5py.File,
    video_id: str,
    features: np.ndarray,
    queries: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    grp = h5_file.require_group("queries")
    if video_id in grp:
        if overwrite:
            del grp[video_id]
        else:
            logging.info("Queries for %s already exist in %s; skipping.", video_id, h5_file.filename)
            return
    vid_group = grp.create_group(video_id)
    vid_group.create_dataset("embeddings", data=features, compression="gzip")
    if queries:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        vid_group.create_dataset("texts", data=np.array(queries, dtype=str_dtype))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ensure_syspath(args.multi_modality_root)
    cfg = load_demo_config(args.config_path, args.pretrained_path, args.device)
    maybe_override_model_inputs(
        cfg,
        num_frames=args.frames_per_clip,
        frame_size=args.frame_size,
        origin_num_frames_override=args.origin_num_frames,
    )

    from demo.utils import setup_internvideo2

    model, _ = setup_internvideo2(cfg)
    model.eval()

    ordered_ids, split_to_ids, video_to_queries = collect_video_metadata(args.train_json, args.val_json)

    video_h5 = prepare_hdf5(args.video_hdf5)
    text_h5 = prepare_hdf5(args.text_hdf5)
    try:
        video_h5.attrs["sample_fps"] = args.sample_fps
        video_h5.attrs["frames_per_clip"] = args.frames_per_clip
        video_h5.attrs["clip_stride"] = args.clip_stride
        video_h5.attrs["frame_size"] = args.frame_size
        text_h5.attrs["max_query_len"] = cfg.max_txt_l
        write_splits(video_h5, split_to_ids)
        write_splits(text_h5, split_to_ids)

        for idx, video_id in enumerate(ordered_ids, start=1):
            video_path = args.video_dir / f"{video_id}.mp4"
            if not video_path.is_file():
                logging.warning("Missing video file for %s", video_id)
                continue

            logging.info("(%d/%d) Encoding video %s", idx, len(ordered_ids), video_id)
            clip_feats, clip_starts = encode_video(
                model,
                video_path,
                sample_fps=args.sample_fps,
                frames_per_clip=args.frames_per_clip,
                clip_stride=args.clip_stride,
                frame_size=args.frame_size,
                batch_size=args.video_batch_size,
                log_interval=args.log_interval,
            )
            store_video_embeddings(
                video_h5,
                video_id,
                clip_feats,
                clip_starts,
                overwrite=args.overwrite,
            )

            logging.info("Encoding %d queries for %s", len(video_to_queries.get(video_id, [])), video_id)
            text_feats = encode_queries(model, video_to_queries.get(video_id, []))
            store_text_embeddings(
                text_h5,
                video_id,
                text_feats,
                video_to_queries.get(video_id, []),
                overwrite=args.overwrite,
            )
    finally:
        video_h5.close()
        text_h5.close()


if __name__ == "__main__":
    main()
