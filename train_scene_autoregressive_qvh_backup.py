#!/usr/bin/env python3
"""
Scene-level autoregressive training script tailored to QVHighlights with
additional support for MSRVTT Untrimmed and ActivityNet CLIP feature dumps.

This script builds a single-file pipeline that:
  * Parses QVHighlights highlight annotations and groups scene descriptions per
    untrimmed base video.
  * Samples untrimmed videos at 1 FPS, forms exhaustive sliding clips, and
    encodes them with the frozen InternVideo2 stage2 vision backbone.
  * Encodes scene texts with the matching frozen InternVideo2 text encoder.
  * Trains a Transformer-based autoregressive latent scene generator that
    predicts scene embeddings conditioned on clip context and supervises
    alignment to text and temporal grounding windows.
  * Provides an inference helper that autoregressively decodes scene embeddings
    until an EOS prototype is predicted.

When launched with the ``msrvtt_untrimmed`` or ``activitynet`` dataset options,
the script skips video decoding and consumes precomputed vision/text features
from disk instead of encoding raw videos on the fly.

All functionality lives in this single file so it can be invoked as:

python train_scene_autoregressive_qvh.py \
    --internvideo-root /dev/ssd1/gjw/prvr/InternVideo \
    --dataset-root /dev/ssd1/gjw/prvr/dataset/qvhighlights \
    --epochs 5 --batch-size 2 --lr 1e-4 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency check
    h5py = None
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


V_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
V_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class VideoSceneItem:
    video_id: str
    video_path: Path
    scene_texts: List[str]
    scene_windows: List[Tuple[float, float]]


@dataclass
class MSRVTTFeatureMeta:
    video_id: str
    vision_path: Path
    text_path: Path
    scene_texts: List[str]
    scene_windows: List[Tuple[float, float]]


@dataclass
class MSRVTTSceneItem:
    video_id: str
    clip_embeddings: torch.Tensor
    clip_times: torch.Tensor
    scene_texts: List[str]
    scene_windows: List[Tuple[float, float]]
    text_embeddings: torch.Tensor


@dataclass
class ActivityNetFeatureMeta:
    video_id: str
    duration: float
    num_frames: int
    fps: float
    scene_texts: List[str]
    timestamps: List[Tuple[float, float]]
    scene_windows: List[Tuple[int, int]]
    text_keys: List[str]


@dataclass
class ActivityNetSceneItem:
    video_id: str
    clip_embeddings: torch.Tensor
    clip_times: torch.Tensor
    scene_texts: List[str]
    scene_windows: List[Tuple[int, int]]
    text_embeddings: torch.Tensor
    timestamps: List[Tuple[float, float]]
    duration: float
    fps: float


def parse_vid_identifier(vid: str) -> Tuple[str, float, float]:
    parts = vid.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid vid format: {vid}")
    base_video = parts[0]
    start_sec = float(parts[1])
    end_sec = float(parts[2])
    return base_video, start_sec, end_sec


class QVHighlightsDataset(Dataset):
    """
    Dataset that groups highlight queries per base untrimmed video and exposes
    scene-level supervision windows in global time coordinates.
    """

    def __init__(self, jsonl_path: Path, concat_root: Path, raw_root: Path):
        self.raw_root = raw_root
        self.items: List[VideoSceneItem] = self._build_items(jsonl_path, concat_root)
        self.total_queries: int = sum(len(item.scene_texts) for item in self.items)
        if not self.items:
            raise RuntimeError(
                f"No valid videos found under {concat_root} using annotations {jsonl_path}"
            )
        logging.info(
            "QVHighlights dataset loaded with %d base videos (%d mapped queries)",
            len(self.items),
            self.total_queries,
        )

    def _build_items(self, jsonl_path: Path, concat_root: Path) -> List[VideoSceneItem]:
        video_segments: Dict[str, Dict[Tuple[float, float], Dict[str, object]]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                vid = row["vid"]
                query = row["query"]
                relevant_windows = row.get("relevant_windows", [])
                if not relevant_windows:
                    continue
                base_video, segment_start, segment_end = parse_vid_identifier(vid)
                segment_key = (segment_start, segment_end)
                per_video = video_segments.setdefault(base_video, {})
                segment_entry = per_video.setdefault(
                    segment_key,
                    {"start": segment_start, "end": segment_end, "queries": []},
                )
                windows: List[Tuple[float, float]] = []
                for window in relevant_windows:
                    if not isinstance(window, (list, tuple)) or len(window) != 2:
                        continue
                    local_start, local_end = float(window[0]), float(window[1])
                    if local_end <= local_start:
                        continue
                    windows.append((local_start, local_end))
                if not windows:
                    continue
                segment_entry["queries"].append({"query": query, "windows": windows})

        items: List[VideoSceneItem] = []
        for video_id, segments in sorted(video_segments.items()):
            video_path = concat_root / f"{video_id}.mp4"
            if not video_path.is_file():
                logging.warning("Missing concatenated video for %s at %s", video_id, video_path)
                continue
            if not segments:
                continue
            sorted_segments = sorted(
                segments.values(), key=lambda seg: float(seg["start"])
            )
            scene_texts: List[str] = []
            scene_windows: List[Tuple[float, float]] = []
            running_offset = 0.0
            for segment in sorted_segments:
                seg_start = float(segment["start"])
                seg_end = float(segment["end"])
                seg_duration = max(0.0, seg_end - seg_start)
                raw_path = self.raw_root / f"{video_id}_{seg_start}_{seg_end}.mp4"
                if not raw_path.is_file():
                    logging.debug(
                        "Raw segment missing for %s [%s, %s]; skipping queries.",
                        video_id,
                        seg_start,
                        seg_end,
                    )
                    continue
                queries = segment["queries"]
                if not queries or seg_duration <= 0:
                    continue
                for query_entry in queries:
                    text = query_entry["query"]
                    for local_start, local_end in query_entry["windows"]:
                        global_start = running_offset + local_start
                        global_end = running_offset + local_end
                        scene_texts.append(text)
                        scene_windows.append((global_start, global_end))
                running_offset += seg_duration
            if not scene_texts:
                continue
            items.append(
                VideoSceneItem(
                    video_id=video_id,
                    video_path=video_path,
                    scene_texts=scene_texts,
                    scene_windows=scene_windows,
                )
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> VideoSceneItem:
        return self.items[idx]


class MSRVTTUntrimmedDataset(Dataset):
    """Dataset that loads precomputed InternVideo2 features for MSRVTT untrimmed videos."""

    def __init__(
        self,
        annotation_path: Path,
        feat_root: Path,
        split: str,
        *,
        cache_features: bool = True,
    ):
        self.annotation_path = annotation_path.expanduser()
        self.feat_root = feat_root.expanduser()
        self.vision_root = self.feat_root / "vision"
        self.text_root = self.feat_root / "text"
        self.split = split
        self.cache_features = cache_features
        self.items: List[MSRVTTFeatureMeta] = self._load_metadata()
        if not self.items:
            raise RuntimeError(
                f"MSRVTT split '{self.split}' has no valid videos using annotation {self.annotation_path}"
            )
        self.total_queries = sum(len(item.scene_texts) for item in self.items)
        self._feature_cache: Dict[str, MSRVTTSceneItem] = {}
        self.feature_dim = self._infer_feature_dim()
        logging.info(
            "MSRVTT dataset loaded: split=%s videos=%d queries=%d feature_dim=%d",
            self.split,
            len(self.items),
            self.total_queries,
            self.feature_dim,
        )

    def _load_metadata(self) -> List[MSRVTTFeatureMeta]:
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("videos", [])
        items: List[MSRVTTFeatureMeta] = []
        for entry in entries:
            if entry.get("split") != self.split:
                continue
            video_id = entry.get("video_id")
            if not video_id:
                continue
            vision_path = self.vision_root / f"{video_id}.pt"
            text_path_primary = self.text_root / f"{video_id}_texts.pt"
            text_path_alt = self.text_root / f"{video_id}.pt"
            text_path = text_path_primary if text_path_primary.is_file() else text_path_alt
            if not vision_path.is_file():
                logging.warning("Skipping %s: missing vision feats at %s", video_id, vision_path)
                continue
            if not text_path.is_file():
                logging.warning(
                    "Skipping %s: missing text feats (checked %s and %s)",
                    video_id,
                    text_path_primary,
                    text_path_alt,
                )
                continue
            queries = entry.get("queries") or []
            windows = entry.get("relevant_windows") or []
            if not queries or not windows:
                logging.warning("Skipping %s: empty queries/windows", video_id)
                continue
            if len(windows) != len(queries):
                logging.warning(
                    "Video %s: query count (%d) != window count (%d); truncating to min length",
                    video_id,
                    len(queries),
                    len(windows),
                )
            limit = min(len(queries), len(windows))
            if limit == 0:
                continue
            scene_texts = [str(queries[i]) for i in range(limit)]
            scene_windows = []
            for i in range(limit):
                window = windows[i]
                if (
                    not isinstance(window, (list, tuple))
                    or len(window) != 2
                ):
                    logging.warning("Video %s: invalid window format at index %d", video_id, i)
                    continue
                start, end = float(window[0]), float(window[1])
                if end <= start:
                    logging.warning(
                        "Video %s: window[%d] non-positive duration (start=%.2f end=%.2f)",
                        video_id,
                        i,
                        start,
                        end,
                    )
                    continue
                scene_windows.append((start, end))
            if not scene_windows or len(scene_windows) != len(scene_texts):
                logging.warning("Skipping %s: mismatched valid windows/texts", video_id)
                continue
            items.append(
                MSRVTTFeatureMeta(
                    video_id=video_id,
                    vision_path=vision_path,
                    text_path=text_path,
                    scene_texts=scene_texts,
                    scene_windows=scene_windows,
                )
            )
        return items

    def _load_sample(self, meta: MSRVTTFeatureMeta) -> MSRVTTSceneItem:
        vision = torch.load(meta.vision_path, map_location="cpu")

        def _extract_tensor(source, keys: Sequence[str]) -> Optional[torch.Tensor]:
            if isinstance(source, torch.Tensor):
                return source
            if isinstance(source, dict):
                for key in keys:
                    value = source.get(key)
                    if isinstance(value, torch.Tensor):
                        return value
            return None

        clip_embeddings = _extract_tensor(vision, ("clip_embeddings", "feat", "features", "clip_feat"))
        if clip_embeddings is None:
            available = list(vision.keys()) if isinstance(vision, dict) else type(vision).__name__
            raise RuntimeError(
                f"Vision feats for {meta.video_id} missing clip embeddings (available={available})"
            )
        clip_embeddings = clip_embeddings.float()

        raw_clip_times = None
        if isinstance(vision, dict):
            for key in ("clip_times", "times", "timestamps"):
                if key in vision:
                    raw_clip_times = vision[key]
                    break
        clip_count = clip_embeddings.shape[0] if clip_embeddings.ndim >= 1 else clip_embeddings.numel()
        if raw_clip_times is None:
            clip_times = torch.arange(clip_count, dtype=torch.float32)
        else:
            clip_times = (
                raw_clip_times.float()
                if isinstance(raw_clip_times, torch.Tensor)
                else torch.tensor(list(raw_clip_times), dtype=torch.float32)
            )
            if clip_times.numel() != clip_count:
                logging.warning(
                    "Video %s: timestamp count (%d) != clip count (%d); generating dummy times.",
                    meta.video_id,
                    clip_times.numel(),
                    clip_count,
                )
                clip_times = torch.arange(clip_count, dtype=torch.float32)

        text_feat = torch.load(meta.text_path, map_location="cpu")
        text_embeddings = _extract_tensor(
            text_feat, ("embeddings", "text_embeddings", "text_feat", "feat", "features")
        )
        if text_embeddings is None:
            available = list(text_feat.keys()) if isinstance(text_feat, dict) else type(text_feat).__name__
            raise RuntimeError(
                f"Text feats for {meta.video_id} missing embeddings (available={available})"
            )
        text_embeddings = text_embeddings.float()

        feature_texts: Optional[List[str]] = None
        if isinstance(text_feat, dict):
            for key in ("texts", "queries", "captions"):
                if key in text_feat and text_feat[key] is not None:
                    feature_texts = [str(entry) for entry in text_feat[key]]
                    break

        scene_texts = meta.scene_texts
        scene_windows = meta.scene_windows
        if feature_texts and len(feature_texts) != len(scene_texts):
            logging.warning(
                "Video %s: text feature count (%d) != annotation count (%d); truncating.",
                meta.video_id,
                len(feature_texts),
                len(scene_texts),
            )
            limit = min(len(scene_texts), len(feature_texts))
            scene_texts = scene_texts[:limit]
            scene_windows = scene_windows[:limit]
            text_embeddings = text_embeddings[:limit]
        if text_embeddings.shape[0] != len(scene_texts):
            limit = min(len(scene_texts), text_embeddings.shape[0])
            logging.warning(
                "Video %s: embeddings/text mismatch (%d vs %d); truncating.",
                meta.video_id,
                text_embeddings.shape[0],
                len(scene_texts),
            )
            text_embeddings = text_embeddings[:limit]
            scene_texts = scene_texts[:limit]
            scene_windows = scene_windows[:limit]

        return MSRVTTSceneItem(
            video_id=meta.video_id,
            clip_embeddings=clip_embeddings,
            clip_times=clip_times,
            scene_texts=scene_texts,
            scene_windows=scene_windows,
            text_embeddings=text_embeddings,
        )

    def _infer_feature_dim(self) -> int:
        if not self.items:
            return 0
        first = self.items[0]
        sample = self._load_sample(first)
        self._feature_cache[first.video_id] = sample
        if sample.clip_embeddings.ndim != 2:
            raise RuntimeError(f"Vision feats for {first.video_id} missing clip dimension.")
        return sample.clip_embeddings.shape[1]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> MSRVTTSceneItem:
        meta = self.items[idx]
        if self.cache_features and meta.video_id in self._feature_cache:
            return self._feature_cache[meta.video_id]
        sample = self._load_sample(meta)
        if self.cache_features:
            self._feature_cache[meta.video_id] = sample
        return sample


class ActivityNetSceneDataset(Dataset):
    """Dataset that reads ActivityNet CLIP features and scene annotations."""

    def __init__(
        self,
        annotation_path: Path,
        video_feat_path: Path,
        text_feat_path: Path,
        *,
        split_name: str = "train",
        cache_features: bool = True,
    ) -> None:
        if h5py is None:
            raise ImportError(
                "h5py is required to read ActivityNet HDF5 features. Please install h5py first."
            )
        self.annotation_path = annotation_path.expanduser()
        self.video_feat_path = video_feat_path.expanduser()
        self.text_feat_path = text_feat_path.expanduser()
        self.split_name = split_name
        self.cache_features = cache_features
        self.items: List[ActivityNetFeatureMeta] = self._load_metadata()
        if not self.items:
            raise RuntimeError(
                f"ActivityNet split '{self.split_name}' has no valid videos (annotation={self.annotation_path})"
            )
        self.total_queries = sum(len(meta.scene_texts) for meta in self.items)
        self._feature_cache: Dict[str, ActivityNetSceneItem] = {}
        self.feature_dim = self._infer_feature_dim()
        logging.info(
            "ActivityNet dataset loaded: split=%s videos=%d queries=%d feature_dim=%d",
            self.split_name,
            len(self.items),
            self.total_queries,
            self.feature_dim,
        )

    def _resolve_text_keys(
        self,
        video_id: str,
        count: int,
        text_file: "h5py.File",
    ) -> Optional[List[str]]:
        if count <= 0:
            return []
        offset: Optional[int] = None
        for candidate in (0, 1):
            key = f"{video_id}#enc#{candidate}"
            if key in text_file:
                offset = candidate
                break
        if offset is None:
            return None
        keys: List[str] = []
        for idx in range(count):
            key = f"{video_id}#enc#{offset + idx}"
            if key not in text_file:
                return None
            keys.append(key)
        return keys

    def _load_metadata(self) -> List[ActivityNetFeatureMeta]:
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"ActivityNet annotation {self.annotation_path} must be a JSON object mapping video ids."
            )
        items: List[ActivityNetFeatureMeta] = []
        assert h5py is not None  # for type checkers
        with h5py.File(self.video_feat_path, "r") as video_file, h5py.File(
            self.text_feat_path, "r"
        ) as text_file:
            available_videos = set(video_file.keys())
            for video_id, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                duration = float(entry.get("duration") or 0.0)
                if duration <= 0:
                    logging.warning(
                        "Skipping %s: invalid duration %.3f (split=%s)",
                        video_id,
                        duration,
                        self.split_name,
                    )
                    continue
                sentences = entry.get("sentences") or []
                timestamps = entry.get("timestamps") or []
                if not sentences or not timestamps:
                    logging.warning("Skipping %s: empty sentences/timestamps", video_id)
                    continue
                if len(sentences) != len(timestamps):
                    limit = min(len(sentences), len(timestamps))
                    logging.warning(
                        "Video %s: mismatch between sentences (%d) and timestamps (%d); truncating to %d",
                        video_id,
                        len(sentences),
                        len(timestamps),
                        limit,
                    )
                    sentences = sentences[:limit]
                    timestamps = timestamps[:limit]
                if video_id not in available_videos:
                    logging.warning(
                        "Skipping %s: missing video features in %s",
                        video_id,
                        self.video_feat_path,
                    )
                    continue
                video_dataset = video_file[video_id]
                if video_dataset.ndim != 2:
                    logging.warning(
                        "Skipping %s: expected 2D video embeddings, got shape=%s",
                        video_id,
                        video_dataset.shape,
                    )
                    continue
                num_frames = int(video_dataset.shape[0])
                if num_frames <= 0:
                    logging.warning("Skipping %s: no video frames stored", video_id)
                    continue
                fps = num_frames / max(duration, 1e-6)
                resolved_timestamps: List[Tuple[float, float]] = []
                frame_windows: List[Tuple[int, int]] = []
                resolved_texts: List[str] = []
                for sent, window in zip(sentences, timestamps):
                    if not isinstance(window, (list, tuple)) or len(window) != 2:
                        continue
                    start = float(window[0])
                    end = float(window[1])
                    if end <= start:
                        continue
                    start_idx = int(math.floor(start * fps))
                    end_idx = int(math.ceil(end * fps))
                    start_idx = max(0, min(start_idx, num_frames - 1))
                    end_idx = max(start_idx + 1, min(end_idx, num_frames))
                    resolved_timestamps.append((start, end))
                    frame_windows.append((start_idx, end_idx))
                    resolved_texts.append(str(sent))
                if not resolved_texts:
                    logging.warning("Skipping %s: no valid timestamp/text pairs", video_id)
                    continue
                text_keys = self._resolve_text_keys(video_id, len(resolved_texts), text_file)
                if text_keys is None:
                    logging.warning("Skipping %s: missing text embeddings in %s", video_id, self.text_feat_path)
                    continue
                items.append(
                    ActivityNetFeatureMeta(
                        video_id=video_id,
                        duration=duration,
                        num_frames=num_frames,
                        fps=fps,
                        scene_texts=resolved_texts,
                        timestamps=resolved_timestamps,
                        scene_windows=frame_windows,
                        text_keys=text_keys,
                    )
                )
        return items

    def _load_sample(self, meta: ActivityNetFeatureMeta) -> ActivityNetSceneItem:
        assert h5py is not None
        with h5py.File(self.video_feat_path, "r") as video_file:
            video_array = np.array(video_file[meta.video_id], dtype=np.float32)
        clip_embeddings = torch.from_numpy(video_array)
        clip_times = torch.arange(meta.num_frames, dtype=torch.float32)
        text_features: List[np.ndarray] = []
        with h5py.File(self.text_feat_path, "r") as text_file:
            for key in meta.text_keys:
                text_features.append(np.array(text_file[key], dtype=np.float32))
        text_embeddings = torch.from_numpy(np.stack(text_features, axis=0))
        return ActivityNetSceneItem(
            video_id=meta.video_id,
            clip_embeddings=clip_embeddings,
            clip_times=clip_times,
            scene_texts=meta.scene_texts,
            scene_windows=meta.scene_windows,
            text_embeddings=text_embeddings,
            timestamps=meta.timestamps,
            duration=meta.duration,
            fps=meta.fps,
        )

    def _infer_feature_dim(self) -> int:
        if not self.items:
            return 0
        first = self.items[0]
        sample = self._load_sample(first)
        self._feature_cache[first.video_id] = sample
        if sample.clip_embeddings.ndim != 2:
            raise RuntimeError(f"ActivityNet vision feats for {first.video_id} missing clip dimension.")
        return int(sample.clip_embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> ActivityNetSceneItem:
        meta = self.items[idx]
        if self.cache_features and meta.video_id in self._feature_cache:
            return self._feature_cache[meta.video_id]
        sample = self._load_sample(meta)
        if self.cache_features:
            self._feature_cache[meta.video_id] = sample
        return sample


def get_video_cache_path(video_cache_root: Path, video_id: str) -> Path:
    return video_cache_root / f"{video_id}.pt"


def load_video_cache(
    video_cache_root: Optional[Path], video_id: str
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if video_cache_root is None:
        return None
    path = get_video_cache_path(video_cache_root, video_id)
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu")
    return data.get("clip_embeddings"), data.get("clip_times")


def save_video_cache(
    video_cache_root: Optional[Path],
    video_id: str,
    clip_embeddings: torch.Tensor,
    clip_times: torch.Tensor,
) -> None:
    if video_cache_root is None:
        return
    path = get_video_cache_path(video_cache_root, video_id)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "clip_embeddings": clip_embeddings.detach().cpu(),
            "clip_times": clip_times.detach().cpu(),
        },
        path,
    )


def get_text_cache_path(text_cache_root: Path, video_id: str) -> Path:
    return text_cache_root / f"{video_id}_texts.pt"


def load_text_cache(text_cache_root: Optional[Path], video_id: str):
    if text_cache_root is None:
        return None
    path = get_text_cache_path(text_cache_root, video_id)
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu")


def save_text_cache(
    text_cache_root: Optional[Path],
    video_id: str,
    texts: Sequence[str],
    embeddings: torch.Tensor,
) -> None:
    if text_cache_root is None:
        return
    path = get_text_cache_path(text_cache_root, video_id)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"texts": list(texts), "embeddings": embeddings.detach().cpu()},
        path,
    )


def log_dataset_samples(name: str, dataset: QVHighlightsDataset, limit: int = 3) -> None:
    if len(dataset) == 0:
        logging.info("%s dataset empty; no samples to display.", name)
        return
    limit = min(limit, len(dataset))
    for idx in range(limit):
        item = dataset.items[idx]
        logging.info(
            "%s sample %d | video_id=%s | video_path=%s | scenes=%d | texts=%s | windows=%s",
            name,
            idx,
            item.video_id,
            item.video_path,
            len(item.scene_texts),
            item.scene_texts,
            item.scene_windows,
        )


def log_msrvtt_dataset_samples(name: str, dataset: MSRVTTUntrimmedDataset, limit: int = 3) -> None:
    if len(dataset) == 0:
        logging.info("%s dataset empty; no samples to display.", name)
        return
    limit = min(limit, len(dataset.items))
    for idx in range(limit):
        item = dataset.items[idx]
        logging.info(
            "%s sample %d | video_id=%s | scenes=%d | texts=%s | windows=%s | vision_feat=%s | text_feat=%s",
            name,
            idx,
            item.video_id,
            len(item.scene_texts),
            item.scene_texts,
            item.scene_windows,
            item.vision_path,
            item.text_path,
        )


def log_activitynet_dataset_samples(name: str, dataset: ActivityNetSceneDataset, limit: int = 3) -> None:
    if len(dataset) == 0:
        logging.info("%s dataset empty; no samples to display.", name)
        return
    limit = min(limit, len(dataset.items))
    for idx in range(limit):
        item = dataset.items[idx]
        logging.info(
            "%s sample %d | video_id=%s | scenes=%d | duration=%.2f | fps=%.3f | windows_idx=%s",
            name,
            idx,
            item.video_id,
            len(item.scene_texts),
            item.duration,
            item.fps,
            item.scene_windows,
        )


def log_batch_schema(name: str, batch: Dict[str, torch.Tensor]) -> None:
    entries: List[str] = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            entries.append(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, list):
            entries.append(f"{key}: list(len={len(value)})")
        else:
            entries.append(f"{key}: type={type(value).__name__}")
    logging.info("%s batch schema -> %s", name, "; ".join(entries))


def preprocess_frame(frame: np.ndarray, frame_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - V_MEAN) / V_STD
    return np.transpose(normalized, (2, 0, 1))


def sample_video_frames(
    video_path: Path,
    *,
    sample_fps: float,
    frame_size: int,
) -> Tuple[List[np.ndarray], List[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or math.isnan(native_fps) or native_fps <= 0:
        native_fps = 30.0
    period = 1.0 / max(sample_fps, 1e-3)
    next_time = 0.0
    sampled_frames: List[np.ndarray] = []
    timestamps: List[float] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / native_fps
        if timestamp + (0.5 / native_fps) >= next_time:
            sampled_frames.append(preprocess_frame(frame, frame_size))
            timestamps.append(timestamp)
            next_time += period
        frame_idx += 1
    cap.release()
    return sampled_frames, timestamps


def build_video_clips(
    frames: List[np.ndarray],
    timestamps: List[float],
    *,
    frames_per_clip: int,
    clip_stride: int,
    frame_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(frames) < frames_per_clip:
        return (
            torch.empty(0, frames_per_clip, 3, frame_size, frame_size),
            torch.empty(0),
        )
    clip_arrays: List[np.ndarray] = []
    clip_times: List[float] = []
    max_start = len(frames) - frames_per_clip + 1
    for start in range(0, max_start, clip_stride):
        clip = np.stack(frames[start : start + frames_per_clip], axis=0)
        time_window = timestamps[start : start + frames_per_clip]
        clip_arrays.append(clip)
        clip_times.append(float(np.mean(time_window)))
    clip_tensor = torch.from_numpy(np.stack(clip_arrays, axis=0)).float()
    clip_times_tensor = torch.tensor(clip_times, dtype=torch.float32)
    return clip_tensor, clip_times_tensor


def _truncate_text(text: str, max_len: int = 48) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _tick_indices(length: int, max_labels: int = 12) -> List[int]:
    if length <= 0:
        return []
    if length <= max_labels:
        return list(range(length))
    step = max(1, math.ceil(length / max_labels))
    return list(range(0, length, step))


def log_scene_text_similarity_matrix(
    video_id: str,
    similarity: np.ndarray,
    scene_texts: Sequence[str],
) -> None:
    if similarity.size == 0:
        return
    scene_count, query_count = similarity.shape
    logging.info(
        "Sim matrix for video %s (scenes=%d, queries=%d)",
        video_id,
        scene_count,
        query_count,
    )
    if scene_count > 0:
        logging.info("Scene indices and texts:")
        for idx in range(scene_count):
            text = scene_texts[idx] if idx < len(scene_texts) else ""
            text = text if text else "(no reference text)"
            logging.info("  Scene %d: %s", idx + 1, _truncate_text(text, 96))
    if query_count > 0:
        logging.info("Query indices and texts:")
        for idx in range(query_count):
            text = scene_texts[idx] if idx < len(scene_texts) else ""
            text = text if text else "(no reference text)"
            logging.info("  Query %d: %s", idx, _truncate_text(text, 96))
    matrix_str = np.array2string(
        similarity,
        precision=3,
        suppress_small=True,
        max_line_width=120,
    )
    logging.info("Scene-text cosine similarities:\n%s", matrix_str)


def save_similarity_heatmap(
    video_id: str,
    similarity: np.ndarray,
    query_texts: Sequence[str],
    output_dir: Optional[Path],
) -> Optional[Path]:
    if output_dir is None or similarity.size == 0:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_count, query_count = similarity.shape
    fig_width = max(6.0, query_count * 0.5)
    fig_height = max(4.0, scene_count * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    img = ax.imshow(
        similarity,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_xlabel("Query index")
    ax.set_ylabel("Scene index")
    xticks = _tick_indices(query_count, max_labels=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [_truncate_text(query_texts[idx], 32) if idx < len(query_texts) else f"Q{idx}" for idx in xticks],
        rotation=45,
        ha="right",
    )
    yticks = list(range(scene_count))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(idx + 1) for idx in yticks])
    fig.colorbar(img, ax=ax, shrink=0.8, label="Cosine similarity")
    ax.set_title(f"Scene-Text Cosine Similarity for {video_id}")
    heatmap_path = output_dir / f"sim_matrix_{video_id}.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)
    return heatmap_path


def log_text_similarity_matrix(
    video_id: str,
    similarity: np.ndarray,
    texts: Sequence[str],
) -> None:
    if similarity.size == 0:
        return
    count = similarity.shape[0]
    logging.info("Text-text sim matrix for video %s (queries=%d)", video_id, count)
    for idx in range(count):
        label = texts[idx] if idx < len(texts) else ""
        logging.info("  Query %d: %s", idx, _truncate_text(label or "(empty)", 96))
    matrix_str = np.array2string(
        similarity,
        precision=3,
        suppress_small=True,
        max_line_width=120,
    )
    logging.info("Text cosine similarities:\n%s", matrix_str)


def save_text_similarity_heatmap(
    video_id: str,
    similarity: np.ndarray,
    texts: Sequence[str],
    output_dir: Optional[Path],
) -> Optional[Path]:
    if output_dir is None or similarity.size == 0:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    count = similarity.shape[0]
    fig_size = max(4.0, count * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    img = ax.imshow(
        similarity,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
        cmap="viridis",
    )
    ticks = list(range(count))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    labels = [_truncate_text(texts[idx], 20) if idx < len(texts) else f"Q{idx}" for idx in ticks]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Query index")
    ax.set_ylabel("Query index")
    ax.set_title(f"Text Cosine Similarity for {video_id}")
    fig.colorbar(img, ax=ax, shrink=0.8, label="Cosine similarity")
    heatmap_path = output_dir / f"text_sim_{video_id}.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)
    return heatmap_path


def compute_teacher_forcing_cosines(
    preds: torch.Tensor,
    targets: torch.Tensor,
    padding_mask: torch.Tensor,
) -> Optional[np.ndarray]:
    if preds.numel() == 0 or targets.numel() == 0:
        return None
    valid_mask = ~padding_mask
    if not torch.any(valid_mask):
        return None
    valid_preds = preds[valid_mask]
    valid_targets = targets[valid_mask]
    pred_norm = F.normalize(valid_preds, dim=-1)
    target_norm = F.normalize(valid_targets, dim=-1)
    cosines = torch.sum(pred_norm * target_norm, dim=-1)
    return cosines.detach().cpu().numpy()


def log_teacher_forcing_cosines(values: np.ndarray, label: str) -> None:
    if values.size == 0:
        return
    mean = float(values.mean())
    std = float(values.std())
    vmin = float(values.min())
    vmax = float(values.max())
    logging.info(
        "Teacher forcing cosine stats [%s]: count=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
        label,
        values.size,
        mean,
        std,
        vmin,
        vmax,
    )
    preview_count = min(32, values.size)
    preview = np.array2string(
        values[:preview_count], precision=3, suppress_small=True, max_line_width=120
    )
    logging.info("Teacher forcing cosine samples [%s] (first %d): %s", label, preview_count, preview)
    hist_counts, hist_bins = np.histogram(values, bins=20, range=(-1.0, 1.0))
    logging.info(
        "Teacher forcing cosine histogram [%s]: bins=%s counts=%s",
        label,
        np.array2string(hist_bins, precision=2, suppress_small=True),
        hist_counts.tolist(),
    )


def summarize_model_parameters(model: nn.Module, label: str) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        "%s parameter counts -> total=%d (%.2fM) trainable=%d (%.2fM)",
        label,
        total,
        total / 1e6,
        trainable,
        trainable / 1e6,
    )
    return total, trainable


def save_loss_plot(
    train_history: Sequence[Tuple[int, float]],
    val_history: Sequence[Tuple[int, float]],
    output_path: Optional[Path],
) -> Optional[Path]:
    if output_path is None or (not train_history and not val_history):
        return None
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if train_history:
        train_steps = [step for step, _ in train_history]
        train_losses = [loss for _, loss in train_history]
        ax.plot(train_steps, train_losses, label="Train loss", color="#1f77b4")
    if val_history:
        val_steps = [step for step, _ in val_history]
        val_losses = [loss for _, loss in val_history]
        ax.plot(val_steps, val_losses, label="Validation loss", color="#d62728", marker="o", linestyle="--")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if train_history or val_history:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved loss plot -> %s", output_path)
    return output_path


def trim_eos_scene_predictions(
    scene_latents: torch.Tensor,
    attention_weights: Optional[torch.Tensor],
    eos_token: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if scene_latents.numel() == 0:
        return scene_latents, attention_weights
    eos_vector = eos_token.detach().to(scene_latents.device)
    last_latent = scene_latents[-1:].clone()
    eos_sim = F.cosine_similarity(last_latent, eos_vector.unsqueeze(0), dim=-1).item()
    if eos_sim >= threshold and scene_latents.shape[0] > 0:
        scene_latents = scene_latents[:-1]
        if attention_weights is not None and attention_weights.numel() > 0:
            attention_weights = attention_weights[:-1]
    return scene_latents, attention_weights


def estimate_total_frames(num_clips: int, frames_per_clip: int, clip_stride: int) -> int:
    if num_clips <= 0:
        return 0
    frames_per_clip = max(1, int(frames_per_clip))
    clip_stride = max(1, int(clip_stride))
    if num_clips == 1:
        return frames_per_clip
    return clip_stride * (num_clips - 1) + frames_per_clip


def compute_frame_attention(
    attention_weights: Optional[np.ndarray],
    *,
    frames_per_clip: int,
    clip_stride: int,
    total_frames: int,
) -> Optional[np.ndarray]:
    if attention_weights is None or attention_weights.size == 0 or total_frames <= 0:
        return None
    scene_count, clip_count = attention_weights.shape
    if clip_count == 0 or scene_count == 0:
        return None
    frames_per_clip = max(1, int(frames_per_clip))
    clip_stride = max(1, int(clip_stride))
    frame_attn = np.zeros((scene_count, total_frames), dtype=np.float32)
    coverage = np.zeros(total_frames, dtype=np.float32)
    clip_ranges: List[Tuple[int, int]] = []
    for clip_idx in range(clip_count):
        start = clip_idx * clip_stride
        if start >= total_frames:
            break
        end = min(start + frames_per_clip, total_frames)
        if end <= start:
            continue
        clip_ranges.append((start, end))
        coverage[start:end] += 1.0
    if not clip_ranges:
        return None
    for clip_idx, (start, end) in enumerate(clip_ranges):
        weight = attention_weights[:, clip_idx : clip_idx + 1]
        frame_attn[:, start:end] += weight
    valid = coverage > 0
    if not np.any(valid):
        return None
    frame_attn[:, valid] /= coverage[valid]
    frame_attn[:, ~valid] = 0.0
    return frame_attn


def retrieve_clip_preview_frames(
    video_path: Path,
    clip_times: Sequence[float],
    *,
    frame_size: int,
) -> List[np.ndarray]:
    if not clip_times:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for visualization: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    frames: List[np.ndarray] = []
    settle = max(0.0, 0.25)
    for time_sec in clip_times:
        seek_time = max(0.0, float(time_sec) - settle)
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_time * 1000.0)
        ok, frame = cap.read()
        if not ok:
            frames.append(np.zeros((frame_size, frame_size, 3), dtype=np.uint8))
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    cap.release()
    return frames


def create_inference_visualization(
    video_id: str,
    clip_times: Sequence[float],
    clip_frames: Sequence[np.ndarray],
    attention_weights: Optional[np.ndarray],
    frame_attention_weights: Optional[np.ndarray],
    scene_texts: Sequence[str],
    text_similarities: Optional[np.ndarray],
    output_dir: Path,
    *,
    max_scene_previews: int = 6,
) -> Optional[Path]:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_count = len(clip_times)
    if attention_weights is not None:
        scene_count = attention_weights.shape[0]
    elif frame_attention_weights is not None:
        scene_count = frame_attention_weights.shape[0]
    else:
        scene_count = 0
    frame_count = (
        frame_attention_weights.shape[1]
        if frame_attention_weights is not None
        else clip_count
    )
    fig_width = max(10.0, frame_count * 0.35)
    fig_height = 6.0 + max(0, scene_count - 1) * 0.3
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.6, 1.0], hspace=0.4)

    ax_attn = fig.add_subplot(grid[0])
    heatmap_source = (
        frame_attention_weights if frame_attention_weights is not None else attention_weights
    )
    if heatmap_source is not None and heatmap_source.size > 0:
        attn_img = ax_attn.imshow(
            heatmap_source,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )
        ax_attn.set_ylabel("Scene step")
        if frame_attention_weights is not None:
            ax_attn.set_xlabel("Frame index")
            xticks = _tick_indices(frame_count)
            ax_attn.set_xticks(xticks)
            ax_attn.set_xticklabels([str(idx) for idx in xticks], rotation=45, ha="right")
        else:
            ax_attn.set_xlabel("Clip timeline (s)")
            xticks = _tick_indices(clip_count)
            ax_attn.set_xticks(xticks)
            ax_attn.set_xticklabels(
                [f"{clip_times[i]:.1f}" for i in xticks],
                rotation=45,
                ha="right",
            )
        yticks = list(range(scene_count))
        ax_attn.set_yticks(yticks)
        ax_attn.set_yticklabels([f"Scene {idx + 1}" for idx in yticks])
        fig.colorbar(attn_img, ax=ax_attn, shrink=0.75, label="Attention")
    else:
        ax_attn.text(0.5, 0.5, "No attention predicted", ha="center", va="center")
        ax_attn.set_axis_off()

    scene_preview_count = min(max_scene_previews, scene_count)
    if scene_preview_count > 0 and clip_frames and attention_weights is not None:
        subgrid = grid[1].subgridspec(1, scene_preview_count, wspace=0.05)
        best_indices = attention_weights.argmax(axis=1)
        for idx in range(scene_preview_count):
            ax_img = fig.add_subplot(subgrid[0, idx])
            clip_idx = int(best_indices[idx]) if idx < best_indices.shape[0] else 0
            time_label = clip_times[clip_idx] if 0 <= clip_idx < len(clip_times) else 0.0
            if 0 <= clip_idx < len(clip_frames):
                ax_img.imshow(clip_frames[clip_idx])
            ax_img.axis("off")
            caption = f"Scene {idx + 1}\n@ {time_label:.1f}s"
            ax_img.set_title(caption, fontsize=9)
    else:
        ax_placeholder = fig.add_subplot(grid[1])
        ax_placeholder.text(0.5, 0.5, "Scene previews unavailable", ha="center", va="center")
        ax_placeholder.set_axis_off()

    ax_sim = fig.add_subplot(grid[2])
    if text_similarities is not None and text_similarities.size > 0:
        sim_img = ax_sim.imshow(
            text_similarities,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            vmin=-1.0,
            vmax=1.0,
        )
        ax_sim.set_ylabel("Scene step")
        ax_sim.set_xlabel("Query")
        xticks = _tick_indices(len(scene_texts), max_labels=8)
        ax_sim.set_xticks(xticks)
        ax_sim.set_xticklabels([_truncate_text(scene_texts[i], 32) for i in xticks], rotation=45, ha="right")
        yticks = list(range(scene_count))
        ax_sim.set_yticks(yticks)
        ax_sim.set_yticklabels([f"Scene {idx + 1}" for idx in yticks])
        fig.colorbar(sim_img, ax=ax_sim, shrink=0.75, label="Text similarity")
    else:
        ax_sim.text(0.5, 0.5, "Query similarities unavailable", ha="center", va="center")
        ax_sim.set_axis_off()

    fig.suptitle(f"{video_id} | Scene attention & query alignment", fontsize=14)
    output_file = output_dir / f"{video_id}_viz.png"
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    return output_file


def ensure_syspath(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"InternVideo2 multi_modality root not found: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def load_internvideo2_config(
    multi_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    *,
    num_frames: int,
    frame_size: int,
    origin_num_frames: Optional[int],
):
    import importlib

    ensure_syspath(multi_root)
    demo_config = importlib.import_module("demo_config")
    cfg = demo_config.Config.from_file(str(config_path.expanduser().resolve()))
    cfg = demo_config.eval_dict_leaf(cfg)
    cfg.device = str(device)
    cfg.pretrained_path = str(checkpoint_path.expanduser().resolve())
    if not os.path.isfile(cfg.pretrained_path):
        raise FileNotFoundError(f"InternVideo2 checkpoint not found: {cfg.pretrained_path}")

    cfg.model.vision_encoder.num_frames = num_frames
    cfg.model.vision_encoder.clip_input_resolution = frame_size
    cfg.num_frames = num_frames
    cfg.num_frames_test = num_frames
    cfg.img_size = frame_size
    cfg.input_size = frame_size
    if origin_num_frames is not None:
        cfg.origin_num_frames = origin_num_frames
    return cfg


def setup_internvideo2_model(cfg):
    from demo.utils import setup_internvideo2

    model, tokenizer = setup_internvideo2(cfg)
    model.eval()
    return model, tokenizer


class InternVideo2VideoBackbone:
    """Frozen wrapper around InternVideo2 stage2 vision encoder."""

    def __init__(self, model, *, clip_batch_size: int):
        self.model = model
        self.device = torch.device(model.config.device)
        self.clip_batch_size = clip_batch_size
        self.dtype = model.dtype

    def encode_clips(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clips: Tensor with shape (num_clips, frames, C, H, W) normalized to
                   the expected statistics.
        Returns:
            clip embeddings of shape (num_clips, D).
        """
        if clips.numel() == 0:
            return torch.empty(0, self.model.embed_dim, device=self.device)
        outputs: List[torch.Tensor] = []
        clips = clips.to(device=self.device, dtype=self.dtype, non_blocking=True)
        for start in range(0, clips.shape[0], self.clip_batch_size):
            chunk = clips[start : start + self.clip_batch_size]
            with torch.no_grad():
                feats = self.model.get_vid_feat(chunk)
            outputs.append(feats.squeeze(1))
        return torch.cat(outputs, dim=0)


class InternVideo2TextBackbone:
    """Frozen wrapper around the InternVideo2 text encoder."""

    def __init__(self, model):
        self.model = model
        self.device = torch.device(model.config.device)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for text in texts:
            with torch.no_grad():
                feat = self.model.get_txt_feat(text)
            feats.append(feat.squeeze(0))
        if not feats:
            return torch.empty(0, self.model.embed_dim, device=self.device)
        return torch.stack(feats, dim=0).to(self.device)


def sinusoidal_position_encoding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if positions.dtype != torch.float32 and positions.dtype != torch.float64:
        positions = positions.float()
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=positions.dtype, device=positions.device)
        * -(math.log(10000.0) / dim)
    )
    angles = positions.unsqueeze(-1) * div_term.unsqueeze(0)
    pe = torch.zeros(positions.shape[0], dim, device=positions.device, dtype=positions.dtype)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class SceneDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, _ = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = self.norm1(tgt + self.dropout(attn_output))

        cross_output, attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = self.norm2(tgt + self.dropout(cross_output))

        ff = self.linear2(F.gelu(self.linear1(tgt)))
        tgt = self.norm3(tgt + self.dropout(ff))
        return tgt, attn_weights


class SceneTransformer(nn.Module):
    """Autoregressive latent scene generator."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.clip_proj = nn.Linear(embed_dim, embed_dim)
        self.target_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.clip_norm = nn.LayerNorm(embed_dim)
        self.target_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            SceneDecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        )
        self.start_token = nn.Parameter(torch.randn(embed_dim))
        self.eos_token = nn.Parameter(torch.randn(embed_dim))

    def project_text_embeddings(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.text_proj(text_embeddings)

    def build_teacher_forcing_inputs(
        self,
        latents: List[torch.Tensor],
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        batch_size = len(latents)
        if batch_size == 0:
            raise ValueError("Cannot build batch inputs from an empty list.")
        scene_lengths = [latent.shape[0] for latent in latents]
        start_token = self.start_token.unsqueeze(0).to(device)
        eos_token = self.eos_token.unsqueeze(0).to(device)
        decoder_input_sequences: List[torch.Tensor] = []
        decoder_target_sequences: List[torch.Tensor] = []
        for latent in latents:
            if latent.ndim != 2 or latent.shape[1] != self.embed_dim:
                raise ValueError("Latent tensors must have shape (num_scenes, embed_dim).")
            if latent.device != device:
                latent = latent.to(device)
            inp_seq = torch.cat([start_token, latent], dim=0)
            tgt_seq = torch.cat([latent, eos_token], dim=0)
            decoder_input_sequences.append(inp_seq)
            decoder_target_sequences.append(tgt_seq)

        decoder_inputs = pad_sequence(decoder_input_sequences, batch_first=True)
        decoder_targets = pad_sequence(decoder_target_sequences, batch_first=True)
        max_len = decoder_inputs.shape[1]
        target_padding = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        scene_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        for idx, length in enumerate(scene_lengths):
            valid_tokens = length + 1  # scenes plus EOS
            target_padding[idx, :valid_tokens] = False
            scene_mask[idx, :length] = True
        return decoder_inputs, decoder_targets, target_padding, scene_mask, scene_lengths

    def _prepare_memory(
        self,
        clip_embeddings: torch.Tensor,
        clip_times: torch.Tensor,
        clip_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = clip_embeddings.shape
        # Use timestamps for positional encoding; padded positions receive zeros.
        pos_list = []
        for i in range(batch):
            valid = ~clip_padding_mask[i]
            times = clip_times[i]
            pos = torch.zeros(seq_len, self.embed_dim, device=clip_embeddings.device, dtype=clip_embeddings.dtype)
            if valid.any():
                encoding = sinusoidal_position_encoding(times[valid], self.embed_dim)
                pos[valid] = encoding
            pos_list.append(pos)
        pos_tensor = torch.stack(pos_list, dim=0)
        memory = self.clip_proj(clip_embeddings) + pos_tensor
        memory = self.clip_norm(self.dropout(memory))
        return memory

    def _prepare_targets(
        self,
        decoder_inputs: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = decoder_inputs.shape
        pos = torch.arange(seq_len, device=decoder_inputs.device, dtype=torch.float32)
        encoding = sinusoidal_position_encoding(pos, self.embed_dim)
        tgt = self.target_proj(decoder_inputs) + encoding.unsqueeze(0)
        tgt = self.target_norm(self.dropout(tgt))
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_inputs.device, dtype=torch.bool), diagonal=1
        )
        return tgt, tgt_mask

    def forward(
        self,
        clip_embeddings: torch.Tensor,
        clip_times: torch.Tensor,
        clip_padding_mask: torch.Tensor,
        decoder_inputs: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self._prepare_memory(clip_embeddings, clip_times, clip_padding_mask)
        tgt, tgt_mask = self._prepare_targets(decoder_inputs, decoder_padding_mask)
        attn_weights: Optional[torch.Tensor] = None
        x = tgt
        for layer in self.layers:
            x, attn = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=clip_padding_mask,
            )
            attn_weights = attn
        outputs = self.output_proj(self.output_norm(x))
        if attn_weights is not None:
            # Average over heads -> (batch, target_len, clip_len)
            attn_weights = attn_weights.mean(dim=1)
        return outputs, attn_weights

    @torch.no_grad()
    def generate(
        self,
        clip_embeddings: torch.Tensor,
        clip_times: torch.Tensor,
        clip_padding_mask: torch.Tensor,
        *,
        max_steps: int,
        eos_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = clip_embeddings.device
        memory = self._prepare_memory(clip_embeddings, clip_times, clip_padding_mask)
        generated = []
        attn_history: List[torch.Tensor] = []
        batch_size = clip_embeddings.shape[0]
        inputs = self.start_token.view(1, 1, -1).to(device).repeat(batch_size, 1, 1)
        padding_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        for _ in range(max_steps):
            tgt, tgt_mask = self._prepare_targets(inputs, padding_mask)
            x = tgt
            attn = None
            for layer in self.layers:
                x, attn = layer(
                    x,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=padding_mask,
                    memory_key_padding_mask=clip_padding_mask,
                )
            pred = self.output_proj(self.output_norm(x[:, -1:, :]))
            pred_latent = pred.squeeze(1)
            generated.append(pred_latent)
            feedback_token = self.project_text_embeddings(pred_latent).unsqueeze(1)
            if attn is not None:
                attn_mean = attn.mean(dim=1)[:, -1, :]
                if clip_padding_mask is not None:
                    attn_mean = attn_mean.masked_fill(clip_padding_mask, 0.0)
                attn_sum = attn_mean.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                attn_history.append(attn_mean / attn_sum)
            else:
                attn_history.append(
                    torch.zeros(
                        clip_embeddings.shape[0],
                        clip_embeddings.shape[1],
                        device=device,
                    )
                )
            inputs = torch.cat([inputs, feedback_token], dim=1)
            pad_row = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([padding_mask, pad_row], dim=1)
            eos_similarity = F.cosine_similarity(
                pred_latent, self.eos_token.view(1, -1).to(device), dim=-1
            )
            if eos_similarity.numel() == 1:
                eos_value = eos_similarity.item()
                if eos_value >= eos_threshold:
                    break
            else:
                if torch.all(eos_similarity >= eos_threshold):
                    break
        if generated:
            return torch.stack(generated, dim=1), torch.stack(attn_history, dim=1)
        empty_embeds = torch.empty(clip_embeddings.shape[0], 0, self.embed_dim, device=device)
        empty_attn = torch.empty(clip_embeddings.shape[0], 0, clip_embeddings.shape[1], device=device)
        return empty_embeds, empty_attn


def representation_alignment_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    keep = ~valid_mask
    if keep.sum() == 0:
        return preds.new_tensor(0.0)
    valid_preds = preds[keep]
    valid_targets = targets[keep]
    if mode == "mse":
        diff = (valid_preds - valid_targets) ** 2
        return diff.sum(dim=-1).mean()
    pred_norm = F.normalize(valid_preds, dim=-1)
    target_norm = F.normalize(valid_targets, dim=-1)
    logits = pred_norm @ target_norm.T / temperature
    labels = torch.arange(logits.shape[0], device=preds.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-6)
    q = q.clamp_min(1e-6)
    return (p * (p.log() - q.log())).sum()


def attention_supervision_loss(
    attn_weights: torch.Tensor,
    clip_times: torch.Tensor,
    clip_padding_mask: torch.Tensor,
    scene_windows: List[List[Tuple[float, float]]],
    scene_lengths: List[int],
) -> torch.Tensor:
    if attn_weights is None:
        return clip_times.new_tensor(0.0)
    total = attn_weights.new_tensor(0.0)
    count = 0
    for b in range(attn_weights.shape[0]):
        clip_mask = ~clip_padding_mask[b]
        if not clip_mask.any():
            continue
        times = clip_times[b, clip_mask]
        for j in range(scene_lengths[b]):
            if j >= attn_weights.shape[1]:
                continue
            window = scene_windows[b][j]
            start, end = float(window[0]), float(window[1])
            gt_mask = (times >= start) & (times <= end)
            if not gt_mask.any():
                continue
            gt = gt_mask.float()
            gt /= gt.sum()
            pred = attn_weights[b, j, clip_mask]
            pred /= pred.sum()
            total = total + kl_divergence(gt, pred)
            count += 1
    if count == 0:
        return total
    return total / count


def monotonicity_loss(
    attn_weights: torch.Tensor,
    clip_padding_mask: torch.Tensor,
    scene_lengths: List[int],
) -> torch.Tensor:
    if attn_weights is None:
        return clip_padding_mask.new_tensor(0.0)
    positions = torch.arange(attn_weights.shape[-1], device=attn_weights.device).float()
    total = attn_weights.new_tensor(0.0)
    count = 0
    for b in range(attn_weights.shape[0]):
        clip_mask = ~clip_padding_mask[b]
        if not clip_mask.any():
            continue
        pos = positions[clip_mask]
        prev_expectation = None
        for j in range(scene_lengths[b]):
            if j >= attn_weights.shape[1]:
                break
            attn = attn_weights[b, j, clip_mask]
            attn = attn / attn.sum()
            expectation = (attn * pos).sum()
            if prev_expectation is not None:
                total = total + F.relu(prev_expectation - expectation)
                count += 1
            prev_expectation = expectation
    if count == 0:
        return total
    return total / count


def coverage_loss(attn_weights: torch.Tensor, clip_padding_mask: torch.Tensor) -> torch.Tensor:
    if attn_weights is None:
        return clip_padding_mask.new_tensor(0.0)
    valid_mask = ~clip_padding_mask
    if not valid_mask.any():
        return clip_padding_mask.new_tensor(0.0)
    coverage = attn_weights.sum(dim=1)
    diff = (coverage - 1.0) ** 2
    return diff[valid_mask].mean()


def run_validation(
    data_loader: DataLoader,
    *,
    video_backbone: Optional[InternVideo2VideoBackbone],
    text_backbone: Optional[InternVideo2TextBackbone],
    scene_model: SceneTransformer,
    args: argparse.Namespace,
    device: torch.device,
    dataset_type: str,
    schema_label: str = "Validation",
    log_schema: bool = False,
) -> Optional[Dict[str, float]]:
    """Compute validation losses without gradient tracking."""

    if data_loader is None:
        return None

    was_training = scene_model.training
    scene_model.eval()
    metrics = {"loss": 0.0, "repr": 0.0, "attn": 0.0, "mono": 0.0, "cov": 0.0}
    batches = 0

    schema_logged = not log_schema
    cosine_values: List[np.ndarray] = []

    with torch.no_grad():
        for batch_samples in data_loader:
            batch = collate_scene_batch(
                batch_samples,
                video_backbone=video_backbone,
                text_backbone=text_backbone,
                scene_model=scene_model,
                args=args,
                device=device,
                dataset_type=dataset_type,
            )
            if batch is None:
                continue

            if not schema_logged:
                log_batch_schema(schema_label, batch)
                schema_logged = True

            preds, attn = scene_model(
                batch["clip_embeddings"],
                batch["clip_times"],
                batch["clip_padding_mask"],
                batch["decoder_inputs"],
                batch["decoder_padding_mask"],
            )
            cosines = compute_teacher_forcing_cosines(
                preds,
                batch["decoder_targets"],
                batch["decoder_padding_mask"],
            )
            if cosines is not None and cosines.size > 0:
                cosine_values.append(cosines)
            rep_loss = representation_alignment_loss(
                preds,
                batch["decoder_targets"],
                batch["decoder_padding_mask"],
                mode=args.alignment_loss,
                temperature=args.infonce_temp,
            )
            attn_loss = attention_supervision_loss(
                attn,
                batch["clip_times"],
                batch["clip_padding_mask"],
                batch["scene_windows"],
                batch["scene_lengths"],
            )
            mono_loss = monotonicity_loss(attn, batch["clip_padding_mask"], batch["scene_lengths"])
            cov_loss = rep_loss.new_tensor(0.0)
            total_loss = rep_loss
            total_loss = total_loss + args.lambda_attn * attn_loss
            total_loss = total_loss + args.lambda_mono * mono_loss
            # coverage loss disabled

            metrics["loss"] += float(total_loss.item())
            metrics["repr"] += float(rep_loss.item())
            metrics["attn"] += float(attn_loss.item())
            metrics["mono"] += float(mono_loss.item())
            metrics["cov"] += float(cov_loss.item())
            batches += 1

    if was_training:
        scene_model.train()

    if batches == 0:
        logging.warning("Validation skipped: dataset produced no valid batches.")
        return None

    for key in metrics:
        metrics[key] /= batches

    if cosine_values:
        tf_values = np.concatenate(cosine_values)
        log_teacher_forcing_cosines(tf_values, schema_label)
    return metrics


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    path = path.expanduser()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_scene_batch(
    samples: List[object],
    *,
    video_backbone: Optional[InternVideo2VideoBackbone],
    text_backbone: Optional[InternVideo2TextBackbone],
    scene_model: SceneTransformer,
    args,
    device: torch.device,
    dataset_type: str,
) -> Optional[Dict[str, torch.Tensor]]:
    clip_embeddings: List[torch.Tensor] = []
    clip_times: List[torch.Tensor] = []
    scene_latents: List[torch.Tensor] = []
    scene_windows: List[List[Tuple[float, float]]] = []
    for sample in samples:
        if dataset_type == "msrvtt_untrimmed":
            if not isinstance(sample, MSRVTTSceneItem):
                raise TypeError("MSRVTT dataset expected MSRVTTSceneItem samples.")
            clip_feats_cpu = sample.clip_embeddings
            clip_times_tensor = sample.clip_times
            text_embeddings = sample.text_embeddings
        elif dataset_type == "activitynet":
            if not isinstance(sample, ActivityNetSceneItem):
                raise TypeError("ActivityNet dataset expected ActivityNetSceneItem samples.")
            clip_feats_cpu = sample.clip_embeddings
            clip_times_tensor = sample.clip_times
            text_embeddings = sample.text_embeddings
        else:
            if video_backbone is None or text_backbone is None:
                raise RuntimeError("InternVideo2 backbones are required for QVHighlights mode.")
            clip_cache = load_video_cache(args.video_cache_root, sample.video_id)
            if (
                clip_cache is not None
                and clip_cache[0] is not None
                and clip_cache[1] is not None
            ):
                clip_feats_cpu, clip_times_tensor = clip_cache
            else:
                try:
                    frames, timestamps = sample_video_frames(
                        sample.video_path,
                        sample_fps=args.sample_fps,
                        frame_size=args.frame_size,
                    )
                except RuntimeError as exc:
                    logging.warning("Skipping %s: %s", sample.video_id, exc)
                    continue
                clip_tensor, clip_times_tensor = build_video_clips(
                    frames,
                    timestamps,
                    frames_per_clip=args.frames_per_clip,
                    clip_stride=args.clip_stride,
                    frame_size=args.frame_size,
                )
                if clip_tensor.numel() == 0:
                    logging.warning("Skipping %s: not enough frames for a single clip", sample.video_id)
                    continue
                clip_feats_cpu = video_backbone.encode_clips(clip_tensor).cpu()
                save_video_cache(
                    args.video_cache_root,
                    sample.video_id,
                    clip_feats_cpu,
                    clip_times_tensor,
                )
            text_embeddings = None
            if args.text_cache_root is not None:
                cached_texts = load_text_cache(args.text_cache_root, sample.video_id)
                if cached_texts is not None:
                    cached_list = cached_texts.get("texts")
                    cached_embeds = cached_texts.get("embeddings")
                    if cached_list == sample.scene_texts and cached_embeds is not None:
                        text_embeddings = cached_embeds
            if text_embeddings is None:
                text_embeddings = text_backbone.encode(sample.scene_texts)
                save_text_cache(
                    args.text_cache_root,
                    sample.video_id,
                    sample.scene_texts,
                    text_embeddings,
                )
        clip_feats = clip_feats_cpu.to(device)
        clip_times_device = clip_times_tensor.to(device)
        text_embeddings = text_embeddings.to(device)
        if text_embeddings.shape[0] == 0:
            logging.warning("Skipping %s: no scene texts", sample.video_id)
            continue
        scene_latent = scene_model.project_text_embeddings(text_embeddings)
        clip_embeddings.append(clip_feats)
        clip_times.append(clip_times_device)
        scene_latents.append(scene_latent)
        scene_windows.append(sample.scene_windows)

    if not clip_embeddings:
        return None

    batch_size = len(clip_embeddings)
    embed_dim = clip_embeddings[0].shape[-1]
    max_clips = max(t.shape[0] for t in clip_embeddings)
    clip_batch = torch.zeros(batch_size, max_clips, embed_dim, device=device)
    clip_padding = torch.ones(batch_size, max_clips, dtype=torch.bool, device=device)
    clip_time_batch = torch.zeros(batch_size, max_clips, device=device)
    for idx, (clips, times) in enumerate(zip(clip_embeddings, clip_times)):
        length = clips.shape[0]
        clip_batch[idx, :length, :] = clips
        clip_padding[idx, :length] = False
        clip_time_batch[idx, :length] = times[:length]

    decoder_inputs, decoder_targets, target_padding, scene_mask, scene_lengths = (
        scene_model.build_teacher_forcing_inputs(scene_latents, device=device)
    )
    return {
        "clip_embeddings": clip_batch,
        "clip_times": clip_time_batch,
        "clip_padding_mask": clip_padding,
        "decoder_inputs": decoder_inputs,
        "decoder_targets": decoder_targets,
        "decoder_padding_mask": target_padding,
        "scene_mask": scene_mask,
        "scene_windows": scene_windows,
        "scene_lengths": scene_lengths,
    }


@torch.no_grad()
def run_inference(
    args: argparse.Namespace,
    dataset: Dataset,
    video_backbone: Optional[InternVideo2VideoBackbone],
    scene_model: SceneTransformer,
    text_backbone: Optional[InternVideo2TextBackbone],
    device: torch.device,
    *,
    output_path: Optional[Path] = None,
    max_videos: Optional[int] = None,
    dataset_type: str,
) -> None:
    resolved_output = output_path or args.inference_output
    output_path = Path(resolved_output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    viz_dir: Optional[Path] = getattr(args, "inference_visualization_dir", None)
    logging.info(
        "Running inference on %d videos; results -> %s",
        len(dataset),
        output_path,
    )
    scene_model.eval()
    processed = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset, start=1):
            if max_videos is not None and processed >= max_videos:
                break
            if dataset_type in {"msrvtt_untrimmed", "activitynet"}:
                if dataset_type == "msrvtt_untrimmed":
                    expected = MSRVTTSceneItem
                else:
                    expected = ActivityNetSceneItem
                if not isinstance(sample, expected):
                    raise TypeError(
                        f"{dataset_type} inference expects {expected.__name__} samples."
                    )
                clip_feats_cpu = sample.clip_embeddings.cpu()
                clip_times_tensor = sample.clip_times.cpu()
                scene_texts = sample.scene_texts
                scene_windows = sample.scene_windows
                video_path = None
                text_embed_source = sample.text_embeddings
            else:
                if video_backbone is None:
                    raise RuntimeError("Video backbone required for QVHighlights inference.")
                clip_cache = load_video_cache(args.video_cache_root, sample.video_id)
                if (
                    clip_cache is not None
                    and clip_cache[0] is not None
                    and clip_cache[1] is not None
                ):
                    clip_feats_cpu, clip_times_tensor = clip_cache
                else:
                    try:
                        frames, timestamps = sample_video_frames(
                            sample.video_path,
                            sample_fps=args.sample_fps,
                            frame_size=args.frame_size,
                        )
                    except RuntimeError as exc:
                        logging.warning("Inference skip %s: %s", sample.video_id, exc)
                        continue
                    clip_tensor, clip_times_tensor = build_video_clips(
                        frames,
                        timestamps,
                        frames_per_clip=args.frames_per_clip,
                        clip_stride=args.clip_stride,
                        frame_size=args.frame_size,
                    )
                    if clip_tensor.numel() == 0:
                        logging.warning("Inference skip %s: insufficient frames.", sample.video_id)
                        continue
                    clip_feats_cpu = video_backbone.encode_clips(clip_tensor).cpu()
                    save_video_cache(
                        args.video_cache_root,
                        sample.video_id,
                        clip_feats_cpu,
                        clip_times_tensor,
                    )
                scene_texts = sample.scene_texts
                scene_windows = sample.scene_windows
                video_path = sample.video_path
                text_embed_source = None
            clip_feats = clip_feats_cpu.to(device).unsqueeze(0)
            clip_times = clip_times_tensor.to(device).unsqueeze(0)
            clip_padding = torch.zeros(1, clip_feats.shape[1], dtype=torch.bool, device=device)
            generations, attn_weights = scene_model.generate(
                clip_feats,
                clip_times,
                clip_padding,
                max_steps=args.max_generation_steps,
                eos_threshold=args.eos_threshold,
            )
            scene_latents = generations.squeeze(0)
            scene_latents_cpu = scene_latents.cpu()
            clip_count = clip_feats.shape[1]
            if attn_weights is not None and attn_weights.numel() > 0:
                attn_tensor_raw = attn_weights.squeeze(0).cpu()
            else:
                attn_tensor_raw = torch.empty(0, clip_count, dtype=torch.float32)
            eos_token_cpu = scene_model.eos_token.detach().to(scene_latents_cpu.device)
            trimmed_latents, trimmed_attn = trim_eos_scene_predictions(
                scene_latents_cpu,
                attn_tensor_raw if attn_tensor_raw.numel() > 0 else None,
                eos_token_cpu,
                args.eos_threshold,
            )
            scene_latents_cpu = trimmed_latents
            attn_tensor = (
                trimmed_attn
                if trimmed_attn is not None
                else torch.empty(0, clip_count, dtype=torch.float32)
            )
            embedding_list = scene_latents_cpu.tolist()
            attn_list = attn_tensor.tolist()
            clip_time_tensor = clip_times.squeeze(0).cpu()
            clip_time_list = clip_time_tensor.tolist()

            text_embeddings_export: List[List[float]] = []
            text_similarities: List[List[float]] = []
            text_text_similarities: List[List[float]] = []
            clip_similarities: List[List[float]] = []
            text_similarity_tensor: Optional[torch.Tensor] = None
            text_similarity_np: Optional[np.ndarray] = None
            sim_heatmap_path: Optional[Path] = None
            text_text_similarity_np: Optional[np.ndarray] = None
            text_text_heatmap_path: Optional[Path] = None
            text_embeds: Optional[torch.Tensor] = None
            if scene_texts:
                if dataset_type in {"msrvtt_untrimmed", "activitynet"}:
                    text_embeds = text_embed_source.to(device)
                else:
                    if text_backbone is None:
                        raise RuntimeError("Text backbone required for QVHighlights inference.")
                    text_embeds = text_backbone.encode(scene_texts).to(device)
            if text_embeds is not None and text_embeds.numel() > 0:
                text_embeddings_export = text_embeds.detach().cpu().tolist()
                text_norm = F.normalize(text_embeds, dim=-1)
                text_text_similarity_tensor = text_norm @ text_norm.T
                text_text_similarity_np = text_text_similarity_tensor.detach().cpu().numpy()
                text_text_similarities = text_text_similarity_tensor.detach().cpu().tolist()
                log_text_similarity_matrix(sample.video_id, text_text_similarity_np, scene_texts)
                text_text_heatmap_path = save_text_similarity_heatmap(
                    sample.video_id,
                    text_text_similarity_np,
                    scene_texts,
                    viz_dir,
                )
                if text_text_heatmap_path is not None:
                    logging.info("Saved text similarity heatmap -> %s", text_text_heatmap_path)
                text_proj = scene_model.project_text_embeddings(text_embeds)
                if scene_latents_cpu.numel() > 0 and text_proj.numel() > 0:
                    text_proj_cpu = text_proj.cpu()
                    scene_norm = F.normalize(scene_latents_cpu, dim=-1)
                    text_norm = F.normalize(text_proj_cpu, dim=-1)
                    text_similarity_tensor = scene_norm @ text_norm.T
                    text_similarities = text_similarity_tensor.tolist()
                    text_similarity_np = text_similarity_tensor.numpy()
                    log_scene_text_similarity_matrix(
                        sample.video_id,
                        text_similarity_np,
                        scene_texts,
                    )
                    sim_heatmap_path = save_similarity_heatmap(
                        sample.video_id,
                        text_similarity_np,
                        scene_texts,
                        viz_dir,
                    )
                    if sim_heatmap_path is not None:
                        logging.info("Saved similarity heatmap -> %s", sim_heatmap_path)

            if scene_latents_cpu.numel() > 0 and clip_feats_cpu.numel() > 0:
                scene_norm = F.normalize(scene_latents_cpu, dim=-1)
                clip_norm = F.normalize(clip_feats_cpu, dim=-1)
                clip_similarities = (scene_norm @ clip_norm.T).tolist()

            scene_best_queries: List[Dict[str, object]] = []
            query_alignment: List[Dict[str, object]] = []
            if text_similarity_tensor is not None and text_similarity_tensor.numel() > 0:
                if text_similarity_np is None:
                    text_similarity_np = text_similarity_tensor.numpy()
                scene_count, query_count = text_similarity_np.shape
                for scene_idx in range(scene_count):
                    if query_count == 0:
                        scene_best_queries.append(
                            {
                                "scene_index": scene_idx,
                                "best_query_index": None,
                                "best_similarity": None,
                            }
                        )
                        continue
                    best_query_idx = int(text_similarity_np[scene_idx].argmax())
                    best_score = float(text_similarity_np[scene_idx, best_query_idx])
                    best_query_text = (
                        scene_texts[best_query_idx]
                        if 0 <= best_query_idx < len(scene_texts)
                        else ""
                    )
                    scene_best_queries.append(
                        {
                            "scene_index": scene_idx,
                            "best_query_index": best_query_idx,
                            "best_similarity": best_score,
                            "best_query_text": best_query_text,
                        }
                    )
                    logging.info(
                        "Video %s scene %d best query -> %s (sim=%.3f)",
                        sample.video_id,
                        scene_idx + 1,
                        best_query_text,
                        best_score,
                    )
                for query_idx, query_text in enumerate(scene_texts):
                    column = text_similarity_np[:, query_idx] if scene_count > 0 else np.empty(0)
                    if column.size == 0:
                        query_alignment.append(
                            {
                                "query": query_text,
                                "best_scene_index": None,
                                "best_similarity": None,
                                "scene_similarities": [],
                            }
                        )
                        continue
                    best_scene_idx = int(column.argmax())
                    best_score = float(column[best_scene_idx])
                    logging.info(
                        "Video %s query[%d] '%s' best scene=%d (sim=%.3f)",
                        sample.video_id,
                        query_idx,
                        query_text,
                        best_scene_idx + 1,
                        best_score,
                    )
                    query_alignment.append(
                        {
                            "query": query_text,
                            "best_scene_index": best_scene_idx,
                            "best_similarity": best_score,
                            "scene_similarities": column.tolist(),
                        }
                    )

            visualization_path: Optional[Path] = None
            need_visualization = viz_dir is not None and (
                (attn_tensor.numel() > 0)
                or (text_similarity_np is not None and text_similarity_np.size > 0)
            )
            clip_frame_images: List[np.ndarray] = []
            if need_visualization and clip_time_list and video_path is not None:
                try:
                    clip_frame_images = retrieve_clip_preview_frames(
                        video_path,
                        clip_time_list,
                        frame_size=args.frame_size,
                    )
                except RuntimeError as exc:
                    logging.warning(
                        "Visualization frames unavailable for %s: %s",
                        sample.video_id,
                        exc,
                    )
            if need_visualization:
                attention_np = attn_tensor.numpy() if attn_tensor.numel() > 0 else None
                frame_attention_np: Optional[np.ndarray] = None
                if (
                    attention_np is not None
                    and attention_np.size > 0
                    and args.dataset == "qvhighlights"
                ):
                    total_frames = estimate_total_frames(
                        attention_np.shape[1],
                        args.frames_per_clip,
                        args.clip_stride,
                    )
                    frame_attention_np = compute_frame_attention(
                        attention_np,
                        frames_per_clip=args.frames_per_clip,
                        clip_stride=args.clip_stride,
                        total_frames=total_frames,
                    )
                visualization_path = create_inference_visualization(
                    sample.video_id,
                    clip_time_list,
                    clip_frame_images,
                    attention_np,
                    frame_attention_np,
                    scene_texts,
                    text_similarity_np,
                    viz_dir,
                    max_scene_previews=args.inference_visualization_max_scenes,
                )

            logging.info(
                "Inference sample %s | scenes=%d | queries=%s | viz=%s | sim_viz=%s | text_sim_viz=%s",
                sample.video_id,
                len(scene_texts),
                scene_texts,
                visualization_path,
                sim_heatmap_path,
                text_text_heatmap_path,
            )
            record = {
                "video_id": sample.video_id,
                "scene_texts": scene_texts,
                "scene_windows": scene_windows,
                "clip_times": clip_time_list,
                "scene_embeddings": embedding_list,
                "attention_weights": attn_list,
                "text_embeddings": text_embeddings_export,
                "scene_text_similarities": text_similarities,
                "text_text_similarities": text_text_similarities,
                "scene_clip_similarities": clip_similarities,
                "scene_best_queries": scene_best_queries,
                "query_alignment": query_alignment,
                "visualization_path": str(visualization_path) if visualization_path else None,
            }
            if hasattr(sample, "timestamps"):
                record["scene_timestamps_seconds"] = getattr(sample, "timestamps")
            if hasattr(sample, "duration"):
                record["video_duration_sec"] = float(getattr(sample, "duration"))
            if hasattr(sample, "fps"):
                record["feature_fps"] = float(getattr(sample, "fps"))
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
            processed += 1
            if idx % max(1, len(dataset) // 10) == 0:
                logging.info("Inference progress: %d/%d videos", idx, len(dataset))
    logging.info(
        "Inference results saved to %s (videos processed: %d)",
        output_path,
        processed,
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_intern_root = repo_root / "InternVideo"
    default_dataset_root = repo_root / "dataset" / "qvhighlights"
    parser = argparse.ArgumentParser(
        description="Train an autoregressive scene embedding model on QVHighlights or MSRVTT."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="qvhighlights",
        choices=("qvhighlights", "msrvtt_untrimmed", "activitynet"),
        help="Dataset to use for training/inference.",
    )
    parser.add_argument(
        "--internvideo-root",
        type=Path,
        default=default_intern_root,
        help="Path to InternVideo repository root.",
    )
    parser.add_argument(
        "--internvideo-config",
        type=Path,
        default=default_intern_root
        / "InternVideo2"
        / "multi_modality"
        / "demo"
        / "internvideo2_stage2_config.py",
        help="Path to InternVideo2 demo config used for loading stage2.",
    )
    parser.add_argument(
        "--internvideo-ckpt",
        type=Path,
        default=default_intern_root
        / "InternVideo2"
        / "ckpt"
        / "InternVideo2-stage2_1b-224p-f4.pt",
        help="Checkpoint file for InternVideo2 stage2.",
    )
    parser.add_argument(
        "--internvideo-origin-num-frames",
        type=int,
        default=None,
        help="Original number of frames used during pretraining (for positional interpolation).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
        help="QVHighlights dataset root directory.",
    )
    default_msrvtt_root = repo_root / "dataset" / "data" / "MSRVTT"
    parser.add_argument(
        "--msrvtt-annotation",
        type=Path,
        default=default_msrvtt_root / "annotation" / "MSRVTT_untrimmed.json",
        help="MSRVTT untrimmed annotation JSON.",
    )
    parser.add_argument(
        "--msrvtt-feat-root",
        type=Path,
        default=default_msrvtt_root / "internvideo_untrimmed_feats",
        help="Root containing precomputed InternVideo2 vision/text feats for MSRVTT.",
    )
    parser.add_argument(
        "--msrvtt-train-split",
        type=str,
        default="train",
        help="Split name used for MSRVTT training.",
    )
    parser.add_argument(
        "--msrvtt-val-split",
        type=str,
        default="val",
        help="Split name used for MSRVTT validation.",
    )
    parser.add_argument(
        "--msrvtt-inference-split",
        type=str,
        default="val",
        help="Split name used for MSRVTT inference/evaluation.",
    )
    default_activitynet_root = repo_root / "dataset" / "activitynet"
    parser.add_argument(
        "--activitynet-root",
        type=Path,
        default=default_activitynet_root,
        help="Root directory containing ActivityNet features and annotations.",
    )
    parser.add_argument(
        "--activitynet-video-features",
        type=Path,
        default=None,
        help="HDF5 file storing CLIP vision features for ActivityNet videos.",
    )
    parser.add_argument(
        "--activitynet-text-features",
        type=Path,
        default=None,
        help="HDF5 file storing CLIP text features for ActivityNet sentences.",
    )
    parser.add_argument(
        "--activitynet-train-json",
        type=Path,
        default=None,
        help="ActivityNet training annotation JSON file.",
    )
    parser.add_argument(
        "--activitynet-val-json",
        type=Path,
        default=None,
        help="ActivityNet validation annotation JSON file.",
    )
    parser.add_argument(
        "--activitynet-inference-json",
        type=Path,
        default=None,
        help="Annotation JSON used for ActivityNet inference (default: val JSON).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Directory containing raw 150s segments (default: dataset_root/raw).",
    )
    parser.add_argument(
        "--concat-root",
        type=Path,
        default=None,
        help="Directory containing concatenated untrimmed videos (default: dataset_root/concat).",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help="QVHighlights train JSONL (default: dataset_root/highlight_train_release.jsonl).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=("train", "inference"),
        help="Run training or inference-only pipeline.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", help="Device for training.")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--clip-stride", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--decoder-ff-dim", type=int, default=2048)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument(
        "--alignment-loss",
        type=str,
        choices=("mse", "infonce"),
        default="infonce",
    )
    parser.add_argument("--infonce-temp", type=float, default=0.07)
    parser.add_argument("--lambda-attn", type=float, default=1.0)
    parser.add_argument("--lambda-mono", type=float, default=0.1)
    parser.add_argument("--lambda-cov", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-generation-steps", type=int, default=12)
    parser.add_argument("--eos-threshold", type=float, default=0.8)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="File to save/load SceneTransformer weights.",
    )
    parser.add_argument(
        "--run-inference-after-train",
        action="store_true",
        help="After training completes, run inference on --inference-jsonl.",
    )
    parser.add_argument(
        "--validation-jsonl",
        type=Path,
        default=None,
        help="QVHighlights validation split (default: --inference-jsonl).",
    )
    parser.add_argument(
        "--inference-jsonl",
        type=Path,
        default=None,
        help="Annotations used for inference (default: highlight_val_release.jsonl).",
    )
    parser.add_argument(
        "--inference-output",
        type=Path,
        default=None,
        help="Where to save inference scene embeddings as JSONL.",
    )
    parser.add_argument(
        "--inference-visualization-dir",
        type=Path,
        default=None,
        help="Directory used to store inference visualizations (default: alongside the JSONL output).",
    )
    parser.add_argument(
        "--inference-visualization-max-scenes",
        type=int,
        default=6,
        help="Maximum number of scene frame previews shown per visualization.",
    )
    parser.add_argument(
        "--disable-inference-visualizations",
        action="store_true",
        help="Skip visualization rendering during inference.",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=0,
        help="Run inference every N global steps during training (0 disables).",
    )
    parser.add_argument(
        "--inference-limit",
        type=int,
        default=None,
        help="Maximum videos to process during periodic inference (default: all).",
    )
    parser.add_argument(
        "--max-sample-infer",
        type=int,
        default=None,
        help="Limit number of videos processed during post-training inference (default: all).",
    )
    parser.add_argument(
        "--video-cache-root",
        type=Path,
        default=None,
        help="Directory to cache InternVideo2 clip embeddings per video.",
    )
    parser.add_argument(
        "--text-cache-root",
        type=Path,
        default=None,
        help="Directory to cache InternVideo2 text embeddings.",
    )
    parser.add_argument(
        "--validation-interval",
        type=float,
        default=0.2,
        help="Fraction of an epoch after which to run validation (0 disables).",
    )
    parser.add_argument(
        "--loss-plot-path",
        type=Path,
        default=None,
        help="Path to save the training/validation loss curve plot.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        logging.warning("CUDA unavailable, using CPU for both backbones and transformer.")

    args.dataset = args.dataset.lower()
    if args.dataset == "qvhighlights":
        args.dataset_root = args.dataset_root.expanduser()
        raw_candidate = args.raw_root or (args.dataset_root / "raw")
        raw_candidate = raw_candidate.expanduser()
        raw_videos = raw_candidate / "videos"
        if raw_videos.is_dir():
            raw_candidate = raw_videos
        args.raw_root = raw_candidate
        args.concat_root = (args.concat_root or (args.dataset_root / "concat")).expanduser()
        args.train_jsonl = (args.train_jsonl or (args.dataset_root / "highlight_train_release.jsonl")).expanduser()
        args.inference_jsonl = (
            args.inference_jsonl or (args.dataset_root / "highlight_val_release.jsonl")
        ).expanduser()
        if args.validation_jsonl is None:
            args.validation_jsonl = args.inference_jsonl
        args.validation_jsonl = args.validation_jsonl.expanduser()
    elif args.dataset == "msrvtt_untrimmed":
        args.msrvtt_annotation = args.msrvtt_annotation.expanduser()
        args.msrvtt_feat_root = args.msrvtt_feat_root.expanduser()
    elif args.dataset == "activitynet":
        args.activitynet_root = args.activitynet_root.expanduser()
        if args.activitynet_video_features is None:
            args.activitynet_video_features = (
                args.activitynet_root
                / "FeatureData"
                / "new_clip_vit_32_activitynet_vid_features.hdf5"
            )
        if args.activitynet_text_features is None:
            args.activitynet_text_features = (
                args.activitynet_root
                / "TextData"
                / "clip_ViT_B_32_activitynet_query_feat.hdf5"
            )
        if args.activitynet_train_json is None:
            args.activitynet_train_json = args.activitynet_root / "TextData" / "train.json"
        if args.activitynet_val_json is None:
            args.activitynet_val_json = args.activitynet_root / "TextData" / "val_1.json"
        args.activitynet_video_features = args.activitynet_video_features.expanduser()
        args.activitynet_text_features = args.activitynet_text_features.expanduser()
        args.activitynet_train_json = args.activitynet_train_json.expanduser()
        args.activitynet_val_json = args.activitynet_val_json.expanduser()
        if args.activitynet_inference_json is None:
            args.activitynet_inference_json = args.activitynet_val_json
        args.activitynet_inference_json = args.activitynet_inference_json.expanduser()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if args.inference_output is None:
        args.inference_output = Path("scene_inference_results.jsonl")
    args.inference_output = args.inference_output.expanduser()
    if args.disable_inference_visualizations:
        args.inference_visualization_dir = None
    else:
        if args.inference_visualization_dir is None:
            args.inference_visualization_dir = args.inference_output.parent / "visualizations"
        args.inference_visualization_dir = args.inference_visualization_dir.expanduser()
    args.inference_visualization_max_scenes = max(1, int(args.inference_visualization_max_scenes))
    if args.checkpoint_path is None:
        args.checkpoint_path = Path("scene_transformer.pt")
    args.checkpoint_path = args.checkpoint_path.expanduser()
    if args.loss_plot_path is None and args.mode == "train":
        args.loss_plot_path = (args.inference_output.parent / "loss_curve.png").expanduser()
    if args.loss_plot_path is not None:
        args.loss_plot_path = args.loss_plot_path.expanduser()
        args.loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    args.inference_interval = max(0, int(args.inference_interval or 0))
    if args.inference_limit is not None and args.inference_limit <= 0:
        args.inference_limit = None
    if args.max_sample_infer is not None and args.max_sample_infer <= 0:
        args.max_sample_infer = None
    if args.video_cache_root is not None:
        args.video_cache_root = args.video_cache_root.expanduser()
        args.video_cache_root.mkdir(parents=True, exist_ok=True)
    if args.text_cache_root is not None:
        args.text_cache_root = args.text_cache_root.expanduser()
        args.text_cache_root.mkdir(parents=True, exist_ok=True)

    msrvtt_dataset_cache: Dict[str, MSRVTTUntrimmedDataset] = {}

    def get_msrvtt_dataset(split: str) -> MSRVTTUntrimmedDataset:
        split = split.strip()
        if not split:
            raise ValueError("MSRVTT split name cannot be empty.")
        if split not in msrvtt_dataset_cache:
            msrvtt_dataset_cache[split] = MSRVTTUntrimmedDataset(
                args.msrvtt_annotation,
                args.msrvtt_feat_root,
                split=split,
            )
        return msrvtt_dataset_cache[split]

    activitynet_dataset_cache: Dict[Path, ActivityNetSceneDataset] = {}

    def get_activitynet_dataset(annotation_path: Path, split_name: str) -> ActivityNetSceneDataset:
        resolved = annotation_path.expanduser()
        if resolved not in activitynet_dataset_cache:
            activitynet_dataset_cache[resolved] = ActivityNetSceneDataset(
                resolved,
                args.activitynet_video_features,
                args.activitynet_text_features,
                split_name=split_name,
            )
        return activitynet_dataset_cache[resolved]

    video_backbone: Optional[InternVideo2VideoBackbone] = None
    text_backbone: Optional[InternVideo2TextBackbone] = None
    embed_dim: Optional[int] = None

    if args.dataset == "qvhighlights":
        multi_root = args.internvideo_root / "InternVideo2" / "multi_modality"
        cfg = load_internvideo2_config(
            multi_root,
            args.internvideo_config,
            args.internvideo_ckpt,
            device,
            num_frames=args.frames_per_clip,
            frame_size=args.frame_size,
            origin_num_frames=args.internvideo_origin_num_frames,
        )
        intern_model, _ = setup_internvideo2_model(cfg)
        video_backbone = InternVideo2VideoBackbone(intern_model, clip_batch_size=args.clip_batch_size)
        text_backbone = InternVideo2TextBackbone(intern_model)
        embed_dim = intern_model.embed_dim
    elif args.dataset == "msrvtt_untrimmed":
        split = args.msrvtt_train_split if args.mode != "inference" else args.msrvtt_inference_split
        primary_dataset = get_msrvtt_dataset(split)
        embed_dim = primary_dataset.feature_dim
    else:
        annotation = args.activitynet_train_json if args.mode != "inference" else args.activitynet_inference_json
        primary_dataset = get_activitynet_dataset(annotation, split_name=args.mode)
        embed_dim = primary_dataset.feature_dim

    if embed_dim is None:
        raise RuntimeError("Failed to determine embedding dimension for the scene model.")

    scene_model = SceneTransformer(
        embed_dim=embed_dim,
        num_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        ff_dim=args.decoder_ff_dim,
        dropout=args.decoder_dropout,
    ).to(device)
    summarize_model_parameters(scene_model, "SceneTransformer")

    inference_dataset_cache: Optional[Dataset] = None
    inference_dataset_error: Optional[Exception] = None

    def ensure_inference_dataset() -> Optional[Dataset]:
        nonlocal inference_dataset_cache
        nonlocal inference_dataset_error
        if inference_dataset_cache is None and inference_dataset_error is None:
            try:
                if args.dataset == "qvhighlights":
                    inference_dataset_cache = QVHighlightsDataset(
                        args.inference_jsonl, args.concat_root, args.raw_root
                    )
                    logging.info(
                        "Inference dataset stats: %d videos, %d queries",
                        len(inference_dataset_cache),
                        inference_dataset_cache.total_queries,
                    )
                    log_dataset_samples("Inference", inference_dataset_cache)
                elif args.dataset == "msrvtt_untrimmed":
                    inference_dataset_cache = get_msrvtt_dataset(args.msrvtt_inference_split)
                    logging.info(
                        "Inference dataset stats: %d videos, %d queries",
                        len(inference_dataset_cache),
                        inference_dataset_cache.total_queries,
                    )
                    log_msrvtt_dataset_samples("Inference", inference_dataset_cache)
                else:
                    inference_dataset_cache = get_activitynet_dataset(
                        args.activitynet_inference_json,
                        split_name="inference",
                    )
                    logging.info(
                        "Inference dataset stats: %d videos, %d queries",
                        len(inference_dataset_cache),
                        inference_dataset_cache.total_queries,
                    )
                    log_activitynet_dataset_samples("Inference", inference_dataset_cache)
            except Exception as exc:
                inference_dataset_error = exc
                if args.mode == "inference":
                    raise
                logging.warning("Inference dataset unavailable: %s", exc)
        return inference_dataset_cache

    def inference_output_for_step(step: int) -> Path:
        base = args.inference_output
        suffix = base.suffix
        stem = base.stem
        return base.with_name(f"{stem}_step{step:06d}{suffix}")

    if args.mode == "inference":
        if not args.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        load_checkpoint(scene_model, args.checkpoint_path, device)
        inference_dataset = ensure_inference_dataset()
        if inference_dataset is None:
            logging.warning("Inference dataset unavailable; exiting without running inference.")
            return
        run_inference(
            args,
            inference_dataset,
            video_backbone,
            scene_model,
            text_backbone,
            device,
            output_path=args.inference_output,
            max_videos=args.inference_limit,
            dataset_type=args.dataset,
        )
        return

    if args.dataset == "qvhighlights":
        dataset = QVHighlightsDataset(args.train_jsonl, args.concat_root, args.raw_root)
        logging.info(
            "Train dataset stats: %d videos, %d queries",
            len(dataset),
            dataset.total_queries,
        )
        log_dataset_samples("Train", dataset)
    elif args.dataset == "msrvtt_untrimmed":
        dataset = get_msrvtt_dataset(args.msrvtt_train_split)
        logging.info(
            "Train dataset stats: %d videos, %d queries",
            len(dataset),
            dataset.total_queries,
        )
        log_msrvtt_dataset_samples("Train", dataset)
    else:
        dataset = get_activitynet_dataset(args.activitynet_train_json, split_name="train")
        logging.info(
            "Train dataset stats: %d videos, %d queries",
            len(dataset),
            dataset.total_queries,
        )
        log_activitynet_dataset_samples("Train", dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )
    optimizer = torch.optim.AdamW(scene_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    validation_loader = None
    if args.validation_interval > 0:
        try:
            if args.dataset == "qvhighlights":
                validation_dataset = QVHighlightsDataset(args.validation_jsonl, args.concat_root, args.raw_root)
                logging.info(
                    "Validation dataset stats: %d videos, %d queries",
                    len(validation_dataset),
                    validation_dataset.total_queries,
                )
                log_dataset_samples("Validation", validation_dataset)
            elif args.dataset == "msrvtt_untrimmed":
                validation_dataset = get_msrvtt_dataset(args.msrvtt_val_split)
                logging.info(
                    "Validation dataset stats: %d videos, %d queries",
                    len(validation_dataset),
                    validation_dataset.total_queries,
                )
                log_msrvtt_dataset_samples("Validation", validation_dataset)
            else:
                validation_dataset = get_activitynet_dataset(
                    args.activitynet_val_json,
                    split_name="val",
                )
                logging.info(
                    "Validation dataset stats: %d videos, %d queries",
                    len(validation_dataset),
                    validation_dataset.total_queries,
                )
                log_activitynet_dataset_samples("Validation", validation_dataset)
        except RuntimeError as exc:
            logging.warning("Validation disabled: %s", exc)
        else:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=lambda batch: batch,
            )

    steps_per_epoch = len(data_loader)
    val_fraction = min(max(args.validation_interval, 0.0), 1.0)
    validation_milestones: List[Tuple[int, float]] = []
    if validation_loader is not None and steps_per_epoch > 0 and val_fraction > 0:
        next_frac = val_fraction
        while next_frac < 1.0 - 1e-9:
            step_threshold = max(1, math.ceil(next_frac * steps_per_epoch))
            if not validation_milestones or step_threshold != validation_milestones[-1][0]:
                validation_milestones.append((step_threshold, next_frac))
            next_frac += val_fraction
        if validation_milestones and validation_milestones[-1][0] == steps_per_epoch:
            validation_milestones[-1] = (steps_per_epoch, 1.0)
        else:
            validation_milestones.append((steps_per_epoch, 1.0))
    elif val_fraction > 0 and validation_loader is None:
        logging.warning("Validation disabled: no validation dataset available.")
    elif val_fraction > 0 and steps_per_epoch == 0:
        logging.warning("Validation disabled: training dataloader has zero batches.")
    validation_enabled = bool(validation_milestones)

    global_step = 0
    logged_train_batch_schema = False
    logged_val_batch_schema = False
    train_history: List[Tuple[int, float]] = []
    val_history: List[Tuple[int, float]] = []
    for epoch in range(1, args.epochs + 1):
        scene_model.train()
        steps_in_epoch = 0
        milestone_idx = 0
        for batch_idx, batch_samples in enumerate(data_loader, start=1):
            batch = collate_scene_batch(
                batch_samples,
                video_backbone=video_backbone,
                text_backbone=text_backbone,
                scene_model=scene_model,
                args=args,
                device=device,
                dataset_type=args.dataset,
            )
            if batch is None:
                continue

            if not logged_train_batch_schema:
                log_batch_schema("Train", batch)
                logged_train_batch_schema = True

            preds, attn = scene_model(
                batch["clip_embeddings"],
                batch["clip_times"],
                batch["clip_padding_mask"],
                batch["decoder_inputs"],
                batch["decoder_padding_mask"],
            )
            rep_loss = representation_alignment_loss(
                preds,
                batch["decoder_targets"],
                batch["decoder_padding_mask"],
                mode=args.alignment_loss,
                temperature=args.infonce_temp,
            )
            attn_loss = attention_supervision_loss(
                attn,
                batch["clip_times"],
                batch["clip_padding_mask"],
                batch["scene_windows"],
                batch["scene_lengths"],
            )
            mono_loss = monotonicity_loss(attn, batch["clip_padding_mask"], batch["scene_lengths"])
            cov_loss = rep_loss.new_tensor(0.0)
            total_loss = rep_loss
            total_loss = total_loss + args.lambda_attn * attn_loss
            total_loss = total_loss + args.lambda_mono * mono_loss
            # coverage loss disabled

            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(scene_model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            steps_in_epoch += 1
            train_history.append((global_step, float(total_loss.item())))

            if global_step % args.log_interval == 0:
                logging.info(
                    "Epoch %d step %d | loss=%.4f repr=%.4f attn=%.4f mono=%.4f cov=%.4f",
                    epoch,
                    global_step,
                    total_loss.item(),
                    rep_loss.item(),
                    attn_loss.item(),
                    mono_loss.item(),
                    cov_loss.item(),
                )

            while (
                validation_enabled
                and milestone_idx < len(validation_milestones)
                and steps_in_epoch >= validation_milestones[milestone_idx][0]
            ):
                frac = validation_milestones[milestone_idx][1]
                metrics = run_validation(
                    validation_loader,
                    video_backbone=video_backbone,
                    text_backbone=text_backbone,
                    scene_model=scene_model,
                    args=args,
                    device=device,
                    dataset_type=args.dataset,
                    schema_label="Validation",
                    log_schema=not logged_val_batch_schema,
                )
                if metrics is not None:
                    if not logged_val_batch_schema:
                        logged_val_batch_schema = True
                    logging.info(
                        "Validation epoch %.2f | loss=%.4f repr=%.4f attn=%.4f mono=%.4f cov=%.4f",
                        epoch - 1 + frac,
                        metrics["loss"],
                        metrics["repr"],
                        metrics["attn"],
                        metrics["mono"],
                        metrics["cov"],
                    )
                    val_history.append((global_step, float(metrics["loss"])))
                milestone_idx += 1

            if args.inference_interval > 0 and global_step % args.inference_interval == 0:
                inference_dataset = ensure_inference_dataset()
                if inference_dataset is None:
                    logging.warning(
                        "Skipping periodic inference at step %d: inference dataset unavailable.",
                        global_step,
                    )
                else:
                    step_output = inference_output_for_step(global_step)
                    logging.info(
                        "Running periodic inference at step %d -> %s",
                        global_step,
                        step_output,
                    )
                    run_inference(
                        args,
                        inference_dataset,
                        video_backbone,
                        scene_model,
                        text_backbone,
                        device,
                        output_path=step_output,
                        max_videos=args.inference_limit,
                        dataset_type=args.dataset,
                    )

        epoch_ckpt = args.checkpoint_path.with_name(
            f"{args.checkpoint_path.stem}_epoch{epoch:03d}{args.checkpoint_path.suffix}"
        )
        save_checkpoint(scene_model, epoch_ckpt)
        logging.info("Saved SceneTransformer checkpoint for epoch %d to %s", epoch, epoch_ckpt)

    save_checkpoint(scene_model, args.checkpoint_path)
    logging.info("Saved SceneTransformer checkpoint to %s", args.checkpoint_path)
    save_loss_plot(train_history, val_history, args.loss_plot_path)

    if args.run_inference_after_train:
        inference_dataset = ensure_inference_dataset()
        if inference_dataset is None:
            logging.warning("Skipping post-training inference: inference dataset unavailable.")
        else:
            run_inference(
                args,
                inference_dataset,
                video_backbone,
                scene_model,
                text_backbone,
                device,
                output_path=args.inference_output,
                max_videos=args.max_sample_infer,
                dataset_type=args.dataset,
            )


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
