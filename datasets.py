from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models import (
    InternVideo2TextBackbone,
    InternVideo2VideoBackbone,
    SceneTransformer,
)

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None

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


@dataclass
class TVRFeatureMeta:
    video_id: str
    duration: float
    scene_texts: List[str]
    timestamps: List[Tuple[float, float]]
    text_keys: List[str]
    num_clips: int


@dataclass
class TVRSceneItem:
    video_id: str
    clip_embeddings: torch.Tensor
    clip_times: torch.Tensor
    scene_texts: List[str]
    scene_windows: List[Tuple[float, float]]
    text_embeddings: torch.Tensor
    timestamps: List[Tuple[float, float]]
    duration: float


def parse_vid_identifier(vid: str) -> Tuple[str, float, float]:
    parts = vid.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid vid format: {vid}")
    base_video = parts[0]
    start_sec = float(parts[1])
    end_sec = float(parts[2])
    return base_video, start_sec, end_sec


class QVHighlightsDataset(Dataset):
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
            sorted_segments = sorted(segments.values(), key=lambda seg: float(seg["start"]))
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
                if not isinstance(window, (list, tuple)) or len(window) != 2:
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
        vision = torch.load(meta.vision_path, map_location="cpu", weights_only=False)

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

        clip_count = clip_embeddings.shape[0] if clip_embeddings.ndim >= 1 else clip_embeddings.numel()
        raw_clip_times = None
        if isinstance(vision, dict):
            for key in ("clip_times", "times", "timestamps"):
                if key in vision:
                    raw_clip_times = vision[key]
                    break
        if raw_clip_times is None:
            if len(meta.scene_windows) == clip_count:
                centers = [0.5 * (float(s) + float(e)) for (s, e) in meta.scene_windows]
                clip_times = torch.tensor(centers, dtype=torch.float32)
            else:
                logging.warning(
                    "Video %s: missing clip_times and scene_windows len (%d) != clip_count (%d); "
                    "falling back to index-based times.",
                    meta.video_id,
                    len(meta.scene_windows),
                    clip_count,
                )
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

        text_feat = torch.load(meta.text_path, map_location="cpu", weights_only=False)
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
        assert h5py is not None
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


class TVRSceneDataset(Dataset):
    def __init__(
        self,
        annotation_path: Path,
        video_feat_path: Path,
        text_feat_path: Path,
        *,
        split_name: str = "train",
        cache_features: bool = True,
        sanity_samples: int = 5,
    ) -> None:
        if h5py is None:
            raise ImportError("h5py is required to read TVR HDF5 features. Please install h5py first.")
        self.annotation_path = annotation_path.expanduser()
        self.video_feat_path = video_feat_path.expanduser()
        self.text_feat_path = text_feat_path.expanduser()
        self.split_name = split_name
        self.cache_features = cache_features
        self.items: List[TVRFeatureMeta] = self._load_metadata()
        if not self.items:
            raise RuntimeError(
                f"TVR split '{self.split_name}' has no valid videos (annotation={self.annotation_path})"
            )
        self.total_queries = sum(len(meta.scene_texts) for meta in self.items)
        self._feature_cache: Dict[str, TVRSceneItem] = {}
        self.feature_dim = self._infer_feature_dim()
        self._run_sanity_checks(samples=sanity_samples)
        logging.info(
            "TVR dataset loaded: split=%s videos=%d queries=%d feature_dim=%d",
            self.split_name,
            len(self.items),
            self.total_queries,
            self.feature_dim,
        )

    def _load_metadata(self) -> List[TVRFeatureMeta]:
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"TVR annotation {self.annotation_path} must be a JSON object mapping video ids."
            )
        items: List[TVRFeatureMeta] = []
        assert h5py is not None
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
                num_clips = int(video_dataset.shape[0])
                if num_clips <= 0:
                    logging.warning("Skipping %s: no video clips stored", video_id)
                    continue
                resolved_timestamps: List[Tuple[float, float]] = []
                resolved_texts: List[str] = []
                for sent, window in zip(sentences, timestamps):
                    if not isinstance(window, (list, tuple)) or len(window) != 2:
                        continue
                    start = float(window[0])
                    end = float(window[1])
                    if math.isnan(start) or math.isnan(end) or end <= start:
                        continue
                    resolved_timestamps.append((start, end))
                    resolved_texts.append(str(sent))
                if not resolved_texts:
                    logging.warning("Skipping %s: no valid timestamp/text pairs", video_id)
                    continue
                text_keys: List[str] = []
                missing_text = False
                missing_key: Optional[str] = None
                for idx in range(len(resolved_texts)):
                    key = f"{video_id}#enc#{idx}"
                    if key not in text_file:
                        missing_text = True
                        missing_key = key
                        break
                    text_keys.append(key)
                if missing_text:
                    logging.warning(
                        "Skipping %s: missing text features for key %s in %s",
                        video_id,
                        missing_key,
                        self.text_feat_path,
                    )
                    continue
                items.append(
                    TVRFeatureMeta(
                        video_id=video_id,
                        duration=duration,
                        scene_texts=resolved_texts,
                        timestamps=resolved_timestamps,
                        text_keys=text_keys,
                        num_clips=num_clips,
                    )
                )
        return items

    def _load_sample(self, meta: TVRFeatureMeta) -> TVRSceneItem:
        assert h5py is not None
        with h5py.File(self.video_feat_path, "r") as video_file:
            video_array = np.array(video_file[meta.video_id], dtype=np.float32)
        clip_embeddings = torch.from_numpy(video_array)
        clip_times = self._build_clip_times(meta.duration, meta.num_clips)
        text_features: List[np.ndarray] = []
        with h5py.File(self.text_feat_path, "r") as text_file:
            for key in meta.text_keys:
                text_features.append(np.array(text_file[key], dtype=np.float32))
        text_embeddings = torch.from_numpy(np.stack(text_features, axis=0))
        return TVRSceneItem(
            video_id=meta.video_id,
            clip_embeddings=clip_embeddings,
            clip_times=clip_times,
            scene_texts=meta.scene_texts,
            scene_windows=meta.timestamps,
            text_embeddings=text_embeddings,
            timestamps=meta.timestamps,
            duration=meta.duration,
        )

    @staticmethod
    def _build_clip_times(duration: float, num_clips: int) -> torch.Tensor:
        if num_clips <= 0:
            return torch.empty(0)
        duration = max(duration, 1e-3)
        edges = torch.linspace(0.0, duration, steps=num_clips + 1, dtype=torch.float32)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers

    def _infer_feature_dim(self) -> int:
        if not self.items:
            return 0
        first = self.items[0]
        sample = self._load_sample(first)
        self._feature_cache[first.video_id] = sample
        if sample.clip_embeddings.ndim != 2:
            raise RuntimeError(f"TVR vision feats for {first.video_id} missing clip dimension.")
        return int(sample.clip_embeddings.shape[1])

    def _run_sanity_checks(self, *, samples: int) -> None:
        if not self.items or samples <= 0:
            return
        rng = random.Random(0)
        sample_count = min(samples, len(self.items))
        indices = rng.sample(range(len(self.items)), sample_count)
        for idx in indices:
            meta = self.items[idx]
            sentences = meta.scene_texts
            timestamps = meta.timestamps
            assert len(sentences) == len(timestamps), (
                f"TVR sanity check failed for {meta.video_id}: sentences/timestamps mismatch"
            )
            if not timestamps:
                continue
            expected = meta.duration / max(1, len(timestamps))
            prev_end = 0.0
            for seg_idx, (start, end) in enumerate(timestamps):
                if start < prev_end - 1e-3:
                    raise AssertionError(
                        f"TVR sanity check failed for {meta.video_id}: segment {seg_idx} overlaps"
                    )
                prev_end = end
                seg_len = end - start
                tolerance = max(1.0, 0.35 * expected)
                if abs(seg_len - expected) > tolerance:
                    raise AssertionError(
                        f"TVR sanity check failed for {meta.video_id}: segment length {seg_len:.3f} differs from expected {expected:.3f}"
                    )
            last_end = timestamps[-1][1]
            if abs(last_end - meta.duration) > max(1.0, 0.05 * meta.duration):
                raise AssertionError(
                    f"TVR sanity check failed for {meta.video_id}: last timestamp {last_end:.3f} != duration {meta.duration:.3f}"
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TVRSceneItem:
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
    data = torch.load(path, map_location="cpu", weights_only=False)
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
    return torch.load(path, map_location="cpu", weights_only=False)


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
        elif dataset_type in {"activitynet", "tvr"}:
            if dataset_type == "activitynet":
                expected = ActivityNetSceneItem
            else:
                expected = TVRSceneItem
            if not isinstance(sample, expected):
                raise TypeError(f"{dataset_type} dataset expected {expected.__name__} samples.")
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
