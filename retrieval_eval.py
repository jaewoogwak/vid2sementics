#!/usr/bin/env python3
"""Text-to-video retrieval evaluation using SceneTransformer scene embeddings."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from datasets import ActivityNetSceneDataset, MSRVTTUntrimmedDataset, TVRSceneDataset
from models import (
    InternVideo2TextBackbone,
    SceneTransformer,
    load_internvideo2_config,
    setup_internvideo2_model,
)
from trainer import load_checkpoint
from utils import trim_eos_scene_predictions

try:
    import h5py  # type: ignore
except ImportError:
    h5py = None
import numpy as np


@dataclass
class RetrievalQuery:
    video_id: str
    text: str
    precomputed_embedding: Optional[torch.Tensor] = None


class ActivityNetRetrievalDataset:
    """Wraps ActivityNetSceneDataset to expose queries per sentence."""

    def __init__(
        self,
        annotation_path: Path,
        video_feat_path: Path,
        text_feat_path: Path,
    ) -> None:
        self.annotation_path = annotation_path.expanduser()
        self.video_feat_path = video_feat_path.expanduser()
        self.text_feat_path = text_feat_path.expanduser()
        self.dataset = ActivityNetSceneDataset(
            annotation_path=annotation_path,
            video_feat_path=video_feat_path,
            text_feat_path=text_feat_path,
            split_name="val",
        )
        self.feature_dim = self.dataset.feature_dim
        self.items = self.dataset.items
        self.video_ids = [meta.video_id for meta in self.items]
        self._index = {meta.video_id: idx for idx, meta in enumerate(self.items)}
        queries: List[RetrievalQuery] = []
        if h5py is None:
            raise ImportError("h5py is required to load ActivityNet text embeddings.")
        with h5py.File(self.text_feat_path, "r") as text_file:
            for meta in self.items:
                for idx, sentence in enumerate(meta.scene_texts):
                    text = sentence.strip()
                    if not text:
                        continue
                    key = meta.text_keys[idx]
                    arr = np.array(text_file[key], dtype=np.float32)
                    tensor = torch.from_numpy(arr)
                    queries.append(
                        RetrievalQuery(
                            video_id=meta.video_id,
                            text=text,
                            precomputed_embedding=tensor,
                        )
                    )
        self.queries = queries

    def get_queries(self, limit: Optional[int] = None, allowed_videos: Optional[Sequence[str]] = None) -> List[RetrievalQuery]:
        pool = self.queries
        if allowed_videos is not None:
            allowed = set(allowed_videos)
            pool = [query for query in pool if query.video_id in allowed]
        if limit is not None:
            return pool[:limit]
        return pool

    def get_video_ids(self, limit: Optional[int] = None) -> List[str]:
        if limit is None:
            return list(self.video_ids)
        return self.video_ids[:limit]

    def load_video_sample(self, video_id: str):
        idx = self._index[video_id]
        return self.dataset[idx]


class MSRVTTUntrimmedRetrievalDataset:
    """Loads JSFusion test queries and MSRVTT untrimmed features."""

    def __init__(
        self,
        annotation_path: Path,
        feat_root: Path,
        csv_path: Path,
        train_split: str,
        val_split: str,
    ) -> None:
        splits = sorted({train_split, val_split})
        datasets: List[MSRVTTUntrimmedDataset] = []
        lookup: Dict[str, Tuple[MSRVTTUntrimmedDataset, int]] = {}
        for split in splits:
            ds = MSRVTTUntrimmedDataset(annotation_path, feat_root, split=split)
            datasets.append(ds)
            for idx, meta in enumerate(ds.items):
                if meta.video_id not in lookup:
                    lookup[meta.video_id] = (ds, idx)
        if not datasets:
            raise RuntimeError("Failed to load any MSRVTT splits for retrieval evaluation.")
        self.datasets = datasets
        self.lookup = lookup
        self.feature_dim = datasets[0].feature_dim
        queries: List[RetrievalQuery] = []
        video_ids: List[str] = []
        seen_videos: set[str] = set()
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = (row.get("video_id") or "").strip()
                sentence = (row.get("sentence") or "").strip()
                if not video_id or not sentence:
                    continue
                if video_id not in lookup:
                    logging.warning("Skipping query for %s: video features unavailable.", video_id)
                    continue
                queries.append(RetrievalQuery(video_id=video_id, text=sentence))
                if video_id not in seen_videos:
                    seen_videos.add(video_id)
                    video_ids.append(video_id)
        if not queries:
            raise RuntimeError("JSFusion CSV did not yield any valid queries.")
        self.queries = queries
        self.video_ids = video_ids

    def get_queries(self, limit: Optional[int] = None, allowed_videos: Optional[Sequence[str]] = None) -> List[RetrievalQuery]:
        pool = self.queries
        if allowed_videos is not None:
            allowed = set(allowed_videos)
            pool = [query for query in pool if query.video_id in allowed]
        if limit is not None:
            return pool[:limit]
        return pool

    def get_video_ids(self, limit: Optional[int] = None) -> List[str]:
        if limit is None:
            return list(self.video_ids)
        return self.video_ids[:limit]

    def load_video_sample(self, video_id: str):
        dataset, idx = self.lookup[video_id]
        return dataset[idx]


class TVRRetrievalDataset:
    def __init__(
        self,
        annotation_path: Path,
        video_feat_path: Path,
        text_feat_path: Path,
    ) -> None:
        self.annotation_path = annotation_path.expanduser()
        self.video_feat_path = video_feat_path.expanduser()
        self.text_feat_path = text_feat_path.expanduser()
        self.dataset = TVRSceneDataset(
            annotation_path=self.annotation_path,
            video_feat_path=self.video_feat_path,
            text_feat_path=self.text_feat_path,
            split_name="val",
        )
        self.feature_dim = self.dataset.feature_dim
        self.items = self.dataset.items
        self.video_ids = [meta.video_id for meta in self.items]
        self._index = {meta.video_id: idx for idx, meta in enumerate(self.items)}
        queries: List[RetrievalQuery] = []
        if h5py is None:
            raise ImportError("h5py is required to load TVR text embeddings.")
        with h5py.File(self.text_feat_path, "r") as text_file:
            for meta in self.items:
                for idx, sentence in enumerate(meta.scene_texts):
                    text = sentence.strip()
                    if not text:
                        continue
                    key = meta.text_keys[idx]
                    arr = np.array(text_file[key], dtype=np.float32)
                    tensor = torch.from_numpy(arr)
                    queries.append(
                        RetrievalQuery(
                            video_id=meta.video_id,
                            text=text,
                            precomputed_embedding=tensor,
                        )
                    )
        self.queries = queries

    def get_queries(
        self,
        limit: Optional[int] = None,
        allowed_videos: Optional[Sequence[str]] = None,
    ) -> List[RetrievalQuery]:
        pool = self.queries
        if allowed_videos is not None:
            allowed = set(allowed_videos)
            pool = [query for query in pool if query.video_id in allowed]
        if limit is not None:
            return pool[:limit]
        return pool

    def get_video_ids(self, limit: Optional[int] = None) -> List[str]:
        if limit is None:
            return list(self.video_ids)
        return self.video_ids[:limit]

    def load_video_sample(self, video_id: str):
        idx = self._index[video_id]
        return self.dataset[idx]


@dataclass
class SceneEmbeddingBank:
    video_ids: List[str]
    video_id_to_index: Dict[str, int]
    scenes: torch.Tensor
    scene_video_index: torch.Tensor


def build_scene_bank(
    embeddings: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    embed_dim: int,
) -> SceneEmbeddingBank:
    video_ids: List[str] = []
    video_id_to_index: Dict[str, int] = {}
    scene_chunks: List[torch.Tensor] = []
    video_index_chunks: List[torch.Tensor] = []
    skipped: List[str] = []
    for video_id, tensor in embeddings.items():
        if tensor.numel() == 0:
            skipped.append(video_id)
            continue
        vid_idx = len(video_ids)
        video_ids.append(video_id)
        video_id_to_index[video_id] = vid_idx
        scene_chunks.append(tensor)
        video_index_chunks.append(
            torch.full((tensor.shape[0],), vid_idx, dtype=torch.long)
        )
    if skipped:
        logging.warning("Skipping %d/%d videos due to empty scene predictions.", len(skipped), len(embeddings))
    if scene_chunks:
        scenes = torch.cat(scene_chunks, dim=0).to(device)
        scene_video_index = torch.cat(video_index_chunks, dim=0).to(device)
    else:
        scenes = torch.empty(0, embed_dim, device=device)
        scene_video_index = torch.empty(0, dtype=torch.long, device=device)
    return SceneEmbeddingBank(
        video_ids=video_ids,
        video_id_to_index=video_id_to_index,
        scenes=scenes,
        scene_video_index=scene_video_index,
    )


@torch.no_grad()
def encode_video_sample(
    sample,
    scene_model: SceneTransformer,
    device: torch.device,
    *,
    max_steps: int,
    eos_threshold: float,
) -> torch.Tensor:
    clip_feats = sample.clip_embeddings.float().to(device)
    if clip_feats.ndim != 2:
        clip_feats = clip_feats.view(-1, clip_feats.shape[-1])
    clip_times = sample.clip_times.float().to(device)
    if clip_times.ndim == 1:
        clip_times = clip_times.unsqueeze(0)
    elif clip_times.ndim > 2:
        clip_times = clip_times.view(1, -1)
    clip_feats = clip_feats.unsqueeze(0)
    seq_len = clip_feats.shape[1]
    if clip_times.shape[-1] != seq_len:
        clip_times = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    clip_padding = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    generated, _ = scene_model.generate(
        clip_feats,
        clip_times,
        clip_padding,
        max_steps=max_steps,
        eos_threshold=eos_threshold,
    )
    latents = generated.squeeze(0)
    trimmed, _ = trim_eos_scene_predictions(latents, None, scene_model.eos_token, eos_threshold)
    if trimmed.numel() == 0:
        return trimmed.cpu()
    normalized = F.normalize(trimmed, dim=-1)
    return normalized.cpu()


def get_cache_path(cache_root: Optional[Path], dataset_name: str, video_id: str) -> Optional[Path]:
    if cache_root is None:
        return None
    return cache_root / dataset_name / f"{video_id}.pt"


def load_cached_scene_embeddings(path: Optional[Path]) -> Optional[torch.Tensor]:
    if path is None or not path.is_file():
        return None
    data = torch.load(path, map_location="cpu")
    tensor = data.get("scene_embeddings")
    if isinstance(tensor, torch.Tensor):
        return tensor
    return None


def save_cached_scene_embeddings(path: Optional[Path], tensor: torch.Tensor) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scene_embeddings": tensor}, path)


def collect_video_embeddings(
    dataset_name: str,
    video_ids: Sequence[str],
    loader,
    scene_model: SceneTransformer,
    device: torch.device,
    *,
    cache_root: Optional[Path],
    max_steps: int,
    eos_threshold: float,
) -> Dict[str, torch.Tensor]:
    results: Dict[str, torch.Tensor] = {}
    for idx, video_id in enumerate(video_ids, start=1):
        cache_path = get_cache_path(cache_root, dataset_name, video_id)
        cached = load_cached_scene_embeddings(cache_path)
        if cached is not None:
            results[video_id] = cached
            continue
        sample = loader(video_id)
        embeddings = encode_video_sample(
            sample,
            scene_model,
            device,
            max_steps=max_steps,
            eos_threshold=eos_threshold,
        )
        results[video_id] = embeddings
        save_cached_scene_embeddings(cache_path, embeddings)
        if idx % 100 == 0 or idx == len(video_ids):
            logging.info("Encoded %d/%d videos", idx, len(video_ids))
    return results


@torch.no_grad()
def encode_queries(
    queries: Sequence[RetrievalQuery],
    text_backbone: Optional[InternVideo2TextBackbone],
    scene_model: SceneTransformer,
    device: torch.device,
) -> Tuple[torch.Tensor, List[str]]:
    embeddings: List[torch.Tensor] = []
    owners: List[str] = []
    for idx, query in enumerate(queries, start=1):
        if query.precomputed_embedding is not None:
            text_tensor = query.precomputed_embedding.to(device)
            if text_tensor.ndim == 1:
                text_tensor = text_tensor.unsqueeze(0)
        else:
            if text_backbone is None:
                raise RuntimeError("Text backbone is required for queries without precomputed embeddings.")
            text_tensor = text_backbone.encode([query.text])
        projected = scene_model.project_text_embeddings(text_tensor.to(device))
        normalized = F.normalize(projected.squeeze(0), dim=-1)
        embeddings.append(normalized)
        owners.append(query.video_id)
        if idx % 500 == 0 or idx == len(queries):
            logging.info("Encoded %d/%d queries", idx, len(queries))
    if embeddings:
        stacked = torch.stack(embeddings, dim=0)
    else:
        stacked = torch.empty(0, scene_model.embed_dim, device=device)
    return stacked, owners


def select_top_videos(
    scene_scores: torch.Tensor,
    scene_video_index: torch.Tensor,
    *,
    top_k: int,
    max_scene_candidates: int,
) -> List[int]:
    if top_k <= 0:
        return []
    total = scene_scores.shape[0]
    if total == 0:
        return []
    candidate_k = min(total, max_scene_candidates)
    top_values, top_indices = torch.topk(scene_scores, candidate_k)
    unique_videos: List[int] = []
    seen = set()
    for idx in top_indices.tolist():
        vid_idx = int(scene_video_index[idx])
        if vid_idx in seen:
            continue
        seen.add(vid_idx)
        unique_videos.append(vid_idx)
        if len(unique_videos) >= top_k:
            break
    if len(unique_videos) >= top_k or candidate_k == total:
        return unique_videos
    full_order = torch.argsort(scene_scores, descending=True)
    for idx in full_order.tolist():
        vid_idx = int(scene_video_index[idx])
        if vid_idx in seen:
            continue
        seen.add(vid_idx)
        unique_videos.append(vid_idx)
        if len(unique_videos) >= top_k:
            break
    return unique_videos


def evaluate_queries(
    query_embeddings: torch.Tensor,
    query_video_ids: Sequence[str],
    scene_bank: SceneEmbeddingBank,
    *,
    max_scene_candidates: int,
    topk: Sequence[int] = (1, 5, 10),
) -> Dict[int, float]:
    hits = {k: 0 for k in topk}
    valid_queries = 0
    for idx in range(query_embeddings.shape[0]):
        query_video_id = query_video_ids[idx]
        gt_index = scene_bank.video_id_to_index.get(query_video_id)
        if gt_index is None:
            logging.warning("Skipping query %s: video not encoded.", query_video_id)
            continue
        scores = scene_bank.scenes @ query_embeddings[idx]
        ranked_videos = select_top_videos(
            scores,
            scene_bank.scene_video_index,
            top_k=max(topk),
            max_scene_candidates=max_scene_candidates,
        )
        valid_queries += 1
        for k in topk:
            if gt_index in ranked_videos[:k]:
                hits[k] += 1
    if valid_queries == 0:
        logging.warning("No valid queries evaluated.")
        return {k: 0.0 for k in topk}
    return {k: hits[k] / valid_queries for k in topk}


def setup_text_encoder(args, device: torch.device) -> InternVideo2TextBackbone:
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
    return InternVideo2TextBackbone(intern_model)


def build_scene_model(args, embed_dim: int, device: torch.device) -> SceneTransformer:
    model = SceneTransformer(
        embed_dim=embed_dim,
        num_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        ff_dim=args.decoder_ff_dim,
        dropout=args.decoder_dropout,
        use_text_projection=not args.disable_text_projection,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_intern_root = repo_root / "InternVideo"
    default_msrvtt_root = repo_root / "dataset" / "data" / "MSRVTT"
    default_activitynet_root = repo_root / "dataset" / "activitynet"
    parser = argparse.ArgumentParser(description="Evaluate text-to-video retrieval using SceneTransformer.")
    parser.add_argument("--dataset", choices=("msrvtt_untrimmed", "activitynet", "tvr"), required=True)
    parser.add_argument("--mode", default="retrieval")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to SceneTransformer checkpoint.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1, help="Unused placeholder for API compatibility.")
    parser.add_argument("--num-workers", type=int, default=0, help="Unused placeholder for API compatibility.")
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--decoder-ff-dim", type=int, default=2048)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument(
        "--disable-text-projection",
        action="store_true",
        help="Skip SceneTransformer text projection (use raw text embeddings).",
    )
    parser.add_argument("--max-generation-steps", type=int, default=12)
    parser.add_argument("--eos-threshold", type=float, default=0.8)
    parser.add_argument("--max-scene-candidates", type=int, default=2048, help="Scene candidates inspected per query before falling back to full sort.")
    parser.add_argument("--scene-cache-root", type=Path, default=None)
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos encoded (debug).")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries evaluated (debug).")
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--internvideo-root", type=Path, default=default_intern_root)
    parser.add_argument(
        "--internvideo-config",
        type=Path,
        default=default_intern_root
        / "InternVideo2"
        / "multi_modality"
        / "demo"
        / "internvideo2_stage2_config.py",
    )
    parser.add_argument(
        "--internvideo-ckpt",
        type=Path,
        default=default_intern_root / "InternVideo2" / "ckpt" / "InternVideo2-stage2_1b-224p-f4.pt",
    )
    parser.add_argument("--internvideo-origin-num-frames", type=int, default=None)
    parser.add_argument(
        "--msrvtt-annotation",
        type=Path,
        default=default_msrvtt_root / "annotation" / "MSRVTT_untrimmed.json",
    )
    parser.add_argument(
        "--msrvtt-feat-root",
        type=Path,
        default=default_msrvtt_root / "internvideo_untrimmed_feats",
    )
    parser.add_argument(
        "--msrvtt-jsfusion-csv",
        type=Path,
        default=default_msrvtt_root / "annotation" / "MSRVTT_JSFUSION_test.csv",
    )
    parser.add_argument("--msrvtt-train-split", type=str, default="train")
    parser.add_argument("--msrvtt-val-split", type=str, default="val")
    parser.add_argument("--activitynet-root", type=Path, default=default_activitynet_root)
    parser.add_argument(
        "--activitynet-video-features",
        type=Path,
        default=default_activitynet_root / "FeatureData" / "new_clip_vit_32_activitynet_vid_features.hdf5",
    )
    parser.add_argument(
        "--activitynet-text-features",
        type=Path,
        default=default_activitynet_root / "TextData" / "clip_ViT_B_32_activitynet_query_feat.hdf5",
    )
    parser.add_argument(
        "--activitynet-val-json",
        type=Path,
        default=default_activitynet_root / "TextData" / "val_1.json",
    )
    default_tvr_root = repo_root / "dataset" / "tvr"
    parser.add_argument("--tvr-root", type=Path, default=default_tvr_root)
    parser.add_argument(
        "--tvr-video-features",
        type=Path,
        default=default_tvr_root / "FeatureData" / "new_clip_vit_32_tvr_vid_features.hdf5",
    )
    parser.add_argument(
        "--tvr-text-features",
        type=Path,
        default=default_tvr_root / "TextData" / "clip_ViT_B_32_tvr_query_feat.hdf5",
    )
    parser.add_argument(
        "--tvr-val-json",
        type=Path,
        default=default_tvr_root / "TextData" / "val.json",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(device_str: str) -> torch.device:
    requested = torch.device(device_str)
    if requested.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA unavailable, falling back to CPU.")
        return torch.device("cpu")
    return requested


def main() -> None:
    args = parse_args()
    configure_logging()
    device = resolve_device(args.device)
    args.checkpoint = args.checkpoint.expanduser()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    dataset_name = args.dataset
    if dataset_name == "msrvtt_untrimmed":
        dataset = MSRVTTUntrimmedRetrievalDataset(
            annotation_path=args.msrvtt_annotation.expanduser(),
            feat_root=args.msrvtt_feat_root.expanduser(),
            csv_path=args.msrvtt_jsfusion_csv.expanduser(),
            train_split=args.msrvtt_train_split,
            val_split=args.msrvtt_val_split,
        )
    elif dataset_name == "activitynet":
        dataset = ActivityNetRetrievalDataset(
            annotation_path=args.activitynet_val_json.expanduser(),
            video_feat_path=args.activitynet_video_features.expanduser(),
            text_feat_path=args.activitynet_text_features.expanduser(),
        )
    else:
        dataset = TVRRetrievalDataset(
            annotation_path=args.tvr_val_json.expanduser(),
            video_feat_path=args.tvr_video_features.expanduser(),
            text_feat_path=args.tvr_text_features.expanduser(),
        )
    video_ids = dataset.get_video_ids(limit=args.max_videos)
    queries = dataset.get_queries(limit=args.max_queries, allowed_videos=video_ids)
    if not queries:
        raise RuntimeError("No queries available after filtering.")
    embed_dim = dataset.feature_dim
    scene_model = build_scene_model(args, embed_dim, device)
    need_text_encoder = any(query.precomputed_embedding is None for query in queries)
    text_backbone: Optional[InternVideo2TextBackbone] = None
    if need_text_encoder:
        text_backbone = setup_text_encoder(args, device)
    logging.info(
        "Dataset: %s | videos=%d | queries=%d", dataset_name, len(video_ids), len(queries)
    )
    if args.scene_cache_root is not None:
        args.scene_cache_root = args.scene_cache_root.expanduser()
    video_embeddings = collect_video_embeddings(
        dataset_name,
        video_ids,
        dataset.load_video_sample,
        scene_model,
        device,
        cache_root=args.scene_cache_root,
        max_steps=args.max_generation_steps,
        eos_threshold=args.eos_threshold,
    )
    scene_bank = build_scene_bank(video_embeddings, device=device, embed_dim=embed_dim)
    if not scene_bank.video_ids:
        raise RuntimeError("No videos produced scene embeddings.")
    query_embeddings, owner_ids = encode_queries(queries, text_backbone, scene_model, device)
    metrics = evaluate_queries(
        query_embeddings,
        owner_ids,
        scene_bank,
        max_scene_candidates=args.max_scene_candidates,
    )
    logging.info("Dataset: %s", dataset_name)
    logging.info("Num queries: %d", len(owner_ids))
    logging.info("Num videos: %d", len(scene_bank.video_ids))
    for k in (1, 5, 10):
        value = metrics.get(k, 0.0) * 100.0
        logging.info("R@%d: %.3f%%", k, value)


if __name__ == "__main__":
    main()
