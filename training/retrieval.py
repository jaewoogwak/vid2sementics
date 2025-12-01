#!/usr/bin/env python3
"""
End-to-end training and HDF5-based evaluation entrypoint for PRVR on QVHighlights.

Two modes are supported:
1. --mode eval_hdf5: load precomputed embeddings from HDF5 files and report R@k metrics.
2. --mode train: decode raw MP4 videos, fine-tune InternVideo2, and periodically evaluate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple, Optional, Set

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    h5py = None

import imageio.v3 as iio
from PIL import Image


# --------------------
# Utility dataclasses
# --------------------

@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_100: float

    @property
    def sum_recall(self) -> float:
        return self.recall_at_1 + self.recall_at_5 + self.recall_at_10 + self.recall_at_100

class ConfigNamespace(SimpleNamespace):
    """SimpleNamespace with dict-like get support."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def to_config_namespace(obj):
    if isinstance(obj, dict):
        return ConfigNamespace(**{k: to_config_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_config_namespace(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_config_namespace(v) for v in obj)
    return obj


def init_distributed_mode(args) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif getattr(args, "local_rank", -1) != -1:
        args.rank = args.local_rank
        args.world_size = 1
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        return

    args.distributed = args.world_size > 1
    if not args.distributed:
        return

    torch.cuda.set_device(args.local_rank)
    dist_backend = getattr(args, "dist_backend", "nccl")
    dist.init_process_group(
        backend=dist_backend,
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def get_amp_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def resolve_ranking_path(base: str | None, default_name: str) -> Path | None:
    if not base:
        return None
    base_path = Path(base)
    if base_path.suffix:
        return base_path
    return base_path / default_name


def gather_list(data: List):
    if not is_dist_avail_and_initialized():
        return data
    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, data)
    merged: List = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def gather_dict_items(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not is_dist_avail_and_initialized():
        return data
    items = list(data.items())
    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, items)
    merged: Dict[str, torch.Tensor] = {}
    for part in gathered:
        if part:
            for vid, tensor in part:
                if vid not in merged:
                    merged[vid] = tensor
    return merged


# --------------------
# Argument parsing
# --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRVR training / evaluation driver.", conflict_handler="resolve")
    parser.add_argument("--mode", choices=["train", "eval_hdf5", "eval_raw"], required=True, help="Execution mode.")
    # eval args
    parser.add_argument("--video_embed_path", type=str, help="Path to video embedding HDF5 file.")
    parser.add_argument("--text_embed_path", type=str, help="Path to text embedding HDF5 file.")
    # train args
    parser.add_argument("--raw_video_dir", type=str, help="Directory with concatenated MP4 videos.")
    parser.add_argument("--jsonl_train_path", type=str, help="JSONL path for training queries.")
    parser.add_argument("--jsonl_val_path", type=str, help="JSONL path for validation queries.")
    parser.add_argument("--internvideo_model_path", type=str, help="Root path of InternVideo2 repo.")
    parser.add_argument("--output_dir", type=str, help="Directory to store checkpoints.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a trained checkpoint (eval_raw or manual load).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--video_clip_bsz", type=int, default=32, help="Sub-batch size for clips per video.")
    parser.add_argument("--lr_vit", type=float, default=3e-5, help="Learning rate for ViT (video encoder).")
    parser.add_argument("--lr_bert", type=float, default=1e-5, help="Learning rate for BERT (text encoder).")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for scheduler.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience based on SumR.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--max_txt_len", type=int, default=48, help="Tokenizer max length.")
    parser.add_argument("--clip_len", type=int, default=8, help="Frames per clip.")
    parser.add_argument("--clip_stride", type=int, default=4, help="Stride between consecutive clips.")
    parser.add_argument("--sample_fps", type=float, default=1.0, help="Sampling rate in FPS.")
    parser.add_argument("--resize", type=int, default=224, help="Spatial resize for frames.")
    parser.add_argument("--device", type=str, default=None, help="Override torch device, e.g. cuda:0.")
    parser.add_argument("--eval_log_interval", type=int, default=0, help="Print interim Recall every N queries (0=off).")
    parser.add_argument("--ranking_output", type=str, default=None, help="Directory or file path to dump top-k retrieval rankings.")
    parser.add_argument("--topk_dump", type=int, default=10, help="How many retrieved video IDs to store per query.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Computation precision for InternVideo2.")
    parser.add_argument("--video_clip_bsz", type=int, default=32, help="Sub-batch size for clip encoding per video.")
    parser.add_argument("--max_clips_train", type=int, default=0, help="Max clips per video during training (0=off).")
    parser.add_argument("--max_clips_eval", type=int, default=0, help="Max clips per video during eval (0=off).")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend.")
    return parser.parse_args()


# --------------------
# Video decoding utils
# --------------------

def _frame_to_tensor(frame: np.ndarray, resize: int) -> torch.Tensor:
    """Convert HWC uint8 frame to normalized CHW tensor."""
    img = Image.fromarray(frame)
    if resize:
        img = img.resize((resize, resize), Image.BILINEAR)
    arr = torch.from_numpy(np.asarray(img).copy()).float() / 255.0  # HWC
    arr = arr.permute(2, 0, 1)  # CHW
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (arr - mean) / std


def extract_clips_from_video(
    video_path: Path,
    target_fps: float,
    clip_len: int,
    stride: int,
    resize: int,
) -> torch.Tensor:
    """Decode video, sample at 1 FPS, and return [num_clips, clip_len, 3, H, W]."""
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    try:
        reader = iio.imiter(video_path, plugin="ffmpeg")
        metadata = iio.immeta(video_path, plugin="ffmpeg")
    except ValueError:
        reader = iio.imiter(video_path)
        metadata = iio.immeta(video_path)
    native_fps = metadata.get("fps", 30.0) or 30.0
    step = max(int(round(native_fps / max(target_fps, 1e-6))), 1)

    frames: List[torch.Tensor] = []
    for idx, frame in enumerate(reader):
        if idx % step == 0:
            frames.append(_frame_to_tensor(frame, resize))
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    # Ensure at least clip_len frames by padding repeats of last frame.
    while len(frames) < clip_len:
        frames.append(frames[-1].clone())

    frame_tensor = torch.stack(frames)  # [num_frames, 3, H, W]
    clip_tensors: List[torch.Tensor] = []
    for start in range(0, frame_tensor.shape[0] - clip_len + 1, stride):
        clip = frame_tensor[start : start + clip_len]
        clip_tensors.append(clip)
    if not clip_tensors:
        clip_tensors.append(frame_tensor[:clip_len])

    clips = torch.stack(clip_tensors)  # [num_clips, clip_len, 3, H, W]
    return clips


# --------------------
# Dataset
# --------------------

class QVHighlightsDataset(Dataset):
    """Flattens (video_id, [queries]) pairs into (video_id, query) entries."""

    def __init__(
        self,
        jsonl_path: Path,
        video_dir: Path,
        tokenizer,
        clip_len: int,
        stride: int,
        target_fps: float,
        resize: int,
        max_txt_len: int,
        cache_video: bool = True,
        missing_log_path: Optional[Path] = None,
    ) -> None:
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.clip_len = clip_len
        self.stride = stride
        self.target_fps = target_fps
        self.resize = resize
        self.max_txt_len = max_txt_len
        self.cache_video = cache_video
        self.video_cache: Dict[str, torch.Tensor] = {}
        self.missing_log_path = missing_log_path or Path("in_training_missing_videos.txt")
        self._missing_logged: Set[str] = set()
        self._max_clips_cap: int = 0
        self._is_train: bool = False
        self.samples, self.unique_video_ids = self._load_pairs(jsonl_path, video_dir)

    def _log_missing_video(self, video_id: str) -> None:
        if video_id in self._missing_logged or self.missing_log_path is None:
            return
        self._missing_logged.add(video_id)
        if not is_main_process():
            return
        self.missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.missing_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{video_id}\n")
        print(f"[WARN] Missing video file for {video_id}; skipping sample (logged to {self.missing_log_path}).")

    def _load_pairs(self, jsonl_path: Path, video_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
        pairs: List[Tuple[str, str]] = []
        videos: set[str] = set()
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                video_id = record["video_id"]
                # ensure video file exists; otherwise skip and log
                video_path = video_dir / f"{video_id}.mp4"
                if not video_path.exists():
                    self._log_missing_video(video_id)
                    continue
                videos.add(video_id)
                for q in record["queries"]:
                    pairs.append((video_id, q))
        if not pairs:
            raise RuntimeError(f"No samples found in {jsonl_path}")
        return pairs, sorted(videos)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_id: str) -> torch.Tensor:
        if self.cache_video and video_id in self.video_cache:
            return self.video_cache[video_id].clone()
        video_path = self.video_dir / f"{video_id}.mp4"
        clips = extract_clips_from_video(
            video_path,
            target_fps=self.target_fps,
            clip_len=self.clip_len,
            stride=self.stride,
            resize=self.resize,
        )
        if self._max_clips_cap > 0 and clips.shape[0] > self._max_clips_cap:
            num_clips = clips.shape[0]
            if self._is_train:
                idx = torch.randperm(num_clips)[: self._max_clips_cap]
                idx, _ = torch.sort(idx)
            else:
                positions = torch.linspace(0, num_clips - 1, steps=self._max_clips_cap)
                idx = positions.round().to(dtype=torch.long)
            clips = clips[idx]
        if self.cache_video:
            self.video_cache[video_id] = clips
        return clips.clone()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_id, query = self.samples[index]
        clips = self._load_video(video_id)
        tokenized = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        sample = {
            "video_id": video_id,
            "query_text": query,
            "video_clips": clips,
        }
        for key, value in tokenized.items():
            sample[key] = value.squeeze(0)
        return sample


class PRVRTrainWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        clip_sub_bsz: int,
        amp_dtype: torch.dtype,
        amp_enabled: bool,
        use_grad_scaler: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.clip_sub_bsz = clip_sub_bsz
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler and torch.cuda.is_available())

    def set_device(self, device: torch.device) -> None:
        self.device = device

    def forward(self, video_clips, text_inputs):
        clip_feats, _ = encode_videos(
            self.backbone,
            video_clips,
            self.device,
            self.clip_sub_bsz,
            self.amp_dtype,
            self.amp_enabled,
        )
        text_embs = encode_queries(
            self.backbone,
            text_inputs,
            self.device,
            self.amp_dtype,
            self.amp_enabled,
        )
        return clip_feats, text_embs


def prvr_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate to keep variable-length clip tensors."""
    keys = [k for k in batch[0].keys() if k not in {"video_clips", "video_id", "query_text"}]
    text_batch = {k: torch.stack([sample[k] for sample in batch], dim=0) for k in keys}
    return {
        "video_clips": [sample["video_clips"] for sample in batch],
        "text_inputs": text_batch,
        "video_ids": [sample["video_id"] for sample in batch],
        "queries": [sample["query_text"] for sample in batch],
    }


# --------------------
# InternVideo2 helpers
# --------------------

def bootstrap_internvideo_repo(repo_root: Path) -> None:
    """Add InternVideo2 repo to sys.path exactly once."""
    repo_root = repo_root.resolve()
    multi_modality = repo_root / "multi_modality"
    if not multi_modality.exists():
        raise FileNotFoundError(f"multi_modality/ not found under {repo_root}")
    paths = [str(repo_root), str(multi_modality)]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def build_internvideo_config(
    repo_root: Path,
    num_frames: int,
    max_txt_len: int,
    device: torch.device,
    precision: str = "fp32",
) -> ConfigNamespace:
    """Construct a minimal config namespace required by InternVideo2_Stage2."""
    ckpt_path = repo_root / "ckpt" / "InternVideo2-stage2_1b-224p-f4.pt"
    bert_path = repo_root / "bert-large-uncased"
    vision_cfg = {
        "name": "pretrain_internvideo2_1b_patch14_224",
        "img_size": 224,
        "num_frames": num_frames,
        "tubelet_size": 1,
        "patch_size": 14,
        "d_model": 1408,
        "clip_embed_dim": 768,
        "clip_teacher_embed_dim": 3200,
        "clip_teacher_final_dim": 768,
        "clip_norm_type": "l2",
        "clip_return_layer": 6,
        "clip_student_return_interval": 1,
        "pretrained": str(ckpt_path),
        "use_checkpoint": False,
        "checkpoint_num": 0,
        "use_flash_attn": False,
        "use_fused_rmsnorm": False,
        "use_fused_mlp": False,
        "clip_input_resolution": 224,
        "clip_teacher": None,
        "clip_teacher_return_interval": 1,
        "video_mask_type": "random",
        "video_mask_ratio": 0.0,
        "image_mask_type": "random",
        "image_mask_ratio": 0.0,
        "sep_image_video_pos_embed": True,
        "keep_temporal": False,
        "only_mask": False,
    }
    text_cfg = {
        "name": "bert_large",
        "pretrained": str(bert_path),
        "config": "configs/config_bert_large.json",
        "d_model": 1024,
        "fusion_layer": 19,
    }
    model_cfg = {
        "model_cls": "InternVideo2_Stage2",
        "vision_encoder": vision_cfg,
        "text_encoder": text_cfg,
        "multimodal": {"enable": True},
        "embed_dim": 512,
        "temp": 0.07,
        "find_unused_parameters": False,
    }
    cfg_dict = {
        "model": model_cfg,
        "gradient_checkpointing": False,
        "use_half_precision": precision == "fp16",
        "use_bf16": precision == "bf16",
        "compile_model": False,
        "max_txt_l": max_txt_len,
        "device": str(device),
        "pretrained_path": str(ckpt_path),
    }
    return to_config_namespace(cfg_dict)


def load_internvideo_model(
    repo_root: Path,
    num_frames: int,
    max_txt_len: int,
    device: torch.device,
    precision: str = "fp32",
):
    """Instantiate InternVideo2_Stage2 and tokenizer."""
    bootstrap_internvideo_repo(repo_root)
    from multi_modality.demo.utils import InternVideo2_Stage2  # type: ignore
    from models.backbones.bert.tokenization_bert import BertTokenizer  # type: ignore

    config = build_internvideo_config(
        repo_root,
        num_frames=num_frames,
        max_txt_len=max_txt_len,
        device=device,
        precision=precision,
    )
    tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained, local_files_only=True)
    model = InternVideo2_Stage2(config=config, tokenizer=tokenizer, is_pretrain=False)
    return model, tokenizer


def apply_finetune_policy(model: nn.Module) -> None:
    """Freeze early layers following the specification."""
    # Video encoder: freeze first 36 transformer blocks
    if hasattr(model, "vision_encoder") and hasattr(model.vision_encoder, "blocks"):
        for idx, block in enumerate(model.vision_encoder.blocks):
            requires_grad = idx >= 36
            for param in block.parameters():
                param.requires_grad = requires_grad
        # Freeze other vision encoder components
        for name, param in model.vision_encoder.named_parameters():
            if "blocks" not in name:
                param.requires_grad = False
    # Allow trainable projection layers
    if hasattr(model, "vision_proj"):
        for param in model.vision_proj.parameters():
            param.requires_grad = True
    # Text encoder: freeze first 12 layers
    base_text = model.get_text_encoder() if hasattr(model, "get_text_encoder") else None
    if base_text is not None and hasattr(base_text, "encoder"):
        for idx, layer in enumerate(base_text.encoder.layer):
            requires_grad = idx >= 12
            for param in layer.parameters():
                param.requires_grad = requires_grad
        if hasattr(base_text, "embeddings"):
            for param in base_text.embeddings.parameters():
                param.requires_grad = False
    if hasattr(model, "text_proj"):
        for param in model.text_proj.parameters():
            param.requires_grad = True
    # Always enable gradients on temperature parameter if it exists
    if hasattr(model, "temp"):
        model.temp.requires_grad = True


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    load_msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    if load_msg.missing_keys:
        print(f"  Missing keys: {len(load_msg.missing_keys)} (showing first 10) {load_msg.missing_keys[:10]}")
    if load_msg.unexpected_keys:
        print(f"  Unexpected keys: {len(load_msg.unexpected_keys)} (showing first 10) {load_msg.unexpected_keys[:10]}")


def collect_param_groups(model: nn.Module, lr_vit: float, lr_bert: float) -> List[Dict]:
    """Create optimizer parameter groups for video/text encoders."""
    vision_params, text_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vision_encoder") or name.startswith("vision_proj"):
            vision_params.append(param)
        elif name.startswith("text_encoder") or name.startswith("text_proj"):
            text_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if vision_params:
        param_groups.append({"params": vision_params, "lr": lr_vit})
    if text_params:
        param_groups.append({"params": text_params, "lr": lr_bert})
    if other_params:
        param_groups.append({"params": other_params, "lr": max(lr_vit, lr_bert)})
    return param_groups


# --------------------
# Training / evaluation helpers
# --------------------

def _to_namespace(batch: Dict[str, torch.Tensor], device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(**{k: v.to(device) for k, v in batch.items()})


def encode_videos(
    model,
    clip_tensors: Sequence[torch.Tensor],
    device: torch.device,
    clip_sub_bsz: int,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> Tuple[List[torch.Tensor], List[int]]:
    out_per_video: List[torch.Tensor] = []
    counts: List[int] = []
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    use_autocast = amp_enabled and device.type == "cuda"
    for clips in clip_tensors:
        feats_this_video: List[torch.Tensor] = []
        num = clips.shape[0]
        for start in range(0, num, clip_sub_bsz):
            chunk = clips[start : start + clip_sub_bsz].to(device, non_blocking=True)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=use_autocast):
                _, pooled = model.encode_vision(chunk, test=True)
                pooled = model.vision_proj(pooled)
            feats_this_video.append(F.normalize(pooled, dim=-1))
            del chunk, pooled
            if device.type == "cuda":
                torch.cuda.empty_cache()
        video_feat = torch.cat(feats_this_video, dim=0) if feats_this_video else torch.empty(0, model.vision_proj.out_features, device=device)
        out_per_video.append(video_feat)
        counts.append(video_feat.shape[0])
    return out_per_video, counts


def encode_queries(
    model,
    text_inputs: Dict[str, torch.Tensor],
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> torch.Tensor:
    namespace = _to_namespace(text_inputs, device)
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    use_autocast = amp_enabled and device.type == "cuda"
    with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=use_autocast):
        _, pooled = model.encode_text(namespace)
        pooled = model.text_proj(pooled)
    return F.normalize(pooled, dim=-1)


def compute_similarity_matrix(
    query_embs: torch.Tensor,
    video_embs: Sequence[torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    bsz = query_embs.shape[0]
    sim = torch.zeros(bsz, bsz, device=query_embs.device)
    for i in range(bsz):
        for j in range(bsz):
            scores = torch.matmul(video_embs[j], query_embs[i])
            sim[i, j] = scores.max()
    temp = temperature.clamp(min=1e-6).exp()
    return sim / temp


def mil_infonce_loss(sim_matrix: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
    loss_i2t = F.cross_entropy(sim_matrix, labels)
    loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def train_one_epoch(
    model: nn.Module,
    optimizer,
    scheduler,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    module = unwrap_model(model)
    backbone = module.backbone
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = module.amp_dtype
    amp_enabled = module.amp_enabled and device.type == "cuda"
    scaler = module.scaler if hasattr(module, "scaler") else None
    model.train()
    progress = tqdm(dataloader, desc="Train", leave=False, disable=not is_main_process())
    total_loss = 0.0
    steps = 0
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            clip_feats, text_embs = model(batch["video_clips"], batch["text_inputs"])
            sim_matrix = compute_similarity_matrix(text_embs, clip_feats, backbone.temp)
            loss = mil_infonce_loss(sim_matrix)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        steps += 1
    loss_tensor = torch.tensor([total_loss, steps], device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(loss_tensor)
    avg_loss = (loss_tensor[0] / loss_tensor[1]).item() if loss_tensor[1] > 0 else 0.0
    return avg_loss


@torch.no_grad()
def extract_embeddings(
    model,
    dataloader: DataLoader,
    device: torch.device,
    clip_sub_bsz: int,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> Tuple[List[torch.Tensor], List[str], Dict[str, torch.Tensor], List[str]]:
    model.eval()
    query_embs: List[torch.Tensor] = []
    query_targets: List[str] = []
    video_cache: Dict[str, torch.Tensor] = {}
    query_texts: List[str] = []
    iterator = tqdm(dataloader, desc="Val", leave=False, disable=not is_main_process())
    for batch in iterator:
        clip_feats, _ = encode_videos(model, batch["video_clips"], device, clip_sub_bsz, amp_dtype, amp_enabled)
        text_embs = encode_queries(model, batch["text_inputs"], device, amp_dtype, amp_enabled)
        for vid, clip_feat in zip(batch["video_ids"], clip_feats):
            if vid not in video_cache:
                video_cache[vid] = clip_feat.detach().cpu()
        for idx, vid in enumerate(batch["video_ids"]):
            query_embs.append(text_embs[idx].detach().cpu())
            query_targets.append(vid)
            query_texts.append(batch["queries"][idx])
    return query_embs, query_targets, video_cache, query_texts


def compute_metrics_from_embeddings(
    query_embs: Sequence[torch.Tensor],
    query_targets: Sequence[str],
    video_embs: Dict[str, torch.Tensor],
    log_interval: int = 0,
    tag: str = "eval",
    queries: Sequence[str] | None = None,
    ranking_path: Path | None = None,
    topk_dump: int = 10,
) -> RetrievalMetrics:
    video_items = list(video_embs.items())
    scores_per_query: List[int] = []
    total_queries = len(query_targets)

    def _recall(ranks: Sequence[int], k: int) -> float:
        hits = sum(rank <= k for rank in ranks)
        return (hits / max(len(ranks), 1)) * 100.0

    ranking_lines: List[str] = []

    for idx, (query_vec, target_vid) in enumerate(zip(query_embs, query_targets), start=1):
        sims = []
        for vid, clip_feats in video_items:
            score = torch.matmul(clip_feats, query_vec).max().item()
            sims.append((vid, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        rank = next((rank_idx + 1 for rank_idx, (vid, _) in enumerate(sims) if vid == target_vid), len(sims) + 1)
        scores_per_query.append(rank)
        if ranking_path is not None and topk_dump > 0:
            label = queries[idx - 1] if queries else f"{target_vid}#q{idx:04d}"
            predicted = [vid for vid, _ in sims[:topk_dump]]
            line = (
                f"Query[{idx}]={label}\tGT={target_vid}\tRank={rank}\tTop{topk_dump}="
                + ",".join(predicted)
            )
            ranking_lines.append(line)
        if log_interval > 0 and idx % log_interval == 0:
            r1 = _recall(scores_per_query, 1)
            r5 = _recall(scores_per_query, 5)
            r10 = _recall(scores_per_query, 10)
            r100 = _recall(scores_per_query, 100)
            print(
                f"[{tag}] processed {idx}/{total_queries} queries -> "
                f"R@1 {r1:.2f} | R@5 {r5:.2f} | R@10 {r10:.2f} | R@100 {r100:.2f}"
            )

    if ranking_path is not None and ranking_lines:
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        with ranking_path.open("w", encoding="utf-8") as f:
            for line in ranking_lines:
                f.write(line + "\n")
        print(f"[{tag}] Saved rankings to {ranking_path}")

    return RetrievalMetrics(
        recall_at_1=_recall(scores_per_query, 1),
        recall_at_5=_recall(scores_per_query, 5),
        recall_at_10=_recall(scores_per_query, 10),
        recall_at_100=_recall(scores_per_query, 100),
    )


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    log_interval: int = 0,
    ranking_path: Path | None = None,
    topk_dump: int = 10,
    clip_sub_bsz: int = 32,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
) -> RetrievalMetrics:
    query_embs, query_targets, video_embs, query_texts = extract_embeddings(
        model,
        dataloader,
        device,
        clip_sub_bsz,
        amp_dtype,
        amp_enabled and device.type == "cuda",
    )
    if is_dist_avail_and_initialized():
        query_embs = gather_list(query_embs)
        query_targets = gather_list(query_targets)
        query_texts = gather_list(query_texts)
        video_embs = gather_dict_items(video_embs)
    if is_main_process():
        print(f"[val] evaluating on {len(video_embs)} videos / {len(query_embs)} queries")
    else:
        ranking_path = None
    metrics = compute_metrics_from_embeddings(
        query_embs,
        query_targets,
        video_embs,
        queries=query_texts,
        log_interval=log_interval if is_main_process() else 0,
        tag="val",
        ranking_path=ranking_path,
        topk_dump=topk_dump,
    )
    if is_dist_avail_and_initialized():
        stats = torch.tensor(
            [metrics.recall_at_1, metrics.recall_at_5, metrics.recall_at_10, metrics.recall_at_100],
            device=device,
        )
        dist.broadcast(stats, src=0)
        if not is_main_process():
            return RetrievalMetrics(*stats.cpu().tolist())
    return metrics


# --------------------
# HDF5 inference path
# --------------------

class HDF5RetrievalEvaluator:
    """Load embeddings from HDF5 and compute retrieval metrics."""

    def __init__(self, video_path: Path, text_path: Path) -> None:
        if h5py is None:
            raise ImportError("h5py is required for eval_hdf5 mode. Please install it first.")
        self.video_path = video_path
        self.text_path = text_path

    def _decode(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _load_split_ids(self, file_handle, split: str) -> List[str]:
        node = file_handle.get(f"splits/{split}")
        if node is None:
            raise KeyError(f"split '{split}' not found in {file_handle.filename}")
        if isinstance(node, h5py.Dataset):
            raw = node[()]
            if raw.ndim == 1:
                return [self._decode(x) for x in raw]
            raise ValueError(f"Unexpected split dataset shape: {raw.shape}")
        if isinstance(node, h5py.Group):
            return [self._decode(k) for k in node.keys()]
        raise TypeError("Unsupported split node type")

    def _load_video_embeddings(self, split_ids: Sequence[str]) -> Dict[str, torch.Tensor]:
        video_embs: Dict[str, torch.Tensor] = {}
        with h5py.File(self.video_path, "r") as hf:
            for vid in split_ids:
                path = f"videos/{vid}/clip_embeddings"
                if path not in hf:
                    print(f"[WARN] video embeddings for {vid} not found; skipping.")
                    continue
                dataset = hf[path][()]
                tensor = torch.from_numpy(dataset).float()
                tensor = F.normalize(tensor, dim=-1)
                video_embs[vid] = tensor
        return video_embs

    def _load_query_embeddings(
        self,
        split_ids: Sequence[str],
        valid_ids: Sequence[str] | None = None,
    ) -> Tuple[List[torch.Tensor], List[str], List[str]]:
        queries: List[torch.Tensor] = []
        targets: List[str] = []
        labels: List[str] = []
        allow = set(valid_ids) if valid_ids is not None else None
        with h5py.File(self.text_path, "r") as hf:
            for vid in split_ids:
                if allow is not None and vid not in allow:
                    continue
                path = f"queries/{vid}/embeddings"
                if path not in hf:
                    print(f"[WARN] text embeddings for {vid} not found; skipping queries.")
                    continue
                dataset = hf[path][()]
                tensor = torch.from_numpy(dataset).float()
                tensor = F.normalize(tensor, dim=-1)
                for row_idx, row in enumerate(tensor):
                    queries.append(row)
                    targets.append(vid)
                    labels.append(f"{vid}#q{row_idx}")
        return queries, targets, labels

    def evaluate(
        self,
        split: str = "val",
        log_interval: int = 0,
        ranking_path: Path | None = None,
        topk_dump: int = 10,
    ) -> RetrievalMetrics:
        with h5py.File(self.video_path, "r") as video_hf:
            split_ids = self._load_split_ids(video_hf, split)
        video_embs = self._load_video_embeddings(split_ids)
        if not video_embs:
            raise RuntimeError("No video embeddings loaded for requested split.")
        query_embs, targets, labels = self._load_query_embeddings(split_ids, valid_ids=video_embs.keys())
        if not query_embs:
            raise RuntimeError("No query embeddings aligned with available videos.")
        print(f"[hdf5-{split}] evaluating on {len(video_embs)} videos / {len(query_embs)} queries")
        return compute_metrics_from_embeddings(
            query_embs,
            targets,
            video_embs,
            queries=labels,
            log_interval=log_interval,
            tag=f"hdf5-{split}",
            ranking_path=ranking_path,
            topk_dump=topk_dump,
        )


# --------------------
# Training orchestration
# --------------------

def run_training(args: argparse.Namespace) -> None:
    required = [
        args.raw_video_dir,
        args.jsonl_train_path,
        args.jsonl_val_path,
        args.internvideo_model_path,
        args.output_dir,
    ]
    if any(path is None for path in required):
        raise ValueError("Train mode requires video dir, jsonl paths, internvideo path, and output_dir.")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(args.internvideo_model_path)
    backbone, tokenizer = load_internvideo_model(
        repo_root=repo_root,
        num_frames=args.clip_len,
        max_txt_len=args.max_txt_len,
        device=device,
        precision=args.precision,
    )
    apply_finetune_policy(backbone)
    amp_dtype = get_amp_dtype(args.precision)
    amp_enabled = amp_dtype != torch.float32 and device.type == "cuda"
    backbone = backbone.to(device=device)

    wrapper = PRVRTrainWrapper(
        backbone,
        device,
        args.video_clip_bsz,
        amp_dtype,
        amp_enabled,
        use_grad_scaler=amp_enabled and amp_dtype == torch.float16,
    )
    wrapper = wrapper.to(device)
    wrapper.set_device(device)
    if args.distributed:
        wrapper = torch.nn.parallel.DistributedDataParallel(
            wrapper,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    missing_log_path = Path("in_training_missing_videos.txt")
    train_ds = QVHighlightsDataset(
        jsonl_path=Path(args.jsonl_train_path),
        video_dir=Path(args.raw_video_dir),
        tokenizer=tokenizer,
        clip_len=args.clip_len,
        stride=args.clip_stride,
        target_fps=args.sample_fps,
        resize=args.resize,
        max_txt_len=args.max_txt_len,
        missing_log_path=missing_log_path,
    )
    train_ds._is_train = True
    train_ds._max_clips_cap = max(0, args.max_clips_train)
    if is_main_process():
        print(f"Train set: {len(train_ds.unique_video_ids)} videos / {len(train_ds)} queries")
    val_ds = QVHighlightsDataset(
        jsonl_path=Path(args.jsonl_val_path),
        video_dir=Path(args.raw_video_dir),
        tokenizer=tokenizer,
        clip_len=args.clip_len,
        stride=args.clip_stride,
        target_fps=args.sample_fps,
        resize=args.resize,
        max_txt_len=args.max_txt_len,
        missing_log_path=missing_log_path,
    )
    val_ds._is_train = False
    val_ds._max_clips_cap = max(0, args.max_clips_eval)
    if is_main_process():
        print(f"Val set: {len(val_ds.unique_video_ids)} videos / {len(val_ds)} queries")

    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False) if args.distributed else None

    train_loader_kwargs = dict(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=prvr_collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader_kwargs = dict(
        dataset=val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=prvr_collate_fn,
        drop_last=False,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    param_groups = collect_param_groups(backbone, lr_vit=args.lr_vit, lr_bert=args.lr_bert)
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)
    total_steps = args.epochs * max(len(train_loader), 1)

    def lr_lambda(step: int) -> float:
        if total_steps == 0:
            return 1.0
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda) if total_steps > 0 else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_sumr = -float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process():
            print(f"Epoch {epoch}/{args.epochs}")
        avg_loss = train_one_epoch(wrapper, optimizer, scheduler, train_loader, device)
        if is_main_process():
            print(f"  Avg loss: {avg_loss:.4f}")
        ranking_path = None
        if args.ranking_output:
            base = Path(args.ranking_output)
            if base.suffix:
                ranking_path = base.with_name(f"{base.stem}_epoch{epoch:02d}{base.suffix}")
            else:
                ranking_path = base / f"val_epoch{epoch:02d}.txt"
        if not is_main_process():
            ranking_path = None
        metrics = evaluate_model(
            backbone,
            val_loader,
            device,
            log_interval=args.eval_log_interval,
            ranking_path=ranking_path,
            topk_dump=args.topk_dump,
            clip_sub_bsz=args.video_clip_bsz,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        if is_main_process():
            print(
                f"  Val R@1 {metrics.recall_at_1:.2f} "
                f"R@5 {metrics.recall_at_5:.2f} "
                f"R@10 {metrics.recall_at_10:.2f} "
                f"R@100 {metrics.recall_at_100:.2f} "
                f"SumR {metrics.sum_recall:.2f}"
            )
        if metrics.sum_recall > best_sumr and is_main_process():
            best_sumr = metrics.sum_recall
            epochs_no_improve = 0
            ckpt_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": backbone.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": metrics.__dict__,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint to {ckpt_path}")
        elif is_main_process():
            epochs_no_improve += 1
        meta = torch.tensor([best_sumr, epochs_no_improve], device=device)
        if is_dist_avail_and_initialized():
            dist.broadcast(meta, src=0)
        best_sumr = meta[0].item()
        epochs_no_improve = int(meta[1].item())
        stop_flag = torch.tensor(
            [1 if (is_main_process() and epochs_no_improve >= args.patience) else 0],
            device=device,
        )
        if is_dist_avail_and_initialized():
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            if is_main_process():
                print("Early stopping triggered.")
            break


def run_eval_hdf5(args: argparse.Namespace) -> None:
    if not args.video_embed_path or not args.text_embed_path:
        raise ValueError("--video_embed_path and --text_embed_path are required for eval_hdf5.")
    ranking_path = resolve_ranking_path(args.ranking_output, "eval_hdf5.txt")
    evaluator = HDF5RetrievalEvaluator(
        video_path=Path(args.video_embed_path),
        text_path=Path(args.text_embed_path),
    )
    metrics = evaluator.evaluate(
        split="val",
        log_interval=args.eval_log_interval,
        ranking_path=ranking_path,
        topk_dump=args.topk_dump,
    )
    print(f"R@1={metrics.recall_at_1:.2f} R@5={metrics.recall_at_5:.2f} R@10={metrics.recall_at_10:.2f} "
          f"R@100={metrics.recall_at_100:.2f} SumR={metrics.sum_recall:.2f}")


def run_eval_raw(args: argparse.Namespace) -> None:
    required = [
        args.raw_video_dir,
        args.jsonl_val_path,
        args.internvideo_model_path,
        args.checkpoint_path,
    ]
    if any(path is None for path in required):
        raise ValueError("eval_raw requires raw_video_dir, jsonl_val_path, internvideo_model_path, and checkpoint_path.")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(args.internvideo_model_path)
    model, tokenizer = load_internvideo_model(
        repo_root=repo_root,
        num_frames=args.clip_len,
        max_txt_len=args.max_txt_len,
        device=device,
        precision=args.precision,
    )
    apply_finetune_policy(model)
    amp_dtype = get_amp_dtype(args.precision)
    amp_enabled = amp_dtype != torch.float32 and device.type == "cuda"
    ckpt_path = Path(args.checkpoint_path)
    if ckpt_path.is_file():
        load_checkpoint_weights(model, ckpt_path, device=device)
    else:
        if is_main_process():
            print(f"[eval_raw] Checkpoint not found at {ckpt_path}. Using zeroshot weights.")
    model = model.to(device=device, memory_format=torch.channels_last)
    model.eval()

    val_ds = QVHighlightsDataset(
        jsonl_path=Path(args.jsonl_val_path),
        video_dir=Path(args.raw_video_dir),
        tokenizer=tokenizer,
        clip_len=args.clip_len,
        stride=args.clip_stride,
        target_fps=args.sample_fps,
        resize=args.resize,
        max_txt_len=args.max_txt_len,
        missing_log_path=Path("in_training_missing_videos.txt"),
    )
    val_ds._is_train = False
    val_ds._max_clips_cap = max(0, args.max_clips_eval)
    if is_main_process():
        print(f"[eval_raw] Dataset: {len(val_ds.unique_video_ids)} videos / {len(val_ds)} queries")
    val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False) if args.distributed else None
    val_loader_kwargs = dict(
        dataset=val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=prvr_collate_fn,
        drop_last=False,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2
    val_loader = DataLoader(**val_loader_kwargs)
    ranking_path = resolve_ranking_path(args.ranking_output, "eval_raw.txt")
    metrics = evaluate_model(
        model,
        val_loader,
        device,
        log_interval=args.eval_log_interval,
        ranking_path=ranking_path,
        topk_dump=args.topk_dump,
        clip_sub_bsz=args.video_clip_bsz,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    print(f"EvalRaw R@1={metrics.recall_at_1:.2f} R@5={metrics.recall_at_5:.2f} R@10={metrics.recall_at_10:.2f} "
          f"R@100={metrics.recall_at_100:.2f} SumR={metrics.sum_recall:.2f}")


def main() -> None:
    args = parse_args()
    init_distributed_mode(args)
    try:
        if args.mode == "eval_hdf5":
            run_eval_hdf5(args)
        elif args.mode == "eval_raw":
            run_eval_raw(args)
        elif args.mode == "train":
            run_training(args)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported mode: {args.mode}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
