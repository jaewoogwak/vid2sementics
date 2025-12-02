from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


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
    if positions.dtype not in (torch.float32, torch.float64):
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
        norm_tgt = self.norm1(tgt)
        attn_output, _ = self.self_attn(
            norm_tgt,
            norm_tgt,
            norm_tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout(attn_output)

        norm_cross = self.norm2(tgt)
        cross_output, attn_weights = self.cross_attn(
            norm_cross,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = tgt + self.dropout(cross_output)

        norm_ff = self.norm3(tgt)
        ff = self.linear2(F.gelu(self.linear1(norm_ff)))
        tgt = tgt + self.dropout(ff)
        return tgt, attn_weights


def build_exhaustive_clip_bank(
    frame_embeds: torch.Tensor,
    clip_times: torch.Tensor,
    clip_padding_mask: Optional[torch.Tensor],
    *,
    min_window: int = 1,
    max_window: Optional[int] = None,
    pooling: str = "mean",
    stride: int = 2,
):
    """Construct an exhaustive sliding-window clip bank.

    Args:
        frame_embeds: Tensor of shape (B, T, D_clip) containing frame-level embeddings.
        clip_times: Tensor of shape (B, T) with timestamps (seconds or indices) for each frame.
        clip_padding_mask: Bool tensor (B, T) where True marks padded frames.
        min_window: Minimum window length (inclusive).
        max_window: Maximum window length (inclusive). If ``None`` uses T-1 per sequence.
        pooling: Aggregation applied within each window ("mean" only).

    Returns:
        clip_bank: (B, N_clips, D_clip)
        clip_centers: (B, N_clips) mean timestamp for each window.
        clip_bank_mask: (B, N_clips) bool mask where True marks padded tokens.
        window_starts: (B, N_clips) start indices per window.
        window_lengths: (B, N_clips) window lengths per token.
        used_frame_fallback: bool flag, True when returning raw frames (no windows possible).
    """

    if pooling != "mean":
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    if clip_times.ndim != 2:
        raise ValueError("clip_times must have shape (B, T)")
    batch_size, seq_len, feat_dim = frame_embeds.shape
    device = frame_embeds.device
    if clip_padding_mask is None:
        clip_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    valid_lengths = (~clip_padding_mask).sum(dim=1)
    max_valid = int(valid_lengths.max().item()) if batch_size > 0 else 0
    if max_valid == 0:
        max_valid = seq_len
    window_min = max(1, int(min_window))
    half_lengths = torch.clamp(valid_lengths // 2, min=1)
    max_half = int(half_lengths.max().item()) if batch_size > 0 else 1
    if max_window is None:
        window_max = max_valid - 1
    else:
        window_max = min(int(max_window), max_valid)
        window_max = max(window_min, window_max)
    window_max = min(window_max, max_half)
    if window_max < window_min:
        window_max = window_min - 1
    stride = max(1, int(stride))
    specs = []
    total_windows = 0
    length = window_min
    while length <= window_max:
        if seq_len < length:
            break
        count = ((seq_len - length) // stride) + 1
        if count <= 0:
            break
        specs.append((length, count))
        total_windows += count
        length *= 2
    use_fallback = total_windows == 0
    if use_fallback:
        starts = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        lengths = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        return (
            frame_embeds,
            clip_times,
            clip_padding_mask,
            starts,
            lengths,
            True,
        )
    bank_chunks = []
    center_chunks = []
    mask_chunks = []
    start_chunks = []
    length_chunks = []
    transposed = frame_embeds.transpose(1, 2)
    time_tensor = clip_times.unsqueeze(1).to(frame_embeds.dtype)
    for length, count in specs:
        pooled = F.avg_pool1d(transposed, kernel_size=length, stride=stride).transpose(1, 2)
        pooled = pooled[:, :count, :]
        bank_chunks.append(pooled)
        time_avg = F.avg_pool1d(time_tensor, kernel_size=length, stride=stride).squeeze(1)
        time_avg = time_avg[:, :count]
        center_chunks.append(time_avg)
        starts = (
            torch.arange(count, device=device, dtype=torch.long)
            .mul(stride)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        start_chunks.append(starts)
        length_chunks.append(torch.full((batch_size, count), length, dtype=torch.long, device=device))
        valid_counts = ((valid_lengths - length) // stride) + 1
        valid_counts = torch.where(valid_lengths >= length, valid_counts, torch.zeros_like(valid_counts))
        valid_counts = valid_counts.clamp_min(0)
        window_positions = torch.arange(count, device=device).unsqueeze(0)
        mask = window_positions >= valid_counts.unsqueeze(1)
        mask_chunks.append(mask)
    clip_bank = torch.cat(bank_chunks, dim=1)
    clip_centers = torch.cat(center_chunks, dim=1)
    clip_bank_mask = torch.cat(mask_chunks, dim=1)
    clip_bank = clip_bank.masked_fill(clip_bank_mask.unsqueeze(-1), 0.0)
    clip_centers = clip_centers.masked_fill(clip_bank_mask, 0.0)
    window_starts = torch.cat(start_chunks, dim=1)
    window_lengths = torch.cat(length_chunks, dim=1)
    return clip_bank, clip_centers, clip_bank_mask, window_starts, window_lengths, False


class SceneTransformer(nn.Module):
    """Autoregressive latent scene generator."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        *,
        use_exhaustive_clip_bank: bool = False,
        use_text_projection: bool = True,
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
        hidden_dim = max(64, embed_dim // 4)
        self.stop_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.use_exhaustive_clip_bank = bool(use_exhaustive_clip_bank)
        self.use_text_projection = bool(use_text_projection)

    def project_text_embeddings(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        if not self.use_text_projection:
            return text_embeddings
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
        start_token = self.start_token.view(1, -1).to(device)
        eos_token = self.eos_token.view(1, -1).to(device)
        decoder_input_seqs: List[torch.Tensor] = []
        decoder_target_seqs: List[torch.Tensor] = []
        scene_lengths: List[int] = []
        for latent in latents:
            if latent.ndim != 2 or latent.shape[1] != self.embed_dim:
                raise ValueError("Latent tensors must have shape (num_scenes, embed_dim).")
            if latent.device != device:
                latent = latent.to(device)
            length = int(latent.shape[0])
            scene_lengths.append(length)
            decoder_input_seq = torch.cat([start_token, latent], dim=0)
            decoder_target_seq = torch.cat([latent, eos_token], dim=0)
            decoder_input_seqs.append(decoder_input_seq)
            decoder_target_seqs.append(decoder_target_seq)
        decoder_inputs = pad_sequence(decoder_input_seqs, batch_first=True, padding_value=0.0)
        decoder_targets = pad_sequence(decoder_target_seqs, batch_first=True, padding_value=0.0)
        max_len = decoder_targets.shape[1]
        target_padding = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        scene_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        for idx, length in enumerate(scene_lengths):
            valid_tokens = min(max_len, length + 1)
            target_padding[idx, :valid_tokens] = False
            scene_mask[idx, :length] = True
        return decoder_inputs, decoder_targets, target_padding, scene_mask, scene_lengths

    def _prepare_memory(
        self,
        clip_embeddings: torch.Tensor,
        clip_times: torch.Tensor,
        clip_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_exhaustive_clip_bank:
            return self._build_frame_level_memory(clip_embeddings, clip_times, clip_padding_mask)
        (
            clip_bank,
            clip_centers,
            clip_bank_mask,
            window_starts,
            window_lengths,
            used_fallback,
        ) = build_exhaustive_clip_bank(
            clip_embeddings,
            clip_times,
            clip_padding_mask,
        )
        if used_fallback:
            return self._build_frame_level_memory(clip_embeddings, clip_times, clip_padding_mask)
        pos_tensor = self._encode_positions(clip_centers, clip_bank_mask, clip_bank.dtype)
        memory = self.clip_proj(clip_bank) + pos_tensor
        memory = self.clip_norm(self.dropout(memory))
        meta = {
            "start": window_starts,
            "lengths": window_lengths,
            "mask": clip_bank_mask,
            "base_mask": clip_padding_mask,
            "base_len": clip_embeddings.shape[1],
        }
        return memory, clip_bank_mask, meta

    def _build_frame_level_memory(
        self,
        clip_embeddings: torch.Tensor,
        clip_times: torch.Tensor,
        clip_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch, seq_len, _ = clip_embeddings.shape
        pos_list = []
        device = clip_embeddings.device
        dtype = clip_embeddings.dtype
        for i in range(batch):
            valid = ~clip_padding_mask[i]
            times = clip_times[i]
            pos = torch.zeros(seq_len, self.embed_dim, device=device, dtype=dtype)
            if valid.any():
                encoding = sinusoidal_position_encoding(times[valid], self.embed_dim)
                pos[valid] = encoding
            pos_list.append(pos)
        pos_tensor = torch.stack(pos_list, dim=0) if pos_list else torch.empty(0, seq_len, self.embed_dim, device=device)
        memory = self.clip_proj(clip_embeddings) + pos_tensor
        memory = self.clip_norm(self.dropout(memory))
        meta = self._identity_window_metadata(seq_len, clip_padding_mask)
        return memory, clip_padding_mask, meta

    def _identity_window_metadata(
        self,
        seq_len: int,
        clip_padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = clip_padding_mask.device
        batch = clip_padding_mask.shape[0]
        starts = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        lengths = torch.ones(batch, seq_len, dtype=torch.long, device=device)
        return {
            "start": starts,
            "lengths": lengths,
            "mask": clip_padding_mask,
            "base_mask": clip_padding_mask,
            "base_len": seq_len,
        }

    def _encode_positions(
        self,
        clip_centers: torch.Tensor,
        clip_bank_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, tokens = clip_centers.shape
        device = clip_centers.device
        if tokens == 0:
            return torch.empty(batch, 0, self.embed_dim, device=device, dtype=dtype)
        pos_list = []
        for i in range(batch):
            valid = ~clip_bank_mask[i]
            pos = torch.zeros(tokens, self.embed_dim, device=device, dtype=dtype)
            if valid.any():
                encoding = sinusoidal_position_encoding(clip_centers[i, valid], self.embed_dim)
                pos[valid] = encoding
            pos_list.append(pos)
        return torch.stack(pos_list, dim=0)

    def _project_attention_to_frames(
        self,
        attn_clip: torch.Tensor,
        meta: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        base_mask = meta.get("base_mask")
        base_len = int(meta.get("base_len", attn_clip.shape[-1]))
        device = attn_clip.device
        batch, tgt_len, _ = attn_clip.shape
        if base_len == 0:
            return torch.empty(batch, tgt_len, 0, device=device, dtype=attn_clip.dtype)
        if attn_clip.shape[-1] == 0:
            zeros = attn_clip.new_zeros(batch, tgt_len, base_len)
            if base_mask is not None:
                zeros = zeros.masked_fill(base_mask.unsqueeze(1), 0.0)
            return zeros
        projected = attn_clip.new_zeros(batch, tgt_len, base_len)
        frame_positions = torch.arange(base_len, device=device)
        starts = meta["start"]
        lengths = meta["lengths"]
        mask = meta["mask"]
        for b in range(batch):
            valid = ~mask[b]
            if not valid.any():
                continue
            starts_b = starts[b, valid]
            lengths_b = lengths[b, valid]
            attn_b = attn_clip[b, :, valid]
            span_mask = (frame_positions.unsqueeze(0) >= starts_b.unsqueeze(1)) & (
                frame_positions.unsqueeze(0) < (starts_b + lengths_b).unsqueeze(1)
            )
            lengths_float = lengths_b.clamp_min(1).to(attn_clip.dtype)
            span_weights = span_mask.to(attn_clip.dtype) / lengths_float.unsqueeze(1)
            distribution = attn_b.unsqueeze(-1) * span_weights.unsqueeze(0)
            projected[b] = distribution.sum(dim=1)
        if base_mask is not None:
            projected = projected.masked_fill(base_mask.unsqueeze(1), 0.0)
        return projected

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
        text_embeddings: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del text_embeddings, text_padding_mask
        memory, memory_mask, memory_meta = self._prepare_memory(
            clip_embeddings,
            clip_times,
            clip_padding_mask,
        )
        tgt, tgt_mask = self._prepare_targets(decoder_inputs, decoder_padding_mask)
        attn_layers: List[torch.Tensor] = []
        x = tgt
        for layer in self.layers:
            x, attn = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=memory_mask,
            )
            if attn is not None:
                attn_layers.append(attn)
        outputs = self.output_proj(self.output_norm(x))
        stop_logits = self.stop_head(outputs).squeeze(-1)
        attn_weights: Optional[torch.Tensor] = None
        if attn_layers:
            stacked = torch.stack(attn_layers, dim=0)
            attn_clip = stacked.mean(dim=0).mean(dim=1)
            attn_weights = self._project_attention_to_frames(attn_clip, memory_meta)
        return outputs, attn_weights, stop_logits

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
        memory, memory_mask, memory_meta = self._prepare_memory(
            clip_embeddings,
            clip_times,
            clip_padding_mask,
        )
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
                    memory_key_padding_mask=memory_mask,
                )
            pred = self.output_proj(self.output_norm(x[:, -1:, :]))
            pred_latent = pred.squeeze(1)
            generated.append(pred_latent)
            feedback_token = pred
            if attn is not None:
                attn_mean = attn.mean(dim=1)[:, -1, :]
                attn_mean = attn_mean.masked_fill(memory_mask, 0.0)
                frame_attn = self._project_attention_to_frames(attn_mean.unsqueeze(1), memory_meta).squeeze(1)
                attn_sum = frame_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                attn_history.append(frame_attn / attn_sum)
            else:
                base_len = int(memory_meta.get("base_len", clip_embeddings.shape[1]))
                zeros = torch.zeros(batch_size, base_len, device=device)
                attn_history.append(zeros)
            stop_logits = self.stop_head(pred_latent).squeeze(-1)
            stop_prob = torch.sigmoid(stop_logits)
            inputs = torch.cat([inputs, feedback_token], dim=1)
            pad_row = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([padding_mask, pad_row], dim=1)
            if bool((stop_prob > 0.5).all()):
                break
            eos_similarity = F.cosine_similarity(
                pred_latent, self.eos_token.view(1, -1).to(device), dim=-1
            )
            if bool((eos_similarity >= eos_threshold).all()):
                break
        if generated:
            return torch.stack(generated, dim=1), torch.stack(attn_history, dim=1)
        empty_embeds = torch.empty(clip_embeddings.shape[0], 0, self.embed_dim, device=device)
        empty_attn = torch.empty(clip_embeddings.shape[0], 0, clip_embeddings.shape[1], device=device)
        return empty_embeds, empty_attn
