from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


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
        hidden_dim = max(64, embed_dim // 4)
        self.stop_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

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
        max_scenes = max(scene_lengths) if scene_lengths else 0
        target_len = max(1, max_scenes + 1)
        decoder_inputs = torch.zeros(batch_size, target_len, self.embed_dim, device=device)
        decoder_targets = torch.zeros(batch_size, target_len, self.embed_dim, device=device)
        target_padding = torch.ones(batch_size, target_len, dtype=torch.bool, device=device)
        scene_mask = torch.zeros(batch_size, target_len, dtype=torch.bool, device=device)
        for idx, latent in enumerate(latents):
            if latent.ndim != 2 or latent.shape[1] != self.embed_dim:
                raise ValueError("Latent tensors must have shape (num_scenes, embed_dim).")
            if latent.device != device:
                latent = latent.to(device)
            length = latent.shape[0]
            decoder_inputs[idx, 0, :] = self.start_token
            if length > 0:
                take = min(length, target_len - 1)
                decoder_inputs[idx, 1 : take + 1, :] = latent[:take]
                decoder_targets[idx, :take, :] = latent[:take]
                scene_mask[idx, :take] = True
            eos_idx = min(length, target_len - 1)
            decoder_targets[idx, eos_idx, :] = self.eos_token
            valid_tokens = min(target_len, length + 1)
            target_padding[idx, :valid_tokens] = False
        return decoder_inputs, decoder_targets, target_padding, scene_mask, scene_lengths

    def _prepare_memory(
        self,
        clip_embeddings: torch.Tensor,
        clip_times: torch.Tensor,
        clip_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = clip_embeddings.shape
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
        text_embeddings: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del text_embeddings, text_padding_mask
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
        stop_logits = self.stop_head(outputs).squeeze(-1)
        if attn_weights is not None:
            attn_weights = attn_weights.mean(dim=1)
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
            feedback_token = pred
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
