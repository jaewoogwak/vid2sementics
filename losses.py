from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
    return F.cross_entropy(logits, labels)


def compute_batch_all_triplet_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.2,
    text_sim_threshold: float = 0.85,
) -> torch.Tensor:
    if preds.numel() == 0 or targets.numel() == 0:
        return preds.new_tensor(0.0)

    anchor = F.normalize(preds, dim=-1)
    positive = F.normalize(targets, dim=-1)
    sim_matrix = anchor @ positive.T
    pos_sim = sim_matrix.diag()

    with torch.no_grad():
        monitor_sim = sim_matrix.clone()
        d_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        monitor_sim.masked_fill_(d_mask, -1e9)
        tgt_sim = positive @ positive.T
        text_mask = tgt_sim > text_sim_threshold
        monitor_sim.masked_fill_(text_mask, -1e9)
        hard_neg_sim, _ = monitor_sim.max(dim=1)
        valid_neg = hard_neg_sim > -100
        if valid_neg.sum() > 0:
            hard_neg_sim[valid_neg].mean().item()
        pos_sim.mean().item()

    triplet_loss = sim_matrix - pos_sim.unsqueeze(1) + margin
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
    tgt_sim = positive @ positive.T
    invalid_neg_mask = tgt_sim > text_sim_threshold
    mask = mask | invalid_neg_mask
    triplet_loss.masked_fill_(mask, 0.0)
    triplet_loss = F.relu(triplet_loss)
    num_positive_triplets = (triplet_loss > 1e-16).sum().float()
    if num_positive_triplets > 0:
        return triplet_loss.sum() / num_positive_triplets
    return preds.new_tensor(0.0)


def compute_hard_triplet_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    text_sim_threshold: float = 0.85,
) -> torch.Tensor:
    if preds.numel() == 0 or targets.numel() == 0:
        return preds.new_tensor(0.0)

    count = min(preds.shape[0], targets.shape[0])
    if count < 2:
        return preds.new_tensor(0.0)

    preds = preds[:count]
    targets = targets[:count]

    anchor = F.normalize(preds, dim=-1)
    positive = F.normalize(targets, dim=-1)

    sim_matrix = anchor @ positive.T
    tgt_sim_matrix = positive @ positive.T
    invalid_neg_mask = tgt_sim_matrix > text_sim_threshold
    sim_matrix.masked_fill_(invalid_neg_mask, float("-inf"))
    hard_neg_idx = sim_matrix.argmax(dim=1)
    negatives = positive[hard_neg_idx]
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    return criterion(anchor, positive, negatives)


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
