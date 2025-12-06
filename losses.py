from __future__ import annotations

import logging
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


def scene_query_matrix_loss(
    scene_preds: List[torch.Tensor],
    text_targets: List[torch.Tensor],
    temperature: float = 0.07,
    text_sim_threshold: float = 0.85,
) -> torch.Tensor:
    if not scene_preds or not text_targets:
        return torch.tensor(0.0)

    device = scene_preds[0].device
    total = scene_preds[0].new_tensor(0.0)
    count = 0

    for sp, tt in zip(scene_preds, text_targets):
        if sp.numel() == 0 or tt.numel() == 0:
            continue

        S = F.normalize(sp, dim=-1)
        T = F.normalize(tt, dim=-1)
        n = min(S.size(0), T.size(0))
        if n < 2:
            continue

        S = S[:n]
        T = T[:n]

        sim = S @ T.T / temperature

        with torch.no_grad():
            text_sim = T @ T.T
            eye = torch.eye(n, device=device, dtype=torch.bool)
            invalid_neg = (text_sim > text_sim_threshold) & (~eye)

        row_logits = sim.clone()
        row_logits.masked_fill_(invalid_neg, -1e9)
        row_labels = torch.arange(n, device=device)
        loss_row = F.cross_entropy(row_logits, row_labels)

        col_logits = sim.T.clone()
        col_logits.masked_fill_(invalid_neg.T, -1e9)
        col_labels = torch.arange(n, device=device)
        loss_col = F.cross_entropy(col_logits, col_labels)

        total = total + 0.5 * (loss_row + loss_col)
        count += 1

    if count == 0:
        return total
    return total / count


def scene_diversity_loss_structured(
    scene_preds: List[torch.Tensor],
    text_targets: List[torch.Tensor],
    gamma: float = 1.0,
) -> torch.Tensor:
    total = None
    fallback_device: Optional[torch.device] = None
    count = 0
    for scenes, texts in zip(scene_preds, text_targets):
        if scenes is None or texts is None:
            continue
        if fallback_device is None and isinstance(scenes, torch.Tensor):
            fallback_device = scenes.device
        if scenes.numel() == 0 or texts.numel() == 0:
            continue
        n = min(scenes.shape[0], texts.shape[0])
        if n <= 1:
            continue
        scenes_use = scenes[:n]
        texts_use = texts[:n]
        if total is None:
            total = scenes_use.new_tensor(0.0)
        scene_norm = F.normalize(scenes_use, dim=-1)
        text_norm = F.normalize(texts_use, dim=-1)
        scene_cos = scene_norm @ scene_norm.T
        text_cos = text_norm @ text_norm.T
        off_mask = ~torch.eye(n, dtype=torch.bool, device=scenes_use.device)
        scene_off = scene_cos[off_mask]
        text_off = text_cos[off_mask]
        weights = (1.0 - text_off).clamp_min(0.0)
        if gamma != 1.0:
            weights = weights ** gamma
        diff = (scene_off - text_off) ** 2
        weighted = weights * diff
        denom = weights.sum().clamp_min(1e-6)
        total = total + weighted.sum() / denom
        count += 1
        if fallback_device is None:
            fallback_device = scenes.device
    if count == 0:
        if total is None:
            device = fallback_device or torch.device("cpu")
            return torch.tensor(0.0, device=device)
        return total
    return total / count


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
            pred = attn_weights[b, j, clip_mask]
            pred = pred / pred.sum().clamp_min(1e-6)
            mass = pred[gt_mask].sum()
            total = total - torch.log(mass + 1e-6)
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
