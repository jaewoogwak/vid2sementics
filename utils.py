from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

matplotlib.use("Agg")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_dataset_samples(name: str, dataset, limit: int = 3) -> None:
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
            getattr(item, "video_path", "N/A"),
            len(item.scene_texts),
            item.scene_texts,
            item.scene_windows,
        )


def log_msrvtt_dataset_samples(name: str, dataset, limit: int = 3) -> None:
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


def log_activitynet_dataset_samples(name: str, dataset, limit: int = 3) -> None:
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


def save_loss_term_plots(
    train_histories: Dict[str, Sequence[Tuple[int, float]]],
    val_histories: Dict[str, Sequence[Tuple[int, float]]],
    output_path: Optional[Path],
) -> Optional[Path]:
    if output_path is None:
        return None
    terms = sorted(
        {
            term
            for term, history in train_histories.items()
            if history
        }.union(
            {
                term
                for term, history in val_histories.items()
                if history
            }
        )
    )
    if not terms:
        return None
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(2, max(1, int(math.ceil(math.sqrt(len(terms))))))
    rows = int(math.ceil(len(terms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
    for idx, term in enumerate(terms):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        train_history = train_histories.get(term, [])
        val_history = val_histories.get(term, [])
        if train_history:
            steps = [step for step, _ in train_history]
            values = [value for _, value in train_history]
            ax.plot(steps, values, label="Train", color="#1f77b4")
        if val_history:
            steps = [step for step, _ in val_history]
            values = [value for _, value in val_history]
            ax.plot(steps, values, label="Validation", color="#d62728", linestyle="--", marker="o")
        ax.set_title(f"{term.capitalize()} loss")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        if train_history or val_history:
            ax.legend()
    # Hide unused subplots
    total_axes = rows * cols
    for idx in range(len(terms), total_axes):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved loss term plots -> %s", output_path)
    return output_path


def trim_eos_scene_predictions(
    scene_latents: torch.Tensor,
    attention_weights: Optional[torch.Tensor],
    eos_token: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if scene_latents.numel() == 0:
        return scene_latents, attention_weights

    eos_vector = eos_token.detach().to(scene_latents.device).unsqueeze(0)
    sims = F.cosine_similarity(scene_latents, eos_vector, dim=-1)

    is_eos = sims >= threshold
    if is_eos.any():
        first_eos_idx = torch.nonzero(is_eos, as_tuple=True)[0][0].item()
        scene_latents = scene_latents[:first_eos_idx]
        if attention_weights is not None and attention_weights.numel() > 0:
            attention_weights = attention_weights[:first_eos_idx]

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
