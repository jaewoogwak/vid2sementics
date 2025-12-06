from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import (
    ActivityNetSceneDataset,
    ActivityNetSceneItem,
    MSRVTTSceneItem,
    MSRVTTUntrimmedDataset,
    QVHighlightsDataset,
    TVRSceneDataset,
    TVRSceneItem,
    build_video_clips,
    collate_scene_batch,
    load_text_cache,
    load_video_cache,
    sample_video_frames,
    save_text_cache,
    save_video_cache,
)
from losses import (
    attention_supervision_loss,
    representation_alignment_loss,
    scene_diversity_loss_structured,
    scene_query_matrix_loss,
)
from models import (
    InternVideo2TextBackbone,
    InternVideo2VideoBackbone,
    SceneTransformer,
    load_internvideo2_config,
    setup_internvideo2_model,
)
from utils import (
    compute_frame_attention,
    compute_teacher_forcing_cosines,
    create_inference_visualization,
    estimate_total_frames,
    log_activitynet_dataset_samples,
    log_batch_schema,
    log_dataset_samples,
    log_msrvtt_dataset_samples,
    log_tvr_dataset_samples,
    log_scene_text_similarity_matrix,
    log_teacher_forcing_cosines,
    log_text_similarity_matrix,
    retrieve_clip_preview_frames,
    save_loss_plot,
    save_loss_term_plots,
    save_similarity_heatmap,
    save_text_similarity_heatmap,
    set_seed,
    trim_eos_scene_predictions,
)


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    path = path.expanduser()
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)


def run_validation(
    data_loader: DataLoader,
    *,
    video_backbone: Optional[InternVideo2VideoBackbone],
    text_backbone: Optional[InternVideo2TextBackbone],
    scene_model: SceneTransformer,
    args,
    device: torch.device,
    dataset_type: str,
    schema_label: str = "Validation",
    log_schema: bool = False,
    viz_video_id: Optional[str] = None,
    viz_dir: Optional[Path] = None,
    epoch: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    if data_loader is None:
        return None

    was_training = scene_model.training
    scene_model.eval()
    metrics = {
        "loss": 0.0,
        "repr": 0.0,
        "attn": 0.0,
        "cov": 0.0,
        "stop": 0.0,
        "sceneq": 0.0,
        "scenediv": 0.0,
    }
    batches = 0

    schema_logged = not log_schema
    viz_logged = False
    viz_epoch_dir: Optional[Path] = None
    if viz_dir is not None and epoch is not None:
        viz_epoch_dir = viz_dir / f"epoch_{epoch:03d}"
        viz_epoch_dir.mkdir(parents=True, exist_ok=True)
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

            preds, attn, stop_logits = scene_model(
                batch["clip_embeddings"],
                batch["clip_times"],
                batch["clip_padding_mask"],
                batch["decoder_inputs"],
                batch["decoder_padding_mask"],
            )
            if (
                not viz_logged
                and viz_epoch_dir is not None
                and viz_video_id is not None
                and batch_samples
                and batch["scene_lengths"]
            ):
                sample_idx = None
                raw_text_embed = None
                for idx_s, sample in enumerate(batch_samples):
                    if getattr(sample, "video_id", None) == viz_video_id:
                        sample_idx = idx_s
                        if hasattr(sample, "text_embeddings") and sample.text_embeddings is not None:
                            raw_text_embed = sample.text_embeddings
                        break
                if sample_idx is not None and sample_idx < len(batch_samples):
                    sample_length = batch["scene_lengths"][sample_idx]
                    sample_texts = getattr(batch_samples[sample_idx], "scene_texts", None)
                    if raw_text_embed is None and sample_texts and text_backbone is not None:
                        raw_text_embed = text_backbone.encode(sample_texts)
                    if raw_text_embed is None:
                        raw_text_embed = batch["decoder_targets"][sample_idx, :sample_length, :].detach()
                    if sample_texts and sample_length > 0:
                        pred_sample = preds[sample_idx, :sample_length, :].detach().cpu()
                        raw_tensor = raw_text_embed[:sample_length].detach().cpu()
                        text_proj = scene_model.project_text_embeddings(raw_text_embed.to(device))
                        target_sample = text_proj[:sample_length].detach().cpu()
                        if pred_sample.numel() > 0 and target_sample.numel() > 0:
                            scene_norm = F.normalize(pred_sample, dim=-1)
                            text_norm = F.normalize(target_sample, dim=-1)
                            scene_text_sim = (scene_norm @ text_norm.T).numpy()
                            text_text_sim = (text_norm @ text_norm.T).numpy()
                            raw_text_norm = F.normalize(raw_tensor, dim=-1)
                            raw_text_sim = (raw_text_norm @ raw_text_norm.T).numpy()
                            log_scene_text_similarity_matrix(
                                viz_video_id,
                                scene_text_sim,
                                sample_texts,
                            )
                            log_text_similarity_matrix(
                                f"{viz_video_id} (raw)",
                                raw_text_sim,
                                sample_texts,
                            )
                            log_text_similarity_matrix(
                                viz_video_id,
                                text_text_sim,
                                sample_texts,
                            )
                            heatmap_scene = save_similarity_heatmap(
                                viz_video_id,
                                scene_text_sim,
                                sample_texts,
                                viz_epoch_dir,
                            )
                            heatmap_text = save_text_similarity_heatmap(
                                viz_video_id,
                                text_text_sim,
                                sample_texts,
                                viz_epoch_dir,
                            )
                            raw_heatmap = save_text_similarity_heatmap(
                                f"raw_{viz_video_id}",
                                raw_text_sim,
                                sample_texts,
                                viz_epoch_dir,
                            )
                            log_path = viz_epoch_dir / f"{viz_video_id.replace('/', '_')}_similarity.txt"
                            with open(log_path, "w", encoding="utf-8") as fh:
                                fh.write(
                                    f"Epoch {epoch} validation video {viz_video_id}\n"
                                    f"Text queries (n={len(sample_texts)}):\n"
                                )
                                for idx_q, text in enumerate(sample_texts):
                                    fh.write(f"  Query {idx_q}: {text}\n")
                                fh.write("Raw text-text cosine similarities:\n")
                                fh.write(
                                    np.array2string(
                                        raw_text_sim,
                                        precision=3,
                                        suppress_small=True,
                                    )
                                    + "\n"
                                )
                                fh.write("Projected text-text cosine similarities:\n")
                                fh.write(
                                    np.array2string(
                                        text_text_sim,
                                        precision=3,
                                        suppress_small=True,
                                    )
                                    + "\n"
                                )
                                fh.write("Scene-text cosine similarities:\n")
                                fh.write(
                                    np.array2string(
                                        scene_text_sim,
                                        precision=3,
                                        suppress_small=True,
                                    )
                                    + "\n"
                                )
                            logging.info(
                                "Validation epoch %d sample %s | scene-text mean=%.4f | text-text mean=%.4f | raw_mean=%.4f | sim_heatmap=%s | text_heatmap=%s | raw_heatmap=%s | log=%s",
                                epoch,
                                viz_video_id,
                                float(scene_text_sim.mean()),
                                float(text_text_sim.mean()),
                                float(raw_text_sim.mean()),
                                heatmap_scene,
                                heatmap_text,
                                raw_heatmap,
                            log_path,
                        )
                        viz_logged = True
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
            valid_mask_tf = ~batch["decoder_padding_mask"]
            valid_preds = preds[valid_mask_tf]
            valid_targets = batch["decoder_targets"][valid_mask_tf]
            attn_loss = attention_supervision_loss(
                attn,
                batch["clip_times"],
                batch["clip_padding_mask"],
                batch["scene_windows"],
                batch["scene_lengths"],
            )
            cov_loss = preds.new_tensor(0.0)
            scene_pred_list: List[torch.Tensor] = []
            text_target_list: List[torch.Tensor] = []
            for idx_b, length in enumerate(batch["scene_lengths"]):
                if length <= 0:
                    continue
                scene_pred_list.append(preds[idx_b, :length, :])
                text_target_list.append(batch["decoder_targets"][idx_b, :length, :])
            if scene_pred_list:
                sceneq_loss = scene_query_matrix_loss(scene_pred_list, text_target_list)
                scenediv_loss = scene_diversity_loss_structured(scene_pred_list, text_target_list)
            else:
                sceneq_loss = preds.new_tensor(0.0)
                scenediv_loss = preds.new_tensor(0.0)
            stop_targets = torch.zeros_like(stop_logits)
            for idx_b, length in enumerate(batch["scene_lengths"]):
                last_idx = min(length, stop_targets.shape[1] - 1)
                stop_targets[idx_b, last_idx] = 1.0
            valid_mask = ~batch["decoder_padding_mask"]
            bce_raw = F.binary_cross_entropy_with_logits(stop_logits, stop_targets, reduction="none")
            denom = valid_mask.float().sum().clamp_min(1.0)
            stop_loss = (bce_raw * valid_mask.float()).sum() / denom
            total_loss = rep_loss
            total_loss = total_loss + args.lambda_attn * attn_loss
            total_loss = total_loss + args.lambda_stop * stop_loss
            total_loss = total_loss + args.lambda_scene_query * sceneq_loss
            total_loss = total_loss + args.lambda_scene_diversity * scenediv_loss

            metrics["loss"] += float(total_loss.item())
            metrics["repr"] += float(rep_loss.item())
            metrics["attn"] += float(attn_loss.item())
            metrics["cov"] += float(cov_loss.item())
            metrics["stop"] += float(stop_loss.item())
            metrics["sceneq"] += float(sceneq_loss.item())
            metrics["scenediv"] += float(scenediv_loss.item())
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


def log_dataset_sample_similarity(
    sample: object,
    *,
    video_backbone: Optional[InternVideo2VideoBackbone],
    text_backbone: Optional[InternVideo2TextBackbone],
    scene_model: SceneTransformer,
    args,
    device: torch.device,
    dataset_type: str,
    viz_dir: Optional[Path],
    epoch: int,
    label: str,
) -> bool:
    video_id = getattr(sample, "video_id", "(unknown)")
    scene_texts = list(getattr(sample, "scene_texts", []) or [])
    was_training = scene_model.training
    scene_model.eval()
    try:
        with torch.no_grad():
            batch = collate_scene_batch(
                [sample],
                video_backbone=video_backbone,
                text_backbone=text_backbone,
                scene_model=scene_model,
                args=args,
                device=device,
                dataset_type=dataset_type,
            )
            if batch is None or not batch["scene_lengths"]:
                logging.warning("%s visualization sample %s produced an empty batch; skipping.", label, video_id)
                return False
            preds, _, _ = scene_model(
                batch["clip_embeddings"],
                batch["clip_times"],
                batch["clip_padding_mask"],
                batch["decoder_inputs"],
                batch["decoder_padding_mask"],
            )
            sample_length = int(batch["scene_lengths"][0])
            if sample_length <= 0:
                logging.warning("%s visualization sample %s has no decoded scenes; skipping.", label, video_id)
                return False
            pred_sample = preds[0, :sample_length, :].detach().cpu()
            target_sample = batch["decoder_targets"][0, :sample_length, :].detach().cpu()
            if pred_sample.numel() == 0 or target_sample.numel() == 0:
                logging.warning(
                    "%s visualization sample %s has empty embeddings; skipping.",
                    label,
                    video_id,
                )
                return False
            scene_norm = F.normalize(pred_sample, dim=-1)
            text_norm = F.normalize(target_sample, dim=-1)
            scene_text_sim = (scene_norm @ text_norm.T).numpy()
            text_text_sim = (text_norm @ text_norm.T).numpy()
            raw_text_sim: Optional[np.ndarray] = None
            raw_tensor = getattr(sample, "text_embeddings", None)
            if raw_tensor is not None and isinstance(raw_tensor, torch.Tensor):
                raw_slice = raw_tensor[:sample_length].detach().cpu()
                if raw_slice.numel() > 0:
                    raw_norm = F.normalize(raw_slice, dim=-1)
                    raw_text_sim = (raw_norm @ raw_norm.T).numpy()
            elif scene_texts and text_backbone is not None:
                try:
                    encoded = text_backbone.encode(scene_texts)
                except Exception as exc:
                    logging.warning(
                        "Failed to encode raw texts for %s visualization sample %s: %s",
                        label,
                        video_id,
                        exc,
                    )
                else:
                    raw_slice = encoded[:sample_length].detach().cpu()
                    if raw_slice.numel() > 0:
                        raw_norm = F.normalize(raw_slice, dim=-1)
                        raw_text_sim = (raw_norm @ raw_norm.T).numpy()
            log_scene_text_similarity_matrix(video_id, scene_text_sim, scene_texts)
            if raw_text_sim is not None:
                log_text_similarity_matrix(f"{video_id} (raw)", raw_text_sim, scene_texts)
            log_text_similarity_matrix(video_id, text_text_sim, scene_texts)
            heatmap_scene = save_similarity_heatmap(video_id, scene_text_sim, scene_texts, viz_dir)
            heatmap_text = save_text_similarity_heatmap(video_id, text_text_sim, scene_texts, viz_dir)
            raw_heatmap = None
            if raw_text_sim is not None:
                raw_heatmap = save_text_similarity_heatmap(f"raw_{video_id}", raw_text_sim, scene_texts, viz_dir)
            raw_mean = f"{float(raw_text_sim.mean()):.4f}" if raw_text_sim is not None else "n/a"
            logging.info(
                "%s epoch %d sample %s | scene-text mean=%.4f | text-text mean=%.4f | raw_mean=%s | sim_heatmap=%s | text_heatmap=%s | raw_heatmap=%s",
                label,
                epoch,
                video_id,
                float(scene_text_sim.mean()) if scene_text_sim.size > 0 else float("nan"),
                float(text_text_sim.mean()) if text_text_sim.size > 0 else float("nan"),
                raw_mean,
                heatmap_scene,
                heatmap_text,
                raw_heatmap,
            )
            return True
    finally:
        if was_training:
            scene_model.train()
    return False


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
            if dataset_type in {"msrvtt_untrimmed", "activitynet", "tvr"}:
                if dataset_type == "msrvtt_untrimmed":
                    expected = MSRVTTSceneItem
                elif dataset_type == "activitynet":
                    expected = ActivityNetSceneItem
                else:
                    expected = TVRSceneItem
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
                if dataset_type in {"msrvtt_untrimmed", "activitynet", "tvr"}:
                    text_embeds = text_embed_source.to(device)
                else:
                    if text_backbone is None:
                        raise RuntimeError("Text backbone required for QVHighlights inference.")
                        text_embeds = text_backbone.encode(scene_texts).to(device)
            if text_embeds is not None and text_embeds.numel() > 0:
                projected_text = scene_model.project_text_embeddings(text_embeds)
                text_embeddings_export = projected_text.detach().cpu().tolist()
                text_norm = F.normalize(projected_text, dim=-1)
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
                if scene_latents_cpu.numel() > 0 and projected_text.numel() > 0:
                    text_proj_cpu = projected_text.detach().cpu()
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
            json_record = json.dumps(record, ensure_ascii=False)
            f.write(json_record + "\n")
            processed += 1
            if idx % max(1, len(dataset) // 10) == 0:
                logging.info("Inference progress: %d/%d videos", idx, len(dataset))
    logging.info(
        "Inference results saved to %s (videos processed: %d)",
        output_path,
        processed,
    )


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
    elif args.dataset == "tvr":
        args.tvr_root = args.tvr_root.expanduser()
        if args.tvr_video_features is None:
            args.tvr_video_features = (
                args.tvr_root / "FeatureData" / "new_clip_vit_32_tvr_vid_features.hdf5"
            )
        if args.tvr_text_features is None:
            args.tvr_text_features = (
                args.tvr_root / "TextData" / "clip_ViT_B_32_tvr_query_feat.hdf5"
            )
        if args.tvr_train_json is None:
            args.tvr_train_json = args.tvr_root / "TextData" / "train.json"
        if args.tvr_val_json is None:
            args.tvr_val_json = args.tvr_root / "TextData" / "val.json"
        args.tvr_video_features = args.tvr_video_features.expanduser()
        args.tvr_text_features = args.tvr_text_features.expanduser()
        args.tvr_train_json = args.tvr_train_json.expanduser()
        args.tvr_val_json = args.tvr_val_json.expanduser()
        if args.tvr_inference_json is None:
            args.tvr_inference_json = args.tvr_val_json
        args.tvr_inference_json = args.tvr_inference_json.expanduser()
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
        args.inference_visualization_dir.mkdir(parents=True, exist_ok=True)
    args.inference_visualization_max_scenes = max(1, int(args.inference_visualization_max_scenes))
    disable_val_viz = bool(getattr(args, "disable_validation_visualizations", False))
    if args.mode == "train" and not disable_val_viz:
        if args.validation_visualization_dir is None:
            args.validation_visualization_dir = args.inference_output.parent / "val_similarity"
        args.validation_visualization_dir = args.validation_visualization_dir.expanduser()
        args.validation_visualization_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.validation_visualization_dir = None
    if disable_val_viz:
        args.validation_visualization_video_id = None
    train_viz_video_id = getattr(args, "train_visualization_video_id", None)
    if train_viz_video_id is not None:
        train_viz_video_id = train_viz_video_id.strip()
        if not train_viz_video_id:
            train_viz_video_id = None
    if args.mode != "train":
        train_viz_video_id = None
    args.train_visualization_video_id = train_viz_video_id
    if args.train_visualization_video_id:
        if args.train_visualization_dir is None:
            args.train_visualization_dir = args.inference_output.parent / "train_similarity"
        args.train_visualization_dir = args.train_visualization_dir.expanduser()
        args.train_visualization_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.train_visualization_dir = None
    val_target_viz_video = getattr(args, "validation_visualization_video_id", None)
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

    tvr_dataset_cache: Dict[Path, TVRSceneDataset] = {}

    def get_tvr_dataset(annotation_path: Path, split_name: str) -> TVRSceneDataset:
        resolved = annotation_path.expanduser()
        if resolved not in tvr_dataset_cache:
            tvr_dataset_cache[resolved] = TVRSceneDataset(
                resolved,
                args.tvr_video_features,
                args.tvr_text_features,
                split_name=split_name,
            )
        return tvr_dataset_cache[resolved]

    def load_sample_by_video_id(dataset_obj: Dataset, video_id: str) -> Optional[object]:
        if not video_id:
            return None
        items = getattr(dataset_obj, "items", None)
        if isinstance(items, list):
            for idx, meta in enumerate(items):
                if getattr(meta, "video_id", None) == video_id:
                    try:
                        return dataset_obj[idx]
                    except Exception as exc:
                        logging.warning("Failed to load dataset sample %s: %s", video_id, exc)
                        return None
            return None
        try:
            total = len(dataset_obj)
        except TypeError:
            return None
        for idx in range(total):
            try:
                sample = dataset_obj[idx]
            except Exception as exc:
                logging.warning("Failed to materialize dataset idx %d for %s: %s", idx, video_id, exc)
                return None
            if getattr(sample, "video_id", None) == video_id:
                return sample
        return None

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
    elif args.dataset == "activitynet":
        annotation = args.activitynet_train_json if args.mode != "inference" else args.activitynet_inference_json
        primary_dataset = get_activitynet_dataset(annotation, split_name=args.mode)
        embed_dim = primary_dataset.feature_dim
    elif args.dataset == "tvr":
        annotation = args.tvr_train_json if args.mode != "inference" else args.tvr_inference_json
        primary_dataset = get_tvr_dataset(annotation, split_name=args.mode)
        embed_dim = primary_dataset.feature_dim
    else:
        raise ValueError(f"Unsupported dataset for embed_dim inference: {args.dataset}")

    if embed_dim is None:
        raise RuntimeError("Failed to determine embedding dimension for the scene model.")

    scene_model = SceneTransformer(
        embed_dim=embed_dim,
        num_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        ff_dim=args.decoder_ff_dim,
        dropout=args.decoder_dropout,
        use_exhaustive_clip_bank=getattr(args, "use_exhaustive_clip_bank", False),
        use_text_projection=not getattr(args, "disable_text_projection", False),
    ).to(device)

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
                elif args.dataset == "activitynet":
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
                else:
                    inference_dataset_cache = get_tvr_dataset(
                        args.tvr_inference_json,
                        split_name="inference",
                    )
                    logging.info(
                        "Inference dataset stats: %d videos, %d queries",
                        len(inference_dataset_cache),
                        inference_dataset_cache.total_queries,
                    )
                    log_tvr_dataset_samples("Inference", inference_dataset_cache)
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
    elif args.dataset == "activitynet":
        dataset = get_activitynet_dataset(args.activitynet_train_json, split_name="train")
        logging.info(
            "Train dataset stats: %d videos, %d queries",
            len(dataset),
            dataset.total_queries,
        )
        log_activitynet_dataset_samples("Train", dataset)
    else:
        dataset = get_tvr_dataset(args.tvr_train_json, split_name="train")
        logging.info(
            "Train dataset stats: %d videos, %d queries",
            len(dataset),
            dataset.total_queries,
        )
        log_tvr_dataset_samples("Train", dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )
    train_viz_sample: Optional[object] = None
    if args.train_visualization_video_id:
        train_viz_sample = load_sample_by_video_id(dataset, args.train_visualization_video_id)
        if train_viz_sample is None:
            logging.warning(
                "Train visualization sample %s not found in training split %s.",
                args.train_visualization_video_id,
                getattr(dataset, "split", getattr(dataset, "split_name", "train")),
            )
    optimizer = torch.optim.AdamW(scene_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    validation_loader = None
    validation_requested = (args.validation_interval > 0) or bool(
        getattr(args, "validation_final_epoch_only", False)
    )
    if validation_requested:
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
            elif args.dataset == "activitynet":
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
            else:
                validation_dataset = get_tvr_dataset(
                    args.tvr_val_json,
                    split_name="val",
                )
                logging.info(
                    "Validation dataset stats: %d videos, %d queries",
                    len(validation_dataset),
                    validation_dataset.total_queries,
                )
                log_tvr_dataset_samples("Validation", validation_dataset)
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
    validation_enabled = validation_loader is not None and steps_per_epoch > 0
    final_epoch_only = bool(getattr(args, "validation_final_epoch_only", False))

    def _record_loss(history: Dict[str, List[Tuple[int, float]]], name: str, step: int, value: float) -> None:
        history.setdefault(name, []).append((step, value))

    global_step = 0
    logged_train_batch_schema = False
    logged_val_batch_schema = False
    train_history: List[Tuple[int, float]] = []
    val_history: List[Tuple[int, float]] = []
    train_loss_terms: Dict[str, List[Tuple[int, float]]] = {}
    val_loss_terms: Dict[str, List[Tuple[int, float]]] = {}
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    early_stopping_patience = max(0, int(getattr(args, "early_stopping_patience", 0) or 0))
    epochs_without_improvement = 0
    stop_training = False
    ss_enabled = bool(getattr(args, "ss_enable", False))
    ss_warmup = int(getattr(args, "ss_warmup_epochs", 0) or 0)
    ss_p_max = float(getattr(args, "ss_p_max", 0.0) or 0.0)
    for epoch in range(1, args.epochs + 1):
        scene_model.train()
        if not ss_enabled:
            ss_p = 0.0
        elif epoch <= ss_warmup:
            ss_p = 0.0
        else:
            denom = max(1, args.epochs - ss_warmup)
            progress = (epoch - ss_warmup) / denom
            progress = max(0.0, min(1.0, progress))
            ss_p = ss_p_max * progress
        if ss_enabled:
            logging.info("Epoch %d scheduled sampling prob: %.4f", epoch, ss_p)
        steps_in_epoch = 0
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

            decoder_inputs_tf = batch["decoder_inputs"]
            with torch.no_grad():
                preds_tf, _, _ = scene_model(
                    batch["clip_embeddings"],
                    batch["clip_times"],
                    batch["clip_padding_mask"],
                    decoder_inputs_tf,
                    batch["decoder_padding_mask"],
                )

            decoder_inputs_mixed = decoder_inputs_tf.clone()
            if ss_p > 0.0 and decoder_inputs_tf.shape[1] > 1:
                B, L, _ = decoder_inputs_tf.shape
                rand = torch.rand(B, L - 1, device=decoder_inputs_tf.device)
                valid_positions = ~batch["decoder_padding_mask"][:, 1:]
                use_pred = (rand < ss_p) & valid_positions
                with torch.no_grad():
                    prev_pred_tokens = preds_tf[:, :-1, :].detach()
                if prev_pred_tokens.numel() > 0:
                    use_pred_expanded = use_pred.unsqueeze(-1)
                    mixed_tail = torch.where(
                        use_pred_expanded,
                        prev_pred_tokens,
                        decoder_inputs_mixed[:, 1:, :],
                    )
                    decoder_inputs_mixed[:, 1:, :] = mixed_tail

            preds, attn, stop_logits = scene_model(
                batch["clip_embeddings"],
                batch["clip_times"],
                batch["clip_padding_mask"],
                decoder_inputs_mixed,
                batch["decoder_padding_mask"],
            )
            rep_loss = representation_alignment_loss(
                preds,
                batch["decoder_targets"],
                batch["decoder_padding_mask"],
                mode=args.alignment_loss,
                temperature=args.infonce_temp,
            )
            valid_mask_tf = ~batch["decoder_padding_mask"]
            valid_preds = preds[valid_mask_tf]
            valid_targets = batch["decoder_targets"][valid_mask_tf]
            attn_loss = attention_supervision_loss(
                attn,
                batch["clip_times"],
                batch["clip_padding_mask"],
                batch["scene_windows"],
                batch["scene_lengths"],
            )
            cov_loss = rep_loss.new_tensor(0.0)
            scene_pred_list = []
            text_target_list = []
            for idx_b, length in enumerate(batch["scene_lengths"]):
                if length <= 0:
                    continue
                scene_pred_list.append(preds[idx_b, :length, :])
                text_target_list.append(batch["decoder_targets"][idx_b, :length, :])
            if scene_pred_list:
                sceneq_loss = scene_query_matrix_loss(scene_pred_list, text_target_list)
                scenediv_loss = scene_diversity_loss_structured(scene_pred_list, text_target_list)
            else:
                sceneq_loss = rep_loss.new_tensor(0.0)
                scenediv_loss = rep_loss.new_tensor(0.0)
            stop_targets = torch.zeros_like(stop_logits)
            for idx_b, length in enumerate(batch["scene_lengths"]):
                last_idx = min(length, stop_targets.shape[1] - 1)
                stop_targets[idx_b, last_idx] = 1.0
            valid_mask = ~batch["decoder_padding_mask"]
            bce_raw = F.binary_cross_entropy_with_logits(stop_logits, stop_targets, reduction="none")
            denom = valid_mask.float().sum().clamp_min(1.0)
            stop_loss = (bce_raw * valid_mask.float()).sum() / denom
            total_loss = rep_loss
            total_loss = total_loss + args.lambda_attn * attn_loss
            total_loss = total_loss + args.lambda_stop * stop_loss
            total_loss = total_loss + args.lambda_scene_query * sceneq_loss
            total_loss = total_loss + args.lambda_scene_diversity * scenediv_loss

            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(scene_model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            steps_in_epoch += 1
            train_history.append((global_step, float(total_loss.item())))
            _record_loss(train_loss_terms, "total", global_step, float(total_loss.item()))
            _record_loss(train_loss_terms, "repr", global_step, float(rep_loss.item()))
            _record_loss(train_loss_terms, "attn", global_step, float(attn_loss.item()))
            _record_loss(train_loss_terms, "stop", global_step, float(stop_loss.item()))
            sceneq_loss_scalar = float(sceneq_loss.item())
            _record_loss(train_loss_terms, "sceneq", global_step, sceneq_loss_scalar)
            scenediv_loss_scalar = float(scenediv_loss.item())
            _record_loss(train_loss_terms, "scenediv", global_step, scenediv_loss_scalar)

            stop_loss_scalar = float(stop_loss.item())
            if global_step % args.log_interval == 0:
                logging.info(
                    "Epoch %d step %d | loss=%.4f repr=%.4f attn=%.4f cov=%.4f stop=%.4f sceneq=%.4f scenediv=%.4f",
                    epoch,
                    global_step,
                    total_loss.item(),
                    rep_loss.item(),
                    attn_loss.item(),
                    cov_loss.item(),
                    stop_loss_scalar,
                    sceneq_loss_scalar,
                    scenediv_loss_scalar,
                )

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
        if stop_training:
            break

        if train_viz_sample is not None and args.train_visualization_video_id:
            viz_epoch_dir: Optional[Path] = None
            if args.train_visualization_dir is not None:
                viz_epoch_dir = args.train_visualization_dir / f"epoch_{epoch:03d}"
                viz_epoch_dir.mkdir(parents=True, exist_ok=True)
            logged = log_dataset_sample_similarity(
                train_viz_sample,
                video_backbone=video_backbone,
                text_backbone=text_backbone,
                scene_model=scene_model,
                args=args,
                device=device,
                dataset_type=args.dataset,
                viz_dir=viz_epoch_dir,
                epoch=epoch,
                label="Train",
            )
            if not logged:
                logging.warning(
                    "Train visualization logging skipped for %s at epoch %d.",
                    args.train_visualization_video_id,
                    epoch,
                )

        if validation_enabled:
            should_validate = True
            if final_epoch_only and epoch != args.epochs:
                should_validate = False
            metrics = None
            if should_validate:
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
                    viz_video_id=val_target_viz_video,
                    viz_dir=args.validation_visualization_dir,
                    epoch=epoch,
                )
            if metrics is not None:
                if not logged_val_batch_schema:
                    logged_val_batch_schema = True
                logging.info(
                    "Validation epoch %d | loss=%.4f repr=%.4f attn=%.4f cov=%.4f stop=%.4f sceneq=%.4f scenediv=%.4f",
                    epoch,
                    metrics["loss"],
                    metrics["repr"],
                    metrics["attn"],
                    metrics["cov"],
                    metrics["stop"],
                    metrics["sceneq"],
                    metrics["scenediv"],
                )
                val_history.append((global_step, float(metrics["loss"])))
                _record_loss(val_loss_terms, "total", global_step, float(metrics["loss"]))
                _record_loss(val_loss_terms, "repr", global_step, float(metrics["repr"]))
                _record_loss(val_loss_terms, "attn", global_step, float(metrics["attn"]))
                _record_loss(val_loss_terms, "stop", global_step, float(metrics["stop"]))
                _record_loss(val_loss_terms, "sceneq", global_step, float(metrics["sceneq"]))
                _record_loss(val_loss_terms, "scenediv", global_step, float(metrics["scenediv"]))
                improved = best_val_loss is None or metrics["loss"] < best_val_loss
                if improved:
                    best_val_loss = float(metrics["loss"])
                    best_epoch = epoch
                    save_checkpoint(scene_model, args.checkpoint_path)
                    logging.info(
                        "New best checkpoint at epoch %d (val loss=%.4f) saved to %s",
                        epoch,
                        best_val_loss,
                        args.checkpoint_path,
                    )
                    epochs_without_improvement = 0
                else:
                    if early_stopping_patience > 0:
                        epochs_without_improvement += 1
                        logging.info(
                            "Validation not improved for %d/%d epochs",
                            epochs_without_improvement,
                            early_stopping_patience,
                        )
                        if epochs_without_improvement >= early_stopping_patience:
                            logging.info(
                                "Early stopping triggered after %d epochs without improvement.",
                                epochs_without_improvement,
                            )
                            stop_training = True
            if should_validate and stop_training:
                break

    if best_epoch is None:
        save_checkpoint(scene_model, args.checkpoint_path)
        logging.info("Saved SceneTransformer checkpoint to %s", args.checkpoint_path)
    else:
        logging.info(
            "Best validation checkpoint (epoch %d, loss=%.4f) stored at %s",
            best_epoch,
            best_val_loss,
            args.checkpoint_path,
        )
    save_loss_plot(train_history, val_history, args.loss_plot_path)
    loss_terms_plot_path = None
    if args.loss_plot_path is not None:
        base_path = Path(args.loss_plot_path)
        loss_terms_plot_path = base_path.with_name(f"{base_path.stem}_terms{base_path.suffix}")
    save_loss_term_plots(train_loss_terms, val_loss_terms, loss_terms_plot_path)

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
