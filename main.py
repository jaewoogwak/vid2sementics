from __future__ import annotations

import argparse
from pathlib import Path

from trainer import train


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
    parser.add_argument("--lambda-cov", type=float, default=0.0)
    parser.add_argument("--lambda-triplet", type=float, default=1.0)
    parser.add_argument(
        "--lambda-stop",
        type=float,
        default=1.0,
        help="Weight applied to the stop-head BCE loss.",
    )
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


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
