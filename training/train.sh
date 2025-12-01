#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-train}
shift || true
EXTRA_ARGS=("$@")

ROOT="/dev/ssd1/gjw/prvr"
RAW_VIDEO_DIR="$ROOT/dataset/qvhighlights/concat"
JSONL_TRAIN="$ROOT/dataset/qvhighlights/queries_by_video_train.jsonl"
JSONL_VAL="$ROOT/dataset/qvhighlights/queries_by_video_val.jsonl"
VIDEO_EMBED_PATH="$ROOT/dataset/qvhighlights/internvideo_video_embeddings.h5"
TEXT_EMBED_PATH="$ROOT/dataset/qvhighlights/internvideo_text_embeddings.h5"
INTERNVIDEO_PATH="$ROOT/InternVideo/InternVideo2"
OUTPUT_DIR="$ROOT/training/checkpoints"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$OUTPUT_DIR/best_model.pt"}

EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
LR_VIT=${LR_VIT:-3e-5}
LR_BERT=${LR_BERT:-1e-5}
WARMUP=${WARMUP:-500}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_TXT_LEN=${MAX_TXT_LEN:-48}
CLIP_LEN=${CLIP_LEN:-8}
CLIP_STRIDE=${CLIP_STRIDE:-4}
SAMPLE_FPS=${SAMPLE_FPS:-1.0}
RESIZE=${RESIZE:-224}
PATIENCE=${PATIENCE:-3}
DEVICE=${DEVICE:-""}
PRECISION=${PRECISION:-fp16}
NUM_GPUS=${NUM_GPUS:-1}
VIDEO_CLIP_BSZ=${VIDEO_CLIP_BSZ:-32}
MAX_CLIPS_TRAIN=${MAX_CLIPS_TRAIN:-36}
MAX_CLIPS_EVAL=${MAX_CLIPS_EVAL:-0}

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  RUNNER_TRAIN=(torchrun --nproc_per_node="${NUM_GPUS}")
else
  RUNNER_TRAIN=(python)
fi

if [[ "${MODE}" == "eval_hdf5" ]]; then
  python training/retrieval.py \
    --mode eval_hdf5 \
    --video_embed_path "${VIDEO_EMBED_PATH}" \
    --text_embed_path "${TEXT_EMBED_PATH}" \
    --precision "${PRECISION}" \
    "${EXTRA_ARGS[@]}"
elif [[ "${MODE}" == "eval_raw" ]]; then
  python training/retrieval.py \
    --mode eval_raw \
    --raw_video_dir "${RAW_VIDEO_DIR}" \
    --jsonl_val_path "${JSONL_VAL}" \
    --internvideo_model_path "${INTERNVIDEO_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_txt_len "${MAX_TXT_LEN}" \
    --clip_len "${CLIP_LEN}" \
    --clip_stride "${CLIP_STRIDE}" \
    --sample_fps "${SAMPLE_FPS}" \
    --resize "${RESIZE}" \
    --video_clip_bsz "${VIDEO_CLIP_BSZ}" \
    --max_clips_eval "${MAX_CLIPS_EVAL}" \
    --precision "${PRECISION}" \
    ${DEVICE:+--device "${DEVICE}"} \
    "${EXTRA_ARGS[@]}"
else
  "${RUNNER_TRAIN[@]}" training/retrieval.py \
    --mode train \
    --raw_video_dir "${RAW_VIDEO_DIR}" \
    --jsonl_train_path "${JSONL_TRAIN}" \
    --jsonl_val_path "${JSONL_VAL}" \
    --internvideo_model_path "${INTERNVIDEO_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr_vit "${LR_VIT}" \
    --lr_bert "${LR_BERT}" \
    --warmup_steps "${WARMUP}" \
    --num_workers "${NUM_WORKERS}" \
    --max_txt_len "${MAX_TXT_LEN}" \
    --clip_len "${CLIP_LEN}" \
    --clip_stride "${CLIP_STRIDE}" \
    --sample_fps "${SAMPLE_FPS}" \
    --resize "${RESIZE}" \
    --patience "${PATIENCE}" \
    --video_clip_bsz "${VIDEO_CLIP_BSZ}" \
    --max_clips_train "${MAX_CLIPS_TRAIN}" \
    --max_clips_eval "${MAX_CLIPS_EVAL}" \
    --precision "${PRECISION}" \
    ${DEVICE:+--device "${DEVICE}"} \
    "${EXTRA_ARGS[@]}"
fi
