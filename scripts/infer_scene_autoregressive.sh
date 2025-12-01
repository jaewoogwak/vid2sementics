#!/usr/bin/env bash
set -euo pipefail

# Wrapper for inference-only runs of the autoregressive scene model.

ROOT_DEFAULT="/dev/ssd1/gjw/prvr"
ROOT="${ROOT:-${ROOT_DEFAULT}}"
DATASET="${DATASET:-msrvtt_untrimmed}"

INTERNVIDEO_ROOT="${INTERNVIDEO_ROOT:-${ROOT}/InternVideo}"
INTERNVIDEO_CONFIG="${INTERNVIDEO_CONFIG:-${INTERNVIDEO_ROOT}/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py}"
INTERNVIDEO_CKPT="${INTERNVIDEO_CKPT:-${INTERNVIDEO_ROOT}/InternVideo2/ckpt/InternVideo2-stage2_1b-224p-f4.pt}"

DATASET_ROOT="${DATASET_ROOT:-${ROOT}/dataset/qvhighlights}"
CONCAT_ROOT="${CONCAT_ROOT:-${DATASET_ROOT}/concat}"
INFERENCE_JSONL="${INFERENCE_JSONL:-${DATASET_ROOT}/highlight_val_release.jsonl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT}/logs/msrvtt_untrimmed/20251125_170748/checkpoints/scene_transformer.ckpt}"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found at ${CHECKPOINT_PATH}. Set CHECKPOINT_PATH to a valid file before running." >&2
  exit 1
fi

MSRVTT_ROOT="${MSRVTT_ROOT:-${ROOT}/dataset/data/MSRVTT}"
MSRVTT_ANNOTATION="${MSRVTT_ANNOTATION:-${MSRVTT_ROOT}/annotation/MSRVTT_untrimmed.json}"
MSRVTT_FEAT_ROOT="${MSRVTT_FEAT_ROOT:-${MSRVTT_ROOT}/internvideo_untrimmed_feats}"
MSRVTT_INFER_SPLIT="${MSRVTT_INFER_SPLIT:-val}"

DEVICE="${DEVICE:-cuda}"
SAMPLE_FPS="${SAMPLE_FPS:-1.0}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-8}"
CLIP_STRIDE="${CLIP_STRIDE:-4}"
FRAME_SIZE="${FRAME_SIZE:-224}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-16}"
MAX_GENERATION_STEPS="${MAX_GENERATION_STEPS:-12}"
EOS_THRESHOLD="${EOS_THRESHOLD:-0.8}"

EXTRA_ARGS=("$@")

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${LOG_DIR:-./logs/${DATASET}/infer_${TIMESTAMP}}"
LOG_FILE="${LOG_DIR}/infer.log"
INFERENCE_OUTPUT="${INFERENCE_OUTPUT:-${LOG_DIR}/val_predictions.jsonl}"
mkdir -p "${LOG_DIR}"

{
  echo "[$(date +"%F %T")] Autoregressive scene inference"
  echo "Working directory: $(pwd)"
  echo "Dataset: ${DATASET}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  if [[ "${DATASET}" == "msrvtt_untrimmed" ]]; then
    echo "MSRVTT annotation: ${MSRVTT_ANNOTATION}"
    echo "MSRVTT feat root: ${MSRVTT_FEAT_ROOT}"
    echo "MSRVTT inference split: ${MSRVTT_INFER_SPLIT}"
  else
    echo "Inference JSONL: ${INFERENCE_JSONL}"
  fi
  echo "Output path: ${INFERENCE_OUTPUT}"
  echo "Sample FPS: ${SAMPLE_FPS}, Frames/clip: ${FRAMES_PER_CLIP}, Stride: ${CLIP_STRIDE}, Frame size: ${FRAME_SIZE}"
  echo "Device: ${DEVICE}"
  echo "Extra args: ${EXTRA_ARGS[*]}"
} >> "${LOG_FILE}"

CMD_ARGS=(
  --mode inference
  --dataset "${DATASET}"
  --internvideo-root "${INTERNVIDEO_ROOT}"
  --internvideo-config "${INTERNVIDEO_CONFIG}"
  --internvideo-ckpt "${INTERNVIDEO_CKPT}"
  --inference-output "${INFERENCE_OUTPUT}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --device "${DEVICE}"
  --sample-fps "${SAMPLE_FPS}"
  --frames-per-clip "${FRAMES_PER_CLIP}"
  --clip-stride "${CLIP_STRIDE}"
  --frame-size "${FRAME_SIZE}"
  --clip-batch-size "${CLIP_BATCH_SIZE}"
  --max-generation-steps "${MAX_GENERATION_STEPS}"
  --eos-threshold "${EOS_THRESHOLD}"
)

if [[ "${DATASET}" == "msrvtt_untrimmed" ]]; then
  CMD_ARGS+=(
    --msrvtt-annotation "${MSRVTT_ANNOTATION}"
    --msrvtt-feat-root "${MSRVTT_FEAT_ROOT}"
    --msrvtt-inference-split "${MSRVTT_INFER_SPLIT}"
  )
else
  CMD_ARGS+=(
    --dataset-root "${DATASET_ROOT}"
    --concat-root "${CONCAT_ROOT}"
    --inference-jsonl "${INFERENCE_JSONL}"
  )
fi

python train_scene_autoregressive_qvh.py \
  "${CMD_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
