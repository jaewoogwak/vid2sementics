#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper for the modular training pipeline (main.py + trainer.py).

ROOT_DEFAULT="/dev/ssd1/gjw/prvr"
ROOT="${ROOT:-${ROOT_DEFAULT}}"

MODE="${MODE:-train}"
DATASET="${DATASET:-msrvtt_untrimmed}"

INTERNVIDEO_ROOT="${INTERNVIDEO_ROOT:-${ROOT}/InternVideo}"
INTERNVIDEO_CONFIG="${INTERNVIDEO_CONFIG:-${INTERNVIDEO_ROOT}/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py}"
INTERNVIDEO_CKPT="${INTERNVIDEO_CKPT:-${INTERNVIDEO_ROOT}/InternVideo2/ckpt/InternVideo2-stage2_1b-224p-f4.pt}"

DATASET_ROOT="${DATASET_ROOT:-${ROOT}/dataset/qvhighlights}"
CONCAT_ROOT="${CONCAT_ROOT:-${DATASET_ROOT}/concat}"
RAW_ROOT="${RAW_ROOT:-${DATASET_ROOT}/raw}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATASET_ROOT}/highlight_train_release.jsonl}"
VALIDATION_JSONL="${VALIDATION_JSONL:-${DATASET_ROOT}/highlight_val_release.jsonl}"
INFERENCE_JSONL="${INFERENCE_JSONL:-${DATASET_ROOT}/highlight_val_release.jsonl}"
INFERENCE_INTERVAL="${INFERENCE_INTERVAL:-0}"
INFERENCE_LIMIT="${INFERENCE_LIMIT:-}" 
VIDEO_CACHE_ROOT="${VIDEO_CACHE_ROOT:-${ROOT}/cache/video}"
TEXT_CACHE_ROOT="${TEXT_CACHE_ROOT:-${ROOT}/cache/text}"

MSRVTT_ROOT="${MSRVTT_ROOT:-${ROOT}/dataset/data/MSRVTT}"
MSRVTT_ANNOTATION="${MSRVTT_ANNOTATION:-${MSRVTT_ROOT}/annotation/MSRVTT_untrimmed.json}"
MSRVTT_FEAT_ROOT="${MSRVTT_FEAT_ROOT:-${MSRVTT_ROOT}/internvideo_untrimmed_feats}"
MSRVTT_TRAIN_SPLIT="${MSRVTT_TRAIN_SPLIT:-train}"
MSRVTT_VAL_SPLIT="${MSRVTT_VAL_SPLIT:-val}"
MSRVTT_INFER_SPLIT="${MSRVTT_INFER_SPLIT:-val}"

ACTIVITYNET_ROOT="${ACTIVITYNET_ROOT:-${ROOT}/dataset/activitynet}"
ACTIVITYNET_VIDEO_FEATURES="${ACTIVITYNET_VIDEO_FEATURES:-${ACTIVITYNET_ROOT}/FeatureData/new_clip_vit_32_activitynet_vid_features.hdf5}"
ACTIVITYNET_TEXT_FEATURES="${ACTIVITYNET_TEXT_FEATURES:-${ACTIVITYNET_ROOT}/TextData/clip_ViT_B_32_activitynet_query_feat.hdf5}"
ACTIVITYNET_TRAIN_JSON="${ACTIVITYNET_TRAIN_JSON:-${ACTIVITYNET_ROOT}/TextData/train.json}"
ACTIVITYNET_VAL_JSON="${ACTIVITYNET_VAL_JSON:-${ACTIVITYNET_ROOT}/TextData/val_1.json}"
ACTIVITYNET_INFER_JSON="${ACTIVITYNET_INFER_JSON:-${ACTIVITYNET_VAL_JSON}}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}" 
INFERENCE_OUTPUT="${INFERENCE_OUTPUT:-}" 
RUN_INFERENCE="${RUN_INFERENCE:-true}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

SAMPLE_FPS="${SAMPLE_FPS:-1.0}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-8}"
CLIP_STRIDE="${CLIP_STRIDE:-4}"
FRAME_SIZE="${FRAME_SIZE:-224}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-16}"

DECODER_LAYERS="${DECODER_LAYERS:-12}"
DECODER_HEADS="${DECODER_HEADS:-8}"
DECODER_FF_DIM="${DECODER_FF_DIM:-2048}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"

ALIGNMENT_LOSS="${ALIGNMENT_LOSS:-infonce}"
INFO_NCE_TEMP="${INFO_NCE_TEMP:-0.07}"
LAMBDA_ATTN="${LAMBDA_ATTN:-20}"
LAMBDA_COV="${LAMBDA_COV:-0.0}"
LAMBDA_STOP="${LAMBDA_STOP:-0.3}"
# LAMBDA_TRIPLET="${LAMBDA_TRIPLET:-0.5}"

MAX_GENERATION_STEPS="${MAX_GENERATION_STEPS:-12}"
EOS_THRESHOLD="${EOS_THRESHOLD:-0.8}"
# MAX_SAMPLE_INFER="${MAX_SAMPLE_INFER:-30}"

EXTRA_ARGS=("$@")

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${LOG_DIR:-./logs/${DATASET}/${TIMESTAMP}_L${DECODER_LAYERS}H${DECODER_HEADS}}"
LOG_FILE="${LOG_DIR}/train.log"
mkdir -p "${LOG_DIR}"
LOSS_PLOT_PATH="${LOSS_PLOT_PATH:-${LOG_DIR}/loss_curve.png}"

# Redirect everything to both stdout and train.log.
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -z "${INFERENCE_OUTPUT}" ]]; then
  INFERENCE_OUTPUT="${LOG_DIR}/val_predictions.jsonl"
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="${LOG_DIR}/checkpoint/scene_transformer.ckpt"
fi
mkdir -p "$(dirname "${CHECKPOINT_PATH}")"

echo "[$(date +"%F %T")] Launching modular autoregressive training"
echo "Working directory: $(pwd)"
echo "Logging to ${LOG_FILE}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --internvideo-root "${INTERNVIDEO_ROOT}"
  --internvideo-config "${INTERNVIDEO_CONFIG}"
  --internvideo-ckpt "${INTERNVIDEO_CKPT}"
  --mode "${MODE}"
  --inference-output "${INFERENCE_OUTPUT}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --sample-fps "${SAMPLE_FPS}"
  --frames-per-clip "${FRAMES_PER_CLIP}"
  --clip-stride "${CLIP_STRIDE}"
  --frame-size "${FRAME_SIZE}"
  --clip-batch-size "${CLIP_BATCH_SIZE}"
  --decoder-layers "${DECODER_LAYERS}"
  --decoder-heads "${DECODER_HEADS}"
  --decoder-ff-dim "${DECODER_FF_DIM}"
  --decoder-dropout "${DECODER_DROPOUT}"
  --alignment-loss "${ALIGNMENT_LOSS}"
  --infonce-temp "${INFO_NCE_TEMP}"
  --lambda-attn "${LAMBDA_ATTN}"
  --lambda-cov "${LAMBDA_COV}"
  --lambda-stop "${LAMBDA_STOP}"
  # --lambda-triplet "${LAMBDA_TRIPLET}"
  --max-generation-steps "${MAX_GENERATION_STEPS}"
  --eos-threshold "${EOS_THRESHOLD}"
  --inference-interval "${INFERENCE_INTERVAL}"
  --loss-plot-path "${LOSS_PLOT_PATH}"
  --video-cache-root "${VIDEO_CACHE_ROOT}"
  --text-cache-root "${TEXT_CACHE_ROOT}"
  # --max-sample-infer "${MAX_SAMPLE_INFER}"
  --seed "${SEED:-42}"
)

if [[ "${RUN_INFERENCE,,}" == "true" ]]; then
  COMMON_ARGS+=(--run-inference-after-train)
fi

if [[ -n "${INFERENCE_LIMIT}" ]]; then
  COMMON_ARGS+=(--inference-limit "${INFERENCE_LIMIT}")
fi

case "${DATASET}" in
  qvhighlights)
    COMMON_ARGS+=(
      --dataset-root "${DATASET_ROOT}"
      --concat-root "${CONCAT_ROOT}"
      --raw-root "${RAW_ROOT}"
      --train-jsonl "${TRAIN_JSONL}"
      --validation-jsonl "${VALIDATION_JSONL}"
      --inference-jsonl "${INFERENCE_JSONL}"
    )
    ;;
  msrvtt_untrimmed)
    COMMON_ARGS+=(
      --msrvtt-annotation "${MSRVTT_ANNOTATION}"
      --msrvtt-feat-root "${MSRVTT_FEAT_ROOT}"
      --msrvtt-train-split "${MSRVTT_TRAIN_SPLIT}"
      --msrvtt-val-split "${MSRVTT_VAL_SPLIT}"
      --msrvtt-inference-split "${MSRVTT_INFER_SPLIT}"
    )
    ;;
  activitynet)
    COMMON_ARGS+=(
      --activitynet-root "${ACTIVITYNET_ROOT}"
      --activitynet-video-features "${ACTIVITYNET_VIDEO_FEATURES}"
      --activitynet-text-features "${ACTIVITYNET_TEXT_FEATURES}"
      --activitynet-train-json "${ACTIVITYNET_TRAIN_JSON}"
      --activitynet-val-json "${ACTIVITYNET_VAL_JSON}"
      --activitynet-inference-json "${ACTIVITYNET_INFER_JSON}"
    )
    ;;
esac

printf "Command: "
printf "python main.py"
for arg in "${COMMON_ARGS[@]}"; do
  printf " %q" "${arg}"
done
for arg in "${EXTRA_ARGS[@]}"; do
  printf " %q" "${arg}"
done
printf "\n"

python main.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
