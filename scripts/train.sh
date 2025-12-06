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

TVR_ROOT="${TVR_ROOT:-${ROOT}/dataset/tvr}"
TVR_VIDEO_FEATURES="${TVR_VIDEO_FEATURES:-${TVR_ROOT}/FeatureData/new_clip_vit_32_tvr_vid_features.hdf5}"
TVR_TEXT_FEATURES="${TVR_TEXT_FEATURES:-${TVR_ROOT}/TextData/clip_ViT_B_32_tvr_query_feat.hdf5}"
TVR_TRAIN_JSON="${TVR_TRAIN_JSON:-${TVR_ROOT}/TextData/train.json}"
TVR_VAL_JSON="${TVR_VAL_JSON:-${TVR_ROOT}/TextData/val.json}"
TVR_INFER_JSON="${TVR_INFER_JSON:-${TVR_VAL_JSON}}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}" 
INFERENCE_OUTPUT="${INFERENCE_OUTPUT:-}" 
RUN_INFERENCE="${RUN_INFERENCE:-true}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

SAMPLE_FPS="${SAMPLE_FPS:-1.0}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-8}"
CLIP_STRIDE="${CLIP_STRIDE:-4}"
FRAME_SIZE="${FRAME_SIZE:-224}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-16}"
SS_ENABLE="${SS_ENABLE:-true}"
SS_P_MAX="${SS_P_MAX:-0.3}"
SS_WARMUP_EPOCHS="${SS_WARMUP_EPOCHS:-3}"

DECODER_LAYERS="${DECODER_LAYERS:-6}"
DECODER_HEADS="${DECODER_HEADS:-16}"
DECODER_FF_DIM="${DECODER_FF_DIM:-2048}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
USE_EXHAUSTIVE_CLIP_BANK="${USE_EXHAUSTIVE_CLIP_BANK:-false}"
DISABLE_TEXT_PROJECTION="${DISABLE_TEXT_PROJECTION:-false}"
VAL_VIS_DIR="${VAL_VIS_DIR:-video0_video1_video2}"
VAL_VIS_VIDEO_ID="${VAL_VIS_VIDEO_ID:-video0_video1_video2}"
TRAIN_VIS_DIR="${TRAIN_VIS_DIR:-video6867_video6868_video6869_video6870}"
TRAIN_VIS_VIDEO_ID="${TRAIN_VIS_VIDEO_ID:-video6867_video6868_video6869_video6870}"

ALIGNMENT_LOSS="${ALIGNMENT_LOSS:-infonce}"
INFO_NCE_TEMP="${INFO_NCE_TEMP:-0.07}"
LAMBDA_ATTN="${LAMBDA_ATTN:-0.1}"
LAMBDA_COV="${LAMBDA_COV:-0.0}"
LAMBDA_STOP="${LAMBDA_STOP:-0.5}"
LAMBDA_SCENE_QUERY="${LAMBDA_SCENE_QUERY:-0}"
LAMBDA_SCENE_DIVERSITY="${LAMBDA_SCENE_DIVERSITY:-0.1}"
# LAMBDA_TRIPLET="${LAMBDA_TRIPLET:-0.5}"

MAX_GENERATION_STEPS="${MAX_GENERATION_STEPS:-10}"
EOS_THRESHOLD="${EOS_THRESHOLD:-0.6}"
# MAX_SAMPLE_INFER="${MAX_SAMPLE_INFER:-30}"

EXTRA_ARGS=("$@")

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${LOG_DIR:-./logs/${DATASET}/${TIMESTAMP}_L${DECODER_LAYERS}H${DECODER_HEADS}}"
LOG_FILE="${LOG_DIR}/train.log"
mkdir -p "${LOG_DIR}"
LOSS_PLOT_PATH="${LOSS_PLOT_PATH:-${LOG_DIR}/loss_curve.png}"

if [[ -z "${INFERENCE_OUTPUT}" ]]; then
  INFERENCE_OUTPUT="${LOG_DIR}/val_predictions.jsonl"
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="${LOG_DIR}/checkpoint/scene_transformer.ckpt"
fi
mkdir -p "$(dirname "${CHECKPOINT_PATH}")"

# Prime the log with the current hyperparameter configuration.
{
  echo "================ TRAIN CONFIG ================"
  echo "date: $(date +"%F %T")"
  echo "script: $0"
  echo "cwd: $(pwd)"
  echo "cmdline: $0 $*"
  echo "LOG_DIR: ${LOG_DIR}"
  echo "MODE: ${MODE}"
  echo "DATASET: ${DATASET}"
  echo "DEVICE: ${DEVICE}"
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  echo "EPOCHS: ${EPOCHS}"
  echo "LR: ${LR}"
  echo "NUM_WORKERS: ${NUM_WORKERS}"
  echo "SAMPLE_FPS: ${SAMPLE_FPS}"
  echo "FRAMES_PER_CLIP: ${FRAMES_PER_CLIP}"
  echo "CLIP_STRIDE: ${CLIP_STRIDE}"
  echo "FRAME_SIZE: ${FRAME_SIZE}"
  echo "CLIP_BATCH_SIZE: ${CLIP_BATCH_SIZE}"
  echo "SS_ENABLE: ${SS_ENABLE}"
  echo "SS_P_MAX: ${SS_P_MAX}"
  echo "SS_WARMUP_EPOCHS: ${SS_WARMUP_EPOCHS}"
  echo "DECODER_LAYERS: ${DECODER_LAYERS}"
  echo "DECODER_HEADS: ${DECODER_HEADS}"
  echo "DECODER_FF_DIM: ${DECODER_FF_DIM}"
  echo "DECODER_DROPOUT: ${DECODER_DROPOUT}"
  echo "EARLY_STOPPING_PATIENCE: ${EARLY_STOPPING_PATIENCE}"
  echo "ALIGNMENT_LOSS: ${ALIGNMENT_LOSS}"
  echo "INFO_NCE_TEMP: ${INFO_NCE_TEMP}"
  echo "LAMBDA_ATTN: ${LAMBDA_ATTN}"
  echo "LAMBDA_COV: ${LAMBDA_COV}"
  echo "LAMBDA_STOP: ${LAMBDA_STOP}"
  echo "LAMBDA_SCENE_QUERY: ${LAMBDA_SCENE_QUERY}"
  echo "LAMBDA_SCENE_DIVERSITY: ${LAMBDA_SCENE_DIVERSITY}"
  echo "MAX_GENERATION_STEPS: ${MAX_GENERATION_STEPS}"
  echo "EOS_THRESHOLD: ${EOS_THRESHOLD}"
  echo "RUN_INFERENCE: ${RUN_INFERENCE}"
  echo "INFERENCE_INTERVAL: ${INFERENCE_INTERVAL}"
  echo "INFERENCE_LIMIT: ${INFERENCE_LIMIT}"
  echo "INFERENCE_OUTPUT: ${INFERENCE_OUTPUT}"
  echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
  echo "LOSS_PLOT_PATH: ${LOSS_PLOT_PATH}"
  echo "VIDEO_CACHE_ROOT: ${VIDEO_CACHE_ROOT}"
  echo "TEXT_CACHE_ROOT: ${TEXT_CACHE_ROOT}"
  echo "MSRVTT_FEAT_ROOT: ${MSRVTT_FEAT_ROOT}"
  echo "ACTIVITYNET_VIDEO_FEATURES: ${ACTIVITYNET_VIDEO_FEATURES}"
  echo "TVR_VIDEO_FEATURES: ${TVR_VIDEO_FEATURES}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    printf "EXTRA_ARGS:"
    for arg in "${EXTRA_ARGS[@]}"; do
      printf " %q" "${arg}"
    done
    printf "\n"
  fi
  echo "=============================================="
} | tee "${LOG_FILE}"

# Redirect everything to both stdout and train.log.
exec > >(tee -a "${LOG_FILE}") 2>&1

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
  --ss-p-max "${SS_P_MAX}"
  --ss-warmup-epochs "${SS_WARMUP_EPOCHS}"
  --decoder-layers "${DECODER_LAYERS}"
  --decoder-heads "${DECODER_HEADS}"
  --decoder-ff-dim "${DECODER_FF_DIM}"
  --decoder-dropout "${DECODER_DROPOUT}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --alignment-loss "${ALIGNMENT_LOSS}"
  --infonce-temp "${INFO_NCE_TEMP}"
  --lambda-attn "${LAMBDA_ATTN}"
  --lambda-cov "${LAMBDA_COV}"
  --lambda-stop "${LAMBDA_STOP}"
  --lambda-scene-query "${LAMBDA_SCENE_QUERY}"
  --lambda-scene-diversity "${LAMBDA_SCENE_DIVERSITY}"
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

if [[ "${USE_EXHAUSTIVE_CLIP_BANK,,}" == "true" ]]; then
  COMMON_ARGS+=(--use-exhaustive-clip-bank)
fi
if [[ "${DISABLE_TEXT_PROJECTION,,}" == "true" ]]; then
  COMMON_ARGS+=(--disable-text-projection)
fi
if [[ "${SS_ENABLE,,}" == "true" ]]; then
  COMMON_ARGS+=(--ss-enable)
fi

if [[ -n "${INFERENCE_LIMIT}" ]]; then
  COMMON_ARGS+=(--inference-limit "${INFERENCE_LIMIT}")
fi
if [[ -n "${VAL_VIS_DIR}" ]]; then
  COMMON_ARGS+=(--validation-visualization-dir "${VAL_VIS_DIR}")
fi
if [[ -n "${VAL_VIS_VIDEO_ID}" ]]; then
  COMMON_ARGS+=(--validation-visualization-video-id "${VAL_VIS_VIDEO_ID}")
fi
if [[ -n "${TRAIN_VIS_DIR}" ]]; then
  COMMON_ARGS+=(--train-visualization-dir "${TRAIN_VIS_DIR}")
fi
if [[ -n "${TRAIN_VIS_VIDEO_ID}" ]]; then
  COMMON_ARGS+=(--train-visualization-video-id "${TRAIN_VIS_VIDEO_ID}")
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
  tvr)
    COMMON_ARGS+=(
      --tvr-root "${TVR_ROOT}"
      --tvr-video-features "${TVR_VIDEO_FEATURES}"
      --tvr-text-features "${TVR_TEXT_FEATURES}"
      --tvr-train-json "${TVR_TRAIN_JSON}"
      --tvr-val-json "${TVR_VAL_JSON}"
      --tvr-inference-json "${TVR_INFER_JSON}"
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
