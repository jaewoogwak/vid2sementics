#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="/dev/ssd1/gjw/prvr"
ROOT="${ROOT:-${ROOT_DEFAULT}}"

DATASET="${DATASET:-tvr}"
MODE="retrieval"
CHECKPOINT="${CHECKPOINT:-/dev/ssd1/gjw/prvr/logs/tvr/20251202_152048_L6H16/checkpoint/scene_transformer.ckpt}"

INTERNVIDEO_ROOT="${INTERNVIDEO_ROOT:-${ROOT}/InternVideo}"
INTERNVIDEO_CONFIG="${INTERNVIDEO_CONFIG:-${INTERNVIDEO_ROOT}/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py}"
INTERNVIDEO_CKPT="${INTERNVIDEO_CKPT:-${INTERNVIDEO_ROOT}/InternVideo2/ckpt/InternVideo2-stage2_1b-224p-f4.pt}"

MSRVTT_ROOT="${MSRVTT_ROOT:-${ROOT}/dataset/data/MSRVTT}"
MSRVTT_ANNOTATION="${MSRVTT_ANNOTATION:-${MSRVTT_ROOT}/annotation/MSRVTT_untrimmed.json}"
MSRVTT_FEAT_ROOT="${MSRVTT_FEAT_ROOT:-${MSRVTT_ROOT}/internvideo_untrimmed_feats}"
MSRVTT_JSFUSION_CSV="${MSRVTT_JSFUSION_CSV:-${MSRVTT_ROOT}/annotation/MSRVTT_JSFUSION_test.csv}"
MSRVTT_TRAIN_SPLIT="${MSRVTT_TRAIN_SPLIT:-train}"
MSRVTT_VAL_SPLIT="${MSRVTT_VAL_SPLIT:-val}"

ACTIVITYNET_ROOT="${ACTIVITYNET_ROOT:-${ROOT}/dataset/activitynet}"
ACTIVITYNET_VIDEO_FEATURES="${ACTIVITYNET_VIDEO_FEATURES:-${ACTIVITYNET_ROOT}/FeatureData/new_clip_vit_32_activitynet_vid_features.hdf5}"
ACTIVITYNET_TEXT_FEATURES="${ACTIVITYNET_TEXT_FEATURES:-${ACTIVITYNET_ROOT}/TextData/clip_ViT_B_32_activitynet_query_feat.hdf5}"
ACTIVITYNET_VAL_JSON="${ACTIVITYNET_VAL_JSON:-${ACTIVITYNET_ROOT}/TextData/val_1.json}"

TVR_ROOT="${TVR_ROOT:-${ROOT}/dataset/tvr}"
TVR_VIDEO_FEATURES="${TVR_VIDEO_FEATURES:-${TVR_ROOT}/FeatureData/new_clip_vit_32_tvr_vid_features.hdf5}"
TVR_TEXT_FEATURES="${TVR_TEXT_FEATURES:-${TVR_ROOT}/TextData/clip_ViT_B_32_tvr_query_feat.hdf5}"
TVR_VAL_JSON="${TVR_VAL_JSON:-${TVR_ROOT}/TextData/val.json}"

SCENE_CACHE_ROOT="${SCENE_CACHE_ROOT:-}" # optional, speeds up repeated runs

DEVICE="${DEVICE:-cuda}"
DECODER_LAYERS="${DECODER_LAYERS:-6}"
DECODER_HEADS="${DECODER_HEADS:-16}"
DECODER_FF_DIM="${DECODER_FF_DIM:-2048}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"
MAX_STEPS="${MAX_STEPS:-12}"
EOS_THRESHOLD="${EOS_THRESHOLD:-0.8}"
MAX_SCENE_CANDIDATES="${MAX_SCENE_CANDIDATES:-2048}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-8}"
FRAME_SIZE="${FRAME_SIZE:-224}"
DISABLE_TEXT_PROJECTION="${DISABLE_TEXT_PROJECTION:-false}"

EXTRA_ARGS=("$@")

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[retrieval.sh] Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

COMMON_ARGS=(
  --mode "${MODE}"
  --dataset "${DATASET}"
  --checkpoint "${CHECKPOINT}"
  --internvideo-root "${INTERNVIDEO_ROOT}"
  --internvideo-config "${INTERNVIDEO_CONFIG}"
  --internvideo-ckpt "${INTERNVIDEO_CKPT}"
  --device "${DEVICE}"
  --decoder-layers "${DECODER_LAYERS}"
  --decoder-heads "${DECODER_HEADS}"
  --decoder-ff-dim "${DECODER_FF_DIM}"
  --decoder-dropout "${DECODER_DROPOUT}"
  --max-generation-steps "${MAX_STEPS}"
  --eos-threshold "${EOS_THRESHOLD}"
  --max-scene-candidates "${MAX_SCENE_CANDIDATES}"
  --frames-per-clip "${FRAMES_PER_CLIP}"
  --frame-size "${FRAME_SIZE}"
)

if [[ -n "${SCENE_CACHE_ROOT}" ]]; then
  COMMON_ARGS+=(--scene-cache-root "${SCENE_CACHE_ROOT}")
fi
if [[ "${DISABLE_TEXT_PROJECTION,,}" == "true" ]]; then
  COMMON_ARGS+=(--disable-text-projection)
fi

case "${DATASET}" in
  msrvtt_untrimmed)
    COMMON_ARGS+=(
      --msrvtt-annotation "${MSRVTT_ANNOTATION}"
      --msrvtt-feat-root "${MSRVTT_FEAT_ROOT}"
      --msrvtt-jsfusion-csv "${MSRVTT_JSFUSION_CSV}"
      --msrvtt-train-split "${MSRVTT_TRAIN_SPLIT}"
      --msrvtt-val-split "${MSRVTT_VAL_SPLIT}"
    )
    ;;
  activitynet)
    COMMON_ARGS+=(
      --activitynet-root "${ACTIVITYNET_ROOT}"
      --activitynet-video-features "${ACTIVITYNET_VIDEO_FEATURES}"
      --activitynet-text-features "${ACTIVITYNET_TEXT_FEATURES}"
      --activitynet-val-json "${ACTIVITYNET_VAL_JSON}"
    )
    ;;
  tvr)
    COMMON_ARGS+=(
      --tvr-root "${TVR_ROOT}"
      --tvr-video-features "${TVR_VIDEO_FEATURES}"
      --tvr-text-features "${TVR_TEXT_FEATURES}"
      --tvr-val-json "${TVR_VAL_JSON}"
    )
    ;;
  *)
    echo "[retrieval.sh] Unsupported dataset: ${DATASET}" >&2
    exit 1
    ;;
esac

echo "Running retrieval evaluation for dataset=${DATASET}"
python retrieval_eval.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
