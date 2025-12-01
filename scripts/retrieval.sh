#!/usr/bin/env bash
set -euo pipefail

python retrieval_eval.py \
  --mode retrieval \
  --dataset activitynet \
  --checkpoint /dev/ssd1/gjw/prvr/logs/activitynet/20251130_190627_L12H8/checkpoint/scene_transformer.ckpt \
  --decoder-layers 12 \
  --decoder-heads 8 \
  --internvideo-root /dev/ssd1/gjw/prvr/InternVideo \
  --device cuda
