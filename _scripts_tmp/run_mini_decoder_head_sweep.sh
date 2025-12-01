#!/usr/bin/env bash
set -euo pipefail

# Mini sweep over decoder layers/heads combinations for quick comparison.
# Usage: ./scripts/run_mini_decoder_head_sweep.sh [extra train_scene arguments]
# Any additional CLI args are forwarded to train_scene_autoregressive.sh.

ROOT_DEFAULT="/dev/ssd1/gjw/prvr"
ROOT="${ROOT:-${ROOT_DEFAULT}}"
REPO_DIR="${ROOT}"

SCRIPT="${REPO_DIR}/train_scene_autoregressive.sh"
if [[ ! -x "${SCRIPT}" ]]; then
  echo "Cannot find executable train_scene_autoregressive.sh at ${SCRIPT}" >&2
  exit 1
fi

declare -a CONFIGS=(
  "baseline 4 8"
  "more_heads 4 16"
  "deeper 6 8"
  "deeper_more_heads 6 16"
)

run_config() {
  local label="$1"
  local layers="$2"
  local heads="$3"
  shift 3
  echo ""
  echo "==== Running sweep config '${label}' (L=${layers}, H=${heads}) ===="
  DECODER_LAYERS="${layers}" \
  DECODER_HEADS="${heads}" \
  "${SCRIPT}" "$@"
}

for entry in "${CONFIGS[@]}"; do
  read -r label layers heads <<< "${entry}"
  run_config "${label}" "${layers}" "${heads}" "$@"
done
