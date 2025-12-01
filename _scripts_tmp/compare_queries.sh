#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Edit these two variables, then run:
#   bash scripts/compare_queries.sh
# ------------------------------------------------------------

# Positive (target) video ID hint: queries mapped to this video will be treated as positives
POSITIVE_HINT="L9cUEhaXnK4"

# Negative (other) video ID hint: queries mapped to this video will be treated as negatives
NEGATIVE_HINT="_i9qWLsZToY"

# Optional: how many top results to print (default: 10)
TOPK="10"

# Optional: if you want to override the video segment file used for encoding
# Leave empty to use the default from scripts/test.py
VIDEO_PATH="/dev/ssd1/gjw/prvr/dataset/qvhighlights/raw/videos/L9cUEhaXnK4_210.0_360.0.mp4"

# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON=${PYTHON:-python}

if [[ -z "$NEGATIVE_HINT" ]]; then
  echo "[ERROR] NEGATIVE_HINT is empty. Please edit scripts/compare_queries.sh and set NEGATIVE_HINT."
  exit 1
fi

CMD=("$PYTHON" "$REPO_ROOT/scripts/test.py" "--negative-hint" "$NEGATIVE_HINT" "--topk" "$TOPK")

if [[ -n "$POSITIVE_HINT" ]]; then
  CMD+=("--positive-hint" "$POSITIVE_HINT")
fi

if [[ -n "$VIDEO_PATH" ]]; then
  CMD+=("--video-path" "$VIDEO_PATH")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"

