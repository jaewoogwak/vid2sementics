#!/usr/bin/env bash
set -euo pipefail

# Batch-concatenate all segments per video_id under raw videos directory.
# Usage:
#   bash scripts/concat_all_segments.sh               # 하드코딩된 경로/병렬수 사용

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 하드코딩: 원시 세그먼트 디렉터리, 출력 디렉터리, 병렬 작업 수
RAW_DIR="/dev/ssd1/gjw/prvr/dataset/qvhighlights/raw/videos"
OUT_DIR="/dev/ssd1/gjw/prvr/dataset/qvhighlights/concat"
JOBS=4   # xargs 병렬 처리 개수

# 기타 설정
PYTHON_BIN="python"
REENCODE_FLAG=""   # 비워두면 스트림 복사, 재인코딩 원하면 아무 값이나 넣기

# Split JSONL paths (used to restrict segments to those listed in splits)
TRAIN_JSONL="/dev/ssd1/gjw/prvr/dataset/qvhighlights/highlight_train_release.jsonl"
VAL_JSONL="/dev/ssd1/gjw/prvr/dataset/qvhighlights/highlight_val_release.jsonl"

if [[ ! -d "$RAW_DIR" ]]; then
  echo "[ERROR] RAW_DIR not found: $RAW_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Preflight: ensure ffmpeg is available; allow override via $FFMPEG_BIN
if [[ -z "${FFMPEG_BIN:-}" ]]; then
  FFMPEG_BIN="$(command -v ffmpeg || true)"
fi
if [[ -z "$FFMPEG_BIN" ]]; then
  echo "[ERROR] ffmpeg not found. Install ffmpeg or set FFMPEG_BIN to its path." >&2
  echo "        Examples:" >&2
  echo "          - Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg" >&2
  echo "          - Conda : conda install -c conda-forge ffmpeg" >&2
  exit 1
fi
export FFMPEG_BIN

# Collect unique base video ids from split JSONLs only
mapfile -t VIDEO_IDS < <(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/list_split_video_ids.py" --which both
)

if (( ${#VIDEO_IDS[@]} == 0 )); then
  echo "[INFO] No MP4 files found in $RAW_DIR"
  exit 0
fi

echo "[INFO] Found ${#VIDEO_IDS[@]} unique video ids"

concat_one() {
  local vid="$1"
  local reencode_opt=()
  if [[ -n "$REENCODE_FLAG" ]]; then
    reencode_opt=("--reencode")
  fi
  # Skip if no segments exist for this id
  if ! compgen -G "$RAW_DIR/${vid}_*.mp4" > /dev/null; then
    echo "[SKIP] No segments found for $vid"
    return 0
  fi
  echo "[INFO] Concatenating: $vid"
  # Use "--" to ensure video IDs starting with '-' are not parsed as options
  "$PYTHON_BIN" "$REPO_ROOT/scripts/concat_segments.py" \
    --raw-dir "$RAW_DIR" \
    --outdir "$OUT_DIR" \
    --segments-jsonl "$TRAIN_JSONL" "$VAL_JSONL" \
    "${reencode_opt[@]}" \
    -- "$vid"
}

export -f concat_one
export REPO_ROOT RAW_DIR OUT_DIR PYTHON_BIN REENCODE_FLAG

printf '%s\n' "${VIDEO_IDS[@]}" | {
  if (( JOBS > 1 )); then
    xargs -r -I{} -P "$JOBS" bash -c 'concat_one "$@"' _ {}
  else
    while IFS= read -r vid; do
      concat_one "$vid"
    done
  fi
}

echo "[DONE] Outputs in: $OUT_DIR"
