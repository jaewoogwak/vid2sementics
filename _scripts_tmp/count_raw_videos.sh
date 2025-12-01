#!/usr/bin/env bash
set -euo pipefail

# Count total number of MP4 files in the raw videos directory.
# Usage:
#   bash scripts/count_raw_videos.sh                 # uses default path
#   bash scripts/count_raw_videos.sh /path/to/dir    # custom directory
#   EXT=avi bash scripts/count_raw_videos.sh         # custom extension

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DIR="${1:-$REPO_ROOT/dataset/qvhighlights/raw/videos}"
EXT="${EXT:-mp4}"

if [[ ! -d "$DIR" ]]; then
  echo "[ERROR] Directory not found: $DIR" >&2
  exit 1
fi

# Use find for robustness; count only regular files with given extension (case-insensitive)
count=$(find "$DIR" -maxdepth 1 -type f \( -iname "*.${EXT}" \) | wc -l | tr -d ' ')

echo "Directory : $DIR"
echo "Extension : .${EXT}"
echo "Total files: $count"

