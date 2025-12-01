#!/usr/bin/env python3
"""
Check if YouTube video IDs are publicly available.

Supports three methods:
  - yt-dlp  : Uses yt_dlp to probe availability (most robust; requires network and yt-dlp)
  - oembed  : Fast heuristic via public oEmbed endpoint (no API key)
  - api     : YouTube Data API v3 (requires API key; accurate statuses)

Inputs:
  - A text file with one ID per line (e.g., missing_train_ids.txt), OR
  - A JSONL file with an id field (e.g., {"video_id": "..."})

Outputs:
  - CSV to stdout by default: id,status,detail
  - Optionally write CSV to a file via --output

Examples:
  python scripts/check_youtube_ids.py --input dataset/qvhighlights/concat_missing/missing_train_ids.txt --method oembed
  YT_API_KEY=xxxx python scripts/check_youtube_ids.py --input ids.txt --method api --output result.csv
  python scripts/check_youtube_ids.py --input ids.txt --method yt-dlp --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def read_ids_from_txt(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def read_ids_from_jsonl(path: Path, field: str) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = obj.get(field)
            if isinstance(v, str) and v:
                ids.append(v)
    return ids


def probe_oembed(video_id: str, timeout: float = 6.0) -> Tuple[str, str]:
    import urllib.request
    import urllib.error

    url = f"https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return ("public_or_unlisted", "oembed:200")
            return ("unknown", f"oembed:{resp.status}")
    except urllib.error.HTTPError as e:
        # Common codes: 401/403/404 for private/removed/age/region restrictions
        return ("unavailable", f"oembed_http:{e.code}")
    except Exception as e:
        return ("error", f"oembed_exc:{type(e).__name__}")


def probe_api(video_id_batch: List[str], api_key: str, timeout: float = 8.0) -> List[Tuple[str, str, str]]:
    import urllib.request
    import urllib.parse
    import urllib.error

    ids_param = ",".join(video_id_batch)
    q = urllib.parse.urlencode({
        "part": "status",
        "id": ids_param,
        "key": api_key,
    })
    url = f"https://www.googleapis.com/youtube/v3/videos?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        return [(vid, "error", f"api_exc:{type(e).__name__}") for vid in video_id_batch]

    found = {item["id"]: item for item in data.get("items", [])}
    results: List[Tuple[str, str, str]] = []
    for vid in video_id_batch:
        item = found.get(vid)
        if not item:
            results.append((vid, "not_found", "api:not_in_items"))
            continue
        st = item.get("status", {})
        privacy = st.get("privacyStatus", "?")
        upload = st.get("uploadStatus", "?")
        made_for_kids = st.get("madeForKids", False)
        emb = st.get("embeddable", True)
        detail = f"privacy={privacy};upload={upload};emb={emb};kids={made_for_kids}"
        results.append((vid, privacy, detail))
    return results


def probe_ytdlp(video_id: str) -> Tuple[str, str]:
    try:
        import yt_dlp  # type: ignore
    except Exception:
        return ("error", "yt_dlp_not_installed")

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "nocheckcertificate": True,
        "extract_flat": True,
        "socket_timeout": 10.0,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        # If extraction succeeds, treat as available (public/unlisted/age)
        return ("available", f"yt_dlp:{info.get('extractor_key','YouTube')}")
    except yt_dlp.utils.DownloadError as e:  # type: ignore
        msg = str(e).lower()
        # Heuristics for common cases
        if "private" in msg:
            return ("private", "yt_dlp:private")
        if "copyright" in msg or "removed" in msg:
            return ("removed", "yt_dlp:removed")
        if "unavailable" in msg:
            return ("unavailable", "yt_dlp:unavailable")
        return ("error", "yt_dlp_download_error")
    except Exception as e:
        return ("error", f"yt_dlp_exc:{type(e).__name__}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check YouTube IDs availability")
    ap.add_argument("--input", type=Path, required=True, help="Path to txt or JSONL file")
    ap.add_argument("--format", choices=["txt", "jsonl"], default="txt")
    ap.add_argument("--id-field", type=str, default="video_id", help="Field name in JSONL records")
    ap.add_argument("--method", choices=["oembed", "api", "yt-dlp"], default="oembed")
    ap.add_argument("--api-key", type=str, default=os.environ.get("YT_API_KEY", ""))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=50, help="API batch size (api method)")
    ap.add_argument("--output", type=Path, default=None, help="Write CSV to this path instead of stdout")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    path = args.input.expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    if args.format == "txt":
        ids = read_ids_from_txt(path)
    else:
        ids = read_ids_from_jsonl(path, args.id_field)

    if not ids:
        print("No IDs found.")
        return

    rows: List[Tuple[str, str, str]] = []

    if args.method == "api":
        if not args.api_key:
            print("YouTube Data API key missing. Set --api-key or $YT_API_KEY.", file=sys.stderr)
            sys.exit(2)
        rows = []
        for i in range(0, len(ids), args.batch):
            batch = ids[i : i + args.batch]
            rows.extend(probe_api(batch, args.api_key))

    elif args.method == "oembed":
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(probe_oembed, vid): vid for vid in ids}
            for fut in as_completed(futs):
                vid = futs[fut]
                status, detail = fut.result()
                rows.append((vid, status, detail))

    else:  # yt-dlp
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(probe_ytdlp, vid): vid for vid in ids}
            for fut in as_completed(futs):
                vid = futs[fut]
                status, detail = fut.result()
                rows.append((vid, status, detail))

    # Output CSV
    out = sys.stdout
    if args.output is not None:
        outp = args.output.expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        out = outp.open("w", encoding="utf-8", newline="")
    writer = csv.writer(out)
    writer.writerow(["id", "status", "detail"])
    for vid, status, detail in rows:
        writer.writerow([vid, status, detail])
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()

