#!/usr/bin/env python3
"""
Utility script to inspect InternVideo QVHighlights embedding files.

Prints the internal structure (groups/datasets with shapes/dtypes) of the
video/text HDF5 files and shows one sample from a dataset in each file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np


DATA_DIR = Path("dataset/qvhighlights")
DEFAULT_FILES = [
    # DATA_DIR / "internvideo_video_embeddings.h5",
    DATA_DIR / "internvideo_text_embeddings.h5",
]


def visit_items(name: str, obj: h5py.Dataset | h5py.Group) -> None:
    indent = "  " * (name.count("/") + 1)
    if isinstance(obj, h5py.Dataset):
        print(
            f"{indent}- dataset `{name}` shape={obj.shape} dtype={obj.dtype}"
        )
    else:
        print(f"{indent}- group `{name}`")


def find_first_dataset_key(h5_file: h5py.File) -> str | None:
    first_key: str | None = None

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        nonlocal first_key
        if first_key is None and isinstance(obj, h5py.Dataset):
            first_key = name

    h5_file.visititems(visitor)
    return first_key


def format_sample(sample: np.ndarray | np.generic | bytes | str | float | int) -> str:
    if isinstance(sample, (bytes, bytearray)):
        try:
            return sample.decode("utf-8")
        except UnicodeDecodeError:
            return repr(sample)

    if isinstance(sample, np.ndarray):
        if sample.ndim == 0:
            return repr(sample.item())
        # Ensure large arrays are truncated to avoid flooding stdout.
        return np.array2string(sample, threshold=16, edgeitems=4, floatmode="maxprec")

    if np.isscalar(sample):
        return repr(sample.item() if isinstance(sample, np.generic) else sample)

    return repr(sample)


def inspect_file(path: Path, sample_index: int) -> None:
    print(f"\n=== {path} ===")
    if not path.exists():
        print("  File not found.")
        return

    with h5py.File(path, "r") as h5_file:
        print("  Top-level keys:", list(h5_file.keys()))
        h5_file.visititems(visit_items)

        first_dataset_key = find_first_dataset_key(h5_file)
        if first_dataset_key is None:
            print("  No datasets found.")
            return

        dataset = h5_file[first_dataset_key]
        if len(dataset) == 0:
            print(f"  Dataset `{first_dataset_key}` is empty.")
            return

        idx = sample_index % len(dataset)
        sample = dataset[idx]
        print(
            f"  Sample index {idx} from `{first_dataset_key}`: {format_sample(sample)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect InternVideo embedding HDF5 files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=DEFAULT_FILES,
        help="One or more HDF5 files to inspect.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index to display from each file (defaults to 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.files:
        inspect_file(path, args.sample_index)


if __name__ == "__main__":
    main()
