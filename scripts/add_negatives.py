#!/usr/bin/env python3
"""Add negative examples (images with no labels) to the dataset.

Copies non-label images into the training set with empty .txt label files,
teaching the model what is NOT a label.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def add_from_source(source_dir: Path, split: str):
    """Copy images from source_dir into dataset with empty label files."""
    img_dst = DATASET / "images" / split
    lbl_dst = DATASET / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    images = [p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"No images found in {source_dir}")
        return 0

    added = 0
    for img in images:
        dest_img = img_dst / f"neg_{img.name}"
        dest_lbl = lbl_dst / f"neg_{img.stem}.txt"
        shutil.copy2(img, dest_img)
        dest_lbl.write_text("")  # empty = no objects
        added += 1

    return added


def add_synthetic(split: str):
    """Generate synthetic negative images (blank, gradient, noise)."""
    img_dst = DATASET / "images" / split
    lbl_dst = DATASET / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    synthetics = {
        "synth_blank_white.jpg": np.ones((640, 640, 3), dtype=np.uint8) * 255,
        "synth_blank_black.jpg": np.zeros((640, 640, 3), dtype=np.uint8),
        "synth_noise.jpg": rng.integers(0, 256, (640, 640, 3), dtype=np.uint8),
        "synth_gradient.jpg": np.tile(
            np.linspace(0, 255, 640, dtype=np.uint8).reshape(1, 640, 1),
            (640, 1, 3),
        ),
    }

    import cv2

    added = 0
    for name, img in synthetics.items():
        cv2.imwrite(str(img_dst / name), img)
        (lbl_dst / name.replace(".jpg", ".txt")).write_text("")
        added += 1

    return added


def main():
    parser = argparse.ArgumentParser(description="Add negative examples to dataset")
    parser.add_argument("--source", type=str, default=None, help="Directory of non-label images")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic negatives")
    args = parser.parse_args()

    if not args.source and not args.synthetic:
        parser.error("Provide --source <dir> and/or --synthetic")

    total = 0
    if args.source:
        src = Path(args.source)
        if not src.exists():
            print(f"Error: Source directory not found: {src}")
            return
        n = add_from_source(src, args.split)
        print(f"Added {n} negative images from {src}")
        total += n

    if args.synthetic:
        n = add_synthetic(args.split)
        print(f"Added {n} synthetic negative images")
        total += n

    print(f"\nTotal negatives added to {args.split}: {total}")
    print("Re-run training to incorporate these examples.")


if __name__ == "__main__":
    main()
