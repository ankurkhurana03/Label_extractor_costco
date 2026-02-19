#!/usr/bin/env python3
"""
Preview augmentations on OBB dataset and save samples with drawn bounding boxes.

Generates augmented samples to visually verify that bounding box transforms
are correct. The same augmentation pipeline can later be plugged into
the training loop.
"""

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


# ---------------------------------------------------------------------------
# Augmentation pipeline — keypoint-safe transforms only
# ---------------------------------------------------------------------------

def get_augmentation_pipeline(imgsz: int = 640) -> A.Compose:
    """
    Build an Albumentations pipeline that transforms both the image
    and the OBB corner keypoints together.

    Every spatial transform here supports keypoint_params.
    """
    return A.Compose(
        [
            # --- Spatial transforms (these move keypoints) ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(
                rotate=(-20, 20),
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                shear=(-10, 10),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.7,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.RandomResizedCrop(
                size=(imgsz, imgsz),
                scale=(0.6, 1.0),
                ratio=(0.8, 1.2),
                p=0.4,
            ),

            # --- Pixel-only transforms (don't affect keypoints) ---
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.6
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.4
            ),
            A.ImageCompression(quality_range=(50, 95), p=0.3),

            # --- Resize to training size ---
            A.Resize(imgsz, imgsz),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",               # (x_pixel, y_pixel)
            remove_invisible=False,     # keep all 4 corners even if clipped
        ),
    )


# ---------------------------------------------------------------------------
# OBB label I/O
# ---------------------------------------------------------------------------

def load_obb_labels(label_path: Path) -> list[tuple[int, list[float]]]:
    """Load YOLO-OBB labels: class x1 y1 x2 y2 x3 y3 x4 y4 (normalised)."""
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))  # 8 floats
        labels.append((cls_id, coords))
    return labels


def denorm_keypoints(coords: list[float], w: int, h: int) -> list[tuple[float, float]]:
    """Convert normalised OBB coords to pixel keypoints [(x,y), ...]."""
    pts = []
    for i in range(0, 8, 2):
        pts.append((coords[i] * w, coords[i + 1] * h))
    return pts


def norm_keypoints(pts: list[tuple[float, float]], w: int, h: int) -> list[float]:
    """Convert pixel keypoints back to normalised flat list."""
    coords = []
    for x, y in pts:
        coords.append(np.clip(x / w, 0.0, 1.0))
        coords.append(np.clip(y / h, 0.0, 1.0))
    return coords


# ---------------------------------------------------------------------------
# Apply augmentation to one image + labels
# ---------------------------------------------------------------------------

def augment_sample(
    image: np.ndarray,
    labels: list[tuple[int, list[float]]],
    pipeline: A.Compose,
) -> tuple[np.ndarray, list[tuple[int, list[float]]]]:
    """
    Augment a single image and its OBB labels.

    All 4 corners of every OBB are fed as keypoints so they
    transform in lockstep with the image.
    """
    h, w = image.shape[:2]

    # Flatten all corners into one keypoint list, track which box they belong to
    all_kps = []
    box_meta = []  # (cls_id, start_idx) for reconstructing labels
    for cls_id, coords in labels:
        start = len(all_kps)
        pts = denorm_keypoints(coords, w, h)
        all_kps.extend(pts)
        box_meta.append((cls_id, start))

    transformed = pipeline(image=image, keypoints=all_kps)
    aug_image = transformed["image"]
    aug_kps = transformed["keypoints"]
    aug_h, aug_w = aug_image.shape[:2]

    # Reconstruct labels from transformed keypoints
    aug_labels = []
    for i, (cls_id, start) in enumerate(box_meta):
        end = box_meta[i + 1][1] if i + 1 < len(box_meta) else len(aug_kps)
        if end - start != 4:
            # Some corners were removed — skip this box
            continue
        pts = aug_kps[start:end]

        # Clamp to image bounds
        pts = [(np.clip(x, 0, aug_w), np.clip(y, 0, aug_h)) for x, y in pts]

        # Skip degenerate boxes (area too small)
        poly = np.array(pts, dtype=np.float32)
        area = cv2.contourArea(poly)
        if area < 100:  # pixels²
            continue

        coords = norm_keypoints(pts, aug_w, aug_h)
        aug_labels.append((cls_id, coords))

    return aug_image, aug_labels


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]


def draw_obb(image: np.ndarray, labels: list[tuple[int, list[float]]]) -> np.ndarray:
    """Draw OBB polygons on a copy of the image."""
    vis = image.copy()
    h, w = vis.shape[:2]
    for cls_id, coords in labels:
        pts = denorm_keypoints(coords, w, h)
        pts_int = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        color = COLORS[cls_id % len(COLORS)]
        cv2.polylines(vis, [pts_int], isClosed=True, color=color, thickness=2)
        # Draw corner dots so we can verify point order
        for j, (px, py) in enumerate(pts):
            radius = 6 - j  # first corner is biggest
            cv2.circle(vis, (int(px), int(py)), radius, color, -1)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preview OBB augmentations")
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of random images to augment",
    )
    parser.add_argument(
        "--augs-per-image", type=int, default=4,
        help="Augmented variants per source image",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for augmentation pipeline",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "augment_preview"),
        help="Output directory for preview images",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = get_augmentation_pipeline(args.imgsz)

    # Gather all training images
    img_dir = DATASET / "images" / "train"
    lbl_dir = DATASET / "labels" / "train"
    image_files = sorted(img_dir.glob("*.*"))
    chosen = random.sample(image_files, min(args.num_samples, len(image_files)))

    total_saved = 0
    for img_path in chosen:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"SKIP: cannot read {img_path.name}")
            continue

        labels = load_obb_labels(lbl_path)
        if not labels:
            print(f"SKIP: no labels for {img_path.name}")
            continue

        # Save original with boxes
        orig_vis = draw_obb(image, labels)
        cv2.imwrite(str(out_dir / f"{stem}_ORIG.jpg"), orig_vis)

        # Generate augmented variants
        for k in range(args.augs_per_image):
            aug_img, aug_labels = augment_sample(image, labels, pipeline)
            aug_vis = draw_obb(aug_img, aug_labels)
            cv2.imwrite(str(out_dir / f"{stem}_AUG{k}.jpg"), aug_vis)
            total_saved += 1

            # Also save the augmented label as text for inspection
            lbl_out = out_dir / f"{stem}_AUG{k}.txt"
            lines = []
            for cls_id, coords in aug_labels:
                coord_str = " ".join(f"{c:.6f}" for c in coords)
                lines.append(f"{cls_id} {coord_str}")
            lbl_out.write_text("\n".join(lines) + "\n")

    print(f"\nSaved {total_saved} augmented samples + originals to {out_dir}/")
    print("Open the images and verify bounding boxes visually.")


if __name__ == "__main__":
    main()
