#!/usr/bin/env python3
"""Validate YOLO OBB model against synthetic and real inputs.

Catches false-positive failure modes: blank images and random noise
should produce zero detections at reasonable confidence thresholds.
"""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "runs" / "costco_label_obb" / "weights" / "best.pt"
TEST_IMAGES = ROOT / "dataset" / "images" / "test"


def count_detections(results, conf_threshold):
    """Count detections above the confidence threshold."""
    count = 0
    max_conf = 0.0
    for r in results:
        if r.obb is not None and len(r.obb):
            confs = r.obb.conf.cpu().numpy()
            above = confs[confs >= conf_threshold]
            count += len(above)
            if len(confs):
                max_conf = max(max_conf, float(confs.max()))
    return count, max_conf


def test_blank(model, conf):
    """Blank white image should produce zero detections."""
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    results = model.predict(img, conf=conf, verbose=False)
    n, mc = count_detections(results, conf)
    passed = n == 0
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Blank white image: {n} detections (max conf {mc:.3f})")
    return passed


def test_noise(model, conf):
    """Random noise image should produce zero detections."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
    results = model.predict(img, conf=conf, verbose=False)
    n, mc = count_detections(results, conf)
    passed = n == 0
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Random noise image: {n} detections (max conf {mc:.3f})")
    return passed


def test_real_images(model, conf):
    """Report detection stats on real test images."""
    if not TEST_IMAGES.exists():
        print(f"  [SKIP] Test images not found at {TEST_IMAGES}")
        return True

    images = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        print(f"  [SKIP] No images in {TEST_IMAGES}")
        return True

    total_det = 0
    detected_count = 0
    for img_path in images:
        results = model.predict(str(img_path), conf=conf, verbose=False)
        n, mc = count_detections(results, conf)
        total_det += n
        if n > 0:
            detected_count += 1
        print(f"    {img_path.name}: {n} detections (max conf {mc:.3f})")

    pct = detected_count / len(images) * 100
    print(f"  [INFO] {detected_count}/{len(images)} images had detections ({pct:.0f}%)")
    print(f"  [INFO] Total detections across test set: {total_det}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO OBB model sanity")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}\n")

    model = YOLO(args.model)
    results = []

    print("Synthetic tests:")
    results.append(test_blank(model, args.conf))
    results.append(test_noise(model, args.conf))

    print("\nReal test images:")
    results.append(test_real_images(model, args.conf))

    print("\n" + "=" * 40)
    if all(results):
        print("PASSED: Model does not fire on blank/noise inputs.")
    else:
        print("FAILED: Model produces false positives on blank/noise inputs.")
        print("Consider adding negative examples and retraining.")
    print("=" * 40)


if __name__ == "__main__":
    main()
