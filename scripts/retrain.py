#!/usr/bin/env python3
"""Import new annotations from Label Studio and retrain the model.

Merges newly exported annotations with the existing dataset and
retrains the YOLO26-OBB model.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def merge_annotations(export_dir: Path):
    """Merge Label Studio YOLO OBB export into the training dataset."""
    export_images = export_dir / "images"
    export_labels = export_dir / "labels"

    if not export_labels.exists():
        # Some exports put everything flat
        export_images = export_dir
        export_labels = export_dir

    added = 0
    for label_file in export_labels.glob("*.txt"):
        dst_label = DATASET / "labels" / "train" / label_file.name
        shutil.copy2(label_file, dst_label)

        # Find and copy matching image
        stem = label_file.stem
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            img = export_images / (stem + ext)
            if img.exists():
                shutil.copy2(img, DATASET / "images" / "train" / img.name)
                break

        added += 1

    return added


def main():
    parser = argparse.ArgumentParser(description="Retrain with new Label Studio annotations")
    parser.add_argument("--annotations", required=True, help="Path to Label Studio export directory")
    parser.add_argument("--model", default=None, help="Base model (default: last best.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    export_dir = Path(args.annotations)
    if not export_dir.exists():
        print(f"Error: Export directory not found: {export_dir}")
        return

    # Merge new annotations
    print(f"Merging annotations from {export_dir}...")
    added = merge_annotations(export_dir)
    print(f"Added {added} new annotations to training set")

    # Determine base model
    if args.model:
        model_path = args.model
    else:
        model_path = ROOT / "runs" / "costco_label_obb" / "weights" / "best.pt"
        if not Path(model_path).exists():
            model_path = "yolo26n-obb.pt"
            print("No previous best.pt found, starting from pretrained weights")

    print(f"Retraining from {model_path}...")
    model = YOLO(str(model_path))

    model.train(
        data=str(ROOT / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        patience=15,
        device=args.device,
        project=str(ROOT / "runs"),
        name="costco_label_obb",
        exist_ok=True,
    )

    print("\nRetraining complete!")
    print(f"Updated model: {ROOT / 'runs' / 'costco_label_obb' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
