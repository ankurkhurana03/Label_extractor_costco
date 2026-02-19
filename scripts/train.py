#!/usr/bin/env python3
"""Train YOLO26-OBB model on the Costco label dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26-OBB model")
    parser.add_argument("--model", default="yolo26n-obb.pt", help="Pretrained model (default: yolo26n-obb.pt)")
    parser.add_argument("--data", default=str(ROOT / "dataset.yaml"), help="Dataset config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", default=None, help="Device: cpu, 0, mps, etc.")
    parser.add_argument("--name", default="costco_label_obb", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        # Augmentation tuned for small dataset
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        scale=0.5,
        translate=0.1,
    )

    print(f"\nTraining complete!")
    print(f"Best model: {ROOT / 'runs' / args.name / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    main()
