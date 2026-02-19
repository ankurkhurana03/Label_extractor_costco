#!/usr/bin/env python3
"""Run inference with a trained YOLO26-OBB model and visualize detections."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "runs" / "costco_label_obb" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="YOLO26-OBB inference")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "predict"), help="Output directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=args.save_dir,
        name="results",
        exist_ok=True,
    )

    for r in results:
        if r.obb is not None and len(r.obb):
            print(f"{Path(r.path).name}: {len(r.obb)} detection(s)")
            for box in r.obb:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  class={cls} conf={conf:.3f}")
        else:
            print(f"{Path(r.path).name}: no detections")

    print(f"\nResults saved to {args.save_dir}/results/")


if __name__ == "__main__":
    main()
