#!/usr/bin/env python3
"""Export YOLO OBB model optimized for mobile deployment (iOS + Android).

Exports:
  - CoreML (.mlpackage) — iOS, uses Apple Neural Engine
  - ONNX (.onnx) — Android via ONNX Runtime Mobile, also cross-platform
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "runs" / "costco_label_obb" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Export model for mobile")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model weights path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size (must match training imgsz)")
    args = parser.parse_args()

    model = YOLO(args.model)
    exports = ROOT / "exports"
    exports.mkdir(exist_ok=True)
    weights_dir = Path(args.model).parent

    # --- iOS: CoreML with FP16 ---
    print("=" * 50)
    print(f"[1/2] CoreML (iOS) — FP16, imgsz={args.imgsz}")
    print("=" * 50)
    model.export(
        format="coreml",
        imgsz=args.imgsz,
        half=True,
    )

    # Move .mlpackage to exports/
    for f in weights_dir.glob("*.mlpackage"):
        dest = exports / f.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(f), str(dest))

    # --- Android: ONNX (use with ONNX Runtime Mobile) ---
    print("\n" + "=" * 50)
    print(f"[2/2] ONNX (Android + cross-platform) — FP16, imgsz={args.imgsz}")
    print("=" * 50)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=True,
        simplify=True,
    )

    # Move .onnx to exports/
    for f in weights_dir.glob("*.onnx"):
        dest = exports / f.name
        f.rename(dest)

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print("Export complete!")
    print(f"{'=' * 50}")
    for f in sorted(exports.iterdir()):
        if f.name.startswith("."):
            continue
        if f.is_file():
            size = f.stat().st_size / (1024 * 1024)
        else:
            size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file()) / (1024 * 1024)
        print(f"  {f.name:40s} {size:.1f} MB")

    print(f"""
Mobile integration:

  iOS (CoreML + Apple Neural Engine):
    - Add best.mlpackage to your Xcode project
    - Use Vision framework VNCoreMLRequest for inference
    - Input: {args.imgsz}x{args.imgsz} RGB image
    - Expected latency: ~5-15ms on iPhone 13+

  Android (ONNX Runtime Mobile):
    - Add best.onnx to app/src/main/assets/
    - Add dependency: com.microsoft.onnxruntime:onnxruntime-android
    - Input: {args.imgsz}x{args.imgsz} RGB, normalized [0,1]
    - Expected latency: ~15-30ms on modern Android (NNAPI)

  To further reduce size, try INT8 quantization:
    yolo export model={args.model} format=onnx imgsz={args.imgsz} int8=True
""")


if __name__ == "__main__":
    main()
