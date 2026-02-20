#!/usr/bin/env python3
"""Compare performance of two YOLO OBB models against test dataset."""

import argparse
from pathlib import Path
import json
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_MODEL = ROOT / "runs" / "costco_label_obb" / "weights" / "best.pt"
DATASET_YAML = ROOT / "dataset.yaml"
TEST_IMAGES = ROOT / "dataset" / "images" / "test"


def validate_model(model_path, dataset_yaml, model_name="Model"):
    """Validate a model against the dataset."""
    print(f"\n{'='*60}")
    print(f"Validating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}\n")
    
    try:
        model = YOLO(model_path)
        results = model.val(data=str(dataset_yaml), imgsz=640, verbose=False)
        
        # Extract metrics
        metrics = {
            'model_name': model_name,
            'model_path': str(model_path),
            'map50': float(results.box.map50) if hasattr(results.box, 'map50') else None,
            'map50_95': float(results.box.map) if hasattr(results.box, 'map') else None,
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else None,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else None,
        }
        
        print(f"\nResults for {model_name}:")
        print(f"  mAP50: {metrics['map50']:.4f}" if metrics['map50'] is not None else "  mAP50: N/A")
        print(f"  mAP50-95: {metrics['map50_95']:.4f}" if metrics['map50_95'] is not None else "  mAP50-95: N/A")
        print(f"  Precision: {metrics['precision']:.4f}" if metrics['precision'] is not None else "  Precision: N/A")
        print(f"  Recall: {metrics['recall']:.4f}" if metrics['recall'] is not None else "  Recall: N/A")
        
        return metrics
        
    except Exception as e:
        print(f"Error validating {model_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare two YOLO OBB models")
    parser.add_argument("--model1", required=True, help="First model path (e.g., /Users/ankur/Downloads/best.pt)")
    parser.add_argument("--model2", default=str(DEFAULT_TEST_MODEL), help="Second model path (default: repo model)")
    parser.add_argument("--dataset", default=str(DATASET_YAML), help="Dataset YAML path")
    args = parser.parse_args()
    
    # Validate paths
    model1_path = Path(args.model1)
    model2_path = Path(args.model2)
    dataset_path = Path(args.dataset)
    
    if not model1_path.exists():
        print(f"Error: Model 1 not found at {model1_path}")
        return
    
    if not model2_path.exists():
        print(f"Error: Model 2 not found at {model2_path}")
        return
    
    if not dataset_path.exists():
        print(f"Error: Dataset config not found at {dataset_path}")
        return
    
    print(f"\nModel Comparison Test")
    print(f"Dataset: {dataset_path}")
    print(f"Test images: {TEST_IMAGES}")
    
    # Validate both models
    metrics1 = validate_model(str(model1_path), str(dataset_path), "Model 1 (Downloaded)")
    metrics2 = validate_model(str(model2_path), str(dataset_path), "Model 2 (Repo)")
    
    # Compare results
    if metrics1 and metrics2:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"{'Metric':<20} {'Model 1':<15} {'Model 2':<15} {'Difference':<15}")
        print("-" * 65)
        
        for key in ['map50', 'map50_95', 'precision', 'recall']:
            v1 = metrics1.get(key)
            v2 = metrics2.get(key)
            if v1 is not None and v2 is not None:
                diff = v1 - v2
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                print(f"{key:<20} {v1:<15.4f} {v2:<15.4f} {diff_str:<15}")
        
        # Winner
        print("\n" + "-" * 65)
        if metrics1.get('map50_95') is not None and metrics2.get('map50_95') is not None:
            if metrics1['map50_95'] > metrics2['map50_95']:
                print("✓ Model 1 (Downloaded) has better mAP50-95")
            elif metrics2['map50_95'] > metrics1['map50_95']:
                print("✓ Model 2 (Repo) has better mAP50-95")
            else:
                print("= Models have equal mAP50-95")


if __name__ == "__main__":
    main()
