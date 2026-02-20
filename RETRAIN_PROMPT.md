# Fix YOLO OBB Model: Costco Label Detection

## Problem Statement

The current YOLO11n-OBB model trained to detect Costco price labels is **fundamentally broken**. It cannot distinguish real labels from blank images or random noise.

### Evidence (tested offline with coremltools on the exported CoreML model)

| Test Input | Max Sigmoid Confidence | Detections > 0.7 |
|---|---|---|
| Blank white image (640x640) | 0.76 | 1,530 |
| Random noise image (640x640) | 0.83 | 7,126 |
| Synthetic label (white rect + black text lines) | 0.89 | 181 |
| Real Costco label (via phone camera) | 0.84 | ~2,000 |

**The model's noise floor (0.76-0.83) is nearly equal to its real detection confidence (0.84).** No threshold can separate signal from noise.

Additionally, all high-confidence detections have **tiny bounding boxes** (w=30-50px on 640px input, ~5-8% of image) instead of encompassing the full label. The model detects small edge/corner features rather than whole labels.

### Raw confidence channel analysis

The model output is shape `(1, 6, 8400)` where channels = `[x, y, w, h, angle, class_conf]`. The raw class confidence values (before sigmoid) have an extremely narrow range:

- Blank image: raw conf range [-0.13, 1.15], mean 0.53
- Noise image: raw conf range [-0.16, 1.56], mean 1.08
- These should ideally be strongly negative (< -5) for non-label images

## Current Setup

### Dataset
- **Location**: `dataset/` directory
- **Size**: 64 training images, 64 test images (same count -- likely some overlap or very small dataset)
- **Classes**: 1 class (`costco_label`)
- **Label format**: YOLO OBB (8 values per box: 4 x,y corner pairs, normalized 0-1)
- **Task**: Oriented Bounding Box detection

### Sample label (3 labels in one image):
```
0 0.529223 0.170597 0.960440 0.167732 0.962291 0.446287 0.531073 0.449153
0 0.053672 0.251412 0.355932 0.230226 0.351695 0.463277 0.043785 0.466102
0 0.875706 0.477401 0.977401 0.478814 0.975989 0.536723 0.874294 0.532486
```

### Previous Training (the broken model in `runs/costco_label_obb/`)
This model was trained with the OLD settings (from `runs/costco_label_obb/args.yaml`):
- **Model**: `yolo11n-obb.pt` (pretrained, nano variant)
- **Epochs**: 100 (with early stopping patience=20, stopped at ~24 epochs)
- **Image size**: 640x640
- **Batch size**: 8
- **Device**: MPS (Apple Silicon)
- **Augmentation**: mosaic=1.0, flipud=0.5, fliplr=0.5, degrees=15, scale=0.5, translate=0.1
- **No mixup, no copy_paste, no dropout, low regularization**

### Updated Training Script (`scripts/train.py`)
The `train.py` script has already been updated with better hyperparameters:
```python
model = YOLO("yolo26n-obb.pt")  # Newer model architecture
results = model.train(
    data=args.data,
    epochs=300,           # More epochs
    imgsz=640,
    batch=8,
    patience=50,          # More patience
    mosaic=1.0,
    flipud=0.5,
    fliplr=0.5,
    degrees=30.0,         # More rotation
    scale=0.5,
    translate=0.2,
    mixup=0.3,            # NEW: mixup augmentation
    copy_paste=0.3,       # NEW: copy-paste augmentation
    hsv_h=0.015,          # Color augmentation
    hsv_s=0.7,
    hsv_v=0.4,
    weight_decay=0.001,   # Stronger regularization
    dropout=0.1,          # NEW: dropout
)
```

### Reported Metrics (suspicious -- don't match real-world behavior)
- mAP50: 0.947
- mAP50-95: 0.804
- Precision: 0.902
- Recall: 0.933

These metrics look good on paper but the model fails completely in practice.

### Export (`scripts/export_mobile.py`)
- Already fixed to export at **640x640** (matching training size)
- Exports both CoreML `.mlpackage` (iOS) and ONNX (Android)

### Deployment
- iOS: CoreML via Vision framework (`VNCoreMLRequest`)
- Backend: ONNX via ultralytics
- Both use the same base `best.pt` weights

## Root Causes

### 1. Dataset is too small (64 images)
For a detection model to learn robust features, you typically need hundreds to thousands of annotated images. 64 images is far too few -- the model memorizes training data without learning generalizable features, leading to:
- High training/val metrics (memorization)
- Terrible real-world performance (no generalization)
- High confidence on everything (can't distinguish positive from negative)

### 2. No negative examples
The dataset appears to contain ONLY images with Costco labels. The model never sees images WITHOUT labels during training, so it never learns what "not a label" looks like. This explains why it fires on blank images and random noise.

### 3. Train/val contamination
Both train and val sets have exactly 64 images. These may overlap or be too similar, giving inflated validation metrics.

### 4. Early stopping too aggressive
Training stopped at epoch 24 out of 100 with patience=20. The model may not have converged. The val loss was still fluctuating.

## Required Fixes

### Fix 1: Expand the dataset significantly
- **Target**: 500+ images minimum, ideally 1000+
- **Include negative examples**: 30-40% of the dataset should be images WITHOUT any Costco labels (grocery stores, random products, other retailers, blank surfaces, etc.)
- **Diversity**: Different lighting, angles, distances, phone cameras, label styles (regular, clearance/asterisk, seasonal, multi-pack)
- **Clean train/val split**: Ensure no overlap, different store visits in train vs val

### Fix 2: Add background/negative class training
The `scripts/add_negatives.py` script is already in place. Use it:
```bash
# Add synthetic negatives (blank, noise, gradient)
python scripts/add_negatives.py --synthetic --split train
python scripts/add_negatives.py --synthetic --split test

# Add real non-label images from a directory
python scripts/add_negatives.py --source /path/to/non-label-images --split train
```

With Ultralytics YOLO, images with empty label files (0 annotations) act as negative examples. This teaches the model to output LOW confidence when no label is present.

### Fix 3: Retrain with the updated `scripts/train.py`
The training script already has the improved hyperparameters. Just run:
```bash
python scripts/train.py --model yolo26n-obb.pt --epochs 300 --patience 50
```

Or if you have more GPU memory, try the small variant for better features:
```bash
python scripts/train.py --model yolo26s-obb.pt --epochs 300 --patience 50
```

### Fix 4: Validate after retraining
The `scripts/validate_model.py` script is already in place:
```bash
python scripts/validate_model.py --conf 0.5
```

This tests blank images, noise images, and real test images. All synthetic tests must PASS (zero detections).

### Fix 5: Export correctly after retraining
The `scripts/export_mobile.py` is already configured correctly (640x640):
```bash
python scripts/export_mobile.py
```

This exports both CoreML (.mlpackage) and ONNX (.onnx) to the `exports/` directory.

## Quick Data Collection Strategy

To get from 64 images to 500+ quickly:
1. **Web scraping**: Search "Costco price tag", "Costco shelf label" for diverse examples
2. **Negative examples**: Use any non-Costco retail images, indoor scenes, random objects
3. **Synthetic augmentation**: Use `scripts/augment_preview.py` with existing 64 images (Albumentations)
4. **In-store photos**: Take 50-100 photos at Costco covering different aisles, lighting, angles
5. **Label Studio**: Already set up at `scripts/setup_label_studio.py` for annotation

## Existing Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/train.py` | Train YOLO OBB model (updated hyperparameters) |
| `scripts/export_mobile.py` | Export to CoreML + ONNX at 640x640 |
| `scripts/validate_model.py` | Validate model against blank/noise/real images |
| `scripts/add_negatives.py` | Add negative examples (synthetic + real) |
| `scripts/retrain.py` | Import Label Studio annotations and retrain |
| `scripts/augment_preview.py` | Preview augmentation on existing images |
| `scripts/predict.py` | Run inference on images |
| `scripts/setup_label_studio.py` | Set up Label Studio for annotation |

## Files to Modify

| File | Change |
|---|---|
| `dataset/` | Add 400+ images (including 30-40% negatives) with annotations |
| `dataset.yaml` | Verify train/val paths are correct and non-overlapping |

The scripts (`train.py`, `export_mobile.py`, `validate_model.py`, `add_negatives.py`) are already updated and ready to use.

## Success Criteria

After retraining, the model must satisfy ALL of these:
- [ ] Zero detections (conf > 0.5) on blank white image
- [ ] Zero detections (conf > 0.5) on random noise image
- [ ] Zero detections (conf > 0.5) on non-Costco retail images
- [ ] Detects real Costco labels with confidence > 0.85
- [ ] Bounding boxes encompass the full label (IoU > 0.7 with ground truth)
- [ ] Works at various distances (close-up, arm's length, 3 feet away)
- [ ] Works in different lighting (fluorescent store lighting, natural light, dim)
- [ ] mAP50 > 0.90 on a clean, non-overlapping validation set

## Step-by-Step Retraining Workflow

```bash
cd /Users/ankur/Documents/GitHub/Label_extractor_costco

# 1. Add synthetic negatives to both splits
python scripts/add_negatives.py --synthetic --split train
python scripts/add_negatives.py --synthetic --split test

# 2. (Optional) Add real negative images
# python scripts/add_negatives.py --source /path/to/non-label-images --split train

# 3. Train with updated settings
python scripts/train.py --model yolo26n-obb.pt --epochs 300 --patience 50

# 4. Validate the retrained model
python scripts/validate_model.py --conf 0.5

# 5. If validation passes, export for mobile
python scripts/export_mobile.py

# 6. Copy exports/best.mlpackage to CostcoCopilot iOS project
# Copy exports/best.onnx to backend for server-side inference
```
