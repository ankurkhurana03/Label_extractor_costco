#!/usr/bin/env python3
"""Set up Label Studio for iterative Costco label OBB annotation correction.

Creates a project, imports images, and attaches pre-annotations from existing
YOLO OBB label files (or from model inference) so you can visually correct
polygon labels and retrain.

Usage:
    # From existing label .txt files (default):
    python scripts/setup_label_studio.py --token <YOUR_API_TOKEN>

    # From YOLO model predictions:
    python scripts/setup_label_studio.py --token <TOKEN> --source model --model runs/costco_label_obb/weights/best.pt

    # Import only test split:
    python scripts/setup_label_studio.py --token <TOKEN> --split test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
PROJECT_NAME = "Costco Labels OBB"

# Label Studio labeling config for polygon annotation (OBB-compatible)
LABELING_CONFIG = """\
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image">
    <Label value="costco_label" background="green"/>
  </PolygonLabels>
</View>
"""

CLASS_NAMES = {0: "costco_label"}


def resolve_token(url: str, token: str) -> dict:
    """Resolve the API token into auth headers.

    Label Studio 1.22+ uses JWT tokens. The Personal Access Token shown in the
    UI is a *refresh* token. We exchange it for a short-lived access token and
    use Bearer auth.  Falls back to the legacy ``Token`` header for older
    versions or classic hex tokens.
    """
    # Try legacy Token auth first (works on older LS versions)
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    r = requests.get(f"{url}/api/projects", headers=headers, params={"page_size": 1}, timeout=5)
    if r.status_code == 200:
        return headers

    # Try exchanging as a JWT refresh token
    try:
        r = requests.post(
            f"{url}/api/token/refresh",
            json={"refresh": token},
            timeout=5,
        )
        if r.status_code == 200:
            access = r.json()["access"]
            return {"Authorization": f"Bearer {access}", "Content-Type": "application/json"}
    except Exception:
        pass

    sys.exit(
        "Error: Could not authenticate with the provided token.\n"
        "Make sure you copied the full Personal Access Token from\n"
        "Label Studio > Account & Settings > Personal Access Token."
    )


def check_server(url: str) -> None:
    """Verify Label Studio is reachable."""
    try:
        r = requests.get(f"{url}/api/version", timeout=5)
        r.raise_for_status()
    except requests.ConnectionError:
        sys.exit(
            f"Error: Cannot reach Label Studio at {url}\n"
            "Start it with:\n"
            "  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\\n"
            "  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/ \\\n"
            "  label-studio start"
        )


def get_or_create_project(url: str, headers: dict) -> int:
    """Return the project ID, creating it if it doesn't exist."""
    r = requests.get(f"{url}/api/projects", headers=headers, params={"page_size": 100})
    r.raise_for_status()
    for proj in r.json().get("results", []):
        if proj["title"] == PROJECT_NAME:
            print(f"Found existing project '{PROJECT_NAME}' (id={proj['id']})")
            return proj["id"]

    r = requests.post(
        f"{url}/api/projects",
        headers=headers,
        json={
            "title": PROJECT_NAME,
            "label_config": LABELING_CONFIG,
        },
    )
    r.raise_for_status()
    pid = r.json()["id"]
    print(f"Created project '{PROJECT_NAME}' (id={pid})")
    return pid


def collect_image_paths(split: str) -> list[Path]:
    """Collect image file paths for the requested split(s)."""
    splits = ["train", "test"] if split == "both" else [split]
    images = []
    for s in splits:
        img_dir = DATASET_DIR / "images" / s
        if not img_dir.exists():
            print(f"Warning: {img_dir} not found, skipping")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            images.extend(img_dir.glob(ext))
    images.sort(key=lambda p: p.name)
    return images


def yolo_obb_to_ls_polygon(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Convert YOLO OBB label file to Label Studio polygon annotations.

    YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
    Label Studio: points [[x1*100, y1*100], ...] (percentage 0-100)
    """
    if not label_path.exists():
        return []

    annotations = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        # Convert normalized coords to percentage
        points = []
        for i in range(0, 8, 2):
            points.append([coords[i] * 100, coords[i + 1] * 100])

        annotations.append({
            "type": "polygonlabels",
            "value": {
                "points": points,
                "polygonlabels": [CLASS_NAMES.get(cls_id, f"class_{cls_id}")],
            },
            "from_name": "label",
            "to_name": "image",
            "original_width": img_w,
            "original_height": img_h,
        })
    return annotations


def predictions_from_model(model_path: str, image_paths: list[Path]) -> dict[str, list[dict]]:
    """Run YOLO model inference and convert results to LS annotations."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Error: ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(model_path)
    preds_map: dict[str, list[dict]] = {}

    print(f"Running inference on {len(image_paths)} images...")
    results = model.predict(
        source=[str(p) for p in image_paths],
        task="obb",
        verbose=False,
    )

    for img_path, result in zip(image_paths, results):
        img_w, img_h = result.orig_shape[1], result.orig_shape[0]
        annotations = []
        if result.obb is not None and len(result.obb):
            for box, cls_tensor, conf_tensor in zip(
                result.obb.xyxyxyxy, result.obb.cls, result.obb.conf
            ):
                cls_id = int(cls_tensor.item())
                conf = float(conf_tensor.item())
                # xyxyxyxy is in pixel coords — normalize to percentage
                pts = box.cpu().numpy()
                points = []
                for pt in pts:
                    points.append([float(pt[0]) / img_w * 100, float(pt[1]) / img_h * 100])

                ann = {
                    "type": "polygonlabels",
                    "value": {
                        "points": points,
                        "polygonlabels": [CLASS_NAMES.get(cls_id, f"class_{cls_id}")],
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": img_w,
                    "original_height": img_h,
                    "score": conf,
                }
                annotations.append(ann)
        preds_map[str(img_path)] = annotations

    return preds_map


def build_tasks(
    image_paths: list[Path],
    source: str,
    model_path: str | None,
) -> list[dict]:
    """Build Label Studio task dicts with pre-annotations."""
    # If using model, run inference first
    model_preds = {}
    if source == "model":
        if not model_path:
            sys.exit("Error: --model is required when --source model")
        model_preds = predictions_from_model(model_path, image_paths)

    tasks = []
    for img_path in image_paths:
        # Image URL for local file serving
        image_url = f"/data/local-files/?d={img_path}"

        # Get image dimensions
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        # Get annotations
        if source == "model":
            annotations = model_preds.get(str(img_path), [])
        else:
            # Derive label path from image path: images/split/name.jpg -> labels/split/name.txt
            rel = img_path.relative_to(DATASET_DIR / "images")
            label_path = DATASET_DIR / "labels" / rel.with_suffix(".txt")
            annotations = yolo_obb_to_ls_polygon(label_path, img_w, img_h)

        task: dict = {"data": {"image": image_url}}
        if annotations:
            task["predictions"] = [{"result": annotations}]
        tasks.append(task)

    return tasks


def import_tasks(url: str, headers: dict, project_id: int, tasks: list[dict]) -> None:
    """Import tasks into the project via API."""
    if not tasks:
        print("No tasks to import.")
        return

    r = requests.post(
        f"{url}/api/projects/{project_id}/import",
        headers=headers,
        json=tasks,
    )
    r.raise_for_status()
    resp = r.json()
    count = resp.get("task_count", len(tasks))
    print(f"Imported {count} tasks into project {project_id}")


def print_workflow(url: str, project_id: int) -> None:
    """Print next-steps workflow instructions."""
    print(f"""
Setup complete!

Next steps:
  1. Open {url}/projects/{project_id} in your browser
  2. Review and correct the polygon labels on each image
  3. Export corrected annotations:
       - UI: Project > Export > YOLO OBB
       - API: curl -H "Authorization: Token <token>" \\
              "{url}/api/projects/{project_id}/export?exportType=YOLO_OBB" -o export.zip
  4. Retrain with corrected labels:
       python scripts/retrain.py --annotations <exported_dir>
  5. Re-import model predictions for another round:
       python scripts/setup_label_studio.py --token <token> --source model \\
              --model runs/costco_label_obb/weights/best.pt
""")


def main():
    parser = argparse.ArgumentParser(
        description="Set up Label Studio for Costco label OBB annotation correction"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("LABEL_STUDIO_API_KEY"),
        help="API token (or set LABEL_STUDIO_API_KEY env var)",
    )
    parser.add_argument(
        "--source",
        choices=["labels", "model"],
        default="labels",
        help="Pre-annotation source: 'labels' from .txt files (default) or 'model' inference",
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "runs" / "costco_label_obb" / "weights" / "best.pt"),
        help="YOLO model path (used with --source model)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to import (default: both)",
    )
    args = parser.parse_args()

    if not args.token:
        sys.exit(
            "Error: --token is required (or set LABEL_STUDIO_API_KEY env var).\n"
            "Find your token at: Label Studio > Account & Settings > Access Token"
        )

    url = args.url.rstrip("/")

    # 1. Check server
    print(f"Checking Label Studio at {url}...")
    check_server(url)
    print("Label Studio is running.")

    # Resolve token (handles both legacy and JWT refresh tokens)
    print("Authenticating...")
    headers = resolve_token(url, args.token)
    print("Authenticated successfully.")

    # 2. Create or find project
    project_id = get_or_create_project(url, headers)

    # 3. Collect images
    image_paths = collect_image_paths(args.split)
    if not image_paths:
        sys.exit(f"No images found for split '{args.split}' in {DATASET_DIR / 'images'}")
    print(f"Found {len(image_paths)} images for split '{args.split}'")

    # 4. Build tasks with pre-annotations
    print(f"Building tasks with pre-annotations from {args.source}...")
    tasks = build_tasks(image_paths, args.source, args.model)
    annotated = sum(1 for t in tasks if "predictions" in t)
    print(f"  {annotated}/{len(tasks)} tasks have pre-annotations")

    # 5. Import
    import_tasks(url, headers, project_id, tasks)

    # 6. Print workflow
    print_workflow(url, project_id)


if __name__ == "__main__":
    main()
