#!/usr/bin/env python3
"""Set up and launch Label Studio for Costco label OBB annotation.

Generates the labeling config XML and prints instructions for:
- Starting Label Studio
- Connecting a YOLO ML backend for model-assisted labeling
- Exporting annotations in YOLO OBB format
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "label_studio_config.xml"

# Label Studio labeling config for polygon annotation (OBB-compatible)
LABELING_CONFIG = """\
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image">
    <Label value="costco_label" background="green"/>
  </PolygonLabels>
</View>
"""


def main():
    # Write labeling config
    CONFIG_PATH.write_text(LABELING_CONFIG)
    print(f"Labeling config written to: {CONFIG_PATH}")

    print("""
╔══════════════════════════════════════════════════════════════╗
║                  Label Studio Setup Guide                    ║
╚══════════════════════════════════════════════════════════════╝

1. START LABEL STUDIO
   label-studio start --port 8080

2. CREATE PROJECT
   - Open http://localhost:8080
   - Create new project "Costco Labels"
   - In Settings > Labeling Interface, paste the config from:
     {config}

3. IMPORT IMAGES
   - Settings > Cloud Storage > Add Source Storage
   - Or drag & drop images directly

4. MODEL-ASSISTED LABELING (optional)
   Set up YOLO ML backend for pre-annotations:

   a. Create ML backend script (ml_backend.py):
      from label_studio_ml.model import LabelStudioMLBase
      from ultralytics import YOLO

   b. Start ML backend:
      label-studio-ml start ml_backend/ --port 9090

   c. Connect in Label Studio:
      Settings > Machine Learning > Add Model
      URL: http://localhost:9090

5. EXPORT ANNOTATIONS
   - Project > Export > YOLO OBB format
   - Or use API: curl http://localhost:8080/api/projects/1/export?exportType=YOLO_OBB

6. RETRAIN WITH NEW DATA
   python scripts/retrain.py --annotations <exported_dir>
""".format(config=CONFIG_PATH))


if __name__ == "__main__":
    main()
