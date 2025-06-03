import os
import cv2
import numpy as np
import joblib
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
YOLO_MODEL_PATH = "runs/segment/fruit_yolo_training/weights/best.pt"
REGRESSOR_MODEL_PATH = "regression_data/feature_regressor.pkl"
OUTPUT_DIR = "inference_outputs"
CONF_THRESHOLD = 0.25

# 🔧 HARD-CODED INPUT (change this as needed)
INPUT_PATH = "inference_inputs/"  # Can be a single image or folder

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
def draw_prediction(image, box, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def is_image_file(f):
    return f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))

# ---------------- INFERENCE ----------------
def run_inference():
    # Load models
    print("🔄 Loading YOLO model and regressor...")
    yolo = YOLO(YOLO_MODEL_PATH)
    regressor = joblib.load(REGRESSOR_MODEL_PATH)

    # Handle image(s)
    if os.path.isfile(INPUT_PATH) and is_image_file(INPUT_PATH):
        image_paths = [INPUT_PATH]
    elif os.path.isdir(INPUT_PATH):
        image_paths = [
            os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if is_image_file(f)
        ]
    else:
        raise ValueError("❌ INPUT_PATH must be a valid image file or directory.")

    print(f"📂 Found {len(image_paths)} image(s)")

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        overlay = image.copy()
        results = yolo(img_path, conf=CONF_THRESHOLD)[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        masks = results.masks.data.cpu().numpy() if results.masks else []
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes else []

        for obj_id, (cls_id, box, mask) in enumerate(zip(class_ids, boxes, masks)):
            x1, y1, x2, y2 = map(int, box)
            class_name = yolo.names[cls_id]

            mask_full = (mask * 255).astype(np.uint8)
            mask_crop = mask_full[y1:y2, x1:x2]
            if mask_crop.shape[0] == 0 or mask_crop.shape[1] == 0:
                continue

            mask_area = int(np.count_nonzero(mask_crop))
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            features = np.array([[mask_area, bbox_width, bbox_height]])
            pred_weight = regressor.predict(features)[0]

            label = f"{class_name}: {pred_weight:.1f}g"
            draw_prediction(overlay, box, label)

        out_path = os.path.join(
            OUTPUT_DIR, f"{Path(img_path).stem}_predicted.jpg")
        cv2.imwrite(out_path, overlay)
        print(f"✅ Saved: {out_path}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    run_inference()