import os
import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- CONFIG ----------------
IMAGE_DIR = "Data/Fruits_Dataset/images/test"
YOLO_MODEL_PATH = "runs/segment/fruit_yolo_training/weights/best.pt"
REGRESSOR_MODEL_PATH = "regression_data/feature_regressor.pkl"
WEIGHTS_CSV = "Data/weights.csv"
OUTPUT_DIR = "regression_data/feature_regression_results"
CONF_THRESHOLD = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
print("🔄 Loading YOLOv11 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("🔄 Loading feature-based regressor...")
regressor = joblib.load(REGRESSOR_MODEL_PATH)

# ---------------- LOAD GROUND TRUTH ----------------
print("🔄 Loading weights.csv...")
gt_df = pd.read_csv(WEIGHTS_CSV)
gt_lookup = {
    (row["image_name"], row["object_id"]): row["weight"]
    for _, row in gt_df.iterrows()
}

# ---------------- INFERENCE ----------------
results_list = []
predictions, actuals = [], []

def draw_prediction(image, box, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def run_inference():
    print("🚀 Running inference on test images...")
    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        overlay = image.copy()

        results = yolo_model(img_path, conf=CONF_THRESHOLD)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        masks = results.masks.data.cpu().numpy() if results.masks else []
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes else []

        for obj_id, (cls_id, box, mask) in enumerate(zip(class_ids, boxes, masks)):
            x1, y1, x2, y2 = map(int, box)
            class_name = yolo_model.names[cls_id]

            mask_full = (mask * 255).astype(np.uint8)
            mask_crop = mask_full[y1:y2, x1:x2]

            if mask_crop.shape[0] == 0 or mask_crop.shape[1] == 0:
                continue

            mask_area = int(np.count_nonzero(mask_crop))
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Predict weight
            features = np.array([[mask_area, bbox_width, bbox_height]])
            pred_weight = regressor.predict(features)[0]

            # Lookup GT
            gt_key = (img_name, obj_id)
            gt_weight = gt_lookup.get(gt_key)
            if gt_weight is None:
                continue

            predictions.append(pred_weight)
            actuals.append(gt_weight)

            error = abs(pred_weight - gt_weight)
            color = (0, 255, 0) if error < 10 else (0, 0, 255)

            label = f"{class_name}: {pred_weight:.1f}g"
            draw_prediction(overlay, box, label, color)

            results_list.append({
                "image_name": img_name,
                "object_id": obj_id,
                "class": class_name,
                "predicted_weight": round(pred_weight, 2),
                "actual_weight": round(gt_weight, 2),
                "abs_error": round(error, 2)
            })

        # Save annotated image
        out_path = os.path.join(OUTPUT_DIR, f"{Path(img_name).stem}_result.jpg")
        cv2.imwrite(out_path, overlay)

    # Save results
    df = pd.DataFrame(results_list)
    csv_out = os.path.join(OUTPUT_DIR, "feature_predictions.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n✅ Prediction results saved to: {csv_out}")

    # Evaluation
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print("\n📊 Evaluation on Test Set:")
    print(f"MAE: {mae:.2f}g")
    print(f"R²:  {r2:.4f}")

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    run_inference()