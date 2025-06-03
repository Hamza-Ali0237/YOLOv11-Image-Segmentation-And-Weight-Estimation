import os
import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
SPLIT_DIR = "Data/ruits_Dataset"
REG_DATA_DIR = "regression_data"
MODEL_NAME = "yolo11l-seg.pt"
YOLO_MODEL_SAVE = "runs/segment/fruit_yolo_training/weights/best.pt"
REGRESSOR_MODEL_SAVE = f"{REG_DATA_DIR}/feature_regressor.pkl"
PLOT_PATH = f"{REG_DATA_DIR}/feature_regression_plot.png"
WEIGHTS_CSV = "Data/weights.csv"
DATA_YAML = f"{SPLIT_DIR}/fruit_data.yaml"
IMG_SIZE = 224
YOLO_EPOCHS = 75
CONF_THRESHOLD = 0.25


# ----------------- STEP 1: TRAIN YOLOv11 -----------------
def train_yolo():
    if not os.path.exists(MODEL_NAME) or os.path.getsize(MODEL_NAME) < 10_000_000:
        print(f"⬇️ Downloading f{MODEL_NAME}...")
        YOLO(MODEL_NAME)  # triggers download
    model = YOLO(MODEL_NAME)
    model.train(data=DATA_YAML, epochs=YOLO_EPOCHS, imgsz=640, name="fruit_yolo_training", task="segment")
    print(f"✅ YOLO model trained and saved to: {YOLO_MODEL_SAVE}")

# ----------------- STEP 2: EXTRACT REGRESSION DATA -----------------
def extract_regression_data():
    os.makedirs(f"{REG_DATA_DIR}/crops", exist_ok=True)
    weights_df = pd.read_csv(WEIGHTS_CSV)
    model = YOLO(YOLO_MODEL_SAVE)

    def get_weight(img_name, obj_id):
        match = weights_df[(weights_df["image_name"] == img_name) & (weights_df["object_id"] == obj_id)]
        return float(match["weight"].values[0]) if not match.empty else None

    all_rows = []
    for split in ["train", "val", "test"]:
        img_dir = f"{SPLIT_DIR}/images/{split}"
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            results = model(img_path, conf=CONF_THRESHOLD)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
            masks = results.masks.data.cpu().numpy() if results.masks else []
            class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes else []

            for obj_id, (cls_id, box, mask) in enumerate(zip(class_ids, boxes, masks)):
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                if crop is None or crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue

                mask_full = (mask * 255).astype(np.uint8)
                mask_crop = mask_full[y1:y2, x1:x2]
                if mask_crop.shape[:2] != crop.shape[:2]:
                    try:
                        mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                    except:
                        continue

                if mask_crop.shape[:2] != crop.shape[:2]:
                    continue

                masked_crop = cv2.bitwise_and(crop, crop, mask=mask_crop)
                weight = get_weight(img_name, obj_id)
                if weight is None:
                    continue

                crop_name = f"{Path(img_name).stem}_obj_{obj_id}_{model.names[cls_id]}.jpg"
                crop_path = f"{REG_DATA_DIR}/crops/{crop_name}"
                cv2.imwrite(crop_path, masked_crop)

                mask_area = int(np.count_nonzero(mask_crop))
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                all_rows.append({
                    "image_name": img_name,
                    "object_id": obj_id,
                    "class": model.names[cls_id],
                    "crop_path": crop_path,
                    "weight": weight,
                    "mask_area": mask_area,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height
                })

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{REG_DATA_DIR}/regression_dataset.csv", index=False)
    print(f"✅ Regression dataset created with {len(df)} samples.")

# ----------------- STEP 3: TRAIN FEATURE REGRESSOR -----------------
def train_feature_regressor():
    df = pd.read_csv(f"{REG_DATA_DIR}/regression_dataset.csv")

    required_cols = ["mask_area", "bbox_width", "bbox_height", "weight"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Column missing: {col}")

    X = df[["mask_area", "bbox_width", "bbox_height"]]
    y = df["weight"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🚀 Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    joblib.dump(model, REGRESSOR_MODEL_SAVE)
    print(f"✅ Feature regressor saved to: {REGRESSOR_MODEL_SAVE}")

    print("\n📊 Final Evaluation:")
    print(f"MAE: {mae:.2f}g")
    print(f"R² : {r2:.4f}")

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_val, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal")
    plt.xlabel("Actual Weight (g)")
    plt.ylabel("Predicted Weight (g)")
    plt.title("Feature Regressor Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()
    print(f"📈 Prediction plot saved to: {PLOT_PATH}")

# ----------------- RUN EVERYTHING -----------------
def run_full_training_pipeline():
    print("🚀 Training YOLOv11 segmentation model...")
    train_yolo()
    print("🔍 Extracting object crops and features...")
    extract_regression_data()
    print("📦 Training feature-based regressor...")
    train_feature_regressor()
    print("🎉 Full pipeline complete!")

if __name__ == "__main__":
    run_full_training_pipeline()