import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms, models
from ultralytics import YOLO
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
YOLO_MODEL_PATH = "runs/segment/fruit_yolo_training/weights/best.pt"
CNN_MODEL_PATH = "regression_data/cnn_weight_regressor.pt"
CSV_PATH = "regression_data/regression_dataset.csv"
TEST_IMAGE_DIR = "Data/Fruits_Dataset/images/test"
OUTPUT_DIR = "regression_data/unified_results"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================= MODEL DEFINITIONS =======================
class FruitCNNRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = torch.nn.Linear(base.fc.in_features, 1)
        self.model = base

    def forward(self, x):
        return self.model(x).squeeze()

def load_cnn_model(path):
    model = FruitCNNRegressor().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

# ======================= HELPER FUNCTIONS ========================
def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path)
    return {
        (row["image_name"], row["object_id"]): row["weight"]
        for _, row in df.iterrows()
    }

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def draw_prediction(image, box, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ======================= MAIN PIPELINE ===========================
def run_pipeline():
    # Load models
    yolo_model = YOLO(YOLO_MODEL_PATH)
    cnn_model = load_cnn_model(CNN_MODEL_PATH)
    ground_truth = load_ground_truth(CSV_PATH)

    results_list = []
    predictions, actuals = [], []

    # Process each test image
    for img_file in os.listdir(TEST_IMAGE_DIR):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue

        image_path = os.path.join(TEST_IMAGE_DIR, img_file)
        original = cv2.imread(image_path)
        if original is None:
            continue
        overlay = original.copy()

        # YOLOv11 inference
        detections = yolo_model(image_path, conf=CONF_THRESHOLD)[0]
        boxes = detections.boxes.xyxy.cpu().numpy() if detections.boxes else []
        class_ids = detections.boxes.cls.cpu().numpy().astype(int) if detections.boxes else []

        for obj_id, (cls_id, box) in enumerate(zip(class_ids, boxes)):
            class_name = yolo_model.names[cls_id]
            x1, y1, x2, y2 = map(int, box)
            crop = original[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            input_tensor = preprocess_image(crop)
            with torch.no_grad():
                pred_weight = cnn_model(input_tensor).item()

            key = (img_file, obj_id)
            true_weight = ground_truth.get(key)

            if true_weight is None:
                continue  # skip unmatched objects

            predictions.append(pred_weight)
            actuals.append(true_weight)

            error = abs(pred_weight - true_weight)
            color = (0, 255, 0) if error < 10 else (0, 0, 255)

            label = f"{class_name}: {pred_weight:.1f}g / GT: {true_weight:.1f}g"
            draw_prediction(overlay, box, label, color)

            results_list.append({
                "image_name": img_file,
                "object_id": obj_id,
                "class": class_name,
                "predicted_weight": round(pred_weight, 2),
                "actual_weight": round(true_weight, 2),
                "abs_error": round(error, 2)
            })

        # Save visualized result
        out_path = os.path.join(OUTPUT_DIR, f"{Path(img_file).stem}_result.jpg")
        cv2.imwrite(out_path, overlay)

    # Save results to CSV
    df = pd.DataFrame(results_list)
    csv_out = os.path.join(OUTPUT_DIR, "cnn_predictions.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n✅ Results saved to {csv_out}")

    # Print evaluation
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f"\n📊 Evaluation on Test Set:")
    print(f"MAE: {mae:.2f}g")
    print(f"R²:  {r2:.4f}")

# ===================== EXECUTION ENTRY POINT =====================
if __name__ == "__main__":
    run_pipeline()