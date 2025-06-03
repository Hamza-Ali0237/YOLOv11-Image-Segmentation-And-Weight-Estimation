import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, r2_score
import torch.nn as nn
import torch.optim as optim

# ----------------- CONFIG -----------------
# Set up all the main paths, filenames, and hyperparameters we'll use throughout the pipeline
SPLIT_DIR = "Data/Fruits_Dataset"
REG_DATA_DIR = "regression_data"
MODEL_NAME = "yolo11l-seg.pt"
YOLO_MODEL_SAVE = "runs/segment/fruit_yolo_training/weights/best.pt"
CNN_MODEL_SAVE = f"{REG_DATA_DIR}/cnn_weight_regressor.pt"
WEIGHTS_CSV = "Data/weights.csv"
DATA_YAML = f"{SPLIT_DIR}/fruit_data.yaml"
IMG_SIZE = 224
YOLO_EPOCHS = 75
EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- STEP 1: TRAIN YOLOv11 -----------------
# This function trains a YOLOv11 segmentation model on our fruit dataset
def train_yolo():
    # Download the YOLOv11 model weights if they're not already present
    if not os.path.exists(MODEL_NAME) or os.path.getsize(MODEL_NAME) < 10_000_000:
        print(f"⬇️ Downloading {MODEL_NAME}...")
        YOLO(MODEL_NAME)  # auto-download
    model = YOLO(MODEL_NAME)
    # Start training the model using our dataset and settings
    model.train(data=DATA_YAML, epochs=YOLO_EPOCHS, imgsz=640, name="fruit_yolo_training", task="segment")
    print(f"✅ YOLO model trained and saved to: {YOLO_MODEL_SAVE}")

# ----------------- STEP 2: EXTRACT REGRESSION DATA -----------------
# This function uses the trained YOLO model to crop out each fruit and save it for regression
def extract_regression_data():
    os.makedirs(f"{REG_DATA_DIR}/crops", exist_ok=True)
    weights_df = pd.read_csv(WEIGHTS_CSV)
    model = YOLO(YOLO_MODEL_SAVE)

    # Helper to get the weight for a specific fruit object in an image
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

            # Run YOLO segmentation on the image
            results = model(img_path, conf=0.25)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
            masks = results.masks.data.cpu().numpy() if results.masks else []
            class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes else []

            # For each detected fruit, crop it out and save the crop and its info
            for obj_id, (cls_id, box, mask) in enumerate(zip(class_ids, boxes, masks)):
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                if crop is None or crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue

                mask_full = (mask * 255).astype(np.uint8)
                mask_crop = mask_full[y1:y2, x1:x2]
                if mask_crop is None or mask_crop.shape[0] == 0 or mask_crop.shape[1] == 0:
                    continue

                # Make sure the mask and crop are the same size
                if mask_crop.shape[:2] != crop.shape[:2]:
                    try:
                        mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                    except:
                        continue
                if mask_crop.shape[:2] != crop.shape[:2]:
                    continue

                # Apply the mask to the crop so we only keep the fruit
                masked_crop = cv2.bitwise_and(crop, crop, mask=mask_crop)
                weight = get_weight(img_name, obj_id)
                if weight is None:
                    continue

                crop_name = f"{Path(img_name).stem}_obj_{obj_id}_{model.names[cls_id]}.jpg"
                crop_path = f"{REG_DATA_DIR}/crops/{crop_name}"
                cv2.imwrite(crop_path, masked_crop)

                all_rows.append({
                    "image_name": img_name,
                    "object_id": obj_id,
                    "class": model.names[cls_id],
                    "crop_path": crop_path,
                    "weight": weight
                })

    # Save all the crop info to a CSV for the regression model
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{REG_DATA_DIR}/regression_dataset.csv", index=False)
    print(f"✅ Regression dataset created with {len(df)} samples.")

# ----------------- CNN REGRESSOR CLASSES -----------------
# Custom dataset for loading fruit crops and their weights
class FruitCropDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = cv2.imread(row["crop_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(img) if self.transform else img
        y = torch.tensor(row["weight"], dtype=torch.float32)
        return x, y

# Simple CNN regressor based on ResNet18, modified for regression
class FruitCNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, 1)
        self.model = base

    def forward(self, x):
        return self.model(x).squeeze()

# ----------------- STEP 3: TRAIN CNN WITH CALLBACKS -----------------
# This function trains the CNN to predict fruit weights from the crops, with early stopping and best model saving
def train_cnn_regressor():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = FruitCropDataset(f"{REG_DATA_DIR}/regression_dataset.csv", transform)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    loader = {
        'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_ds, batch_size=BATCH_SIZE)
    }

    model = FruitCNNRegressor().to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    # ---------- Callbacks ----------
    # Early stopping: stop training if validation MAE doesn't improve for a while
    class EarlyStopping:
        def __init__(self, patience=30):
            self.patience = patience
            self.counter = 0
            self.best_score = float('inf')
            self.early_stop = False

        def __call__(self, score):
            if score < self.best_score:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    # Save the model whenever it achieves a new best MAE on validation
    class BestModelSaver:
        def __init__(self, path):
            self.best_mae = float('inf')
            self.path = path
            self.best_preds = []
            self.best_truths = []

        def update(self, mae, preds, truths, model):
            if mae < self.best_mae:
                self.best_mae = mae
                self.best_preds = preds
                self.best_truths = truths
                torch.save(model.state_dict(), self.path)

    stopper = EarlyStopping()
    saver = BestModelSaver(CNN_MODEL_SAVE)

    # ---------- Training ----------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in loader['train']:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)

        avg_train_loss = train_loss / len(train_ds)
        train_losses.append(avg_train_loss)

        # Validation: check how well the model is doing on unseen data
        model.eval()
        val_loss = 0.0
        preds, truths = [], []
        with torch.no_grad():
            for xb, yb in loader['val']:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu().numpy()
                preds.extend(pred)
                truths.extend(yb.numpy())
                loss = loss_fn(torch.tensor(pred), yb)
                val_loss += loss.item() * xb.size(0)

        avg_val_loss = val_loss / len(val_ds)
        val_losses.append(avg_val_loss)

        mae = mean_absolute_error(truths, preds)
        r2 = r2_score(truths, preds)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MAE: {mae:.2f} | R²: {r2:.3f}")

        saver.update(mae, preds, truths, model)
        stopper(mae)
        if stopper.early_stop:
            print(f"⛔ Early stopping at epoch {epoch+1}")
            break

    print(f"\n✅ Best model saved to: {CNN_MODEL_SAVE}")
    final_mae = mean_absolute_error(saver.best_truths, saver.best_preds)
    final_r2 = r2_score(saver.best_truths, saver.best_preds)
    print(f"\n📊 Final Evaluation:\nMAE: {final_mae:.2f}g\nR²:  {final_r2:.4f}")

    # ---------- Plot ----------
    # Plot the training and validation loss curves for visualization
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("📉 CNN Training Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plot_path = os.path.join(REG_DATA_DIR, "cnn_training_curve.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"📈 Training curve saved to: {plot_path}")

# ----------------- RUN EVERYTHING -----------------
# This function runs the full pipeline: train YOLO, extract crops, train CNN
def run_full_training_pipeline():
    print("🚀 Training YOLOv11 segmentation model...")
    train_yolo()
    print("🔍 Extracting object crops for CNN training...")
    extract_regression_data()
    print("📦 Training CNN regressor with callbacks...")
    train_cnn_regressor()
    print("🎉 Pipeline complete!")

if __name__ == "__main__":
    run_full_training_pipeline()