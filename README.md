# 🍎 Fruit Weight Estimation using YOLOv11 + Regressor

This project detects and segments fruits (currently apples and bananas) in images using **YOLOv11 Segmentation**, and estimates their **weight (in grams)** using a separate regression model — either a **CNN** or a **Gradient Boosting Regressor (GBR)**.

---

## 📁 Project Structure

```
FINAL DELIVERABLES/
├── Data/
│   ├── Fruits_Dataset/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── test/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── test/
│   │   │   └── val/
│   │   ├── classes.txt
│   │   ├── fruit_data.yaml         # YOLOv11 dataset config
│   │   ├── notes.json
│   │   └── Fruits_Dataset.zip
│   └── weights.csv                 # Ground truth weights (per object)
│
├── Notebooks/
│   └── Training_And_Inference_Pipeline_Notebook.ipynb
│
├── Scripts/
│   ├── CNN Regressor/
│   │   ├── yolo_cnn_train_pipeline.py          # YOLOv11 + CNN training
│   │   └── yolo_cnn_inference_pipeline.py      # Inference using CNN regressor
│   └── GB Regressor/
│       ├── yolo_gbr_train_pipeline.py          # YOLOv11 + GBR training
│       ├── yolo_gbr_inference_pipeline.py      # Inference using GBR
│       └── inference_custom_imgs.py            # Inference on custom images
│
├── requirements.txt
└── README.md
```

---

## ✅ Features

- 🧠 **YOLOv11** for fruit detection and segmentation  
- 📐 Feature extraction: mask area, bounding box width & height  
- ⚖️ Two regression methods:
  - **CNN** (trained on masked fruit crops)
  - **Gradient Boosting Regressor (GBR)** (trained on extracted features)
- 🖼️ Real-time inference on test and custom images
- 🧪 Annotated results with bounding boxes and weight predictions

---

## 🔄 Workflow Overview

1. **Train YOLOv11** segmentation model on fruit dataset  
2. **Run segmentation** to extract fruit instances  
3. **Prepare regression dataset**:
   - Use **object crops** for CNN
   - Use **geometric features** for GBR  
4. **Train regression model** (CNN or GBR)  
5. **Run unified inference**: YOLOv11 + regressor  

---

## 🚀 Getting Started

### 🔧 Setup (Local)

Install dependencies:

```bash
pip install -r requirements.txt
```

### 🏋️‍♂️ Training Pipelines

**CNN Regressor:**

```bash
python3 Scripts/CNN\ Regressor/yolo_cnn_train_pipeline.py
```

**Gradient Boosting Regressor (GBR):**

```bash
python3 Scripts/GB\ Regressor/yolo_gbr_train_pipeline.py
```

### 🔎 Inference Pipelines (on Test Set)

**Using CNN:**

```bash
python3 Scripts/CNN\ Regressor/yolo_cnn_inference_pipeline.py
```

**Using GBR:**

```bash
python3 Scripts/GB\ Regressor/yolo_gbr_inference_pipeline.py
```

📍 *Output predictions with visualized bounding boxes are saved in:*  
`regression_data/unified_results/`

### 🖼️ Inference on Custom Images

```bash
python3 Scripts/GB\ Regressor/inference_custom_imgs.py
```

📌 *Place your custom images in the folder specified inside the script (e.g., `inference_inputs/`). Results will be saved in `inference_outputs/`.*

---

## 💻 Google Colab Setup

1. Upload `Training_And_Inference_Pipeline_Notebook.ipynb` to Colab  
2. Upload zipped `Fruits_Dataset/` folder  
3. Upload `weights.csv`  
4. Upload YOLOv11 model weights (e.g., `yolov11l-seg.pt`)  
5. Run the notebook cells sequentially

> ⚠️ Ensure GPU is enabled in Colab and paths are correctly set.

---

## 📊 Evaluation Metrics

| Model               | Mean Absolute Error (MAE) | R² Score |
|---------------------|---------------------------|----------|
| CNN Regressor       | 6.17 g                    | 0.43     |
| GBR (Feature-Based) | ~6–9 g                    | ~0.30–0.40 |

---

## 📌 Notes

- Dataset is in YOLO-format with segmentation labels  
- `weights.csv` maps `(image_name, object_id)` pairs to known fruit weights  
- YOLO dataset config is in: `Data/Fruits_Dataset/fruit_data.yaml`
