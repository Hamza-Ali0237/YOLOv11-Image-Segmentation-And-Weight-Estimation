# Fruit Weight Estimation with YOLOv11 + Regressor

This project detects and segments fruits (apples and bananas) in images using YOLOv11 Segmentation, and estimates their weight (in grams) using a separate regression model (either a CNN or a Gradient Boosting Regressor).

FINAL DELIVERABLES/
├── Data/
│ ├── Fruits_Dataset/
│ │ ├── images/
│ │ │ ├── train/
│ │ │ ├── test/
│ │ │ └── val/
│ │ ├── labels/
│ │ │ ├── train/
│ │ │ ├── test/
│ │ │ └── val/
│ │ ├── classes.txt
│ │ ├── fruit_data.yaml # YOLOv11 dataset config
│ │ ├── notes.json
│ │ └── Fruits_Dataset.zip
│ └── weights.csv # Ground truth weights (per object)
│
├── Notebooks/
│ └── Training_And_Inference_Pipeline_Notebook.ipynb
│
├── Scripts/
│ ├── CNN Regressor/
│ │ ├── yolo_cnn_train_pipeline.py # YOLOv11 + CNN training pipeline
│ │ └── yolo_cnn_inference_pipeline.py # Inference pipeline using CNN regressor
│ └── GB Regressor/
│ ├── yolo_gbr_train_pipeline.py # YOLOv11 + feature regressor training
│ ├── yolo_gbr_inference_pipeline.py # Inference pipeline using GB regressor
│ └── inference_custom_imgs.py # Inference on new unseen images
│
├── requirements.txt
└── README.md

## ✅ Features

1. 📦 YOLOv11 Segmentation model for fruit detection & segmentation
2. 📐 Feature extraction: mask area, bounding box width & height
3. ⚖️ Two regression options:
   - CNN: learns from masked image crops
   - GBR: uses extracted features
4. 🔍 Supports real-time inference on both test and custom images
5. 🧪 Annotated results with predicted weight overlays

## 📈 Workflow Overview

1. Train YOLOv11 on segmented fruit dataset
2. Run segmentation on dataset to extract object instances
3. Create regression dataset with either:
   - Object crops (for CNN)
   - Object features (for GBR)
4. Train regression model
   - CNN: train on object crops
   - GBR: train on extracted features
5. Run unified inference combining YOLO + regressor

## 🚀 How to Use

### On Local Environment

📦 Install Requirements:

```bash
pip3 install -r requirements.txt
```

1. Training Pipelines

- CNN-Based Regressor:

```bash
python3 Scripts/CNN\ Regressor/yolo_cnn_train_pipeline.py
```

```bash
python3 Scripts/CNN\ Regressor/yolo_cnn_train_pipeline.py
```

- Feature-Based Regressor (GBR):

```bash
python3 Scripts/GB\ Regressor/yolo_gbr_train_pipeline.py
```

```bash
python3 Scripts/CNN\ Regressor/yolo_cnn_train_pipeline.py
```

- Feature-Based Regressor (GBR):

```bash
python3 Scripts/GB\ Regressor/yolo_gbr_train_pipeline.py
```

2. Inference Pipelines (on Test Set)

- Using CNN Regressor:

```bash
python3 Scripts/CNN\ Regressor/yolo_cnn_inference_pipeline.py
```

- Using Gradient Boosting Regressor:

```bash
python3 Scripts/GB\ Regressor/yolo_gbr_inference_pipeline.py # Output predictions with visualized bounding boxes will be saved to: regression_data/unified_results/
```

3. Custom Image Inference (No Ground Truth)

```bash
python3 Scripts/GB\ Regressor/inference_custom_imgs.py
# Just place your images in the hardcoded folder inside the script (e.g. inference_inputs/)
# Results are saved in inference_outputs/
```

### On Google Colab:

1. Upload Training_And_Inference_Pipeline_Notebook.ipynb file to Google Colab.
2. Zip "Fruits_Dataset" folder and upload to Google Colab.
3. Upload the `weights.csv` file to Google Colab.
4. Upload the YOLOv11 (e.g., `yolo11l-seg.pt`) file to Google Colab.
5. Run the notebook cells sequentially.

NOTE: The notebook is designed to run on Google Colab with GPU support. Ensure you have the necessary permissions and resources. You might need to adjust the paths and configurations based on your environment (also applicable for Python scripts).

📊 Evaluation Metrics

- **CNN Regressor**

  - Mean Absolute Error (MAE): 6.17 g
  - R² Score: 0.43

- **GBR (Feature-Based)**

  - Mean Absolute Error (MAE): ~6–9 g
  - R² Score: ~0.30–0.40

📌 Notes

- Dataset includes YOLO-format segmentation labels
- weights.csv maps (image_name, object_id) to the known fruit weight
- YOLO config file is located at: Data/Fruits_Dataset/fruit_data.yaml
