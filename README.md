# Deep Learning Based Traffic Sign Detection and Recognition System

An end-to-end computer vision and deep learning system for real-time localization, classification, and safety interpretation of traffic signs from road-scene imagery.

---

## 📌 1. Pipeline Architecture

The system utilizes a modular two-stage cascaded deep learning architecture:

```
                  Input Road Scene Image (Any Resolution)
                                    ↓
                 Stage 1: YOLOv8 Object Detector
                        ("Where is the sign?")
                                    ↓
                    Traffic Sign Bounding Box [x1, y1, x2, y2]
                                    ↓
                     Contextual Crop & 128x128 Scaling
                                    ↓
                 Stage 2: MobileNetV2 Classifier
                       ("What sign is it?")
                                    ↓
                     43-Class Softmax Probability
                                    ↓
              Human-Readable Safety & Driving Action Lookup
                                    ↓
               Annotated Visual Output with Class & Confidence
```

---

## 📂 2. Project Directory Structure

```
traffic-sign-project/
│
├── classification/                    # MobileNetV2 Classifier Module
│   ├── model.py                       # Architecture with transfer learning & fine-tuning
│   ├── preprocessing.py               # Augmentation & data generator utilities
│   ├── predict_classifier.py          # Standalone classifier inference script
│   └── traffic_sign_info.py           # 43-class interpretation & knowledge base
│
├── detection/                         # YOLOv8 Detector Module
│   ├── predict_yolo.py                # Standalone road-scene detection script
│   └── data.yaml                      # YOLO dataset configuration
│
├── integration/                       # End-to-End Integrated Pipeline
│   └── detect_and_classify.py         # Full two-stage cascaded inference pipeline
│
├── models/                            # Trained Model Weights & Mappings
│   ├── traffic_sign_classifier.keras  # Fine-tuned MobileNetV2 weights (96.17% Val Acc)
│   ├── best_yolo.pt                   # Best YOLOv8n detector weights (97.1% mAP50)
│   └── class_indices.json             # Keras-aligned class-to-index mapping
│
├── results/                           # Evaluation & Inference Visualizations
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_history.png
│   └── classification_report.txt
│
├── requirements.txt                   # Dependency specifications
└── README.md                          # Project documentation
```

---

## ⚡ 3. Performance & Evaluation Metrics

### MobileNetV2 Classifier (43 Classes on GTSRB)
- **Validation Accuracy**: `96.17%`
- **Macro Precision**: `0.9672`
- **Macro Recall**: `0.9590`
- **Macro F1-Score**: `0.9623`
- **Micro-average ROC-AUC**: `0.9998`
- **Macro-average ROC-AUC**: `0.9998`

### YOLOv8n Detector (Road Scene Localization)
- **Precision**: `0.952`
- **Recall**: `0.956`
- **mAP@50**: `0.971 (97.1%)`
- **mAP@50-95**: `0.786`

---

## 💻 4. Local Execution & PowerShell Commands

### A. Environment Setup
```powershell
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

### B. Running Standalone Classification Inference
```powershell
python classification/predict_classifier.py --image test_images/sample_sign.png
```

### C. Running Standalone YOLO Road Scene Detection
```powershell
python detection/predict_yolo.py --image test_images/road_scene.jpg
```

### D. Running Full End-to-End Integrated Pipeline
```powershell
python integration/detect_and_classify.py --image test_images/road_scene.jpg
```
