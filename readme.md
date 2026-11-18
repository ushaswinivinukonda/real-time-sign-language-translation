# Real Time Sign Language Translation

An end-to-end computer vision and deep learning application that translates American Sign Language (ASL) hand gestures into text in real time using your webcam, powered by **TensorFlow**, **MobileNetV2** transfer learning, and **OpenCV**.

---

##  Project Overview

Communication barriers exist between the deaf and hard-of-hearing community and individuals unfamiliar with sign language. This project provides a lightweight, high-performance solution capable of classifying **36 distinct alphanumeric ASL gestures (0–9 and A–Z)** directly from a standard webcam feed without requiring expensive GPU infrastructure.

### Key Capabilities

* **Automated Data Pipeline:** Programmatically fetches and extracts the ASL dataset via the Kaggle API, applying robust data generators with validation splits and pixel normalization (`1./255`).
* **Transfer Learning Backbone:** Leverages a pre-trained **MobileNetV2** model (ImageNet weights) with frozen feature extraction layers, paired with a custom dense classification head.
* **Modular Codebase:** Cleanly separates training (`train.py`) and live inference (`predict.py`), auto-saving class index mappings (`class_indices.json`) alongside model weights for flawless label alignment.
* **Real-Time Inference Engine:** Captures live webcam frames, preprocesses them on the fly, and runs sub-second model predictions with confidence scoring.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    DATA & INGESTION LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  Kaggle API → asl-dataset.zip → Extracted Images (350x350)  │
│  ImageDataGenerator: Train/Validation Split (80/20)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┤
│                  DEEP LEARNING MODEL LAYER                  │
├─────────────────────────────────────────────────────────────┤
│  MobileNetV2 (Frozen Base) + GlobalAveragePooling2D         │
│  Dense (128 units, ReLU) → Softmax Output (36 Classes)      │
│  Saved Artifacts: saved_models/asl_model.h5                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┤
│                 REAL-TIME DEPLOYMENT LAYER                  │
├─────────────────────────────────────────────────────────────┤
│  OpenCV VideoCapture(0) → Frame Resize & Normalization     │
│  Model Inference → Predicted Sign & Confidence Display       │
└─────────────────────────────────────────────────────────────┘

```
---

## 🚀 Quick Start (5 Minutes)

### 1. Clone & Setup Repository

Clone or download this repository and open your terminal inside the root folder:

```bash
cd real-time-sign-language-translation

```

### 2. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt

```

### 3. Add Kaggle Credentials

Place your `kaggle.json` API token file directly into the root project directory so `train.py` can automatically authenticate and download the dataset.

### 4. Train the Model

Run the training script to fetch the dataset, train the transfer learning network for 10 epochs, and export model artifacts:

```bash
python train.py

```

### 5. Run Real-Time Webcam Detection

Launch the live prediction script:

```bash
python predict.py

```

*(Press **`q`** at any time to close the live webcam window).*

---

## 📊 Complete Workflow

1. **Training (`train.py`)**:
* Connects to Kaggle, downloads `a-ayuraj/asl-dataset`, and extracts it.
* Feeds images through `ImageDataGenerator` configured with pixel scaling and an 80/20 train-validation split.
* Trains the MobileNetV2 architecture and dumps the resulting `.h5` model and class mapping JSON into `saved_models/`.


2. **Inference (`predict.py`)**:
* Loads `asl_model.h5` and `class_indices.json`.
* Captures live frames from your default webcam (`cv2.VideoCapture(0)`).
* Resizes frames to `350x350`, normalizes pixel values, passes them through the model, and overlays the predicted sign and confidence score in real time.


---

## 🐛 Troubleshooting

* **Webcam Not Opening (`predict.py`)**: Verify that no other application (like Zoom or Teams) is locking your webcam, or change `cv2.VideoCapture(0)` to `1` if you use an external camera.
* **`kaggle.json` Authentication Error**: Ensure your `kaggle.json` file is placed directly in the root directory and contains active username and key fields.
* **Model Not Found Error**: Make sure you have run `python.py` successfully at least once so that the `saved_models/asl_model.h5` file exists before launching `predict.py`.
* **Low Inference FPS**: If live detection lags, check if TensorFlow is utilizing your GPU or reduce the input image resolution in the scripts.

---

## 📋 Checklist Before Running

* [ ] Python 3.8+ installed
* [ ] `kaggle.json` placed in root directory
* [ ] Dependencies installed via `pip install -r requirements.txt`
* [ ] `train.py` executed successfully and `saved_models/` folder populated
* [ ] Webcam connected and functional

---

## 🔮 Future Enhancements

* [ ] Build a web-based dashboard using Streamlit for browser-based sign translation.
* [ ] Implement full sentence composition by accumulating sequential letter predictions.
* [ ] Fine-tune convolutional blocks for higher accuracy on ambiguous hand shapes.
* [ ] Package the app into a desktop executable for offline accessibility use.

---

## 📄 License & Attribution

This project uses open-source components including TensorFlow, OpenCV, and the ASL Dataset provided via Kaggle.