# X-Pathology V2.0: Multi-Organ Histopathological Diagnostic Model

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)

## Overview
X-Pathology V2.0 is a lightweight, high-performance deep learning model engineered to classify microscopic histopathological scans of lung and colon tissues. Moving beyond standard binary classification, this model serves as a multi-organ diagnostic tool capable of distinguishing between five highly specific clinical categories with exceptional accuracy.

Designed with edge-deployment in mind, the model balances computational efficiency with robust predictive power, making it ideal for integration into web applications and clinical diagnostic pipelines.

This model is currently the CNN Engine of project **X-Pathology:** https://x-pathology.vercel.app/!

## Clinical Categories
The model has been trained to identify the following 5 tissue classes:
1. **Colon Adenocarcinoma** (`colon_aca`) - Malignant
2. **Colon Benign Tissue** (`colon_n`) - Benign
3. **Lung Adenocarcinoma** (`lung_aca`) - Malignant
4. **Lung Benign Tissue** (`lung_n`) - Benign
5. **Lung Squamous Cell Carcinoma** (`lung_scc`) - Malignant

## Architecture & Engineering
This project tackles common pitfalls in medical AI (such as spatial memorization and overconfidence) through a mathematically strict training pipeline:

* **Lightweight Backbone:** Utilizes a **MobileNetV2** architecture implemented via the Keras Functional API. This ensures rapid inference times and a small memory footprint (`.keras` format) suitable for real-time browser or lightweight server deployment.
* **Hostile Data Augmentation:** A rigorous preprocessing pipeline (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`) was applied during training to prevent the model from memorizing pristine dataset conditions, forcing it to learn actual cellular structures.
* **Confidence Calibration (Label Smoothing):** The model avoids the "100% confidence" overfitting trap by utilizing `CategoricalCrossentropy` with a `label_smoothing` factor of 0.1. This mathematical penalty forces realistic probability distributions across all 5 classes.
* **Heavy Regularization:** A `Dropout(0.5)` layer was integrated immediately prior to the classification head, penalizing over-reliance on specific nodes and ensuring generalized feature extraction.
* **Multi-Phase Fine-Tuning:** Training was executed in two distinct phases: an initial phase with a frozen backbone to map macroscopic features, followed by a microscopic learning rate (`1e-5`) unfreeze phase to capture complex glandular and cellular micro-textures.

## Dataset
Trained on the **LC25000 (Lung and Colon Cancer Histopathological Images)** dataset. 
* **Total Images:** 25,000
* **Resolution:** Originally 768x768 (Downsampled to 224x224 for MobileNetV2 efficiency)
* **Validation Split:** 80% Training / 20% Validation

## Performance Metrics
* **Validation Accuracy:** ~98.5%
* **Loss Convergence:** Demonstrated zero overfitting during the fine-tuning phase, with training and validation loss tracking cleanly.

## Installation & Usage

### 1. Requirements
```bash
pip install tensorflow numpy matplotlib pillow
```

### 2. Running Inference
You can load the .keras model directly into your Python environment. Note: The custom_objects parameter is required to deserialize the embedded MobileNetV2 preprocessing Lambda layer.
```bash
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Load the model with the required custom preprocessing function
model = tf.keras.models.load_model(
    'xpathology_v2_5class_finetuned.keras',
    custom_objects={'preprocess_input': preprocess_input}
)

# 2. Load and format the image
image_path = 'path/to/scan.jpeg'
img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create batch axis

# 3. Predict (Pass raw array; model handles internal scaling)
predictions = model.predict(img_array)[0]
classes = ['Colon ACA', 'Colon Benign', 'Lung ACA', 'Lung Benign', 'Lung SCC']

print(f"Primary Diagnosis: {classes[np.argmax(predictions)]}")
print(f"Confidence: {np.max(predictions) * 100:.2f}%")
```

## ⚠️ Disclaimer
**Intended Use:** This model and repository are provided strictly for educational, research, and portfolio demonstration purposes. It is not an FDA-approved medical device and must not be used for actual clinical diagnosis, treatment planning, or patient care without verification by a certified human pathologist.
EOF
