# 05 Player Tracking

This directory contains scripts for Player Tracking using YOLO models.

---

## **Input Requirements**
To execute the scripts, ensure the following are available in the correct locations:
- **00_Dataset**: The dataset folder must contain the data in the correct format, which can be downloaded from the Kaggle dataset.

---

## **Files Overview**

### **0. `0_Create_YOLO_Dataset.ipynb`**
This notebook:
- Converts the existing dataset to YOLO format, including only the player annotations.

### **1. `1_Train_YOLO_Player_Only.py`**
This script:
- Trains two different YOLO versions for player detection.

### **2. `2_Create_YOLO_Dataset_Both.ipynb`**
This notebook:
- Converts the existing dataset to YOLO format, including both player and ball annotations.

### **3. `3_Hyperparameter_Tuning.py`**
This script:
- Utilizes Weights & Biases (WandB) to optimize hyperparameters, addressing class imbalance in the dataset.

### **4. `4_Train_YOLO_Both.py`**
This script:
- Applies the optimized hyperparameters to train models capable of detecting both players and the ball.

### **5. `5_Generate_Results.ipynb`**
This notebook:
- Analyzes and compares the results of the different YOLO models.

---

## **Output**
After running the scripts:
1. Trained models for player detection and combined player+ball detection.
2. Detailed results and comparative analysis of model performance.

---