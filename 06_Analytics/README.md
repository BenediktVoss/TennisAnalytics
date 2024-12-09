# 06 Analytics

This directory contains scripts for Trajectory Smoothing and Bounce and hit proediction

---

## **Input Requirements**
To execute the scripts, ensure the following are available in the correct locations:
- **00_Dataset**: The dataset folder must contain the data in the correct format, which can be downloaded from the Kaggle dataset.
- Model results from ball tracking

---

## **Files Overview**

### **1. `1_Trajectory_Smoothing.ipynb`**
This notebook:
- Test different approaches for trajectory smoothing
- Creates a post processing pipeline 
- Evaluates the pipleine on the modle results

### **2. `2_BounceAndHitPrediction.py`**
This notebook:
- Visualizes Bounces and Hits on a example clip
- Tests different models for Classification
- Evaluates results on the Dataset

---

## **Output**
After running the scripts:
1. Postprocessing Pipeline
2. Trained Models for bounce and Hit prediction

---