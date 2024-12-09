# 02 Explorative Data Analysis

This directory contains scripts for analyzing the dataset in the context of Explorative Data Analysis (EDA).

---

## **Input Requirements**
To execute the scripts, ensure the following are available in the correct locations:
- **00_Dataset**: The dataset folder must contain the data in the correct format, which can be downloaded from the Kaggle dataset.

---

## **Files Overview**

### **1. `1_General_Visualization.ipynb`**
This script:
- Visualizes the annotations on the frames.
- Performs general analytics on object attributes.

### **2. `2_Court_Homography.ipynb`**
This script:
- Demonstrates how to calculate the homography between a court model and annotated court keypoints.
- Projects players and the ball onto the court model using the calculated homography.
- Tests various configurations to determine the best approach for homography calculation.

### **3. `3_Court_Keypoint_Analysis.ipynb`**
This script:
- Analyzes keypoints in the amateur and court datasets.
- Compares camera angles through visualization.

---

## **Output**
After running the scripts:
1. Various graphs are generated during the EDA process.
2. Three court model PNG files are created for use in minimap visualizations.

---



