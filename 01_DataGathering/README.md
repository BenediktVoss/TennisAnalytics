# 01 Data Gathering

This directory contains scripts to generate the `annotations.json` file and create the train, validation, and test splits for dataset preparation. 

> **Note**: As the process requires raw data files, this folder is intended for documentation and demonstration purposes only.

---

## **Input Requirements**
To execute the scripts, ensure the following are available in the correct locations:
- **Raw clips**: Original video clips or image data.
- **Annotations.xml**: The corresponding annotation file.

---

## **Files Overview**

### **1. `1_Dataset_Creation`**
This script:
- Combines data from three datasets.
- Generates the `annotations.json` file.

### **2. `2_Create_Splits`**
This script:
- Updates the `annotations.json` file.
- Creates train, validation, and test splits based on the specified ratio.

---

## **Output**
After running the scripts:
1. The `00_Dataset` folder will be populated with frames organized into the appropriate folder structure for each subset (train, validation, and test).
2. An `annotations.json` file will be created, including:
   - Correct split ratios.
   - Task-specific annotation files, each containing only the relevant frames.

---




