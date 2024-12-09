# 07 User Interface

This directory contains the **GRADIO Application**, designed to:

- Provide **easy insights** into the dataset.
- Enable **inference on the dataset**.
- Allow **inference on your own clips**.

> **HINT**: It is highly recommended to run this on a system with **GPU support** for efficient inference.  
> **Note**: Inference on CPU is significantly slower.  

### **Performance Requirements**
- Approximately **2GB of GPU memory** is required.  
- **FPS**: ~10 (for complete inference, including minimap and video generation).

---

## **Requirements**

Ensure the following are available in the correct locations:

- **`../00_Dataset/`**:  
  Contains the dataset in the required format. (Downloadable from Kaggle.)
  
- **`models/`**:  
  Populated with the models for inference. (Downloadable from Kaggle.)

---

## **How to Run**

1. Execute the `app.py` script.  
2. Access the application at **localhost:8080** in your browser.  
3. Start testing your clips or exploring the dataset!

---