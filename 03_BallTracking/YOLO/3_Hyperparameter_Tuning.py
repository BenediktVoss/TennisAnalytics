from ultralytics import YOLO

def tune_yolo_model(model_name, data_path, img_size, devices, batch_size, seed):
    """
    Function to perform hyperparameter tuning for the specified YOLO model.
    """
    try:
        print(f"Starting hyperparameter tuning for model: {model_name}")
        model = YOLO(model_name)  # Load the specified YOLO model

        # Perform hyperparameter tuning
        results = model.tune(
            data=data_path,
            imgsz=img_size,
            device=devices,
            batch=batch_size,
            project="ball-tracking-hyperparams",
            seed=seed,
            epochs=20, 
            iterations=15, 
            optimizer="AdamW", 
            plots=False, 
            save=False, 
            val=False,
            patience=8,
            name=f"tune_{model_name.split('.')[0]}"  # Set a custom name for the tuning run
        )
        print(f"Hyperparameter tuning completed for model: {model_name}")
        return results
    except Exception as e:
        print(f"An error occurred while tuning model {model_name}: {e}")
        return None

def tune_yolo_versions():
    """
    Function to perform hyperparameter tuning for YOLOv5s and YOLOv11s models.
    """
    # List of YOLO model names to tune
    yolo_model_names = [
        "yolov5su.pt",
        "yolo11s.pt"
    ]

    # Common tuning parameters
    data_path = "./datasets/YOLODataset/annotations.yaml"
    img_size = 640
    devices = [0,1]
    batch_size = 128
    seed = 42

    # Tune each model in the list
    for model_name in yolo_model_names:
        tune_yolo_model(
            model_name=model_name,
            data_path=data_path,
            img_size=img_size,
            devices=devices,
            batch_size=batch_size,
            seed=seed
        )

if __name__ == "__main__":
    tune_yolo_versions()