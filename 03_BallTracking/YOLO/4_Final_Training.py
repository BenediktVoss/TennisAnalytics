from ultralytics import YOLO
import yaml

def train_yolo_model(model_name, data_path, epochs, img_size, devices, batch_size, seed, hyp_path):
    try:
        # Load hyperparameters from the YAML file
        with open(hyp_path, 'r') as file:
            hyperparameters = yaml.safe_load(file)
            
        print(f"Starting training for model: {model_name}")
        model = YOLO(model_name)  # Load the specified YOLO model

        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            device=devices,
            batch=batch_size,
            project="ball-tracking",
            seed=seed,
            patience=8,
            name=f"train_tuned_{model_name.split('.')[0]}",  # Set a custom name for the training run
            **hyperparameters  # Pass the loaded hyperparameters
        )
        print(f"Training completed for model: {model_name}")
        return results
    except Exception as e:
        print(f"An error occurred while training model {model_name}: {e}")
        return None

def test_yolo_versions():
    # List of YOLO model names to train
    yolo_model_names = [
        ("yolov5su.pt", "./ball-tracking-hyperparams/Tune_Results_YoloV11/best_hyperparameters.yaml"),
        ("yolo11s.pt", "./ball-tracking-hyperparams/Tune_Results_YoloV5/best_hyperparameters.yaml")
    ]

    # Common training parameters
    data_path = "./datasets/YOLODataset/annotations.yaml"
    epochs = 100
    img_size = 640
    devices = [0]
    batch_size = -1
    seed = 42

    # Train each model in the list
    for model_name, path in yolo_model_names:
        train_yolo_model(
            model_name=model_name,
            data_path=data_path,
            epochs=epochs,
            img_size=img_size,
            devices=devices,
            batch_size=batch_size,
            seed=seed,
            hyp_path=path)

if __name__ == "__main__":
    test_yolo_versions()