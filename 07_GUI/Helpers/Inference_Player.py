from ultralytics import YOLO
import torch
import os
import pandas as pd
import gc

def select_top2_confidences(predictions):
    top2_predictions = []

    for frame in predictions:
        # Sort detections by confidence in descending order
        sorted_frame = sorted(frame, key=lambda x: x["confidence"], reverse=True)

        # Select the top 2 detections
        top2_frame = sorted_frame[:2]
        top2_predictions.append(top2_frame)

    return top2_predictions


def get_input_images_in_sequence(image_path):
    # get all image file names in the path
    image_files = [
        file for file in os.listdir(image_path)
        if os.path.isfile(os.path.join(image_path, file)) and file.lower().endswith('.jpg')
    ]

    # create a dataframe from the image files
    data = pd.DataFrame(image_files, columns=['file'])

    # create columns  without the extension
    data['frame'] = data['file'].apply(lambda x: int(x.split('.')[0]))

    # sort by frame
    data = data.sort_values('frame').reset_index(drop=True) 

    # Generate full paths and return as a list
    image_paths = [image_path + "/" + file for file in data['file']]
    return image_paths

# Clear GPU memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def get_player_predictions(model_path, image_path):
    clear_gpu_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize an empty list to store results
    data = []

    model = YOLO(model_path)

    image_path_ordered = get_input_images_in_sequence(image_path)

    # loop through the images
    for i, image in enumerate(image_path_ordered):
        predictions = model.predict(
            source=image,
            conf=0.3,       
            save=False,
            stream=True,
            batch=1,
            device=device,
            verbose=False
        )

        # Initialize an empty list for detections in the current image
        detections = []

        for result in predictions:
            boxes = result.boxes  # Bounding box outputs
            if boxes is not None:
                for box in boxes:
                    # Extract information for each detected object
                    bbox = box.xyxy[0].tolist()  # Bounding box (x1, y1, x2, y2)
                    confidence = box.conf[0].item()  # Confidence score
                    class_label = int(box.cls[0].item())  # Class label
                    if class_label == 0:
                        # Add to the results list
                        detections.append({
                            "xtl": bbox[0],
                            "ytl": bbox[1],
                            "xbr": bbox[2],
                            "ybr": bbox[3],
                            "confidence": confidence,
                            "class_label": class_label
                        })

        data.append(detections)
    
    clear_gpu_memory()

    return select_top2_confidences(data)


if __name__ == "__main__":
    model_path = "models/YOLOv11_Player.pt"
    image_path = "../00_Dataset/Amateur/Video_1/clip_5"

    ball_predictions = get_player_predictions(model_path, image_path)
    print(ball_predictions)