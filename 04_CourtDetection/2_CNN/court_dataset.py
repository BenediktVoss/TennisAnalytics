from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import numpy as np
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import KeypointParams
import random
import matplotlib.pyplot as plt


def scale_points(point, scale):
    # scale and round to int
    return (int(point[0] * scale), int(point[1] * scale))

def convert_json_to_df(json_data, split='train'):
    rows = []
    for subset in json_data['subsets']:
        for video in subset['videos']:
            for clip in video['clips']:
                for frame_number, frame in clip['frames_with_objects'].items():    
                    if frame['split'] != split:
                        continue
                    #calculate center point
                    keypoints = frame['keypoints'][0]['points']
                    center_point = calculate_center_point(keypoints)
                    #add center point to keypoints dict
                    keypoints['center_point'] = center_point
                    # add to rows
                    rows.append({
                        'subset': subset['name'],
                        'video': video['name'],
                        'clip': clip['name'],
                        'frame': frame_number,  
                        'points': keypoints
                    })

    df = pd.DataFrame(rows)
    # sort by subset, video, clip, int(frame)
    df['frame'] = df['frame'].astype(int)
    df = df.sort_values(by=['subset', 'video', 'clip', 'frame'])
    return pd.DataFrame(rows)

def calculate_center_point(points):
    # get points
    top_left_corner = points['top_left_corner']
    bottom_right_corner = points['bottom_right_corner']
    top_right_corner = points['top_right_corner']
    bottom_left_corner = points['bottom_left_corner']

    # line from top left to bottom right
    m1 = (top_left_corner[1] - bottom_right_corner[1]) / (top_left_corner[0] - bottom_right_corner[0])
    b1 = top_left_corner[1] - m1 * top_left_corner[0]

    # line from top right to bottom left
    m2 = (top_right_corner[1] - bottom_left_corner[1]) / (top_right_corner[0] - bottom_left_corner[0])
    b2 = top_right_corner[1] - m2 * top_right_corner[0]

    # calculate intersection
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return (x, y)

def reorder_keypoints(points, point_names):
    # helper to swap right for left
    def swap_left_right(name):
        """Helper function to swap 'left' and 'right' in key names."""
        if 'left' in name:
            return name.replace('left', 'right')
        elif 'right' in name:
            return name.replace('right', 'left')
        return name

    # Create dynamic mapping for flipping
    flipped_mapping = {name: swap_left_right(name) for name in point_names}

    # Reorder the points based on the flipped mapping
    reordered_points = [points[point_names.index(flipped_mapping[name])] for name in point_names]

    return reordered_points


class CourtDataset(Dataset):
    def __init__(self, path, split, input_height=720, input_width=1280, model_height=288, model_width=512, augment=False, selected_points=None):
        self.input_height = input_height
        self.input_width = input_width
        self.model_height = model_height
        self.model_width = model_width
        self.path = path
        self.selected_points = selected_points

        # Load JSON file
        with open(self.path + "/annotations_court_filtered.json", 'r') as f:
            data = json.load(f)

        # convert json to dataframe
        self.data = convert_json_to_df(data, split)


        if augment:
            # Define the Albumentations transformation pipeline
            self.base_transform = A.ReplayCompose([
                    # Flip horizontally to simulate changes in court side
                    A.HorizontalFlip(p=0.5),

                    # Adjust brightness, contrast, and saturation to handle varying lighting conditions
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),

                    # Slight rotations to account for camera angle variations
                    A.Rotate(limit=5, p=0.2, border_mode=0),
                    A.Perspective(scale=(0.02, 0.05), fit_output=True, p=0.2),

                    # Small translations and scaling to simulate camera jitter or slight positional changes
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=0),

                    # random erase
                    A.CoarseDropout(max_holes=5, max_height=32, max_width=32, min_holes=2, min_height=5, min_width=5, p=0.1),

                    # Apply Gaussian blur and/or noise to handle camera quality variations and simulate motion blur
                    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),

                    # random resize crop to size
                    A.RandomResizedCrop(model_height, model_width, scale=(0.7, 1.0), p=1),
                    
                    # Normalize for resnet
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    
                    # Convert the image to a PyTorch tensor
                    ToTensorV2()
            ],
                keypoint_params=KeypointParams(format='xy', remove_invisible=False)
            )
        else:
            # Define the Albumentations transformation pipeline
            self.base_transform = A.ReplayCompose([
                # Resize to the desired input dimensions
                A.Resize(model_height, model_width),

                # Normalize for resnet
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

                # Convert the image to a PyTorch tensor
                ToTensorV2()
            ], 
                keypoint_params=KeypointParams(format='xy', remove_invisible=False)
            )

        # Filter data by subset (e.g., train or val)
        print(f'Samples: {len(self.data)}')

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]

        img_path = os.path.join(
            self.path,
            entry['subset'],
            entry['video'],
            entry['clip'],
            str(int(entry['frame'])) + ".jpg"
        )
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

        keypoints = entry['points']

        # Filter points
        if self.selected_points is not None:
            keypoints = {k: v for k, v in keypoints.items() if k in self.selected_points}

        # Convert dictionary to list for augmentation
        point_names = list(keypoints.keys())  # Save the key order
        points_list = [keypoints[name] for name in point_names]  # Create a list of points
        points_array = np.array(points_list)  # Convert to numpy array for augmentation

        # Apply transformations with replay
        augmented = self.base_transform(image=img, keypoints=points_array)
        img = augmented['image']
        points_augmented = augmented['keypoints']  # Augmented keypoints list

        # Check if horizontal flip was applied and adjust keypoint indices
        transforms = augmented['replay']['transforms']
        horizontal_flip_applied = any(
            transform['__class_fullname__'].endswith('HorizontalFlip') and transform['applied']
            for transform in transforms
        )
        if  horizontal_flip_applied:
            points_augmented = reorder_keypoints(points_augmented, point_names)

        # Convert points to a PyTorch tensor
        points_array = np.array(points_augmented) 
        points_tensor = torch.tensor(points_array, dtype=torch.float32)

        # Convert idx to tensor
        idx = torch.tensor(idx)

        return img, points_tensor, idx
        
    
def main():

    # Define dataset parameters
    dataset_path = "./FinalDataset"
    split = "train" 
    input_height = 720
    input_width = 1280
    model_height = 288
    model_width = 512
    augment = True
    selected_points = None

    # Initialize the dataset
    dataset = CourtDataset(
        path=dataset_path,
        split=split,
        input_height=input_height,
        input_width=input_width,
        model_height=model_height,
        model_width=model_width,
        augment=augment,
        selected_points=selected_points
    )

    # Test a single sample
    sample_idx = random.randint(0, len(dataset))
    img, points, idx = dataset[sample_idx]

    print(f"Sample index: {idx}")
    print(f"Image shape: {img.shape}")  # Expecting PyTorch tensor shape (C, H, W)
    print(f"Keypoints: {points}")

    # Convert the image tensor to a numpy array for visualization
    img_np = img.permute(1, 2, 0).numpy()  # Convert to HWC for visualization
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Plot the image with keypoints
    plt.figure(figsize=(8, 6))
    plt.imshow(img_np)

    # Scatter the keypoints on the image
    for point in points:
        plt.scatter(point[0], point[1], c="red", s=20)

    # Set the title and turn off the axis
    plt.title(f"Image with Keypoints - Sample {idx}")
    plt.axis("off")

    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    main()
