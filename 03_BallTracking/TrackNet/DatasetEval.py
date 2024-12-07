from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import numpy as np
import json
from tqdm import tqdm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math


def scale_points(point, scale):
    # scale and round to int
    return (int(point[0] * scale), int(point[1] * scale))


def convert_json_to_df(json_data, split='train', resolution=[1280, 720]):
    rows = []
    for subset in json_data['subsets']:
        resolution_subset= subset['resolution']
        if 'ball' in subset['objects']:
            for video in subset['videos']:
                for clip in video['clips']:
                    for frame_number, frame in clip['frames_with_objects'].items():    
                        if frame['split'] != split:
                            continue
                        points = []
                        for ball in frame['balls']:
                            if ball['visibility'] not in ['Outside'] and ball['trajectory'] not in ['', 'Static']:
                                point = scale_points([ball['x'],ball['y']], resolution[0] / resolution_subset[0])
                                point_object = {'x': point[0], 'y': point[1], 'visibility': ball['visibility']}
                                points.append(point_object)
                        rows.append({
                            'subset': subset['name'],
                            'video': video['name'],
                            'clip': clip['name'],
                            'frame': frame_number,  
                            'points': points
                        })

    df = pd.DataFrame(rows)
    # sort by subset, video, clip, int(frame)
    df['frame'] = df['frame'].astype(int)
    df = df.sort_values(by=['subset', 'video', 'clip', 'frame'])
    return pd.DataFrame(rows)


def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g
    

def create_gaussian(size, variance, range=255):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array =  gaussian_kernel_array * range/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
    return gaussian_kernel_array
    

def create_binary_circle(radius, range=1):
    diameter = 2 * radius + 1
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    circle = np.zeros((diameter, diameter), dtype=np.float32)
    circle[mask] = range
    return circle
    

def apply_kernel(heatmap, kernel, x, y, width, height):
    size = kernel.shape[0] // 2
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            if (
                0 <= x + i < width and 
                0 <= y + j < height and 
                0 <= i + size < kernel.shape[0] and 
                0 <= j + size < kernel.shape[1]
            ):
                temp = kernel[i + size][j + size]
                if temp > 0:
                    heatmap[y + j, x + i] = max(heatmap[y + j, x + i], temp)
                    

def create_label_arrays(df, path_output, size, variance, width, height, method="Gaussian", range=1):
    if method == "Gaussian":
        kernel = create_gaussian(size, variance, range)
    elif method == "Circle":
        kernel = create_binary_circle(size, range)
    else:
        raise ValueError("Invalid method. Use 'Gaussian' or 'Circle'.")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing frames"):
        subset = row['subset']
        video = row['video']
        clip = row['clip']
        frame = row['frame']
        points = row['points']  # List of (x, y) tuples

        # Create output paths
        path_out_clip = os.path.join(path_output, subset, video, clip)

        # Ensure directories exist
        os.makedirs(path_out_clip, exist_ok=True)

        # Initialize blank 2D heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Draw heatmaps for each point in 'points'
        for (x, y) in points:
            x = int(x)
            y = int(y)
            apply_kernel(heatmap, kernel, x, y, width, height)

        # Save heatmap as a 2D NumPy array with the frame name
        output_filename = f"{frame}.npy"
        np.save(os.path.join(path_out_clip, output_filename), heatmap)
        

def create_window_index(df, output_size):
    # create a new column for the window index
    df['window_index'] = 0
    window_counter = 0
    output_counter = 0
    current_subset = df.iloc[0]['subset']
    current_video = df.iloc[0]['video']
    current_clip = df.iloc[0]['clip']
    # loop over over data and create a window index
    for row in df.iterrows():
        # check if still in the right subset, video and clip
        if (row[1]['subset'] != current_subset or row[1]['video'] != current_video or row[1]['clip'] != current_clip) & output_counter > 0:
            window_counter += 1
            output_counter = 0
        # update current subset
        current_subset = row[1]['subset']
        current_video = row[1]['video']
        current_clip = row[1]['clip']

        # set row window index
        df.at[row[0], 'window_index'] = window_counter
        output_counter += 1

        #check if end of window is reached
        if output_counter == output_size:
            window_counter += 1
            output_counter = 0

    return df


class BallDataSet(Dataset):
    def __init__(self, path, split, input_height=720, input_width=1280, model_height=288, model_width=512, input_size=3, output_size=3, augment=False, heatmap_size=5, recreate_heatmap=True):
        self.input_height = input_height
        self.input_width = input_width
        self.model_height = model_height
        self.model_width = model_width
        self.path = path
        self.input_size = input_size
        self.output_size = output_size

        # Load JSON file
        with open(self.path + "/annotations_ball.json", 'r') as f:
            data = json.load(f)

        # convert json to dataframe
        self.data = convert_json_to_df(data, split)
        self.data = create_window_index(self.data, self.output_size)

        if recreate_heatmap:
            create_label_arrays(self.data, path_output=self.path + "/heatmaps", method="Circle", size=heatmap_size, variance=10, width=1280, height=720)

        if augment:
            # Define the Albumentations transformation pipeline
            self.base_transform = A.ReplayCompose([
                    # Flip horizontally to simulate changes in court side
                    A.HorizontalFlip(p=0.3),

                    # Adjust brightness, contrast, and saturation to handle varying lighting conditions
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),

                    # Slight rotations to account for camera angle variations
                    A.Rotate(limit=5, p=0.2),
                    A.Perspective(scale=(0.02, 0.05), p=0.1),

                    # Small translations and scaling to simulate camera jitter or slight positional changes
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.2),

                    # random erase
                    A.CoarseDropout(max_holes=5, max_height=32, max_width=32, min_holes=2, min_height=5, min_width=5, p=0.1),

                    # Apply Gaussian blur and/or noise to handle camera quality variations and simulate motion blur
                    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),

                    # random resize crop to size
                    A.RandomResizedCrop(model_height, model_width, scale=(0.7, 1.0), p=1),

                    # Convert the image to a PyTorch tensor
                    ToTensorV2()
            ])
        else:
            # Define the Albumentations transformation pipeline
            self.base_transform = A.ReplayCompose([
                # Resize to the desired input dimensions
                A.Resize(model_height, model_width),

                # Convert the image to a PyTorch tensor
                ToTensorV2()
            ])
        size = self.data['window_index'].max() + 1
        # Filter data by subset (e.g., train or val)
        print(f'Windows: {size}')

    
    def __len__(self):
        # get max window index
        size = self.data['window_index'].max() + 1
        return size

    
    def __getitem__(self, idx):
        # get entries for window index
        entries = self.data[self.data['window_index'] == idx]
        
        #select the entry with the highest frame number
        entry = entries.iloc[-1]
        #get idx of entry
        idx = entries.index[-1]

        input_lag = -self.input_size +1
        output_lag = -self.output_size +1

        # get the 2 images before and the current image
        images = []
        for i in range(input_lag, 1):
            if idx + i < 0 or idx + i >= len(self.data):  # Check for boundary conditions
                img = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
                images.append(img)
                continue
                 
            entry_range = self.data.iloc[idx + i]

            img_path = os.path.join(
                self.path,
                entry_range['subset'],
                entry_range['video'],
                entry_range['clip'],
                str(int(entry['frame']) + i) + ".jpg"
            )

            # check if the entry is in the same video to avoid data leakage at break points
            if entry_range['video'] != entry['video'] or entry_range['clip'] != entry['clip']:
                # if not, use a blank image
                img = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
            # check if file exists 
            elif not os.path.isfile(img_path):
                # if the file does not exist, use a blank image
                img = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
            else:
                # load the image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

            images.append(img)

        # load the corresponding heatmaps
        heatmap_paths = [
            os.path.join(
                self.path,
                "heatmaps",
                entry['subset'],
                entry['video'],
                entry['clip'],
                str(int(entry['frame']) + i) + ".npy"
            )
            for i in range(output_lag, 1)
        ]

        # Load and stack heatmaps
        heatmaps = self._load_heatmaps_numpy(heatmap_paths)

        # Generate a single set of transformation parameters
        # Apply the transformation to one image to get consistent params
        augmented = self.base_transform(image=images[0], mask=heatmaps[0])
        transform_params = augmented['replay']

        # apply transformations to heatmaps
        heatmaps = torch.stack([
            A.ReplayCompose.replay(transform_params, image=images[0], mask=heatmap)['mask']
            for heatmap in heatmaps
        ])

        # Apply these parameters to all images in the sequence
        transformed_images = []
        for img in images:
            augmented = A.ReplayCompose.replay(transform_params, image=img)
            transformed_images.append(augmented['image'])

        images = torch.stack([img.clone().detach().float() if isinstance(img, torch.Tensor) else torch.tensor(img, dtype=torch.float32) for img in transformed_images], dim=0)

        # get all points with output lag
        points = []
        for i in range(output_lag, 1):
            point_lag = self.data.iloc[idx + i]['points']
            if len(point_lag) == 0:
                points.append([-1, -1])  # Placeholder for missing points
            else:
                scaled_point = scale_points((point_lag[0]['x'], point_lag[0]['y']) , self.model_width / self.input_width)
                points.append(scaled_point)

        points = torch.tensor(points, dtype=torch.float32)  # Points should also be float32

        return images, heatmaps, points, entries

    def _load_heatmaps_numpy(self, heatmap_paths):
        heatmaps = []
        for path in heatmap_paths:
            if os.path.exists(path):
                heatmap = np.load(path)
                heatmaps.append(heatmap)
                # print size of heatmap
                
            else:
                heatmaps.append(np.zeros((self.input_height, self.input_width), dtype=np.float32))

        return np.stack(heatmaps)
        
