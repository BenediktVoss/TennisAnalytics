import torch.nn as nn
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import functional as F
from collections import OrderedDict
import gc


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class Court_TN(nn.Module):
    def __init__(self, input_size=1, output_size=15):
        super(Court_TN, self).__init__()

        input_layers = input_size * 3

        # Define layers
        self.conv1 = ConvBlock(input_layers, 64)
        self.conv2 = ConvBlock(64, 64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv11 = ConvBlock(512 + 256, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv14 = ConvBlock(256 + 128, 128)
        self.conv15 = ConvBlock(128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv16 = ConvBlock(128 + 64, 64)
        self.conv17 = ConvBlock(64, 64)

        self.conv18 = nn.Conv2d(64, output_size, kernel_size=1, padding=0)

        self._init_weights()
    

    def forward(self, x):
        
        x1 = self.conv1(x)
        
        x1 = self.conv2(x1)
        x2 = self.pool1(x1)
    
        x2 = self.conv3(x2)
    
        x2 = self.conv4(x2)
        x3 = self.pool2(x2)
    
        x3 = self.conv5(x3)
        x3 = self.conv6(x3)
        x3 = self.conv7(x3)
        x4 = self.pool3(x3)
    
        x4 = self.conv8(x4)
        x4 = self.conv9(x4)
        x4 = self.conv10(x4)
    
        # Upsampling path
        x5 = self.up1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.conv11(x5)
        x5 = self.conv12(x5)
        x5 = self.conv13(x5)
    
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.conv14(x6)
        x6 = self.conv15(x6)
    
        x7 = self.up3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.conv16(x7)
        x7 = self.conv17(x7)

        output = torch.sigmoid(self.conv18(x7))
        return output

    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)  

def process_images(path):
    # gat all image file names in the path
    image_files = [
        file for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.lower().endswith('.jpg')
    ]

    # create a dataframe from the image files
    data = pd.DataFrame(image_files, columns=['file'])

    # create columns  without the extension
    data['frame'] = data['file'].apply(lambda x: int(x.split('.')[0]))

    # sort by frame
    data = data.sort_values('frame').reset_index(drop=True)

    # keep only every 50th frame
    data = data[data['frame'] % 50 == 0]

    return data.reset_index(drop=True)



class CourtDatasetHeatmap(Dataset):
    def __init__(self, path, input_height=720, input_width=1280, model_height=288, model_width=512):
        self.input_height = input_height
        self.input_width = input_width
        self.model_height = model_height
        self.model_width = model_width
        self.path = path

        self.data = process_images(path)

        # Define the Albumentations transformation pipeline
        self.base_transform = A.Compose([
            # Resize to the desired input dimensions
            A.Resize(model_height, model_width),

            # Normalize for resnet
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            # Convert the image to a PyTorch tensor
            ToTensorV2()
        ])


    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]

        img_path = os.path.join(
                self.path,
                entry['file']
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transformations with replay
        augmented = self.base_transform(image=img)
        img = augmented['image']

        return img
    
def calculate_position(output, threshold_probability=0.5):
    # Convert to binary map based on the threshold
    binary_map = (output > threshold_probability).astype(np.uint8)

    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    if num_labels > 1:  # Exclude the background label (label 0)
        # Find the largest connected component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Offset by 1 due to excluding background
        largest_centroid = centroids[largest_label]
        return int(largest_centroid[0]), int(largest_centroid[1])
    else:
        # No connected component found
        return np.nan, np.nan
    

def calculate_average_positions(data):
    num_indices = len(data[0])  # Number of positions per frame
    averages = []

    for idx in range(num_indices):
        x_values = []
        y_values = []

        # Collect valid positions
        for frame in data:
            x, y = frame[idx]
            if (x, y) != (np.nan, np.nan):  
                x_values.append(x)
                y_values.append(y)

        # Calculate the average, or (-2, -2) if no valid positions
        if x_values and y_values:
            avg_x = int(np.mean(x_values))
            avg_y = int(np.mean(y_values))
            averages.append((avg_x, avg_y))
        else:
            averages.append((np.nan, np.nan))  # No valid data for this index

    return averages

# Clear GPU memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def inference_ball_tracknet(images_path, model_path, image_width):
    clear_gpu_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = Court_TN(input_size=1, output_size=15)
    # Load the model's state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Check if keys have "module." prefix (happens when using DataParallel)
    if any(key.startswith("module.") for key in state_dict.keys()):
        # Remove "module." prefix
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module."
            new_state_dict[new_key] = value
        state_dict = new_state_dict

    # Load the state dict into the model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create a dataset
    dataset = CourtDatasetHeatmap(images_path, model_height=288, model_width=512)

    # calculate updscaling ratio
    ratio = image_width / 512

    #create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    positions = []
    for batch_idx, images in enumerate(dataloader):
        with torch.no_grad():
            images = images.to(device)
            output = model(images)

            keypoints = []

            for i in range(output.shape[1]):
                output_element = output[0, i].cpu().numpy()

                # Calculate the position of the ball
                keypoint = calculate_position(output_element, threshold_probability=0.5)

                # scale the position
                keypoint = (int(keypoint[0] * ratio), int(keypoint[1] * ratio)) if keypoint != (np.nan, np.nan) else keypoint

                # Append the position to the list
                keypoints.append(keypoint)
            
            # Append the keypoints to the list
            positions.append(keypoints)

    positions = calculate_average_positions(positions)
    clear_gpu_memory()

    # return the positions
    return positions

    
def main():
    # Define dataset parameters
    dataset_path = "../00_Dataset/Amateur/Video_1/clip_5"
    model_path = "models/CourtTracking.pth"
    print(inference_ball_tracknet(dataset_path, model_path, 1280))
    
if __name__ == "__main__":
    main()
 