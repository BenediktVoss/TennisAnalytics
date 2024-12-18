import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
import cv2
import torch.nn as nn
import gc

def create_window_index(df, output_size):
    # create a new column for the window index
    df['window_index'] = 0
    window_counter = 0
    output_counter = 0

    # loop over over data and create a window index
    for row in df.iterrows():
        # set row window index
        df.at[row[0], 'window_index'] = window_counter
        output_counter += 1

        #check if end of window is reached
        if output_counter == output_size:
            window_counter += 1
            output_counter = 0

    return df


class BallDataSet(Dataset):
    def __init__(self, path, model_height=288, model_width=512, input_size=5, output_size=5):
        self.model_height = model_height
        self.model_width = model_width
        self.path = path
        self.input_size = input_size
        self.output_size = output_size

        # gat all image file names in the path
        image_files = [
            file for file in os.listdir(path)
            if os.path.isfile(os.path.join(path, file)) and file.lower().endswith('.jpg')
        ]

        # create a dataframe from the image files
        self.data = pd.DataFrame(image_files, columns=['file'])

        # create columns  without the extension
        self.data['frame'] = self.data['file'].apply(lambda x: int(x.split('.')[0]))

        # sort by frame
        self.data = self.data.sort_values('frame').reset_index(drop=True)

        # convert json to dataframe
        self.data = create_window_index(self.data, self.output_size)

        # Define the Albumentations transformation pipeline
        self.base_transform = A.Compose([
            # Resize to the desired input dimensions
            A.Resize(model_height, model_width),

            # Convert the image to a PyTorch tensor
            ToTensorV2()
        ])
        self.size = self.data['window_index'].max() + 1

    
    def __len__(self):
        return self.size

    
    def __getitem__(self, idx):
        # get entries for window index
        entries = self.data[self.data['window_index'] == idx]
        
        #get idx of entry
        idx = entries.index[-1]
        input_lag = -self.input_size +1

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
                entry_range['file']
            )
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)

        # Apply these parameters to all images in the sequence
        transformed_images = []
        for img in images:
            augmented = self.base_transform(image=img)
            transformed_images.append(augmented['image'])

        images = torch.stack([img.clone().detach().float() if isinstance(img, torch.Tensor) else torch.tensor(img, dtype=torch.float32) for img in transformed_images], dim=0)

        return images
    
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

class BallTrackNet(nn.Module):
    def __init__(self, input_size=5, output_size=5):
        super(BallTrackNet, self).__init__()

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

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size, num_frames * channels, height, width)

        # Downsampling path
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

def calculate_position(output, threshold):
    # Binarize the heatmap
    binary_map = (output > threshold).cpu().numpy().astype(np.uint8)

    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    if num_labels > 1:  # Exclude background label 0
        # Find the largest region (excluding background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Index 1 is the first component
        largest_centroid = centroids[largest_label]
        return int(largest_centroid[0]), int(largest_centroid[1])
    else:
        # No region found
        return -1, -1
    
# Clear GPU memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def inference_ball_tracknet(images_path, model_path, image_width):
    clear_gpu_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = BallTrackNet(input_size=5, output_size=5)
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
    dataset = BallDataSet(images_path, model_height=288, model_width=512, input_size=5, output_size=5)

    # calculate updscaling ratio
    ratio = image_width / 512

    #create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    positions = []
    for batch_idx, images in enumerate(dataloader):
        with torch.no_grad():
            images = images.to(device)
            output = model(images)

            for i in range(output.shape[1]):
                output_element = output[0, i].cpu()

                # Calculate the position of the ball
                position = calculate_position(output_element, threshold=0.5)

                # scale the position
                position = (int(position[0] * ratio), int(position[1] * ratio))

                # Append the position to the list
                positions.append(position)

    clear_gpu_memory()
    # return the positions
    return positions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(inference_ball_tracknet("../00_Dataset/Amateur/Video_1/clip_1", "models/BallTracking.pt", 1280))
