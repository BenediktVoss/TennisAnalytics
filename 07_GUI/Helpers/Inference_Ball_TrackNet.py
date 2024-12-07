import os
import torch
import pandas as pd
import numpy as np
from BallTrackNet import BallTrackNet
from Ball_Dataset import BallDataSet
from collections import OrderedDict
import cv2

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

def inference_ball_tracknet(images_path, model_path, device):
    # Load the model
    model = BallTrackNet(input_size=3, output_size=3)
    # Load the model's state dict
    state_dict = torch.load(model_path, map_location=device)

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
    dataset = BallDataSet(images_path, model_height=288, model_width=512, input_size=3, output_size=3)

    #create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    positions = []
    for batch_idx, images in enumerate(dataloader):
        with torch.no_grad():
            images = images.to(device)
            output = model(images)

            # Calculate the position of the ball
            position = calculate_position(output[0, 0], threshold=0.5)

            # Append the position to the list
            positions.append(position)

    # return the positions
    return positions


# test in main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(inference_ball_tracknet("00_Dataset/New/Video_1/clip_1", "07_GUI\models\model_best.pt", device))