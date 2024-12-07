from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import numpy as np
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
    def __init__(self, path, model_height=288, model_width=512, input_size=3, output_size=3):
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
    

if __name__ == "__main__":
    dataset = BallDataSet("00_Dataset/New/Video_1/clip_1", model_height=288, model_width=512, input_size=3, output_size=3)
    # get example batch
    batch = dataset[0]
    print(batch.shape)



        
