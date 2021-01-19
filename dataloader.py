import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd


class ElbowxrayDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, xlsx_file, root_dir, transform=None, num_classes =2):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patient_info = pd.read_excel(xlsx_file).iloc[1:,:] # Dataframe from pandas
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.patient_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_number = os.path.join(self.root_dir, self.patient_info.iloc[idx, 0])
        image = Image.open(patient_number + '.tiff').convert('RGB')
        label = int(self.patient_info.iloc[idx, 1])
        # label_onehot = torch.zeros(self.num_classes)
        # label_onehot[label] = 1
        # label = label_onehot
        # readout = self.patient_info.iloc[idx, 4]
        sample = {'image': image, 'label': label}

        if self.transform:
            image = self.transform(image)
            sample = {'image': image, 'label': label}

        return sample