import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd


class ElbowxrayDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, xlsx_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patient_info = pd.read_excel(xlsx_file).iloc[1:,:] # Dataframe from pandas
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.patient_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_number = os.path.join(self.root_dir, self.patient_info.iloc[idx, 0])
        rl = self.patient_info.iloc[idx,3]                                  # XXX check type, must be string
        rl = '' if rl == 'n' else rl
        image = Image.open(patient_number + rl + '.tiff')
        label = float(self.patient_info.iloc[idx, 1])
        # readout = self.patient_info.iloc[idx, 4]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample