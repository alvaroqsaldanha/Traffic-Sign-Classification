import torch
import torchvision.transforms as transforms 
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class GTRSBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 7])
        image = Image.open(img_path)
        image = image.resize((32,32))
        y_label = torch.tensor(int(self.annotations.iloc[index, 6]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)