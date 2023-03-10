import torch
import torchvision.transforms as transforms 
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import cv2

class GTRSBDataset(Dataset):
    def __init__(self, csv_file, root_dir,img_size=32, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 7])
        image = cv2.imread(img_path)
        image = cv2.resize(image,(self.img_size,self.img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        y_label = torch.tensor(int(self.annotations.iloc[index, 6]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)