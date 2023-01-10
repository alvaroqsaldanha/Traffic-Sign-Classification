from DatasetLoader import GTRSBDataset
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import os
import cv2
import numpy
import matplotlib.pyplot as plt
import cv2
batch_size = 256

class CLAHE:
    def __call__(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        img = clahe.apply(img)
        img = img.reshape(img.shape + (1,))
        return img

class TORGB:
     def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

clahe_transforms = transforms.Compose([
    CLAHE(),
    transforms.ToTensor()
])

normal_transforms = transforms.Compose([
    TORGB(),
    transforms.ToTensor()
])

if not os.path.exists(os.getcwd() + os.sep + "serialized_data/"):
    os.makedirs(os.getcwd() + os.sep + "serialized_data/")

train_set = GTRSBDataset('data/train.csv','data',transform=normal_transforms) 
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
pickle.dump(train_loader, open("serialized_data/train_data_loader", "wb"))

test_set = GTRSBDataset('data/test.csv','data',transform=normal_transforms) 
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
pickle.dump(test_loader, open("serialized_data/test_data_loader", "wb"))

train_set_clahe = GTRSBDataset('data/train.csv','data',transform=clahe_transforms) 
train_loader_clahe = DataLoader(dataset=train_set_clahe, batch_size=64, shuffle=True)
pickle.dump(train_loader_clahe, open("serialized_data/train_data_loader_clahe", "wb"))

test_set_clahe = GTRSBDataset('data/test.csv','data',transform=clahe_transforms) 
test_loader_clahe = DataLoader(dataset=test_set_clahe, batch_size=1, shuffle=True)
pickle.dump(test_loader_clahe, open("serialized_data/test_data_loader_clahe", "wb"))
