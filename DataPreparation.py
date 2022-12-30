from DatasetLoader import GTRSBDataset
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import os

batch_size = 256

if not os.path.exists(os.getcwd() + os.sep + "serialized_data/"):
    os.makedirs(os.getcwd() + os.sep + "serialized_data/")

train_set = GTRSBDataset('data/train.csv','data',transform=transforms.ToTensor()) 
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
pickle.dump(train_loader, open("serialized_data/train_data_loader", "wb"))

test_set = GTRSBDataset('data/test.csv','data',transform=transforms.ToTensor()) 
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
pickle.dump(test_loader, open("serialized_data/test_data_loader", "wb"))

