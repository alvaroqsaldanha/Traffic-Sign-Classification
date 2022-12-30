import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=320, kernel_size=3,padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=320, out_channels=256, kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(inplace=True),
        )
        
        self.classification = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16*256, 600),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=600, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        X = self.conv(x)
        X = X.view(X.shape[0], -1)
        Y = self.classification(X)
        return Y