from model import TrafficSignCNN
from torch import optim, nn
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from matplotlib.pyplot import Axes, gca, figure, savefig
from torch.utils.data import DataLoader
import pickle

def train(model, train_set, opt, criterion):
    epoch_ls = 0
    epoch_acc = 0
    model.train()
    for batch_idx, (x, y) in enumerate(train_set):
        opt.zero_grad()
        y_pred= model(x)
        loss = criterion(y_pred,y)
        loss.backward()
        final_pred = y_pred.argmax(axis=1)
        accuracy = accuracy_score(final_pred,y)
        opt.step()
        epoch_ls += loss.item()
        epoch_acc += accuracy
    return epoch_ls / len(train_set), epoch_acc / len(train_set)

num_classes = 43
batch_size = 256
learning_rate = 0.001
EPOCHS = 15

model = TrafficSignCNN(num_classes)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
transform = transforms.ToTensor()

train_set = pickle.load(open("serialized_data/train_data_loader", "rb"))

train_loss = []
train_accuracy = []

for epoch in range(EPOCHS):
    print("Epoch: ",epoch)
    train_ls, train_acc = train(model, train_set, optimizer, criterion)
    train_loss.append(train_ls)
    train_accuracy.append(train_acc)
    print(train_ls)
    print(train_acc)

ax = gca()
ax.set_title('Loss over time')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_xlim(0, EPOCHS)
ax.set_xticks([i for i in range(EPOCHS)])
legend: list = []
ax.plot([1 in range(EPOCHS+1)], train_loss)
legend.append('Loss')
