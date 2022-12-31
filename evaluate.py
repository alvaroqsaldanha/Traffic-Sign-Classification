from model import TrafficSignCNN
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report
import torch
import pickle
import codecs, json

test_set = pickle.load(open("serialized_data/test_data_loader", "rb"))
model = TrafficSignCNN(43)
model.load_state_dict(torch.load('serialized_data/model.pt'))

label_json = codecs.open("DataProfiling/label_names.json", 'r', encoding='utf-8').read()
label_names = json.loads(label_json)
#criterion = nn.CrossEntropyLoss()

true_labels = []
pred_labels = []

with torch.no_grad():
    model.eval()
    y_right = 0
    for idx, (x, y) in enumerate(test_set):
        y_pred = model(x)
        final_pred = y_pred.argmax(axis=1)
        if y.item() == final_pred.item():
            y_right += 1
        pred_labels.append(final_pred.item())
        true_labels.append(y.item())

print(f"Correctly classified images: {y_right}")
print(f"Incorrectly classified images: {len(test_set)-y_right}")
print(f"Final Model Accuracy: {y_right/len(test_set)}")
print(classification_report(true_labels,pred_labels))



