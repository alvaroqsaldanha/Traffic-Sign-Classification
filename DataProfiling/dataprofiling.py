import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random
import os
import codecs, json

IMAGE_SIZE = 32

train_path = os.getcwd() + os.sep + 'data' + os.sep + 'train' + os.sep
test_path = os.getcwd() + os.sep + 'data' + os.sep + 'test' + os.sep
num_classes = 43
train_data = []
train_labels = []
test_data = []

label_json = codecs.open("label_names.json", 'r', encoding='utf-8').read()
label_names = json.loads(label_json)

# Loading Training Data
for i in range(num_classes):
    img_path = train_path + str(i)
    images = os.listdir(img_path)
    print(f"Loading images for class {i}...")
    for image in tqdm(images, colour='green'): 
        img = Image.open(img_path + '/' + image)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = np.array(img)
        train_data.append(img)
        train_labels.append(i)

# Loading Test Data
images = os.listdir(test_path)
print("Loading images for test set...")
for image in tqdm(images, colour='blue'):
    try: 
        img = Image.open(test_path + '/' + image)
        img = img.resize((32,32))
        img = np.array(img)
        test_data.append(img)
    except:
        print(f"Couldn't load file: {image}.")

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)

print("\nShape of training data: ", train_data.shape)
print("Shape of test data: ", test_data.shape)
print(f"Number of classes: {num_classes}")
print(f"Images resized to {IMAGE_SIZE} x {IMAGE_SIZE}")

plot_size = 3
random_idxs = [random.randint(0, train_data.shape[0]) for i in range(plot_size**2)]
fig = plt.figure(figsize=(15, 15))
for i, index in enumerate(random_idxs):
    a=fig.add_subplot(plot_size,plot_size, i+1)
    imgplot = plt.imshow(train_data[index])
    a.set_title(label_names[str(train_labels[index]+1)])
plt.show()

# Serializing data
if not os.path.exists(os.getcwd() + os.sep + "serialized_data/"):
    os.makedirs(os.getcwd() + os.sep + "serialized_data/")

json.dump(train_data.tolist(), codecs.open("serialized_data/train_serialized.json", 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4) 
json.dump(train_labels.tolist(), codecs.open("serialized_data/trainlabels_serialized.json", 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4) 
json.dump(test_data.tolist(), codecs.open("serialized_data/test_serialized.json", 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4) 









