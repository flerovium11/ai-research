"""
The code in this file is very unprofessinal, I'm sorry. 
I was running out of time with my final project; 
tried using keras-tuner for the other models but didn't get it to work.

Command to run it from ssh shell overnight: nohup /opt/conda/bin/python -u diy_hyperparam_search.py &> logs/hyperparam_search.log &
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from PIL import Image
import cv2
import glob

dataset_path = 'data/GTSRB'
classes_path = os.path.join(dataset_path, 'classes.txt')
classes_german_path = os.path.join(dataset_path, 'classes-german.txt')
img_size = 32

with open(classes_path, 'r') as file:
    classes = [line.strip() for line in file.readlines()]

with open(classes_german_path, 'r') as file:
    classes_german = [line.strip() for line in file.readlines()]

images_path = os.path.join(dataset_path, 'Training')
class_folders = [folder for folder in os.listdir(
    images_path) if os.path.isdir(os.path.join(images_path, folder))]
data = []

for i, folder in enumerate(class_folders):
    folder_path = os.path.join(images_path, folder)
    files = glob.glob(os.path.join(folder_path, '*.ppm'))
    label = int(folder)

    for file in files:
        image = plt.imread(file)
        image = cv2.resize(image, (img_size, img_size))
        data.append((image, label))

print()
print(f'Loaded {len(data)} images')

print('Category distribution:')
for i, class_name in enumerate(classes):
    print(
        f'{class_name}: {len(list(filter(lambda entry: entry[1] == i, data)))}')


# original search space
# layer1_filters = [32, 64, 128]
# layer1_kernel_sizes = [(3, 3)]
# layer2_neurons = [128, 256, 512]
# layer1_dropouts = [0, 0.1, 0.2, 0.25, 0.3, 0.35]
# layer2_dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# batch_sizes = [32, 64, 128, 256]


# for layer1_filter_num in layer1_filters:
#     for layer1_kernel_size in layer1_kernel_sizes:
#         for layer2_neuron_num in layer2_neurons:
#             for layer1_dropout in layer1_dropouts:
#                 for layer2_dropout in layer2_dropouts:
#                     for batch_size in batch_sizes:
#                         print(f'Trying values {layer1_filter_num} {layer1_kernel_size} {layer2_neuron_num} {layer1_dropout} {layer2_dropout} {batch_size}')

#                         accuracies = []
#                         for i in range(1):
#                             print(f'Trial {i + 1}')

#                             model = Sequential()
#                             model.add(Input(shape=(img_size, img_size, 3)))
#                             model.add(Conv2D(layer1_filter_num, layer1_kernel_size, activation='relu'))
#                             model.add(MaxPooling2D((2, 2)))
#                             model.add(Dropout(layer1_dropout))
#                             model.add(Flatten())
#                             model.add(Dense(layer2_neuron_num, activation='relu'))
#                             model.add(Dropout(layer2_dropout))
#                             model.add(Dense(len(classes), activation='softmax'))
#                             model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#                             split_train_val = 0.8
#                             split_index = int(len(data) * split_train_val)
#                             np.random.shuffle(data)
#                             train_data, val_data = data[:split_index], data[split_index:]
#                             train_images, train_labels = np.array([entry[0] for entry in train_data]), np.array([entry[1] for entry in train_data])
#                             val_images, val_labels = np.array([entry[0] for entry in val_data]), np.array([entry[1] for entry in val_data])

#                             history = model.fit(train_images,
#                                 train_labels,
#                                 epochs=20,
#                                 batch_size=batch_size,
#                                 validation_data=(val_images, val_labels),
#                                 verbose=0)

#                             accuracies.append(history.history["val_accuracy"][-1])

#                         print(f'Reached average_val_accuracy {sum(accuracies) / (len(accuracies))}')

# best performing configs
configs = [
    (32, (3, 3), 512, 0.1, 0.3, 256),
    (32, (3, 3), 512, 0.25, 0.2, 256),
    (64, (3, 3), 128, 0.25, 0.1, 128),
    (32, (3, 3), 512, 0.3, 0.1, 256),
    (32, (3, 3), 256, 0.2, 0.2, 256),
    (32, (3, 3), 512, 0, 0.5, 64),
    (32, (3, 3), 512, 0.2, 0.4, 256),
    (32, (3, 3), 512, 0.35, 0.2, 256),
    (64, (3, 3), 256, 0.1, 0.1, 256),
    (64, (3, 3), 256, 0.1, 0.2, 128),
    (64, (3, 3), 256, 0.25, 0.3, 256)
]

for config in configs:
    layer1_filter_num, layer1_kernel_size, layer2_neuron_num, layer1_dropout, layer2_dropout, batch_size = config
    print(
        f'Trying values {layer1_filter_num} {layer1_kernel_size} {layer2_neuron_num} {layer1_dropout} {layer2_dropout} {batch_size}')

    accuracies = []
    for i in range(5):
        print(f'Trial {i + 1}')

        model = Sequential()
        model.add(Input(shape=(img_size, img_size, 3)))
        model.add(Conv2D(layer1_filter_num,
                  layer1_kernel_size, activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(layer1_dropout))
        model.add(Flatten())
        model.add(Dense(layer2_neuron_num, activation='relu'))
        model.add(Dropout(layer2_dropout))
        model.add(Dense(len(classes), activation='softmax'))
        model.compile(
            optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        split_train_val = 0.8
        split_index = int(len(data) * split_train_val)
        np.random.shuffle(data)
        train_data, val_data = data[:split_index], data[split_index:]
        train_images, train_labels = np.array(
            [entry[0] for entry in train_data]), np.array([entry[1] for entry in train_data])
        val_images, val_labels = np.array([entry[0] for entry in val_data]), np.array([
            entry[1] for entry in val_data])

        history = model.fit(train_images,
                            train_labels,
                            epochs=20,
                            batch_size=batch_size,
                            validation_data=(val_images, val_labels),
                            verbose=0)

        accuracies.append(history.history["val_accuracy"][-1])

    print(
        f'Reached average_val_accuracy {sum(accuracies) / (len(accuracies))}')
