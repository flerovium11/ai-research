import matplotlib.pyplot as plt
from copy import deepcopy
import os
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
import random
import numpy as np
from PIL import Image
import yaml

class DataType(Enum):
    TRAIN = 1
    TEST = 2
    VALID = 3

class Label:
    def __init__(self, raw_data: str) -> None:
        split_data = raw_data.split(' ')
        self.category = int(split_data[0])
        self.center_x, self.center_y, self.width, self.height = map(float, split_data[1:])

class Entry:
    def __init__(self, image: np.array, labels: list[Label], image_name: str) -> None:
        self.image = image
        self.labels = labels
        self.image_name = image_name


dataset_path = 'data/traffic-signs-detection'
info_file = os.path.join(dataset_path, 'car/data.yaml')
categories = yaml.load(open(info_file), Loader=yaml.FullLoader)['names']

img_size = 128
min_bounding_box_size = 0.2
only_single_label = True
forbidden_file_prefixes = ['FisheyeCamera', 'road']
grayscale = False

def load_image_data(type: DataType):
    data_path = os.path.join(dataset_path, 'car', type.name.lower())
    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')

    entries = []
    files_in_folder = os.listdir(images_path)
    print(f'Scanning {len(files_in_folder)} entries from {images_path} and {labels_path}...')

    for image_name in files_in_folder:
        image = plt.imread(os.path.join(images_path, image_name))
        image = np.array(Image.fromarray(image).resize((img_size, img_size)))

        if grayscale:
            image = np.mean(image, axis=2)
        
        image = image / 255.0
        labels_raw = open(os.path.join(labels_path, image_name.replace('.jpg', '.txt'))).read().split('\n')
        labels = [Label(label) for label in labels_raw if label]

        if (
            only_single_label and len(labels) > 1
            or len(labels) == 0
            or any([image_name.startswith(prefix) for prefix in forbidden_file_prefixes])
            or any([label.width < min_bounding_box_size or label.height < min_bounding_box_size for label in labels])
        ):
            continue

        entries.append(Entry(image, labels, image_name))

    return entries

train_data = load_image_data(DataType.TRAIN)
validate_data = load_image_data(DataType.VALID)
test_data = load_image_data(DataType.TEST)

print(f'Loaded {len(train_data)} training images, {len(validate_data)} validation images and {len(test_data)} test images')

def get_data(data: list[Entry]):
    images = np.array([entry.image for entry in data])
    labels = np.array([entry.labels[0].category for entry in data])
    return images, labels

train_images, train_labels = get_data(train_data)
validate_images, validate_labels = get_data(validate_data)
test_images, test_labels = get_data(test_data)

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class CustomModel(kt.HyperModel):
    def build(self, hp):
        tf.keras.backend.clear_session()
        model = Sequential()

        # Define input shape
        model.add(Input(shape=(img_size, img_size, 3)))
        model.add(Flatten())
        
        # Search for number of layers between 1 and 3
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu'))
            model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0, max_value=0.5, step=0.05)))

        # Output layer
        model.add(Dense(len(categories), activation='softmax'))

        # Compile the model with an optimizer that has a tunable learning rate
        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-2, sampling='log')), 
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    
    # thanks to https://github.com/keras-team/keras-tuner/issues/122#issuecomment-544648268
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [2 ** n for n in range(1, 11)]),
            verbose=2,
            **kwargs,
        )

# Set up the tuner
tuner = kt.Hyperband(
    CustomModel(),
    objective='val_accuracy',
    max_epochs=200,
    hyperband_iterations=6,
    directory='models',
    project_name='ann_hyperparam_search'
)

class ClearGPUCallback(tf.keras.callbacks.Callback):
    def on_train_end(self):
        tf.keras.backend.clear_session()

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
gpu_clear_callback = ClearGPUCallback()

# Search for the best hyperparameters
tuner.search(train_images, train_labels,
             validation_data=(validate_images, validate_labels),
             callbacks=[lr_scheduler, early_stop])

# Get the best model
best_model = tuner.get_best_models()[0]

# Evaluate the model
loss, accuracy = best_model.evaluate(test_images, test_labels)

# Plot training and validation accuracy
history = best_model.history
plt.plot(history.history['accuracy'], color='red', label='Training')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
print(f'Accuracy: {accuracy * 100:.2f}%')

# nohup /opt/conda/bin/python /home/oinnerednib/ai-research/ann_hyperparam_search.py >> logs/ann_hyperparam_search.log 2>&1 &
