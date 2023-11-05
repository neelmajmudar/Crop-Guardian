import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import PIL 
import cv2
import os 
import numpy as np
import seaborn as sns
import random
import shutil

"""
data_dir = "Data"
train_dir = "train"
test_dir = "test"
split_ratio = 0.8  #80% for training, 20% for testing


# Create training and testing directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    images = os.listdir(class_path)
    num_images = len(images)
    num_train = int(num_images * split_ratio)

    random.shuffle(images)

# Copy images to training directory
    for img in images[:num_train]:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, class_name, img)
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        shutil.copy(src, dst)

# Copy images to testing directory
    for img in images[num_train:]:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_dir, class_name, img)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        shutil.copy(src, dst)

"""
"""
train_dir = 'train'
test_dir = 'test'
batch_size = 16
img_height = 256
img_width = 256


# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.1,
    rotation_range=25,
    width_shift_range=0.05,
    height_shift_range=0.05
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dense(128, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[early_stopping]
)

# Save the model
model.save('guava_disease_model.h5')
"""

# Load the pre-trained model
#model = keras.models.load_model('apple_disease_model85.h5')

# Load the pre-trained model
model = keras.models.load_model('guava_disease_model81.h5')

# Define a function to predict an image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training data

    # Make the prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    # Define class labels based on your mapping
    class_labels = {
        0: 'Phytopthora',
        1: 'Root',
        2: 'Scab'
    }

    predicted_class = class_labels[class_index]
    confidence = np.max(prediction)

    return predicted_class, confidence

# Example usage:
image_path = 'train/Scab/Scab 1.jpg'
predicted_class, confidence = predict_image(image_path)
print(f'Predicted class: {predicted_class} with confidence: {confidence:.2f}')
