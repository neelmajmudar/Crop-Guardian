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
import PIL 
import cv2
import os 
import numpy as np
import seaborn as sns
import random
import shutil

traindir = 'train'
testdir = 'test'
batch_size = 32
img_h = 300
img_w = 300 



#data_dir = "Data"
#train_dir = "train"
#test_dir = "test"
#split_ratio = 0.8  #80% for training, 20% for testing


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
class CauliflowerML:
    def __init__(self, traindir, testdir, batch_size):
        self.traindir = traindir
        self.testdir = testdir
        self.batch_size = batch_size
        self.model = self.build_model()
        self.history = None 

# Inside the create_data_generators method
    def create_data_generators(self):
        training_generator = ImageDataGenerator(
        rescale=1.0/255,
        zoom_range=0.1,
        rotation_range=25,
        width_shift_range=0.05,
        height_shift_range=0.05
        )
        validation_generator = ImageDataGenerator()

    # Load the image data and one-hot encode the labels
        train_data = self.load_image_data(self.traindir)
        test_data = self.load_image_data(self.testdir)

        self.training_iterator = training_generator.flow(train_data[0], to_categorical(train_data[1], num_classes=4), batch_size=self.batch_size)
        self.testing_iterator = validation_generator.flow(test_data[0], to_categorical(test_data[1], num_classes=4), batch_size=self.batch_size)

    # Check the shape of labels
        print("Shape of training data:", train_data[0].shape)
        print("Shape of testing data:", test_data[0].shape)
        print("Shape of training labels:", train_data[1].shape)
        print("Shape of testing labels:", test_data[1].shape)

    def load_image_data(self, data_dir):
    # Load image data and corresponding labels
        data = []
        labels = []
    
        class_to_int = {'No Disease': 0, 'Mildew': 1, 'Black Rot': 2, 'Spot Rot': 3}  # Define a mapping from class labels to integers

        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            images = os.listdir(class_path)
        
            for img in images:
                img_path = os.path.join(class_path, img)
                img_data = cv2.imread(img_path)
                img_data = cv2.resize(img_data, (300, 300))
                data.append(img_data)
                labels.append(class_to_int[class_name])  # Map class label to integer

        data = np.array(data)
        labels = np.array(labels)
    
        return data, labels


def build_model(self):
    self.model = Sequential()
    # tf.random.set_seed(42)
    self.model.add(layers.Conv2D(8, (3, 3), activation="relu", input_shape=(300, 300, 3))
    self.model.add(layers.MaxPooling2D(2, 2))
    self.model.add(layers.Conv2D(16, (3, 3), activation="relu"))
    self.model.add(layers.MaxPooling2D(2, 2))
    self.model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    self.model.add(layersMaxPooling2D(2, 2))
    self.model.add(layers.Dropout(0.2))
    self.model.add(layers.Dense(128, activation='relu'))
    self.model.add(layers.Dropout(0.2))
    self.model.add(layers.Dense(4, activation='softmax'))  # Adjusted output shape

    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    ))

    def train_model(self, epochs=10):
    # Early stopping to prevent accuracy from plateau/overfitting
        es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)
        self.history = self.model.fit(
        self.training_iterator,
        epochs=epochs,
        steps_per_epoch=len(self.training_iterator),
        validation_data=self.testing_iterator,
        validation_steps=len(self.testing_iterator),
        callbacks=[es]
        )
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.testing_iterator)
        print("Model Loss:", loss)
        print("Model Accuracy:", accuracy)

    def predict(self, image):
        img = load_img(image, target_size=(300,300,3), color_mode='rgb')
        image_array = img_to_array(img)
        image_array = image_array / 255
        image_array = np.expand_dims(image_array, axis=0)
        prediction = self.model.predict(image_array)
        pclass = np.argmax(prediction)
        print(pclass)

    def plot_epoch_accuracy(self):
        plt.figure(figsize=(10,6))
        plt.plot(range(1,11),self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(range(1,11),self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(1, 11, 1))
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def plot_epoch_loss(self):
        plt.figure(figsize=(10,6))
        plt.plot(range(1,11),self.history.history['loss'], label='Training Loss')
        plt.plot(range(1,11),self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(np.arange(1, 11, 1))
        plt.legend()
        plt.show()

    def plot_prediction_actual(self):
        predictions = []
        actualvals = []
        plt.figure(figsize=(8,8))
        for _ in  range(66):
            test, actual = self.testing_iterator.next()
            actualvals.extend(np.argmax(actual, axis=1))

            prediction = self.model.predict(test)
            predictions.extend(np.argmax(prediction, axis=1))
        cmatrix = confusion_matrix(actualvals, predictions)

        sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Actual Classification')
        plt.ylabel('Predicted Classification')
        plt.title('Confusion Matrix')
        plt.show()

    def on_train_end(self):
        self.model.save('model.h5')

model = CauliflowerML(traindir,testdir, batch_size)
model.create_data_generators()
model.build_model()
model.train_model()
model.evaluate_model()
model.on_train_end()
"""



# Define your data directories
train_dir = 'train'
test_dir = 'test'
batch_size = 32
img_height = 300
img_width = 300



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
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

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
model.save('cabbage_disease_model.h5')



# Load the pre-trained model
model = keras.models.load_model('cabbage_disease_model.h5')

# Define a function to predict an image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training data

    # Make the prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    # Define class labels based on your mapping
    class_labels = {
        0: 'No Disease',
        1: 'Downy Mildew',
        2: 'Black Rot',
        3: 'Spot Rot'
    }

    predicted_class = class_labels[class_index]
    confidence = np.max(prediction)

    return predicted_class, confidence


image_path = 'train/Mildew/Downy Mildew. (16).jpeg'
predicted_class, confidence = predict_image(image_path)
print(f'Predicted class: {predicted_class} with confidence: {confidence:.2f}')