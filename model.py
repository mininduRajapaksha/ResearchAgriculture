import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


#defining the paths
train_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
test_dir = 'D:/Research project/Datasets/Banana Dataset/Test'
valid_dir = 'D:/Research project/Datasets/Banana Dataset/Valid'

#data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Load training dataset
train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(416, 416),
    batch_size=32,
    class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)

# Load validation dataset
val_set = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(416, 416),
    batch_size=32,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

# Load test dataset
test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(416, 416),
    batch_size=32,
    class_mode='categorical')

# Print class indices
print("Class indices:", train_set.class_indices)

# Display a few sample images from the training set
sample_images, sample_labels = next(train_set)
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(sample_images[i])
    # Get class label name from the generator's class_indices dictionary
    class_indices = {v: k for k, v in train_set.class_indices.items()}
    plt.title(class_indices[np.argmax(sample_labels[i])])
    plt.axis("off")
plt.tight_layout()
plt.show()

# Build the CNN model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(416, 416, 3)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    # Flatten and dense layers
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax') # 3 classes: fresh, rotten, unripe
])