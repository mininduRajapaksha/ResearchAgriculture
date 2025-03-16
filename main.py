import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
test_dir = 'D:/Research project/Datasets/Banana Dataset/Test'
valid_dir = 'D:/Research project/Datasets/Banana Dataset/Valid'

# Set image dimensions and batch size
img_height = 416
img_width = 416
batch_size = 32

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',  # Use 'categorical' for multi-class classification
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True
)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=False
)

# Store class names before preprocessing
class_names = train_ds.class_names

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

print("Class names:", class_names)

# Display a few sample images from the training set
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[tf.argmax(labels[i]).numpy()])
        plt.axis("off")
plt.show()