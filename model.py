import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# import cv2
import numpy as np
import tensorflow as tf
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


