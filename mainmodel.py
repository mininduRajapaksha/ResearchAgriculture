import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import json


#defining the paths
train_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
test_dir = 'D:/Research project/Datasets/Banana Dataset/Test'
valid_dir = 'D:/Research project/Datasets/Banana Dataset/Valid'

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

#data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,)

# Load training dataset
train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)

# Load validation dataset
val_set = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

# Load test dataset
test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Print class indices
print("Class indices:", train_set.class_indices)

# Display a few sample images from the training set
# sample_images, sample_labels = next(train_set)
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i+1)
#     plt.imshow(sample_images[i])
#     # Get class label name from the generator's class_indices dictionary
#     class_indices = {v: k for k, v in train_set.class_indices.items()}
#     plt.title(class_indices[np.argmax(sample_labels[i])])
#     plt.axis("off")
# plt.tight_layout()
# plt.show()

# Build the CNN model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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

# Display the model's architecture
model.summary()

# Compile the model
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = np.ceil(train_set.samples // train_set.batch_size)
validation_steps = np.ceil(val_set.samples // val_set.batch_size)

#early stopping
# early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)



# Train the model
epochs = 15
history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=epochs,
    # callbacks=[early_stop]  
)

# Save the traing history to a JSON file
with open('history.json', 'w') as f:
    json.dump(history.history, f)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_set)
print("Test Accuracy:", test_accuracy)

# Load the training history from a JSON file
with open('history.json', 'r') as f:
    history = json.load(f)

# Plot Training and Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()


# Save the model
model.save('D:/Quality Control System/banana_quality_model.h5')

# Load the trained model
model = load_model('banana_quality_model.h5')

