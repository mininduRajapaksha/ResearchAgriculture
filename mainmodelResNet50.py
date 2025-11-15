import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import json

# Define the paths for your datasets
train_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
test_dir = 'D:/Research project/Datasets/Banana Dataset/Test'
valid_dir = 'D:/Research project/Datasets/Banana Dataset/Valid'

# Set image dimensions and batch size
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)
train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Preprocessing for validation
val_datagen = ImageDataGenerator(rescale=1./255)
val_set = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Preprocessing for test
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Print class indices to confirm
print("Class indices:", train_set.class_indices)

# Build the model using ResNet50
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',     
    include_top=False,      
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False 

# Add custom classification layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dropout(0.5)(x)             
x = Dense(128, activation='relu')(x) 
predictions = Dense(3, activation='softmax')(x)

# Combine base model and custom head into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Display the model's architecture
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Optionally, set up early stopping if needed
# early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Train the model
epochs = 15
history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=epochs,
    # callbacks=[early_stop]  # Uncomment if using early stopping
)

# Save the training history to a JSON file
with open('history.json', 'w') as f:
    json.dump(history.history, f)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_set)
print("Test Accuracy:", test_accuracy)

# Load the training history from the JSON file for plotting
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
model.save('D:/Quality Control System/banana_quality_model_version2.h5')

# Optionally, load the model later with:
# model = load_model('D:/Quality Control System/banana_quality_model.h5')
