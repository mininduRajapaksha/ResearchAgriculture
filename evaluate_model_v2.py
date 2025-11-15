import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import json
import os

def evaluate_model_v2():
    # List all .h5 files in current directory
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    print("\nAvailable model files:", model_files)

    # Try to load the model
    model_path = 'banana_quality_model.h5'  # Default model name
    if not os.path.exists(model_path):
        print(f"\nError: Could not find model at {model_path}")
        print("Please ensure the model file exists in the current directory")
        return

    try:
        model = load_model(model_path)
        print(f"\nModel loaded successfully from {model_path}!")
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()

        # Load test dataset
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        test_dir = 'D:/Research project/Datasets/Banana Dataset/Test'
        test_set = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        # Evaluate model
        print("\nEvaluating model on test set...")
        test_loss, test_accuracy = model.evaluate(test_set)
        
        print(f"\nTest Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")

    except Exception as e:
        print(f"\nError loading or evaluating model: {str(e)}")

if __name__ == "__main__":
    evaluate_model_v2()