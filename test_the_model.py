import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.models import load_model
import cv2
import json
import matplotlib.pyplot as plt

def load_model_and_classes():
    # Load the trained model
    model = load_model('D:/Quality Control System/banana_quality_model.h5')
    print("Model loaded successfully.")
    
    # Load class categories from the training directory
    source_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
    categories = sorted(os.listdir(source_dir))
    print("Classes:", categories)
    return model, categories

def prepare_image(image_path, target_size=(224, 224)):
    # Load and preprocess the image for prediction
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    return img_array

def predict_banana_quality(model, image_path, categories):
    # Prepare image for prediction
    img_array = prepare_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get the predicted class index and confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    # Map the index to the class name
    predicted_class = categories[class_idx]
    
    # Print detailed results
    print("\nPrediction Results:")
    print("-" * 50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Print all class probabilities
    print("\nClass Probabilities:")
    for category, prob in zip(categories, predictions[0]):
        print(f"{category}: {prob * 100:.2f}%")
    
    return predicted_class, confidence

def show_prediction_on_image(image_path, predicted_class, confidence, output_size=(600, 600)):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image:", image_path)
        return
    # Resize image to the fixed output size
    image = cv2.resize(image, output_size)
    
    # Create the text to overlay
    text = f"{predicted_class}: {confidence:.2f}%"
    
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    color = (0, 255, 0)
    
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 10
    # Draw a rectangle for text background
    cv2.rectangle(image, (0, 0), (text_width + 2 * padding, text_height + 2 * padding), (0, 0, 0), -1)
    
    # Put the prediction text on the image
    cv2.putText(image, text, (padding, text_height + padding), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Show the image with overlay
    cv2.imshow('Prediction Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the model and class categories
    model, categories = load_model_and_classes()
    
    test_images = [
        'D:/Quality Control System/rottenes.jpg',
        'D:/Quality Control System/mult.jpg',
        'D:/Quality Control System/mulrot.jpg',
        'D:/Quality Control System/UnripeBananas.jpg',
    ]
    
    for image_path in test_images:
        predicted_class, confidence = predict_banana_quality(model, image_path, categories)
        show_prediction_on_image(image_path, predicted_class, confidence, output_size=(600, 600))
