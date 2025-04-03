import tensorflow as tf
import os
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import cv2

def load_model_and_classes():
    # Load the trained model
    model = load_model('D:/Quality Control System/banana_quality_model.h5')
    print("Model loaded successfully.")
    
    # Load class categories
    source_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
    categories = sorted(os.listdir(source_dir))
    print("Classes:", categories)
    return model, categories

def prepare_image(image_path, target_size=(224, 224)):
    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize only once
    img_array = img_array / 255.0
    return img_array

def predict_banana_quality(model, image_path, categories):
    # Prepare image
    img_array = prepare_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get class index and confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    # Get class name
    predicted_class = categories[class_idx]
    
    # Print detailed results
    print("\nPrediction Results:")
    print("-" * 50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Print all class probabilities
    print("\nClass Probabilities:")
    for i, (category, prob) in enumerate(zip(categories, predictions[0])):
        print(f"{category}: {prob*100:.2f}%")
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Load model and classes
    model, categories = load_model_and_classes()
    
    # Test multiple images
    test_images = [
        'D:/Quality Control System/rotten2.jpg',
        # Add more test images here
    ]
    
    for image_path in test_images:
        predicted_class, confidence = predict_banana_quality(model, image_path, categories)
    
#show_the_image_with_text

img = cv2.imread('D:/Quality Control System/rotten2.jpg')
font=cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, predicted_class, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def show_prediction_on_image(image_path, predicted_class, confidence):
#     # Read image
#     image = cv2.imread(image_path)
#     # Resize if image is too large
#     scale_percent = 60
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     image = cv2.resize(image, (width, height))
    
#     # Create text to display
#     text = f"{predicted_class}: {confidence:.2f}%"
    
#     # Set text properties
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.8
#     font_thickness = 2
#     font_color = (0, 255, 0)  # Green color
    
#     # Get text size
#     (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
#     # Calculate text position
#     padding = 10
#     rect_height = text_height + 2 * padding
#     rect_position = (0, 0, text_width + 2 * padding, rect_height)
    
#     # Draw black rectangle behind text
#     cv2.rectangle(image, 
#                  (rect_position[0], rect_position[1]), 
#                  (rect_position[2], rect_position[3]), 
#                  (0, 0, 0), 
#                  -1)
    
#     # Add text
#     cv2.putText(image, 
#                 text, 
#                 (padding, text_height + padding//2), 
#                 font, 
#                 font_scale, 
#                 font_color, 
#                 font_thickness)
    
#     # Show image
#     cv2.imshow('Prediction Result', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Update the main block
# if __name__ == "__main__":
#     # Load model and classes
#     model, categories = load_model_and_classes()
    
#     # Test multiple images
#     test_images = [
#         'D:/Quality Control System/rotten2.jpg',
#         # Add more test images here
#     ]
    
#     for image_path in test_images:
#         predicted_class, confidence = predict_banana_quality(model, image_path, categories)
#         show_prediction_on_image(image_path, predicted_class, confidence)