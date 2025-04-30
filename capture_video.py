import os
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
from keras.models import load_model

# Avoid OpenMP duplicate runtime issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define image dimensions (adjust as needed)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load the trained model
model_path = 'D:/Quality Control System/banana_quality_model.h5'
model = load_model(model_path)
print("Model loaded successfully from", model_path)

# Load class categories from your training directory
source_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
categories = sorted(os.listdir(source_dir))
print("Classes:", categories)

def prepare_frame(frame, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Preprocess the frame:
    - Resize to target size,
    - Convert from BGR to RGB,
    - Convert to array, expand dimensions, and normalize.
    """
    resized_frame = cv2.resize(frame, target_size)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img_array = img_to_array(rgb_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Start video capture from the default webcam (device 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Starting real-time quality prediction. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame for prediction
    processed_frame = prepare_frame(frame)
    
    # Get predictions from the model
    predictions = model.predict(processed_frame, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    predicted_class = categories[class_idx]
    
    # Perform color segmentation to detect yellow regions (common for ripe bananas)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define HSV range for yellow color (tune these values as needed)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours in the mask to detect the banana region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filter out small contours; adjust threshold as needed
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw a bounding box around the detected banana region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Prepare text to display the prediction
    text = f"Class: {predicted_class}, Confidence: {confidence:.2f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Show the frame with the overlay and highlighted regions
    cv2.imshow("Banana Quality Control", frame)
    
    # Break loop on 'q' key press (ensure the video window is focused)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
