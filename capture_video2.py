import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import img_to_array

# Avoid OpenMP duplicate runtime issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Config
IMG_HEIGHT = 224
IMG_WIDTH = 224
CONFIDENCE_THRESHOLD = 0.6  # below this show "Unknown"

# Load model and class names
model_path = 'D:/Quality Control System/banana_quality_model.h5'
model = load_model(model_path)
print("Loaded model from", model_path)

source_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
categories = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
print("Classes:", categories)

def preprocess_frame(frame, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    # Resize, convert to RGB, normalize
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = img_to_array(rgb)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

def draw_text_box(img, text, origin=(10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    padding = 6
    cv2.rectangle(img, (x - padding, y - h - padding), (x + w + padding, y + padding//2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Real-time quality prediction. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prediction on full frame
        input_img = preprocess_frame(frame)
        preds = model.predict(input_img, verbose=0)[0]  # softmax
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "Unknown"
        else:
            predicted_class = categories[class_idx]
        disp_conf = confidence * 100

        # Yellow segmentation overlay (visual only)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Optional: show mask in corner
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        small_mask = cv2.resize(mask_bgr, (160, 120))
        frame[0:120, 0:160] = small_mask  # overlay mask

        # Display prediction
        if predicted_class == "Unknown":
            text = f"Prediction: {predicted_class}"
        else:
            text = f"{predicted_class} ({disp_conf:.1f}%)"
        draw_text_box(frame, text)

        cv2.imshow("Banana Quality Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("snapshot.png", frame)
            print("Saved snapshot.png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
