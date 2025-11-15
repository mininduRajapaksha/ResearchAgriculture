import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import img_to_array
import threading
import queue
import time
from banana_detector import BananaDetector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

MODEL_PATH = 'D:/Quality Control System/banana_quality_model.h5'
TRAIN_DIR = 'D:/Research project/Datasets/Banana Dataset/Train'
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.6

# Load model and prepare fast inference function
model = load_model(MODEL_PATH)
infer = tf.function(lambda x: model(x, training=False))  # compiled

categories = sorted(os.listdir(TRAIN_DIR))

# Thread-safe queues
frame_queue = queue.Queue(maxsize=4)
result_queue = queue.Queue(maxsize=4)
stop_event = threading.Event()

def preprocess_roi(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, IMG_SIZE)
    arr = img_to_array(roi)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr.astype(np.float32)

def get_banana_bboxes_small(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask_yellow, mask_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = frame.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:  # tune threshold
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)
        if 0.3 < aspect_ratio < 3.0:
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def capture_thread(cap, detector):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect bananas using YOLO
        boxes = detector.detect(frame)
        
        # Only process frames with detected bananas
        if boxes:
            try:
                frame_queue.put_nowait((frame, boxes))
            except queue.Full:
                pass  # drop if backlog

def inference_thread():
    while not stop_event.is_set():
        try:
            frame, boxes = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        detections = []
        for (x, y, w, h) in boxes:
            # Extract ROI for quality classification
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue
                
            inp = preprocess_roi(roi)
            preds = infer(inp)[0].numpy()
            idx = int(np.argmax(preds))
            confidence = preds[idx]
            label = categories[idx] if confidence >= CONF_THRESHOLD else "Unknown"
            detections.append(((x, y, w, h), label, float(confidence)))
            
        result_queue.put((frame, detections))

def main():
    # Initialize YOLO detector
    banana_detector = BananaDetector(conf_threshold=0.4)
    
    cap = cv2.VideoCapture(0)  # or path to video
    if not cap.isOpened():
        print("Cannot open video source")
        return

    # warm-up
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    _ = infer(dummy)

    # start threads
    t1 = threading.Thread(target=capture_thread, args=(cap, banana_detector), daemon=True)
    t2 = threading.Thread(target=inference_thread, daemon=True)
    t1.start()
    t2.start()

    class_counts = {'fresh': 0, 'rotten': 0, 'unripe': 0, 'Unknown': 0}
    last_update = time.time()

    while True:
        try:
            frame, detections = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # annotate
        for (x, y, w, h), label, conf in detections:
            # Draw YOLO box in blue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw quality classification in green
            quality_text = f"{label} {conf*100:.1f}%"
            cv2.putText(frame, quality_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # simple counting: you could improve with better tracking
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts["Unknown"] += 1

        # overlay counts (reset periodically to avoid runaway)
        if time.time() - last_update > 5:  # every 5 seconds reset
            class_counts = {k: 0 for k in class_counts}
            last_update = time.time()
        y0 = 30
        for i, (cls, cnt) in enumerate(class_counts.items()):
            cv2.putText(frame, f"{cls}: {cnt}", (10, y0 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Real-time Banana Quality", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    t1.join()
    t2.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
