# banana_detector.py

import os
import cv2
import numpy as np
import torch
from typing import List, Tuple

class BananaDetector:
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        conf_threshold: float = 0.3
    ):
        """
        Initialize the YOLO banana detector.
        model_path: path to a YOLOv8 weights file (e.g. 'yolov8n.pt').
        conf_threshold: minimum confidence for detections.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: could not import ultralytics.YOLO.")
            print("Install with: pip install ultralytics torch torchvision opencv-python")
            raise

        self.conf_threshold = conf_threshold
        print(f"Loading YOLO model from {model_path} …")
        self.model = YOLO(model_path)        # load weights
        self.model.to('cpu')                 # force CPU
        print("YOLO loaded and moved to CPU!")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Run banana detection on a single BGR image.
        Returns a list of bounding boxes (x, y, w, h).
        """
        try:
            results = self.model(frame, verbose=False)[0]
            boxes: List[Tuple[int,int,int,int]] = []
            for *box, score, class_id in results.boxes.data.tolist():
                if score < self.conf_threshold:
                    continue
                if int(class_id) != 46:  # COCO class 46 = banana
                    continue
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                boxes.append((int(x1), int(y1), int(w), int(h)))
            return boxes

        except Exception as e:
            print("Detection error:", e)
            return []

if __name__ == "__main__":
    detector = BananaDetector(conf_threshold=0.4)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit(1)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for (x, y, w, h) in detector.detect(frame):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Banana", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Banana Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
