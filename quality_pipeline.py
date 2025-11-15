import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

def check_versions():
    import sys
    print("Python version:", sys.version)
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"Error importing NumPy: {e}")
        sys.exit(1)
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"Error importing TensorFlow: {e}")
        sys.exit(1)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch OpenMP available: {torch.backends.openmp.is_available()}")
    except ImportError as e:
        print(f"Error importing PyTorch: {e}")
        sys.exit(1)

    # Compatibility checks
    if np.__version__.startswith("2."):
        raise RuntimeError("NumPy 2.x detected. Please use NumPy 1.x")
    if not tf.__version__.startswith("2.10"):
        print("Warning: TensorFlow version may not be optimal. Recommended: 2.10.0")
    return True

# Version check at startup
check_versions()

import cv2
import numpy as np
import time
import queue
import threading
from tensorflow.keras.models import load_model
from banana_detector import BananaDetector

class QualityPipeline:
    def __init__(self):
        print("Initializing QualityPipeline...")
        # load our YOLO-based Banana detector
        self.detector = BananaDetector(conf_threshold=0.3)
        print("Detector initialized successfully")
        # load the quality-classifier
        model_path = 'banana_quality_model.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.classifier = load_model(model_path)
        print("Classifier model loaded successfully")
        # queues for producer/consumer threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        print("Pipeline initialized successfully")

    def preprocess_roi(self, roi):
        """Resize + normalize ROI for classification."""
        roi = cv2.resize(roi, (224, 224))
        roi = roi.astype(np.float32) / 255.0
        return np.expand_dims(roi, axis=0)

    def process_frame(self):
        """Worker thread: detect bananas, classify quality, push results."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            boxes = self.detector.detect(frame)
            results = []
            for (x, y, w, h) in boxes:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                inp = self.preprocess_roi(roi)
                preds = self.classifier.predict(inp, verbose=0)[0]
                quality = int(np.argmax(preds))
                confidence = float(preds[quality])
                results.append((x, y, w, h, quality, confidence))

            # push (frame, results) tuple
            try:
                self.result_queue.put((frame, results), timeout=0.1)
            except queue.Full:
                pass

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # start the classification thread
        process_thread = threading.Thread(target=self.process_frame, daemon=True)
        process_thread.start()

        fps_buffer = []
        quality_labels = ['Fresh', 'Rotten', 'Unripe']
        quality_colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]
        last_results = []  # cache of last non-empty detection+classification

        try:
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                # hand frame off to worker
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

                # try to get new results; if none arrive, reuse last_results
                try:
                    _, new_results = self.result_queue.get(timeout=0.02)
                    if new_results:
                        last_results = new_results
                except queue.Empty:
                    pass

                # draw whatever is in last_results
                for (x, y, w, h, quality, conf) in last_results:
                    color = quality_colors[quality]
                    label = f"{quality_labels[quality]}: {conf:.2f}"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # compute & display FPS
                fps_buffer.append(1.0 / (time.time() - t0))
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer)
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow("Banana Quality Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop_event.set()
            process_thread.join(timeout=1.0)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = QualityPipeline()
    pipeline.run()
