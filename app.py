import os
import csv
import datetime
from flask import Flask, Response, render_template, jsonify, request, send_file
import pathlib
import cv2
import numpy as np
from banana_detector import BananaDetector
from tensorflow.keras.models import load_model
import time
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define PerformanceMonitor class first
class PerformanceMonitor:
    def __init__(self, buffer_size=100):
        self.frame_times = deque(maxlen=buffer_size)
        self.inference_times = deque(maxlen=buffer_size)
        self.true_labels = []
        self.pred_labels = []
        
    def add_frame_time(self, time_ms):
        self.frame_times.append(time_ms)
        
    def add_inference_time(self, time_ms):
        self.inference_times.append(time_ms)
        
    def add_prediction(self, true_label, pred_label):
        # Simulate ground truth for demo (replace with actual ground truth in production)
        self.true_labels.append(true_label)
        self.pred_labels.append(pred_label)
        
    def get_fps(self):
        return 1000 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        
    def get_latency(self):
        return sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0

# Initialize Flask and other components
app = Flask(__name__)
video_capture = None

# Initialize models
detector = BananaDetector(conf_threshold=0.4)
classifier = load_model('banana_quality_model.h5')
CLASS_NAMES = ['fresh','rotten','unripe']

# Session tracking
session_counts = {'fresh':0,'rotten':0,'unripe':0,'unknown':0}
session_start = None
next_banana_id = 0
tracked_bananas = {}   # id -> centroid

# Initialize performance monitor
performance_monitor = PerformanceMonitor(buffer_size=100)

def preprocess_roi(roi):
    roi = cv2.resize(roi,(224,224))
    return np.expand_dims(roi.astype('float32')/255.0,0)

def get_centroid(x,y,w,h):
    return (int(x+w/2), int(y+h/2))

def gen_frames():
    global video_capture, session_counts, session_start, next_banana_id, tracked_bananas, performance_monitor

    if session_start is None:
        session_start = datetime.datetime.now()
        session_counts = dict.fromkeys(session_counts, 0)
        tracked_bananas = {}

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Cannot open camera")

    performance_monitor = PerformanceMonitor(buffer_size=100)

    while True:
        frame_start = time.time()
        ret, frame = video_capture.read()
        if not ret:
            break

        # Initialize current_centroids for this frame
        current_centroids = []  # Add this line

        # detect bananas
        inference_start = time.time()
        boxes = detector.detect(frame)
        
        for (x,y,w,h) in boxes:
            try:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:  # Check if ROI is valid
                    continue
                    
                pred = classifier.predict(preprocess_roi(roi), verbose=0)[0]
                idx = int(np.argmax(pred))
                conf = float(pred[idx])
                label = CLASS_NAMES[idx] if conf>=0.6 else 'unknown'
                
                # Make sure we only collect valid predictions
                if label in CLASS_NAMES:  # Only collect when we have a valid class
                    # For demo, use the predicted label as ground truth
                    # In production, replace with actual ground truth
                    performance_monitor.add_prediction(label, label)
                    print(f"Added prediction: {label}")  # Debug print
                
                # Track inference time
                inference_time = (time.time() - inference_start) * 1000
                performance_monitor.add_inference_time(inference_time)
                
                # Add prediction to performance monitor
                performance_monitor.add_prediction(label, label)  # Using predicted as truth for demo
                
                # track centroids to avoid double-count
                centroid = get_centroid(x,y,w,h)
                current_centroids.append((centroid,label))

                # check if this banana is new
                already_tracked = False
                for b_id, (c_prev, _) in tracked_bananas.items():
                    if abs(c_prev[0]-centroid[0]) < 50 and abs(c_prev[1]-centroid[1]) < 50:
                        tracked_bananas[b_id] = (centroid,label)
                        already_tracked = True
                        break

                if not already_tracked:
                    tracked_bananas[next_banana_id] = (centroid,label)
                    session_counts[label] += 1
                    next_banana_id += 1

                # draw
                color = (0,255,0) if label=='fresh' else (0,0,255) if label=='rotten' else (255,255,0)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,f"{label}:{conf:.2f}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                
            except Exception as e:
                print(f"Error processing ROI: {e}")
                continue

        # Track total frame time
        frame_time = (time.time() - frame_start) * 1000
        performance_monitor.add_frame_time(frame_time)
        
        # stream out
        try:
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret: 
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Error encoding frame: {e}")
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_counts')
def detection_counts():
    return jsonify(session_counts)

@app.route('/stop_video')
def stop_video():
    global video_capture, session_start, session_counts, tracked_bananas
    try:
        if video_capture is not None:
            video_capture.release()
            video_capture = None

        session_end = datetime.datetime.now()

        if session_start:
            # Save session data
            with open('banana_sessions.csv','a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    session_start.isoformat(),
                    session_end.isoformat(),
                    session_counts['fresh'],
                    session_counts['rotten'],
                    session_counts['unripe'],
                    session_counts['unknown']
                ])
            
            # Generate and save figures using the correct function name
            save_performance_figures()
            
            session_start = None
            session_counts = dict.fromkeys(session_counts, 0)
            tracked_bananas = {}
            return jsonify({"status":"stopped", "saved":True})
        
        return jsonify({"status":"stopped", "saved":False})
        
    except Exception as e:
        print(f"Error stopping video: {e}")
        return jsonify({"status":"error", "message":str(e)}), 500

@app.route('/session_info')
def session_info():
    if session_start:
        duration = (datetime.datetime.now() - session_start).total_seconds()
        return jsonify({
            "start_time": session_start.isoformat(),
            "duration_seconds": duration,
            "counts": session_counts
        })
    return jsonify({"status": "no active session"})

def ensure_csv_exists():
    csv_file = 'banana_sessions.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'session_start',
                'session_end',
                'fresh_count',
                'rotten_count',
                'unripe_count',
                'unknown_count'
            ])

ensure_csv_exists()

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global video_capture
    try:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
            
        if session_start:
            stop_video()
            
        if 'werkzeug.server.shutdown' in request.environ:
            request.environ['werkzeug.server.shutdown']()
            return jsonify({"status": "Server shutting down..."})
        
        os._exit(0)
        
    except Exception as e:
        print(f"Shutdown error: {e}")
        os._exit(1)

@app.route('/generate_performance_plot')
def generate_performance_plot():
    if not performance_monitor or not performance_monitor.frame_times:
        return jsonify({"error": "No performance data available"}), 404

    plt.style.use('seaborn')
    plt.figure(figsize=(12, 8))
    
    # Plot FPS over time
    fps_values = [1000/t for t in performance_monitor.frame_times]
    avg_fps = sum(fps_values)/len(fps_values)
    
    plt.subplot(2, 1, 1)
    plt.plot(fps_values, 'b-', label='Instantaneous FPS', alpha=0.6)
    plt.axhline(y=avg_fps, color='r', linestyle='--', 
                label=f'Average FPS: {avg_fps:.1f}')
    plt.title('Frame Rate Performance', pad=15)
    plt.ylabel('Frames per Second')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot latency over time
    avg_latency = performance_monitor.get_latency()
    plt.subplot(2, 1, 2)
    plt.plot(performance_monitor.inference_times, 'g-', 
             label='Processing Time', alpha=0.6)
    plt.axhline(y=avg_latency, color='r', linestyle='--',
                label=f'Average Latency: {avg_latency:.1f}ms')
    plt.title('Processing Latency', pad=15)
    plt.xlabel('Frame Number')
    plt.ylabel('Milliseconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return send_file('static/performance.png', mimetype='image/png')

@app.route('/generate_confusion_matrix')
def generate_confusion_matrix():
    if not performance_monitor or not performance_monitor.true_labels:
        return jsonify({"error": "No classification data available"}), 404
        
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 8))
    
    cm = confusion_matrix(
        performance_monitor.true_labels, 
        performance_monitor.pred_labels,
        labels=CLASS_NAMES
    )
    
    accuracy = np.trace(cm) / np.sum(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized, 
        annot=cm,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        square=True,
        cbar_kws={'label': 'Normalized Predictions'}
    )
    
    plt.title(f'Banana Quality Classification Results\nAccuracy: {accuracy:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return send_file('static/confusion_matrix.png', mimetype='image/png')

def save_performance_figures():
    if not performance_monitor or not performance_monitor.frame_times:
        print("No performance data available")
        return False

    print(f"Number of predictions collected: {len(performance_monitor.true_labels)}")
    print(f"Labels collected: {set(performance_monitor.true_labels)}")
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Generate performance plot
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 8))
    
    # Plot FPS over time
    fps_values = [1000/t for t in performance_monitor.frame_times]
    avg_fps = sum(fps_values)/len(fps_values) if fps_values else 0
    
    plt.subplot(2, 1, 1)
    plt.plot(fps_values, 'b-', label='Instantaneous FPS', alpha=0.6)
    plt.axhline(y=avg_fps, color='r', linestyle='--', 
                label=f'Average FPS: {avg_fps:.1f}')
    plt.title('Frame Rate Performance', pad=15)
    plt.ylabel('Frames per Second')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot latency over time
    avg_latency = performance_monitor.get_latency()
    plt.subplot(2, 1, 2)
    plt.plot(performance_monitor.inference_times, 'g-', 
             label='Processing Time', alpha=0.6)
    plt.axhline(y=avg_latency, color='r', linestyle='--',
                label=f'Average Latency: {avg_latency:.1f}ms')
    plt.title('Processing Latency', pad=15)
    plt.xlabel('Frame Number')
    plt.ylabel('Milliseconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('static/performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate confusion matrix if we have predictions
    if performance_monitor.true_labels and performance_monitor.pred_labels:
        print("Generating confusion matrix...")
        try:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(
                performance_monitor.true_labels, 
                performance_monitor.pred_labels,
                labels=CLASS_NAMES
            )
            
            # Print confusion matrix for debugging
            print("\nConfusion Matrix:")
            print(cm)
            
            sns.heatmap(
                cm, 
                annot=True,
                fmt='d',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap='YlOrRd'
            )
            
            plt.title('Banana Quality Classification Results')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Confusion matrix saved successfully")
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
    else:
        print("No prediction data available for confusion matrix")
        print(f"True labels: {len(performance_monitor.true_labels)}")
        print(f"Predicted labels: {len(performance_monitor.pred_labels)}")
    
    return True

if __name__ == '__main__':
    try:
        # Ensure the static directory exists
        os.makedirs('static', exist_ok=True)
        
        # Initialize CSV file
        ensure_csv_exists()
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
