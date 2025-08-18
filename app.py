import os
import csv
import datetime
from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
from banana_detector import BananaDetector
from tensorflow.keras.models import load_model

app = Flask(__name__)
video_capture = None

# Initialize models
detector   = BananaDetector(conf_threshold=0.4)
classifier = load_model('banana_quality_model.h5')
CLASS_NAMES = ['fresh','rotten','unripe']

# Session‐level counters & timestamps
session_counts = {'fresh':0,'rotten':0,'unripe':0,'unknown':0}
session_start  = None

def preprocess_roi(roi):
    roi = cv2.resize(roi,(224,224))
    return np.expand_dims(roi.astype('float32')/255.0,0)

def gen_frames():
    global video_capture, session_counts, session_start

    # On first frame, mark start time and zero out session_counts
    if session_start is None:
        session_start = datetime.datetime.now()
        session_counts = dict.fromkeys(session_counts, 0)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # detect & classify
        boxes = detector.detect(frame)
        for (x,y,w,h) in boxes:
            roi  = frame[y:y+h, x:x+w]
            pred = classifier.predict(preprocess_roi(roi), verbose=0)[0]
            idx  = int(np.argmax(pred))
            conf = float(pred[idx])
            label = CLASS_NAMES[idx] if conf>=0.6 else 'unknown'

            # draw
            color = (0,255,0) if label=='fresh' else (0,0,255) if label=='rotten' else (255,255,0)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,f"{label}:{conf:.2f}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            # **Accumulate session total**
            session_counts[label] += 1

        # stream out
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_counts')
def detection_counts():
    # returns last‐frame counts if you still track those,
    # or you could omit this now that you have session_counts
    return jsonify(session_counts)

@app.route('/stop_video')
def stop_video():
    global video_capture, session_start, session_counts
    try:
        if video_capture is not None:
            video_capture.release()
            video_capture = None

        session_end = datetime.datetime.now()

        if session_start:  # Only save if session was actually started
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
            
            session_start = None
            session_counts = dict.fromkeys(session_counts, 0)
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

# Add this line after Flask app initialization
ensure_csv_exists()

# Add this new route after your other routes
@app.route('/shutdown', methods=['POST'])
def shutdown():
    global video_capture
    try:
        # Release camera
        if video_capture is not None:
            video_capture.release()
            video_capture = None
            
        # Save final session if active
        if session_start:
            stop_video()
            
        # Shutdown the server
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return jsonify({"status": "Server shutting down..."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
