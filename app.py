# app.py
import os
import csv
import time
import datetime
from collections import deque

from dotenv import load_dotenv
from flask import Flask, Response, render_template, jsonify, request, send_file
from pymongo import MongoClient
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from banana_detector import BananaDetector  # your YOLO wrapper

# -----------------------
# Load env + Mongo setup
# -----------------------
load_dotenv()  # loads .env in the same folder as app.py

MONGO_URI = (os.getenv("MONGO_URI") or "").strip()
MONGO_DB  = (os.getenv("MONGO_DB")  or "quality_control").strip()

mongo_client = None
db = None
sessions_col = None

if not MONGO_URI:
    print("[WARN] MONGO_URI not set; history will use CSV fallback only.")
else:
    try:
        # Verify the connection early so failures are obvious
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")
        db = mongo_client[MONGO_DB]
        sessions_col = db["sessions"]
        sessions_col.create_index([("session_start", 1)])
        print(f"[Mongo] Connected to DB '{MONGO_DB}', collection 'sessions'.")
    except Exception as e:
        print(f"[Mongo] Connection failed: {e}")
        mongo_client = db = sessions_col = None

CSV_PATH = "banana_sessions.csv"  # optional fallback CSV

# -----------------------
# App / Globals
# -----------------------
CLASS_NAMES = ["fresh", "rotten", "unripe"]

app = Flask(__name__)
video_capture = None

# Load models once (avoid Flask reloader duplicate by running with use_reloader=False)
detector = BananaDetector(conf_threshold=0.3)
classifier = load_model("banana_quality_model.h5")

session_counts = {"fresh": 0, "rotten": 0, "unripe": 0, "unknown": 0}
session_start = None
next_banana_id = 0
tracked_bananas = {}  # id -> (centroid, label)

# -----------------------
# Performance monitor
# -----------------------
class PerformanceMonitor:
    def __init__(self, buffer_size=100):
        self.frame_times = deque(maxlen=buffer_size)       # ms per frame (overall)
        self.pipeline_times = deque(maxlen=buffer_size)    # ms detect+classify
        self.true_labels = []   # demo only; same as predicted
        self.pred_labels = []

    def add_frame_time(self, ms): 
        self.frame_times.append(ms)

    def add_pipeline_time(self, ms): 
        self.pipeline_times.append(ms)

    def add_prediction(self, true_label, pred_label):
        self.true_labels.append(true_label)
        self.pred_labels.append(pred_label)

    def get_fps(self):
        if not self.frame_times: 
            return 0.0
        avg_ms = sum(self.frame_times) / len(self.frame_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def get_latency(self):
        if not self.pipeline_times: 
            return 0.0
        return sum(self.pipeline_times) / len(self.pipeline_times)

performance_monitor = PerformanceMonitor(buffer_size=100)

# -----------------------
# Utils
# -----------------------
def ensure_csv_exists():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "session_start", "session_end",
                "fresh_count", "rotten_count", "unripe_count", "unknown_count"
            ])

def preprocess_roi(roi):
    roi = cv2.resize(roi, (224, 224))
    roi = roi.astype("float32") / 255.0
    return np.expand_dims(roi, axis=0)

def get_centroid(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def parse_date_param(value, end_of_day=False):
   
    # Accepts 'YYYY-MM-DD' or full ISO. Returns naive datetime.
    # If end_of_day=True and only date given, returns date + 23:59:59.999999
    
    if not value:
        return None
    try:
        # Try full ISO datetime first
        return datetime.datetime.fromisoformat(value)
    except Exception:
        pass
    try:
        d = datetime.datetime.strptime(value, "%Y-%m-%d")
        if end_of_day:
            return d + datetime.timedelta(days=1) - datetime.timedelta(microseconds=1)
        return d
    except Exception:
        return None

def save_session_to_mongo(start_dt, end_dt, counts):
    # Explicit None check to avoid PyMongo truthiness error
    if sessions_col is not None:
        doc = {
            "session_start": start_dt,
            "session_end": end_dt,
            "duration_sec": round((end_dt - start_dt).total_seconds(), 2),
            "fresh_count": int(counts.get("fresh", 0)),
            "rotten_count": int(counts.get("rotten", 0)),
            "unripe_count": int(counts.get("unripe", 0)),
            "unknown_count": int(counts.get("unknown", 0)),
            "total_count": int(sum(counts.values())),
            "created_at": datetime.datetime.now(),
        }
        try:
            sessions_col.insert_one(doc)
            return True
        except Exception as e:
            print(f"[MongoDB] Save error: {e}")
    return False

def fetch_history_from_mongo(start_dt=None, end_dt=None):
    # Explicit None check — collections don't implement truthiness
    if sessions_col is None:
        return None
    q = {}
    if start_dt or end_dt:
        rng = {}
        if start_dt: rng["$gte"] = start_dt
        if end_dt:   rng["$lte"] = end_dt
        q["session_start"] = rng
    cursor = sessions_col.find(q).sort("session_start", -1)
    rows = []
    for d in cursor:
        rows.append({
            "session_start": d.get("session_start").strftime("%Y-%m-%d %H:%M:%S") if d.get("session_start") else "",
            "session_end": d.get("session_end").strftime("%Y-%m-%d %H:%M:%S") if d.get("session_end") else "",
            "duration_sec": d.get("duration_sec", 0),
            "fresh_count": d.get("fresh_count", 0),
            "rotten_count": d.get("rotten_count", 0),
            "unripe_count": d.get("unripe_count", 0),
            "unknown_count": d.get("unknown_count", 0),
        })
    return rows

# -----------------------
# Streaming generator
# -----------------------
def gen_frames():
    global video_capture, session_counts, session_start, next_banana_id, tracked_bananas, performance_monitor

    if session_start is None:
        session_start = datetime.datetime.now()
        session_counts = {k: 0 for k in session_counts}
        tracked_bananas = {}

    video_capture = cv2.VideoCapture(1)
    if not video_capture.isOpened():
        raise RuntimeError("Cannot open camera")

    performance_monitor = PerformanceMonitor(buffer_size=100)

    while True:
        frame_t0 = time.time()
        ret, frame = video_capture.read()
        if not ret:
            break

        pipe_t0 = time.time()
        boxes = detector.detect(frame)

        for (x, y, w, h) in boxes:
            try:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                preds = classifier.predict(preprocess_roi(roi), verbose=0)[0]
                idx = int(np.argmax(preds))
                conf = float(preds[idx])
                label = CLASS_NAMES[idx] if conf >= 0.6 else "unknown"

                # demo: predicted == truth
                if label in CLASS_NAMES:
                    performance_monitor.add_prediction(label, label)

                # centroid tracker to avoid double count
                centroid = get_centroid(x, y, w, h)
                already_tracked = False
                for b_id, (c_prev, _) in list(tracked_bananas.items()):
                    if abs(c_prev[0] - centroid[0]) < 50 and abs(c_prev[1] - centroid[1]) < 50:
                        tracked_bananas[b_id] = (centroid, label)
                        already_tracked = True
                        break
                if not already_tracked:
                    tracked_bananas[next_banana_id] = (centroid, label)
                    session_counts[label] += 1
                    next_banana_id += 1

                # draw
                color = (0, 255, 0) if label == "fresh" else (0, 0, 255) if label == "rotten" else (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label}:{conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                print(f"[ROI] {e}")

        performance_monitor.add_pipeline_time((time.time() - pipe_t0) * 1000.0)
        performance_monitor.add_frame_time((time.time() - frame_t0) * 1000.0)

        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        except Exception as e:
            print(f"[Encode] {e}")

# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/detection_counts")
def detection_counts():
    return jsonify(session_counts)

@app.get("/session_info")
def session_info():
    if session_start:
        duration = (datetime.datetime.now() - session_start).total_seconds()
        return jsonify({
            "start_time": session_start.isoformat(),
            "duration_seconds": duration,
            "counts": session_counts
        })
    return jsonify({"status": "no active session"})

@app.get("/api/history")
def api_history():
    try:
        start_date = parse_date_param(request.args.get("start_date"))
        end_date = parse_date_param(request.args.get("end_date"), end_of_day=True)

        # Use Mongo if available
        if sessions_col is not None:
            rows = fetch_history_from_mongo(start_date, end_date)
            return jsonify(rows)

        # Fallback to CSV
        ensure_csv_exists()
        rows = []
        with open(CSV_PATH, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    t0 = datetime.datetime.fromisoformat(row["session_start"])
                    t1 = datetime.datetime.fromisoformat(row["session_end"])
                except Exception:
                    continue

                if start_date and t0 < start_date:  # outside lower bound
                    continue
                if end_date and t1 > end_date:      # outside upper bound
                    continue

                duration = (t1 - t0).total_seconds()
                rows.append({
                    "session_start": t0.strftime("%Y-%m-%d %H:%M:%S"),
                    "session_end":   t1.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_sec":  round(duration),
                    "fresh_count":   int(row.get("fresh_count", 0)),
                    "rotten_count":  int(row.get("rotten_count", 0)),
                    "unripe_count":  int(row.get("unripe_count", 0)),
                    "unknown_count": int(row.get("unknown_count", 0)),
                })
        # newest first (ISO strings sort correctly; explicit is fine too)
        rows.sort(key=lambda x: x["session_start"], reverse=True)
        return jsonify(rows)

    except Exception as e:
        print(f"[History] {e}")
        return jsonify([])

@app.get("/history.csv")  # optional direct CSV download
def history_csv():
    ensure_csv_exists()
    return send_file(CSV_PATH, as_attachment=False, download_name="banana_sessions.csv")

@app.get("/generate_performance_plot")
def generate_performance_plot():
    if not performance_monitor.frame_times:
        return jsonify({"error": "No performance data"}), 404

    plt.figure(figsize=(12, 8))

    fps_values = [1000.0 / t for t in performance_monitor.frame_times if t > 0]
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0

    plt.subplot(2, 1, 1)
    plt.plot(fps_values, "b-", alpha=0.6, label="Instantaneous FPS")
    plt.axhline(y=avg_fps, color="r", linestyle="--", label=f"Average FPS: {avg_fps:.1f}")
    plt.title("Frame Rate Performance")
    plt.ylabel("FPS")
    plt.grid(True, alpha=0.3)
    plt.legend()

    avg_lat = performance_monitor.get_latency()
    plt.subplot(2, 1, 2)
    plt.plot(performance_monitor.pipeline_times, "g-", alpha=0.6, label="Per-frame Pipeline Time (ms)")
    plt.axhline(y=avg_lat, color="r", linestyle="--", label=f"Average Latency: {avg_lat:.1f} ms")
    plt.title("Processing Latency")
    plt.xlabel("Frame #")
    plt.ylabel("ms")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    out_path = "static/performance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return send_file(out_path, mimetype="image/png")

@app.get("/generate_confusion_matrix")
def generate_confusion_matrix():
    if not performance_monitor.true_labels:
        return jsonify({"error": "No classification data"}), 404

    cm = confusion_matrix(
        performance_monitor.true_labels,
        performance_monitor.pred_labels,
        labels=CLASS_NAMES
    )
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Banana Quality Classification Results")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    out_path = "static/confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return send_file(out_path, mimetype="image/png")

@app.get("/stop_video")
def stop_video():
    global video_capture, session_start, session_counts, tracked_bananas
    try:
        if video_capture is not None:
            video_capture.release()
            video_capture = None

        session_end = datetime.datetime.now()
        if session_start:
            # Save to Mongo if connected
            saved_to_mongo = False
            if sessions_col is not None:
                saved_to_mongo = save_session_to_mongo(session_start, session_end, session_counts)

            # Always append to CSV as fallback/audit
            ensure_csv_exists()
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    session_start.isoformat(),
                    session_end.isoformat(),
                    session_counts["fresh"],
                    session_counts["rotten"],
                    session_counts["unripe"],
                    session_counts["unknown"]
                ])

            # reset for next session
            session_start = None
            session_counts = {k: 0 for k in session_counts}
            tracked_bananas = {}

            return jsonify({"status": "stopped", "saved": True, "mongo": saved_to_mongo})

        return jsonify({"status": "stopped", "saved": False})
    except Exception as e:
        print(f"[Stop] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    ensure_csv_exists()
    # Disable reloader so models load only once
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
