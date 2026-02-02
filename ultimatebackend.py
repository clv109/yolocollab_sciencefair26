import os
import sys
import argparse
import time
import threading
import sqlite3
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, Response, request

# --- CONFIGURATION ---
DB_NAME = "parking_data.db"
STAT_LOG_INTERVAL = 5  # Log statistics every 5 seconds

# --- HARDWARE SETUP (I2C LCD) ---
lcd = None
try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD('PCF8574', 0x27)
    lcd.clear()
    lcd.write_string("AI Parking DB")
    print("✅ LCD Display Connected!")
except Exception:
    print(f"⚠️ LCD Display not found.")
    lcd = None

# --- GLOBAL VARIABLES ---
TOTAL_CAR_COUNT = 0
TOTAL_CAPACITY = 10
ACTIVE_IDS = []
outputFrame = None 
lock = threading.Lock() 

app = Flask(__name__)

# --- DATABASE FUNCTIONS ---
def init_db():
    """Creates the database and 3 tables if they don't exist"""
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        # Table 1: Periodic Statistics
        c.execute('''CREATE TABLE IF NOT EXISTS detection_stats (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     occupied_count INTEGER,
                     total_capacity INTEGER
                     )''')
        # Table 2: Capacity Changes
        c.execute('''CREATE TABLE IF NOT EXISTS capacity_logs (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     old_value INTEGER,
                     new_value INTEGER
                     )''')
        # Table 3: New Vehicle Entries
        c.execute('''CREATE TABLE IF NOT EXISTS vehicle_logs (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     vehicle_id INTEGER
                     )''')
        conn.commit()
    print(f"✅ Database initialized: {DB_NAME}")

def log_statistic(occupied, capacity):
    """Logs the current count/capacity snapshot"""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO detection_stats (timestamp, occupied_count, total_capacity) VALUES (?, ?, ?)",
                      (timestamp, occupied, capacity))
            conn.commit()
    except Exception as e:
        print(f"DB Error (Stats): {e}")

def log_capacity_change(old_val, new_val):
    """Logs when user changes capacity setting"""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO capacity_logs (timestamp, old_value, new_value) VALUES (?, ?, ?)",
                      (timestamp, old_val, new_val))
            conn.commit()
        print(f"💾 Logged Capacity Change: {old_val} -> {new_val}")
    except Exception as e:
        print(f"DB Error (Capacity): {e}")

def log_new_vehicle(veh_id):
    """Logs a new unique vehicle ID"""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO vehicle_logs (timestamp, vehicle_id) VALUES (?, ?)",
                      (timestamp, veh_id))
            conn.commit()
        print(f"💾 Logged New Vehicle: ID {veh_id}")
    except Exception as e:
        print(f"DB Error (Vehicle): {e}")

# --- HELPER: Colors ---
def get_id_color(id_num):
    np.random.seed(int(id_num) * 50)
    return np.random.randint(0, 255, size=3).tolist()

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('mergedashboard.html')

@app.route('/data')
def get_data():
    return jsonify({
        'count': TOTAL_CAR_COUNT,
        'capacity': TOTAL_CAPACITY,
        'ids': ACTIVE_IDS,
        'status': 'Active'
    })

@app.route('/set_capacity', methods=['POST'])
def set_capacity():
    global TOTAL_CAPACITY
    data = request.get_json()
    if data and 'capacity' in data:
        try:
            new_cap = int(data['capacity'])
            old_cap = TOTAL_CAPACITY
            
            # Update Global
            TOTAL_CAPACITY = new_cap
            print(f"🔄 Capacity updated: {old_cap} -> {new_cap}")
            
            # --- DB LOGGING (Table 2) ---
            if old_cap != new_cap:
                log_capacity_change(old_cap, new_cap)
            
            return jsonify({'status': 'success', 'new_capacity': TOTAL_CAPACITY})
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid number'}), 400
    return jsonify({'status': 'error', 'message': 'No capacity provided'}), 400

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                time.sleep(0.01)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- AI LOGIC LOOP ---
def run_ai_logic(args):
    global TOTAL_CAR_COUNT, ACTIVE_IDS, outputFrame, lock, lcd
    
    # Init Database
    init_db()

    # Load Model
    if not os.path.exists(args.model):
        print(f"ERROR: Model {args.model} not found.")
        return
    print(f"Loading Model: {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading YOLO: {e}")
        return

    # Source Handling
    img_source = args.source
    source_type = 'unknown'
    if 'usb' in img_source:
        source_type = 'usb'
        usb_idx = int(img_source.replace('usb', ''))
    elif img_source.isdigit(): 
        source_type = 'usb'
        usb_idx = int(img_source)
    elif os.path.isfile(img_source):
        source_type = 'video'
    
    cap = None
    if source_type in ['video', 'usb']:
        cap = cv2.VideoCapture(usb_idx if source_type == 'usb' else img_source)
        if args.resolution:
            w, h = map(int, args.resolution.split('x'))
            cap.set(3, w)
            cap.set(4, h)
            
    count_buffer = deque(maxlen=10)
    
    # Tracking State
    logged_session_ids = set() # To ensure we only log a specific ID once per session
    last_stat_log_time = time.time()

    print("--- AI Tracking Loop Started ---")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            if source_type == 'video': cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            else: break

        # TRACKING
        # Note: ensuring we use the custom config we created earlier
        results = model.track(
            frame, 
            persist=True, 
            tracker="custom_tracker.yaml", 
            verbose=False, 
            conf=float(args.thresh), 
            iou=float(args.iou)
        )
        
        current_ids = []
        current_count = 0

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                
                # Filter by Area
                area = (x2 - x1) * (y2 - y1)
                if int(args.min_area) > 0 and area < int(args.min_area): continue

                current_ids.append(track_id)
                current_count += 1
                
                # --- DB LOGGING (Table 3) ---
                if track_id not in logged_session_ids:
                    log_new_vehicle(track_id)
                    logged_session_ids.add(track_id)

                # Draw Visuals
                color = get_id_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Update Globals
        count_buffer.append(current_count)
        if count_buffer:
            TOTAL_CAR_COUNT = int(round(sum(count_buffer) / len(count_buffer)))
        else:
            TOTAL_CAR_COUNT = current_count
        
        ACTIVE_IDS = current_ids

        # --- DB LOGGING (Table 1: Periodic Stats) ---
        if time.time() - last_stat_log_time > STAT_LOG_INTERVAL:
            log_statistic(TOTAL_CAR_COUNT, TOTAL_CAPACITY)
            last_stat_log_time = time.time()

        # Update LCD
        if lcd:
            try:
                lcd.cursor_pos = (0, 0)
                lcd.write_string(f"Occ: {TOTAL_CAR_COUNT}/{TOTAL_CAPACITY}    ")
                lcd.cursor_pos = (1, 0)
                lcd.write_string(f"IDs: {len(ACTIVE_IDS)}     ")
            except: pass

        # Draw Overlay
        cv2.putText(frame, f'COUNT: {TOTAL_CAR_COUNT} / {TOTAL_CAPACITY}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        with lock:
            outputFrame = frame.copy()
        
        if not args.headless:
            cv2.imshow('AI Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    if cap: cap.release()
    cv2.destroyAllWindows()
    os._exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--capacity', type=int, default=10)
    parser.add_argument('--thresh', default=0.5)
    parser.add_argument('--iou', default=0.5)
    parser.add_argument('--min_area', default=0)
    parser.add_argument('--resolution', default=None)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record', action='store_true')
    args, _ = parser.parse_known_args()

    TOTAL_CAPACITY = args.capacity

    t = threading.Thread(target=run_ai_logic, args=(args,))
    t.daemon = True
    t.start()

    print("Starting Web Server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
