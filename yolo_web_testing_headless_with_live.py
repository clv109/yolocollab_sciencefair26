import os
import sys
import argparse
import glob
import time
import threading
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
# Added 'request' to allow Flask to receive the capacity update
from flask import Flask, render_template, jsonify, Response, request


lcd = None
try:
    from RPLCD.i2c import CharLCD
    
    lcd = CharLCD('PCF8574', 0x27)
    lcd.clear()
    
    lcd.cursor_pos = (0, 0)
    lcd.write_string("AI Parking")
    
   
    lcd.cursor_pos = (1, 0)
    lcd.write_string("Starting...")
    print("✅ LCD Display Connected!")
except Exception as e:
    print(f"LCD Display not found. Running without it.")
    lcd = None

# --- GLOBAL VARIABLES ---
TOTAL_CAR_COUNT = 0
TOTAL_CAPACITY = 10  # This will be updated by the website
outputFrame = None 
lock = threading.Lock() 

# --- INITIALIZE FLASK ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/data')
def get_data():
    return jsonify({
        'count': TOTAL_CAR_COUNT,
        'capacity': TOTAL_CAPACITY,
        'status': 'Active'
    })

#route to update capacity
@app.route('/set_capacity', methods=['POST'])
def set_capacity():
    global TOTAL_CAPACITY
    
    #grab json data from website
    data = request.get_json()
    
    if data and 'capacity' in data:
        try:
            new_cap = int(data['capacity'])
            TOTAL_CAPACITY = new_cap
            print(f"Capacity updated via Web to: {TOTAL_CAPACITY}")
            return jsonify({'status': 'success', 'new_capacity': TOTAL_CAPACITY})
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid number'}), 400
            
    return jsonify({'status': 'error', 'message': 'No capacity provided'}), 400

def generate():
    """ Video streaming generator function."""
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                time.sleep(0.01)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ai model logic
def run_ai_logic(args):
    global TOTAL_CAR_COUNT, outputFrame, lock, lcd
    # access total capacity in this thread

    model_path = args.model
    img_source = args.source
    min_thresh = float(args.thresh)
    iou_thresh = float(args.iou)
    min_area = int(args.min_area)
    user_res = args.resolution
    is_headless = args.headless

    if not os.path.exists(model_path):
        print(f"ERROR: Model {model_path} not found.")
        return

    print(f"Loading Model: {model_path}...")
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load YOLO model. {e}")
        return
        
    source_type = 'unknown'
    usb_idx = 0
    img_ext_list = ['.jpg','.JPG','.jpeg','.png','.PNG','.bmp']
    vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

    if os.path.isdir(img_source): source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext in img_ext_list: source_type = 'image'
        elif ext in vid_ext_list: source_type = 'video'
    elif 'usb' in img_source:
        source_type = 'usb'
        usb_idx = int(img_source.replace('usb', ''))
    elif img_source.isdigit(): 
        source_type = 'usb'
        usb_idx = int(img_source)
    elif 'picamera' in img_source: source_type = 'picamera'
    
    # camera set up 
    cap = None
    if source_type in ['video', 'usb']:
        cap_arg = img_source if source_type == 'video' else usb_idx
        cap = cv2.VideoCapture(cap_arg)
        if not cap.isOpened():
            print(f"CRITICAL ERROR: Could not open video source {cap_arg}.")
            return
        if user_res:
            resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
            cap.set(3, resW)
            cap.set(4, resH)

    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        resW, resH = 640, 480
        if user_res: resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
        config = cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)})
        cap.configure(config)
        cap.start()

    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                   (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
    
    count_buffer = deque(maxlen=10)

    print("--- AI Detection Loop Starting ---")
    if is_headless:
        print("Running in HEADLESS mode. View results at http://[Your-Pi-IP]:5000")
    
    while True:
        if source_type in ['usb', 'video']:
            ret, frame = cap.read()
            if not ret: break
        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        else: break

        results = model.track(frame, persist=True, verbose=False, conf=min_thresh, iou=iou_thresh)
        detections = results[0].boxes

        current_frame_count = 0
        valid_indices = []

        if detections.id is not None:
            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                
                area = (xmax - xmin) * (ymax - ymin)
                if area < min_area: continue

                current_frame_count += 1
                valid_indices.append(i)
        
        count_buffer.append(current_frame_count)
        
        if len(count_buffer) > 0:
            avg_count = sum(count_buffer) / len(count_buffer)
            TOTAL_CAR_COUNT = int(round(avg_count))
        else:
            TOTAL_CAR_COUNT = current_frame_count

        # update lcd display with new information
        if lcd:
            free_spaces = max(0, TOTAL_CAPACITY - TOTAL_CAR_COUNT)
            try:
                # Top Row: "Occupied: 5/20"
                lcd.cursor_pos = (0, 0)
                line1 = f"Occupied: {TOTAL_CAR_COUNT}/{TOTAL_CAPACITY}"
                lcd.write_string(f"{line1:<16}")
                
                # Bottom Row: "Free: 15"
                lcd.cursor_pos = (1, 0)
                line2 = f"Free: {free_spaces}"
                lcd.write_string(f"{line2:<16}")
            except Exception:
                pass 

        for i in valid_indices:
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            classidx = int(detections[i].cls.item())
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label = f'{detections[i].conf.item():.2f}'
            cv2.putText(frame, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show global capacity on the live video
        cv2.putText(frame, f'COUNT: {TOTAL_CAR_COUNT} / {TOTAL_CAPACITY}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        with lock:
            outputFrame = frame.copy()
        
        if not is_headless:
            cv2.imshow('YOLO Web Cam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    if cap: 
        if source_type == 'picamera': cap.stop()
        else: cap.release()
    cv2.destroyAllWindows()
    os._exit(0)

# main execution(arguments) for code to run
if __name__ == '__main__':
    # 1. PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to .pt file', required=True)
    parser.add_argument('--source', help='Image source (0, usb0, video.mp4)', required=True)
    parser.add_argument('--capacity', help='Total max parking spots', type=int, default=10)
    parser.add_argument('--thresh', help='Confidence (0.0-1.0)', default=0.5)
    parser.add_argument('--iou', help='IOU threshold (0.0-1.0)', default=0.5)
    parser.add_argument('--min_area', help='Min pixel area', default=0)
    parser.add_argument('--resolution', help='WxH', default=None)
    parser.add_argument('--headless', help='Run without window', action='store_true')
    parser.add_argument('--record', action='store_true')
    
    args, unknown = parser.parse_known_args()

    # --- DEBUGGING ARGUMENTS ---
    if len(unknown) > 0:
        print(f" WARNING: IGNORED UNKNOWN ARGUMENTS: {unknown}")
        print("   (Check your dashes! Use standard '--' dashes)")

    # setting global capacity
    TOTAL_CAPACITY = args.capacity
    print(f"🅿️  PARKING CAPACITY SET TO: {TOTAL_CAPACITY}")
    print(f"----------------------------------------")

    # start threads
    ai_thread = threading.Thread(target=run_ai_logic, args=(args,))
    ai_thread.start()
    
    print("Starting Web Server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
