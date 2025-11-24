"""
Flask Web Application for Smart Classroom Automation
Based on persondist.py with clean video feed (no text overlays)
"""
import cv2
import numpy as np
import time
import serial
import serial.tools.list_ports
import threading
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
import io
import csv
import os


app = Flask(__name__)

# Global variables for sharing data between detection thread and Flask
current_frame = None
frame_lock = threading.Lock()
detection_active = False
detection_thread = None

# System state - All relays start OFF
# relay1 = Zone 1 Light, relay2 = Zone 2 Light, relay3 = Zone 1 Fan, relay4 = Zone 2 Fan
relay_state = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
manual_override = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
automation_paused = False
current_distance = 0
active_zones = []
zones = []
focal_length_yolo = None
calibration_complete = False
zone_setup_complete = False
current_zone_setup = {'start': None, 'end': None}

# Tracking variables
appliance_timestamps = {
    'relay1': {'start': None, 'end': None, 'total_time': 0, 'events': []},  # Zone 1 Light
    'relay2': {'start': None, 'end': None, 'total_time': 0, 'events': []},  # Zone 2 Light
    'relay3': {'start': None, 'end': None, 'total_time': 0, 'events': []},  # Zone 1 Fan
    'relay4': {'start': None, 'end': None, 'total_time': 0, 'events': []}   # Zone 2 Fan
}

# Power consumption in watts
power_consumption = {
    'relay1': 60,   # Zone 1 Light: 60W
    'relay2': 60,   # Zone 2 Light: 60W
    'relay3': 75,   # Zone 1 Fan: 75W
    'relay4': 75    # Zone 2 Fan: 75W
}

# Energy cost per kWh (in INR)
energy_cost_per_kwh = 6.0

# CO₂ emission factor (kg CO₂ per kWh)
co2_emission_factor = 0.82

# Event log
event_log = []



# Detection parameters (from your original code)
min_confidence = 0.6
nms_threshold = 0.4
cap = None
arduino = None
net = None
output_layers = None
classes = None

def find_arduino_port():
    """Try to find Arduino by listing available ports"""
    # Get list of available ports
    available_ports = [port.device for port in serial.tools.list_ports.comports()]
    print(f"Available serial ports: {available_ports}")
    
    if not available_ports:
        print("No serial ports found.")
        return None
    
    # Return the first available port (usually the connected Arduino)
    # You can manually change this if you have multiple serial devices
    return available_ports[0]

def initialize_system():
    """Initialize the detection system components"""
    global cap, arduino, net, output_layers, classes, relay_state
    
    try:
        print("Initializing camera...")
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Failed to initialize camera"
        print("Camera initialized successfully")
        
        print("Initializing Arduino...")
        # Try to find Arduino port automatically
        arduino_port = find_arduino_port()
        if arduino_port:
            print(f"Using Arduino on {arduino_port}")
            arduino = serial.Serial(arduino_port, 9600, timeout=1)
        else:
            # Fallback to COM9 if auto-detection fails
            print("Auto-detection failed, trying COM9...")
            raise serial.SerialException("No serial ports found. Please connect Arduino and try again.")
        time.sleep(2)
        print("Arduino connected successfully")
        
        # Initialize all relays to OFF state
        relay_state = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
        send_arduino_command(relay_state)  # Send OFF command to all relays
        print("All relays initialized to OFF state")
        
        print("Loading YOLO model...")
        # Initialize YOLO
        net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print("YOLO model loaded successfully")
        
        return True, "System initialized successfully"
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        return False, f"Initialization error: {str(e)}"

def apply_nms(boxes, confidences, img_width, img_height):
    """Apply Non-Maximum Suppression (from your original code)"""
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, nms_threshold)
    if len(indexes) > 0:
        if isinstance(indexes[0], (list, np.ndarray)):
            indexes = [idx[0] for idx in indexes]
        filtered_boxes = [boxes[i] for i in indexes]
        return filtered_boxes
    return []

def estimate_distance_from_yolo(box_width_pixels):
    """Estimate distance from camera using YOLO bounding box width (from your original code)"""
    if focal_length_yolo and box_width_pixels > 0:
        REAL_PERSON_WIDTH_CM = 45
        return (REAL_PERSON_WIDTH_CM * focal_length_yolo) / box_width_pixels
    return 0

def update_appliance_timestamp(relay_name, is_on):
    """Update appliance timestamps and log events"""
    current_time = time.time()
    timestamp = datetime.now()
    
    if is_on and appliance_timestamps[relay_name]['start'] is None:
        # Appliance turning ON
        appliance_timestamps[relay_name]['start'] = current_time
        appliance_timestamps[relay_name]['end'] = None
        
        event = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'appliance': relay_name,
            'action': 'ON',
            'type': 'automatic' if not manual_override[relay_name] else 'manual'
        }
        appliance_timestamps[relay_name]['events'].append(event)
        event_log.append(event)
        
    elif not is_on and appliance_timestamps[relay_name]['start'] is not None:
        # Appliance turning OFF
        appliance_timestamps[relay_name]['end'] = current_time
        duration = current_time - appliance_timestamps[relay_name]['start']
        appliance_timestamps[relay_name]['total_time'] += duration
        
        event = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'appliance': relay_name,
            'action': 'OFF',
            'duration': f"{duration:.1f}s",
            'type': 'automatic' if not manual_override[relay_name] else 'manual'
        }
        appliance_timestamps[relay_name]['events'].append(event)
        event_log.append(event)
        
        # Reset for next cycle
        appliance_timestamps[relay_name]['start'] = None
        appliance_timestamps[relay_name]['end'] = None

def analyze_energy_patterns():
    """Analyze energy usage patterns and generate recommendations"""
    
    # Calculate current usage statistics
    total_consumption = 0
    total_cost = 0
    zone_usage = {'zone1': {'light_time': 0, 'fan_time': 0, 'consumption': 0}, 
                  'zone2': {'light_time': 0, 'fan_time': 0, 'consumption': 0}}
    
    for relay_name, data in appliance_timestamps.items():
        if data['total_time'] > 0:
            consumption = (data['total_time'] / 3600) * power_consumption[relay_name]
            total_consumption += consumption
            total_cost += consumption * (energy_cost_per_kwh / 1000)
            
            # Zone analysis
            if relay_name in ['relay1', 'relay3']:  # Zone 1 appliances
                zone_usage['zone1']['light_time' if relay_name == 'relay1' else 'fan_time'] = data['total_time'] / 3600
                zone_usage['zone1']['consumption'] += consumption
            elif relay_name in ['relay2', 'relay4']:  # Zone 2 appliances
                zone_usage['zone2']['light_time' if relay_name == 'relay2' else 'fan_time'] = data['total_time'] / 3600
                zone_usage['zone2']['consumption'] += consumption
    
    # Generate recommendations based on patterns
    recommendations = []
    
    # High usage recommendations
    if total_consumption > 500:  # More than 500Wh
        recommendations.append({
            'type': 'warning',
            'title': 'High Energy Consumption Detected',
            'message': f'Current consumption is {total_consumption:.1f}Wh. Consider optimizing usage patterns.',
            'suggestion': 'Try to reduce unnecessary appliance usage and ensure proper zone-based control.'
        })
    
    # Zone imbalance recommendations
    zone1_usage = zone_usage['zone1']['consumption']
    zone2_usage = zone_usage['zone2']['consumption']
    if abs(zone1_usage - zone2_usage) > 100:  # Significant imbalance
        recommendations.append({
            'type': 'info',
            'title': 'Zone Usage Imbalance',
            'message': f'Zone 1: {zone1_usage:.1f}Wh, Zone 2: {zone2_usage:.1f}Wh',
            'suggestion': 'Consider redistributing activities to balance energy usage across zones.'
        })
    
    # Manual override recommendations
    manual_overrides = sum(1 for override in manual_override.values() if override)
    if manual_overrides > 0:
        recommendations.append({
            'type': 'info',
            'title': 'Manual Overrides Active',
            'message': f'{manual_overrides} appliances are under manual control',
            'suggestion': 'Manual control is active. Remember to turn off appliances when not needed.'
        })
    
    # Efficiency recommendations
    if total_consumption > 0:
        efficiency = ((total_consumption * 0.3) / total_consumption) * 100  # Rough efficiency calculation
        if efficiency < 70:
            recommendations.append({
                'type': 'suggestion',
                'title': 'Energy Efficiency Tips',
                'message': 'Consider these energy-saving strategies:',
                'suggestion': '• Use natural lighting when possible\n• Turn off fans when leaving the area\n• Optimize zone-based automation\n• Regular maintenance of appliances'
            })
    
    # Cost-saving recommendations
    if total_cost > 5:  # More than ₹5
        recommendations.append({
            'type': 'suggestion',
            'title': 'Cost Optimization',
            'message': f'Current cost: ₹{total_cost:.2f}',
            'suggestion': 'To reduce costs: Use LED lights, optimize fan speeds, and ensure proper automation settings.'
        })
    
    return recommendations


def send_arduino_command(new_relay):
    """Send relay commands to Arduino (active-LOW configuration)"""
    if arduino and arduino.is_open:
        try:
            command = 0
            # Active-LOW configuration: bit=1 means relay OFF, bit=0 means relay ON
            # relay1 = Zone 1 Light, relay2 = Zone 2 Light, relay3 = Zone 1 Fan, relay4 = Zone 2 Fan
            command |= (0 if new_relay['relay1'] else 1) << 0  # Zone 1 Light (relay1)
            command |= (0 if new_relay['relay2'] else 1) << 1  # Zone 2 Light (relay2)
            command |= (0 if new_relay['relay3'] else 1) << 2  # Zone 1 Fan (relay3)
            command |= (0 if new_relay['relay4'] else 1) << 3  # Zone 2 Fan (relay4)
            arduino.write(bytes([command]))
            print(f"Arduino Command: {command:04b} - relay1:{new_relay['relay1']}, relay2:{new_relay['relay2']}, relay3:{new_relay['relay3']}, relay4:{new_relay['relay4']}")
        except Exception as e:
            print(f"Error sending command to Arduino: {e}")

def detection_worker():
    """Main detection loop with manual zone setup (adapted from your original code)"""
    global current_frame, current_distance, active_zones, relay_state, focal_length_yolo, zones
    
    if not cap or not cap.isOpened():
        return
    
    # Step 1: Calibration (from your original code)
    print("Starting calibration...")
    REAL_PERSON_WIDTH_CM = 45
    known_distance_cm = 100
    
    # Calibration loop
    while detection_active and cap.isOpened() and not calibration_complete:
        success, img = cap.read()
        if not success:
            continue
            
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > min_confidence and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        person_boxes_filtered = apply_nms(boxes, confidences, width, height)
        
        if person_boxes_filtered:
            x, y, w, h = max(person_boxes_filtered, key=lambda b: b[2])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            focal_length_yolo = (w * known_distance_cm) / REAL_PERSON_WIDTH_CM
            
            # Add calibration text to frame
            cv2.putText(img, f"Focal Length: {int(focal_length_yolo)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, "Click 'Complete Calibration' when stable", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Update frame for streaming
            with frame_lock:
                current_frame = img.copy()
        else:
            cv2.putText(img, "No person detected for calibration", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            with frame_lock:
                current_frame = img.copy()
        
        time.sleep(0.1)
    
    if not focal_length_yolo:
        print("Calibration failed")
        return
    
    # Step 2: Zone Setup (from your original code)
    print("Starting zone setup...")
    zones = []
    current_zone = {}
    num_zones = 2
    
    # Zone setup loop
    while detection_active and cap.isOpened() and not zone_setup_complete:
        success, img = cap.read()
        if not success:
            continue

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > min_confidence and classes[class_id] == "person":
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        person_boxes_filtered = apply_nms(boxes, confidences, width, height)
        
        if person_boxes_filtered:
            x, y, w, h = max(person_boxes_filtered, key=lambda b:b[2])
            dist = estimate_distance_from_yolo(w)
            current_distance = dist
            cv2.putText(img, f"{int(dist)} cm", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        else:
            current_distance = 0

        cv2.putText(img, f"Zones: {len(zones)}/{num_zones}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(img, "Use buttons to mark zones", (30,130), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        
        # Update frame for streaming
        with frame_lock:
            current_frame = img.copy()
        
        time.sleep(0.1)
    
    print(f"Zones defined: {zones}")
    
    # If zones not properly set, use defaults
    if len(zones) < 2:
        zones = [
            {'start': 50, 'end': 150},   # Zone 1
            {'start': 150, 'end': 300}   # Zone 2
        ]
    
    # Main detection loop - INTEGRATED WITH WORKING RELAY LOGIC
    fan_timer = [None] * 2
    no_person_timer = None
    margin = 7
    activation_delay = 4  # 4 seconds as in reference code
    turn_off_delay = 3
    
    while detection_active and cap.isOpened():
        success, img = cap.read()
        if not success:
            continue
            
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > min_confidence and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        person_boxes_filtered = apply_nms(boxes, confidences, width, height)
        
        new_relay = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
        detected_in_any_zone = False
        active_zones = [False] * 2
        current_distance = 0
        
        if person_boxes_filtered:
            # Iterate through all detected persons (multiple people detection)
            for x, y, w, h in person_boxes_filtered:
                distance = estimate_distance_from_yolo(w)
                current_distance = distance
                
                # Check all defined zones to see if the person is within any of them
                for i, z in enumerate(zones):
                    if (z['start'] - margin) <= distance <= (z['end'] + margin):
                        active_zones[i] = True
                        detected_in_any_zone = True
                        break
                
                # Draw bounding box and distance for each person
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{int(distance)} cm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Apply automation logic (from reference code)
        if not automation_paused and detected_in_any_zone:
            no_person_timer = None
            
            # Update relay states based on active zones
            for i in range(2):
                if active_zones[i]:
                    if i == 0:  # Zone 1 - relay1 (Light) and relay3 (Fan)
                        new_relay['relay1'] = True  # Light for Zone 1
                        if fan_timer[0] is None:
                            fan_timer[0] = time.time()
                        if (time.time() - fan_timer[0]) >= activation_delay:
                            new_relay['relay3'] = True  # Fan for Zone 1 after 4s
                    elif i == 1:  # Zone 2 - relay2 (Light) and relay4 (Fan)
                        new_relay['relay2'] = True  # Light for Zone 2
                        if fan_timer[1] is None:
                            fan_timer[1] = time.time()
                        if (time.time() - fan_timer[1]) >= activation_delay:
                            new_relay['relay4'] = True  # Fan for Zone 2 after 4s
                else:
                    fan_timer[i] = None
        
        # Apply manual overrides - manual override takes precedence
        for relay_name in manual_override:
            if manual_override[relay_name]:
                new_relay[relay_name] = True
        
        # Handle no person detection
        if not detected_in_any_zone and not automation_paused:
            if no_person_timer is None:
                no_person_timer = time.time()
            if (time.time() - no_person_timer) >= turn_off_delay:
                fan_timer = [None] * 2
                # Turn off all relays (except manual overrides)
                new_relay = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
                # Apply manual overrides
                for relay_name in manual_override:
                    if manual_override[relay_name]:
                        new_relay[relay_name] = True
        
        # Update timestamps when relay states change
        for relay_name in new_relay:
            if new_relay[relay_name] != relay_state[relay_name]:
                update_appliance_timestamp(relay_name, new_relay[relay_name])
        
        # Send commands to Arduino (using working reference logic)
        if new_relay != relay_state:
            send_arduino_command(new_relay)
            relay_state.update(new_relay)
            print(f"Relay state updated: {relay_state}")
        
        # Update frame for streaming (CLEAN - no text overlays)
        with frame_lock:
            current_frame = img.copy()
        
        time.sleep(0.1)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the camera feed - CLEAN (no text overlays)"""
    def generate():
        while detection_active:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start the detection system"""
    global detection_active, detection_thread
    
    if not detection_active:
        success, message = initialize_system()
        if not success:
            return jsonify({'status': 'error', 'message': message})
        
        detection_active = True
        detection_thread = threading.Thread(target=detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Detection started'})
    else:
        return jsonify({'status': 'success', 'message': 'Detection already running'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop the detection system"""
    global detection_active
    
    detection_active = False
    if cap:
        cap.release()
    if arduino:
        arduino.close()
    
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

@app.route('/status')
def get_status():
    """Get current system status - comprehensive JSON response"""
    # Calculate energy metrics
    total_consumption = 0
    total_cost = 0
    total_co2 = 0
    
    for relay_name, data in appliance_timestamps.items():
        if data['total_time'] > 0:
            consumption = (data['total_time'] / 3600) * power_consumption[relay_name]
            cost = consumption * (energy_cost_per_kwh / 1000)
            co2 = consumption * (co2_emission_factor / 1000)
            
            total_consumption += consumption
            total_cost += cost
            total_co2 += co2
    
    # Get active zone names
    active_zone_names = []
    if active_zones:
        for i, active in enumerate(active_zones):
            if active:
                active_zone_names.append(f"Zone {i + 1}")
    
    return jsonify({
        'relay_state': relay_state,
        'manual_override': manual_override,
        'automation_paused': automation_paused,
        'current_distance': current_distance,
        'active_zones': active_zones,
        'active_zone_names': active_zone_names,
        'zones': zones,
        'energy_metrics': {
            'total_consumption_wh': total_consumption,
            'total_cost_inr': total_cost,
            'total_co2_kg': total_co2
        },
        'instructions': {
            'start': 'Press Start Detection to begin',
            'stop': 'Press Stop Detection to end',
            'pause': 'Press Pause/Resume to control automation',
            'manual': 'Use toggle buttons for manual control'
        }
    })

@app.route('/toggle_relay', methods=['POST'])
def toggle_relay():
    """Toggle manual override for a relay"""
    data = request.json
    relay_name = data.get('relay')
    
    if relay_name in manual_override and relay_name in relay_state:
        # Toggle manual override state
        manual_override[relay_name] = not manual_override[relay_name]
        
        if manual_override[relay_name]:
            # Turn ON the relay manually
            relay_state[relay_name] = True
            update_appliance_timestamp(relay_name, True)
            print(f"Manual override ON: {relay_name} turned ON")
        else:
            # Turn OFF the relay and let automation take control
            relay_state[relay_name] = False
            update_appliance_timestamp(relay_name, False)
            print(f"Manual override OFF: {relay_name} turned OFF")
        
        # Send command to Arduino
        send_arduino_command(relay_state)
        
        return jsonify({
            'status': 'success', 
            'relay_state': relay_state, 
            'manual_override': manual_override,
            'message': f"{relay_name} turned {'ON' if relay_state[relay_name] else 'OFF'}"
        })
    
    return jsonify({'status': 'error', 'message': 'Invalid relay'})

@app.route('/pause_automation', methods=['POST'])
def pause_automation():
    """Pause/resume automation"""
    global automation_paused
    
    automation_paused = not automation_paused
    return jsonify({'status': 'success', 'paused': automation_paused})

@app.route('/stop_automation', methods=['POST'])
def stop_automation():
    """Stop automation - turn off all relays and clear manual overrides"""
    global relay_state, manual_override, automation_paused
    
    # Store previous states to update timestamps
    previous_states = relay_state.copy()
    
    # Turn off all relays
    relay_state = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
    
    # Clear all manual overrides
    manual_override = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
    
    # Pause automation
    automation_paused = True
    
    # Update timestamps for relays that were previously on
    for relay_name in relay_state:
        if previous_states.get(relay_name, False):
            update_appliance_timestamp(relay_name, False)
    
    # Send command to Arduino to turn off all relays
    send_arduino_command(relay_state)
    
    print("Automation stopped - all relays turned OFF")
    
    return jsonify({
        'status': 'success', 
        'message': 'Automation stopped - all appliances turned off',
        'relay_state': relay_state,
        'manual_override': manual_override,
        'paused': automation_paused
    })

@app.route('/get_events')
def get_events():
    """Get recent events"""
    return jsonify({'events': event_log[-20:]})  # Last 20 events

@app.route('/get_recommendations')
def get_recommendations():
    """Get recommendations based on current energy usage"""
    recommendations = analyze_energy_patterns()
    return jsonify({'recommendations': recommendations})



@app.route('/complete_calibration', methods=['POST'])
def complete_calibration():
    """Complete the calibration step"""
    global calibration_complete
    calibration_complete = True
    return jsonify({'status': 'success', 'message': 'Calibration completed'})

@app.route('/start_zone', methods=['POST'])
def start_zone():
    """Start marking a zone"""
    global current_zone_setup
    current_zone_setup['start'] = current_distance
    return jsonify({'status': 'success', 'message': f'Zone start marked at {int(current_distance)} cm'})

@app.route('/end_zone', methods=['POST'])
def end_zone():
    """End marking a zone"""
    global current_zone_setup, zones, zone_setup_complete
    if current_zone_setup['start'] is not None:
        current_zone_setup['end'] = current_distance
        zones.append(current_zone_setup.copy())
        current_zone_setup = {'start': None, 'end': None}
        
        if len(zones) >= 2:
            zone_setup_complete = True
        
        return jsonify({'status': 'success', 'message': f'Zone {len(zones)} marked: {int(zones[-1]["start"])}-{int(zones[-1]["end"])} cm'})
    return jsonify({'status': 'error', 'message': 'No zone start marked'})

@app.route('/get_setup_status')
def get_setup_status():
    """Get current setup status"""
    return jsonify({
        'calibration_complete': calibration_complete,
        'zone_setup_complete': zone_setup_complete,
        'zones': zones,
        'current_distance': current_distance,
        'current_zone_setup': current_zone_setup
    })

@app.route('/browser_report')
def browser_report():
    """Generate comprehensive report for browser display"""
    # Calculate consumption data
    total_consumption = 0
    total_cost = 0
    total_co2 = 0
    total_savings = 0
    
    # Detailed appliance analysis with user-friendly names
    appliance_analysis = {
        'relay1': {'name': 'Zone 1 Light', 'display_name': 'Zone 1 Light', 'type': 'light', 'zone': 1, 'time_on': 0, 'consumption': 0, 'cost': 0, 'events': 0},
        'relay2': {'name': 'Zone 2 Light', 'display_name': 'Zone 2 Light', 'type': 'light', 'zone': 2, 'time_on': 0, 'consumption': 0, 'cost': 0, 'events': 0},
        'relay3': {'name': 'Zone 1 Fan', 'display_name': 'Zone 1 Fan', 'type': 'fan', 'zone': 1, 'time_on': 0, 'consumption': 0, 'cost': 0, 'events': 0},
        'relay4': {'name': 'Zone 2 Fan', 'display_name': 'Zone 2 Fan', 'type': 'fan', 'zone': 2, 'time_on': 0, 'consumption': 0, 'cost': 0, 'events': 0}
    }
    
    # Zone-wise analysis
    zone_analysis = {
        'zone1': {'light_time': 0, 'fan_time': 0, 'light_consumption': 0, 'fan_consumption': 0, 'light_cost': 0, 'fan_cost': 0, 'total_time': 0, 'total_consumption': 0, 'total_cost': 0},
        'zone2': {'light_time': 0, 'fan_time': 0, 'light_consumption': 0, 'fan_consumption': 0, 'light_cost': 0, 'fan_cost': 0, 'total_time': 0, 'total_consumption': 0, 'total_cost': 0}
    }
    
    # Type-wise analysis (lights vs fans)
    type_analysis = {
        'lights': {'time_on': 0, 'consumption': 0, 'cost': 0, 'appliances': []},
        'fans': {'time_on': 0, 'consumption': 0, 'cost': 0, 'appliances': []}
    }
    
    session_start_time = time.time()
    
    # Calculate session duration
    if appliance_timestamps['relay1']['events']:
        # Convert string timestamps to time objects for comparison
        from datetime import datetime
        timestamps = []
        for event in appliance_timestamps['relay1']['events']:
            try:
                # Parse the timestamp string
                dt = datetime.strptime(event['timestamp'], '%Y-%m-%d %H:%M:%S')
                timestamps.append(dt.timestamp())
            except (ValueError, KeyError):
                continue
        if timestamps:
            session_start_time = min(timestamps)
    
    session_duration_hours = (time.time() - session_start_time) / 3600
    
    for relay_name, data in appliance_timestamps.items():
        if data['total_time'] > 0:
            consumption = (data['total_time'] / 3600) * power_consumption[relay_name]
            cost = consumption * (energy_cost_per_kwh / 1000)
            co2 = consumption * (co2_emission_factor / 1000)
            
            total_consumption += consumption
            total_cost += cost
            total_co2 += co2
            
            # Calculate potential savings (if appliances were always on during session)
            max_possible_consumption = session_duration_hours * power_consumption[relay_name]
            savings = max_possible_consumption - consumption
            total_savings += savings
            
            # Update appliance analysis
            if relay_name in appliance_analysis:
                appliance_analysis[relay_name]['time_on'] = data['total_time']
                appliance_analysis[relay_name]['consumption'] = consumption
                appliance_analysis[relay_name]['cost'] = cost
                appliance_analysis[relay_name]['events'] = len(data['events'])
            
            # Zone analysis
            if relay_name in ['relay1', 'relay3']:  # Zone 1
                if relay_name == 'relay1':
                    zone_analysis['zone1']['light_time'] = data['total_time'] / 3600
                    zone_analysis['zone1']['light_consumption'] = consumption
                    zone_analysis['zone1']['light_cost'] = cost
                else:  # relay3
                    zone_analysis['zone1']['fan_time'] = data['total_time'] / 3600
                    zone_analysis['zone1']['fan_consumption'] = consumption
                    zone_analysis['zone1']['fan_cost'] = cost
                
                zone_analysis['zone1']['total_time'] += data['total_time'] / 3600
                zone_analysis['zone1']['total_consumption'] += consumption
                zone_analysis['zone1']['total_cost'] += cost
                
            elif relay_name in ['relay2', 'relay4']:  # Zone 2
                if relay_name == 'relay2':
                    zone_analysis['zone2']['light_time'] = data['total_time'] / 3600
                    zone_analysis['zone2']['light_consumption'] = consumption
                    zone_analysis['zone2']['light_cost'] = cost
                else:  # relay4
                    zone_analysis['zone2']['fan_time'] = data['total_time'] / 3600
                    zone_analysis['zone2']['fan_consumption'] = consumption
                    zone_analysis['zone2']['fan_cost'] = cost
                
                zone_analysis['zone2']['total_time'] += data['total_time'] / 3600
                zone_analysis['zone2']['total_consumption'] += consumption
                zone_analysis['zone2']['total_cost'] += cost
            
            # Type analysis (lights vs fans)
            if relay_name in ['relay1', 'relay2']:  # Lights
                type_analysis['lights']['time_on'] += data['total_time']
                type_analysis['lights']['consumption'] += consumption
                type_analysis['lights']['cost'] += cost
                type_analysis['lights']['appliances'].append(relay_name)
            elif relay_name in ['relay3', 'relay4']:  # Fans
                type_analysis['fans']['time_on'] += data['total_time']
                type_analysis['fans']['consumption'] += consumption
                type_analysis['fans']['cost'] += cost
                type_analysis['fans']['appliances'].append(relay_name)
    
    # Calculate total savings cost
    total_savings_cost = total_savings * (energy_cost_per_kwh / 1000)
    
    # Calculate maximum possible consumption
    max_possible_consumption = session_duration_hours * (power_consumption['relay1'] + power_consumption['relay2'] + power_consumption['relay3'] + power_consumption['relay4'])
    max_possible_cost = max_possible_consumption * (energy_cost_per_kwh / 1000)
    
    return jsonify({
        'session_duration_hours': session_duration_hours,
        'total_consumption': total_consumption,
        'total_cost': total_cost,
        'total_co2': total_co2,
        'total_savings': total_savings,
        'total_savings_cost': total_savings_cost,
        'max_possible_consumption': max_possible_consumption,
        'max_possible_cost': max_possible_cost,
        'appliance_analysis': appliance_analysis,
        'zone_analysis': zone_analysis,
        'type_analysis': type_analysis,
        'efficiency_percentage': ((total_savings / max_possible_consumption) * 100) if max_possible_consumption > 0 else 0
    })

@app.route('/generate_report')
def generate_report():
    """Generate comprehensive consumption report with detailed analytics"""
    # Calculate consumption data
    total_consumption = 0
    total_cost = 0
    total_co2 = 0
    total_savings = 0
    
    # Detailed appliance analysis with user-friendly names
    appliance_analysis = {
        'relay1': {'name': 'Zone 1 Light', 'display_name': 'Zone 1 Light', 'type': 'light', 'zone': 1, 'time_on': 0, 'consumption': 0, 'cost': 0, 'events': 0},
        'relay2': {'name': 'Zone 2 Light', 'display_name': 'Zone 2 Light', 'type': 'light', 'zone': 2, 'time_on': 0, 'consumption': 0, 'cost': 0, 'events': 0},
        'relay3': {'name': 'Zone 1 Fan', 'display_name': 'Zone 1 Fan', 'type': 'fan', 'zone': 1, 'time_on': 0, 'consumption': 0, 'cost': 0, 'events': 0},
        'relay4': {'name': 'Zone 2 Fan', 'display_name': 'Zone 2 Fan', 'type': 'fan', 'zone': 2, 'time_on': 0, 'consumption': 0, 'cost': 0, 'events': 0}
    }
    
    # Zone-wise analysis
    zone_analysis = {
        'zone1': {'light_time': 0, 'fan_time': 0, 'light_consumption': 0, 'fan_consumption': 0, 'light_cost': 0, 'fan_cost': 0, 'total_time': 0, 'total_consumption': 0, 'total_cost': 0},
        'zone2': {'light_time': 0, 'fan_time': 0, 'light_consumption': 0, 'fan_consumption': 0, 'light_cost': 0, 'fan_cost': 0, 'total_time': 0, 'total_consumption': 0, 'total_cost': 0}
    }
    
    # Type-wise analysis (lights vs fans)
    type_analysis = {
        'lights': {'time_on': 0, 'consumption': 0, 'cost': 0, 'appliances': []},
        'fans': {'time_on': 0, 'consumption': 0, 'cost': 0, 'appliances': []}
    }
    
    report_data = []
    session_start_time = time.time()
    
    # Calculate session duration
    if appliance_timestamps['relay1']['events']:
        # Convert string timestamps to time objects for comparison
        from datetime import datetime
        timestamps = []
        for event in appliance_timestamps['relay1']['events']:
            try:
                # Parse the timestamp string
                dt = datetime.strptime(event['timestamp'], '%Y-%m-%d %H:%M:%S')
                timestamps.append(dt.timestamp())
            except (ValueError, KeyError):
                continue
        if timestamps:
            session_start_time = min(timestamps)
    
    session_duration_hours = (time.time() - session_start_time) / 3600
    
    for relay_name, data in appliance_timestamps.items():
        if data['total_time'] > 0:
            consumption = (data['total_time'] / 3600) * power_consumption[relay_name]
            cost = consumption * (energy_cost_per_kwh / 1000)
            co2 = consumption * (co2_emission_factor / 1000)
            
            total_consumption += consumption
            total_cost += cost
            total_co2 += co2
            
            # Calculate potential savings (if appliances were always on during session)
            max_possible_consumption = session_duration_hours * power_consumption[relay_name]
            savings = max_possible_consumption - consumption
            total_savings += savings
            
            # Update appliance analysis
            if relay_name in appliance_analysis:
                appliance_analysis[relay_name]['time_on'] = data['total_time']
                appliance_analysis[relay_name]['consumption'] = consumption
                appliance_analysis[relay_name]['cost'] = cost
                appliance_analysis[relay_name]['events'] = len(data['events'])
            
            # Zone analysis
            if relay_name in ['relay1', 'relay3']:  # Zone 1
                if relay_name == 'relay1':
                    zone_analysis['zone1']['light_time'] = data['total_time'] / 3600
                    zone_analysis['zone1']['light_consumption'] = consumption
                    zone_analysis['zone1']['light_cost'] = cost
                else:  # relay3
                    zone_analysis['zone1']['fan_time'] = data['total_time'] / 3600
                    zone_analysis['zone1']['fan_consumption'] = consumption
                    zone_analysis['zone1']['fan_cost'] = cost
                
                zone_analysis['zone1']['total_time'] += data['total_time'] / 3600
                zone_analysis['zone1']['total_consumption'] += consumption
                zone_analysis['zone1']['total_cost'] += cost
                
            elif relay_name in ['relay2', 'relay4']:  # Zone 2
                if relay_name == 'relay2':
                    zone_analysis['zone2']['light_time'] = data['total_time'] / 3600
                    zone_analysis['zone2']['light_consumption'] = consumption
                    zone_analysis['zone2']['light_cost'] = cost
                else:  # relay4
                    zone_analysis['zone2']['fan_time'] = data['total_time'] / 3600
                    zone_analysis['zone2']['fan_consumption'] = consumption
                    zone_analysis['zone2']['fan_cost'] = cost
                
                zone_analysis['zone2']['total_time'] += data['total_time'] / 3600
                zone_analysis['zone2']['total_consumption'] += consumption
                zone_analysis['zone2']['total_cost'] += cost
            
            # Type analysis (lights vs fans)
            if relay_name in ['relay1', 'relay2']:  # Lights
                type_analysis['lights']['time_on'] += data['total_time']
                type_analysis['lights']['consumption'] += consumption
                type_analysis['lights']['cost'] += cost
                type_analysis['lights']['appliances'].append(relay_name)
            elif relay_name in ['relay3', 'relay4']:  # Fans
                type_analysis['fans']['time_on'] += data['total_time']
                type_analysis['fans']['consumption'] += consumption
                type_analysis['fans']['cost'] += cost
                type_analysis['fans']['appliances'].append(relay_name)
            
            report_data.append({
                'appliance': relay_name,
                'appliance_name': f"Zone {1 if relay_name in ['relay1', 'relay3'] else 2} {'Light' if relay_name in ['relay1', 'relay2'] else 'Fan'}",
                'total_time_hours': data['total_time'] / 3600,
                'total_time_minutes': (data['total_time'] / 60),
                'consumption_wh': consumption,
                'cost_inr': cost,
                'co2_kg': co2,
                'savings_wh': savings,
                'savings_cost': savings * (energy_cost_per_kwh / 1000),
                'events_count': len(data['events']),
                'efficiency': f"{((savings / max_possible_consumption) * 100):.1f}%" if max_possible_consumption > 0 else "0%"
            })
    
    # Calculate total savings cost
    total_savings_cost = total_savings * (energy_cost_per_kwh / 1000)
    
    # Create comprehensive CSV report
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['SMART CLASSROOM AUTOMATION - COMPREHENSIVE ENERGY REPORT'])
    writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow(['Session Duration:', f"{session_duration_hours:.2f} hours"])
    writer.writerow(['Report Type:', 'Detailed Analytics with Smart Automation Savings'])
    writer.writerow([])
    
    # Executive Summary
    writer.writerow(['=== EXECUTIVE SUMMARY ==='])
    writer.writerow(['Total Energy Consumed:', f"{total_consumption:.2f} Wh"])
    writer.writerow(['Total Cost:', f"₹{total_cost:.2f}"])
    writer.writerow(['Total CO2 Emissions:', f"{total_co2:.3f} kg"])
    writer.writerow(['Energy Saved by Smart Automation:', f"{total_savings:.2f} Wh"])
    writer.writerow(['Cost Saved by Smart Automation:', f"₹{total_savings_cost:.2f}"])
    writer.writerow(['Overall Efficiency:', f"{((total_savings / (total_consumption + total_savings)) * 100):.1f}%"] if (total_consumption + total_savings) > 0 else ['Overall Efficiency:', '0%'])
    writer.writerow([])
    
    # Smart Automation Savings Analysis
    writer.writerow(['=== SMART AUTOMATION SAVINGS ANALYSIS ==='])
    writer.writerow(['Smart Automation Benefits:'])
    writer.writerow(['• Automatic ON/OFF based on occupancy detection'])
    writer.writerow(['• Zone-based control for targeted energy usage'])
    writer.writerow(['• 4-second fan delay to prevent unnecessary usage'])
    writer.writerow(['• 3-second delay before turning off appliances'])
    writer.writerow(['• Real-time monitoring and optimization'])
    writer.writerow(['• Reduced energy waste through intelligent switching'])
    writer.writerow([])
    
    # Type-wise Analysis (Lights vs Fans)
    writer.writerow(['=== APPLIANCE TYPE ANALYSIS ==='])
    writer.writerow(['LIGHTS ANALYSIS:'])
    writer.writerow(['Total Light ON Time:', f"{type_analysis['lights']['time_on']/3600:.2f} hours ({type_analysis['lights']['time_on']/60:.1f} minutes)"])
    writer.writerow(['Total Light Consumption:', f"{type_analysis['lights']['consumption']:.2f} Wh"])
    writer.writerow(['Total Light Cost:', f"₹{type_analysis['lights']['cost']:.2f}"])
    writer.writerow(['Light Appliances:', ', '.join([appliance_analysis[app]['name'] for app in type_analysis['lights']['appliances']])])
    writer.writerow([])
    writer.writerow(['FANS ANALYSIS:'])
    writer.writerow(['Total Fan ON Time:', f"{type_analysis['fans']['time_on']/3600:.2f} hours ({type_analysis['fans']['time_on']/60:.1f} minutes)"])
    writer.writerow(['Total Fan Consumption:', f"{type_analysis['fans']['consumption']:.2f} Wh"])
    writer.writerow(['Total Fan Cost:', f"₹{type_analysis['fans']['cost']:.2f}"])
    writer.writerow(['Fan Appliances:', ', '.join([appliance_analysis[app]['name'] for app in type_analysis['fans']['appliances']])])
    writer.writerow([])
    
    # Zone-wise Analysis
    writer.writerow(['=== ZONE-WISE ANALYSIS ==='])
    for zone_name, zone_data in zone_analysis.items():
        zone_num = zone_name.replace('zone', '')
        total_zone_consumption = zone_data['light_consumption'] + zone_data['fan_consumption']
        total_zone_cost = zone_data['light_cost'] + zone_data['fan_cost']
        
        writer.writerow([f'ZONE {zone_num.upper()} ANALYSIS:'])
        writer.writerow(['  Total Zone ON Time:', f"{zone_data['total_time']:.2f} hours ({zone_data['total_time']*60:.1f} minutes)"])
        writer.writerow(['  Light ON Time:', f"{zone_data['light_time']:.2f} hours ({zone_data['light_time']*60:.1f} minutes)"])
        writer.writerow(['  Fan ON Time:', f"{zone_data['fan_time']:.2f} hours ({zone_data['fan_time']*60:.1f} minutes)"])
        writer.writerow(['  Light Consumption:', f"{zone_data['light_consumption']:.2f} Wh"])
        writer.writerow(['  Fan Consumption:', f"{zone_data['fan_consumption']:.2f} Wh"])
        writer.writerow(['  Total Zone Consumption:', f"{total_zone_consumption:.2f} Wh"])
        writer.writerow(['  Total Zone Cost:', f"₹{total_zone_cost:.2f}"])
        writer.writerow(['  Zone Efficiency:', f"{((zone_data['total_time'] / session_duration_hours) * 100):.1f}%" if session_duration_hours > 0 else '0%'])
        writer.writerow([])
    
    # Individual Appliance Analysis
    writer.writerow(['=== INDIVIDUAL APPLIANCE ANALYSIS ==='])
    writer.writerow(['Appliance', 'Name', 'Type', 'Zone', 'ON Time (Hours)', 'ON Time (Minutes)', 'Consumption (Wh)', 'Cost (₹)', 'Events', 'Efficiency'])
    
    for relay_name in ['relay1', 'relay2', 'relay3', 'relay4']:
        if relay_name in appliance_analysis:
            app_data = appliance_analysis[relay_name]
            time_hours = app_data['time_on'] / 3600
            time_minutes = app_data['time_on'] / 60
            
            # Calculate efficiency (time on vs session duration)
            efficiency = f"{((app_data['time_on'] / 3600) / session_duration_hours * 100):.1f}%" if session_duration_hours > 0 else "0%"
            
            writer.writerow([
                relay_name,
                app_data['name'],
                app_data['type'].title(),
                f"Zone {app_data['zone']}",
                f"{time_hours:.2f}",
                f"{time_minutes:.1f}",
                f"{app_data['consumption']:.2f}",
                f"{app_data['cost']:.2f}",
                app_data['events'],
                efficiency
            ])
    writer.writerow([])
    
    # Detailed Appliance Report
    writer.writerow(['=== DETAILED APPLIANCE REPORT ==='])
    writer.writerow(['Appliance', 'Name', 'ON Time (Hours)', 'ON Time (Minutes)', 'Consumption (Wh)', 'Cost (₹)', 'CO2 (kg)', 'Savings (Wh)', 'Savings Cost (₹)', 'Efficiency', 'Events'])
    
    for data in report_data:
        writer.writerow([
            data['appliance'],
            data['appliance_name'],
            f"{data['total_time_hours']:.2f}",
            f"{data['total_time_minutes']:.1f}",
            f"{data['consumption_wh']:.2f}",
            f"{data['cost_inr']:.2f}",
            f"{data['co2_kg']:.3f}",
            f"{data['savings_wh']:.2f}",
            f"{data['savings_cost']:.2f}",
            data['efficiency'],
            data['events_count']
        ])
    
    writer.writerow([])
    writer.writerow(['TOTALS', '', f"{sum(d['total_time_hours'] for d in report_data):.2f}", 
                     f"{sum(d['total_time_minutes'] for d in report_data):.1f}", 
                     f"{total_consumption:.2f}", f"{total_cost:.2f}", f"{total_co2:.3f}", 
                     f"{total_savings:.2f}", f"{total_savings_cost:.2f}", 
                     f"{((total_savings / (total_consumption + total_savings)) * 100):.1f}%" if (total_consumption + total_savings) > 0 else "0%", 
                     f"{sum(d['events_count'] for d in report_data)}"])
    
    writer.writerow([])
    
    # Smart Automation Impact Analysis
    writer.writerow(['=== SMART AUTOMATION IMPACT ANALYSIS ==='])
    writer.writerow(['Without Smart Automation (if all appliances were always ON):'])
    max_possible_consumption = session_duration_hours * (power_consumption['relay1'] + power_consumption['relay2'] + power_consumption['relay3'] + power_consumption['relay4'])
    max_possible_cost = max_possible_consumption * (energy_cost_per_kwh / 1000)
    writer.writerow(['  Maximum Possible Consumption:', f"{max_possible_consumption:.2f} Wh"])
    writer.writerow(['  Maximum Possible Cost:', f"₹{max_possible_cost:.2f}"])
    writer.writerow([])
    writer.writerow(['With Smart Automation:'])
    writer.writerow(['  Actual Consumption:', f"{total_consumption:.2f} Wh"])
    writer.writerow(['  Actual Cost:', f"₹{total_cost:.2f}"])
    writer.writerow([])
    writer.writerow(['Smart Automation Savings:'])
    writer.writerow(['  Energy Saved:', f"{total_savings:.2f} Wh"])
    writer.writerow(['  Cost Saved:', f"₹{total_savings_cost:.2f}"])
    writer.writerow(['  Percentage Saved:', f"{((total_savings / max_possible_consumption) * 100):.1f}%" if max_possible_consumption > 0 else '0%'])
    writer.writerow([])
    
    # Time Analysis Summary
    writer.writerow(['=== TIME ANALYSIS SUMMARY ==='])
    writer.writerow(['Session Duration:', f"{session_duration_hours:.2f} hours ({session_duration_hours*60:.1f} minutes)"])
    writer.writerow(['Total Light ON Time:', f"{type_analysis['lights']['time_on']/3600:.2f} hours ({type_analysis['lights']['time_on']/60:.1f} minutes)"])
    writer.writerow(['Total Fan ON Time:', f"{type_analysis['fans']['time_on']/3600:.2f} hours ({type_analysis['fans']['time_on']/60:.1f} minutes)"])
    writer.writerow(['Total Appliance ON Time:', f"{(type_analysis['lights']['time_on'] + type_analysis['fans']['time_on'])/3600:.2f} hours"])
    writer.writerow(['Average Light Usage:', f"{((type_analysis['lights']['time_on']/3600) / session_duration_hours * 100):.1f}%" if session_duration_hours > 0 else '0%'])
    writer.writerow(['Average Fan Usage:', f"{((type_analysis['fans']['time_on']/3600) / session_duration_hours * 100):.1f}%" if session_duration_hours > 0 else '0%'])
    writer.writerow([])
    
    # Cost Breakdown
    writer.writerow(['=== COST BREAKDOWN ==='])
    writer.writerow(['Light Cost:', f"₹{type_analysis['lights']['cost']:.2f} ({((type_analysis['lights']['cost']/total_cost)*100):.1f}%)" if total_cost > 0 else '₹0.00 (0%)'])
    writer.writerow(['Fan Cost:', f"₹{type_analysis['fans']['cost']:.2f} ({((type_analysis['fans']['cost']/total_cost)*100):.1f}%)" if total_cost > 0 else '₹0.00 (0%)'])
    writer.writerow(['Total Cost:', f"₹{total_cost:.2f}"])
    writer.writerow(['Cost per Hour:', f"₹{total_cost/session_duration_hours:.2f}" if session_duration_hours > 0 else '₹0.00'])
    writer.writerow([])
    
    # Recommendations
    writer.writerow(['=== ENERGY OPTIMIZATION RECOMMENDATIONS ==='])
    if type_analysis['lights']['time_on'] > type_analysis['fans']['time_on']:
        writer.writerow(['• Lights are used more than fans - consider LED bulbs for better efficiency'])
    else:
        writer.writerow(['• Fans are used more than lights - consider energy-efficient fan models'])
    
    if total_savings > 0:
        writer.writerow(['• Smart automation is working effectively - continue current setup'])
        writer.writerow(['• Consider extending automation to other areas'])
    else:
        writer.writerow(['• Consider optimizing zone boundaries for better detection'])
        writer.writerow(['• Review automation settings for better efficiency'])
    
    writer.writerow(['• Regular maintenance of detection system for optimal performance'])
    writer.writerow(['• Monitor usage patterns to identify optimization opportunities'])
    writer.writerow([])
    
    # Footer
    writer.writerow(['=== REPORT FOOTER ==='])
    writer.writerow(['Report Generated by Smart Classroom Automation System'])
    writer.writerow(['Energy Cost Rate:', f"₹{energy_cost_per_kwh} per kWh"])
    writer.writerow(['CO2 Emission Factor:', f"{co2_emission_factor} kg CO2 per kWh"])
    writer.writerow(['System Version:', 'Smart Zone Mapping 2.0'])
    writer.writerow(['Report Type:', 'Comprehensive Energy Analytics'])
    
    # Save to file
    filename = f"smart_classroom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as f:
        f.write(output.getvalue())
    
    return send_file(filename, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    print("Smart Classroom Automation Web Dashboard")
    print("=" * 50)
    print("Starting Flask application...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
