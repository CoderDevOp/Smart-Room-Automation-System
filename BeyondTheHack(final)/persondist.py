import cv2
import numpy as np
import time
import serial
from cvzone.FaceMeshModule import FaceMeshDetector

# Setup
min_confidence = 0.6
nms_threshold = 0.4
detector = FaceMeshDetector(maxFaces=1)
cap = cv2.VideoCapture(0)
arduino = serial.Serial('COM9', 9600, timeout=1)
time.sleep(2)

# YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to apply Non-Maximum Suppression (NMS) to the bounding boxes
def apply_nms(boxes, confidences, img_width, img_height):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, nms_threshold)
    if len(indexes) > 0:
        # Ensure indexes is a flat list of integers
        if isinstance(indexes[0], (list, np.ndarray)):
            indexes = [idx[0] for idx in indexes]
        filtered_boxes = [boxes[i] for i in indexes]
        return filtered_boxes
    return []

# Step 1: Calibration (YOLO-based)
# This step calibrates the camera's focal length using a known real-world person width.
# The user places a person at a known distance, and the system calculates the focal length
# based on the person's detected width in pixels.
print("\n--- Calibration ---")
print("Place a person at a known distance (e.g., 100 cm) from the camera.")
print("Press 'q' when the person's bounding box is stable and visible.")
REAL_PERSON_WIDTH_CM = 45  # Average shoulder width of a person in cm
focal_length_yolo = None
known_distance_cm = 100  # Example known distance for calibration

while True:
    success, img = cap.read()
    if not success:
        break
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
    
    # Apply NMS to filter out overlapping bounding boxes
    person_boxes_filtered = apply_nms(boxes, confidences, width, height)
    if person_boxes_filtered:
        # Take the largest bounding box for calibration to ensure it's the primary person
        x, y, w, h = max(person_boxes_filtered, key=lambda b: b[2])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"Width: {w} px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Calculate focal length using the pinhole camera model formula
        focal_length_yolo = (w * known_distance_cm) / REAL_PERSON_WIDTH_CM
        cv2.putText(img, f"Focal Length: {int(focal_length_yolo)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, "Press 'q' when stable", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(img, "No person detected for calibration", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("YOLO Calibration", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
if not focal_length_yolo:
    print("❌ Calibration failed. Focal length not determined.")
    exit()
print(f"✅ Calibration complete. Focal Length: {int(focal_length_yolo)}")

# Step 2: Zone Setup
# This step allows the user to define multiple zones based on distance from the camera.
# Each zone is defined by a start and end distance (in cm).
num_zones = 2 # Updated to 2 zones (Zone 1, Zone 2)
zones = []
current_zone = {}
print("\n--- Zone Setup ---")
print("Press 's' = start, 'e' = end, 'q' = finish setup")
print("Zone 1: Light (Relay 1) + Fan (Relay 3)")
print("Zone 2: Light (Relay 2) + Fan (Relay 4)")

# Function to estimate distance from the camera using YOLO bounding box width
def estimate_distance_from_yolo(box_width_pixels):
    if focal_length_yolo and box_width_pixels > 0:
        # Distance = (Real_World_Width * Focal_Length) / Pixel_Width
        return (REAL_PERSON_WIDTH_CM * focal_length_yolo) / box_width_pixels
    return 0

while True:
    success, img = cap.read()
    if not success: break

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
        # Take the largest bounding box after NMS for distance estimation
        x, y, w, h = max(person_boxes_filtered, key=lambda b:b[2])
        dist = estimate_distance_from_yolo(w)
        cv2.putText(img, f"{int(dist)} cm", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    else:
        dist = 0 # No person detected

    cv2.putText(img, f"Zones: {len(zones)}/{num_zones}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(img, "S=start, E=end, Q=quit", (30,130), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.imshow("Zone Setup", img)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('s') and dist > 0: # Ensure person is detected for zone start
        current_zone["start"] = dist
        print(f"✅ Zone {len(zones)+1} START: {int(dist)} cm")
    elif key==ord('e') and "start" in current_zone and dist > 0:
        current_zone["end"] = dist
        zones.append(current_zone.copy())
        print(f"✅ Zone {len(zones)} recorded: {int(current_zone['start'])}-{int(current_zone['end'])} cm")
        current_zone.clear()
    elif key==ord('q') or len(zones)==num_zones:
        break

cv2.destroyAllWindows()
print(f"✅ Zones defined: {zones}")

# Step 3: Real-time Person Detection, Zone Mapping, and Arduino Control
# This loop continuously detects persons, estimates their distance, assigns them to zones,
# and controls electrical appliances via Arduino relays based on zone occupancy and predefined delays.

relay_state = {'relay1':False,'relay2':False,'relay3':False,'relay4':False}
fan_timer = [None]*num_zones
no_person_timer = None
margin = 7 # cm tolerance for zone detection
activation_delay = 4 # seconds before fan (relay3/relay4) activates after person detection
turn_off_delay = 3 # seconds before all relays turn off when no person is detected

# Appliance tracking for timestamps and consumption
appliance_timestamps = {
    'relay1': {'start': None, 'end': None, 'total_time': 0},  # Zone 1 Light
    'relay2': {'start': None, 'end': None, 'total_time': 0},  # Zone 2 Light
    'relay3': {'start': None, 'end': None, 'total_time': 0},  # Zone 1 Fan
    'relay4': {'start': None, 'end': None, 'total_time': 0}   # Zone 2 Fan
}

# Power consumption in watts (dummy values)
power_consumption = {
    'relay1': 60,   # Zone 1 Light: 60W
    'relay2': 60,   # Zone 2 Light: 60W
    'relay3': 75,   # Zone 1 Fan: 75W
    'relay4': 75    # Zone 2 Fan: 75W
}

# Zone consumption tracking
zone_consumption = {
    'zone1': {'light_time': 0, 'fan_time': 0, 'total_consumption': 0},
    'zone2': {'light_time': 0, 'fan_time': 0, 'total_consumption': 0}
}

# Helper functions for timestamp and consumption tracking
def update_appliance_timestamp(relay_name, is_on):
    """Update appliance timestamps when state changes"""
    current_time = time.time()
    
    if is_on and appliance_timestamps[relay_name]['start'] is None:
        # Appliance turning ON
        appliance_timestamps[relay_name]['start'] = current_time
        appliance_timestamps[relay_name]['end'] = None
        print(f"🟢 {relay_name} turned ON at {time.strftime('%H:%M:%S', time.localtime(current_time))}")
        
    elif not is_on and appliance_timestamps[relay_name]['start'] is not None:
        # Appliance turning OFF
        appliance_timestamps[relay_name]['end'] = current_time
        duration = current_time - appliance_timestamps[relay_name]['start']
        appliance_timestamps[relay_name]['total_time'] += duration
        print(f"🔴 {relay_name} turned OFF at {time.strftime('%H:%M:%S', time.localtime(current_time))} (Duration: {duration:.1f}s)")
        
        # Update zone consumption
        if relay_name in ['relay1', 'relay3']:  # Zone 1 appliances
            if relay_name == 'relay1':
                zone_consumption['zone1']['light_time'] += duration
            else:
                zone_consumption['zone1']['fan_time'] += duration
        elif relay_name in ['relay2', 'relay4']:  # Zone 2 appliances
            if relay_name == 'relay2':
                zone_consumption['zone2']['light_time'] += duration
            else:
                zone_consumption['zone2']['fan_time'] += duration
        
        # Reset for next cycle
        appliance_timestamps[relay_name]['start'] = None
        appliance_timestamps[relay_name]['end'] = None

def calculate_zone_consumption():
    """Calculate and display electricity consumption for each zone"""
    print("\n" + "="*50)
    print("📊 ELECTRICITY CONSUMPTION REPORT")
    print("="*50)
    
    for zone_name, zone_data in zone_consumption.items():
        print(f"\n{zone_name.upper()}:")
        
        # Calculate consumption for each appliance
        if zone_name == 'zone1':
            light_time = zone_data['light_time']
            fan_time = zone_data['fan_time']
            light_consumption = (light_time / 3600) * power_consumption['relay1']  # Convert to hours, multiply by watts
            fan_consumption = (fan_time / 3600) * power_consumption['relay3']
        else:
            light_time = zone_data['light_time']
            fan_time = zone_data['fan_time']
            light_consumption = (light_time / 3600) * power_consumption['relay2']
            fan_consumption = (fan_time / 3600) * power_consumption['relay4']
        
        total_consumption = light_consumption + fan_consumption
        
        print(f"  💡 Light: {light_time:.1f}s ({light_consumption:.3f} Wh)")
        print(f"  🌪  Fan: {fan_time:.1f}s ({fan_consumption:.3f} Wh)")
        print(f"  ⚡ Total: {total_consumption:.3f} Wh")
        
        # Update total consumption
        zone_consumption[zone_name]['total_consumption'] = total_consumption
    
    print("="*50)

print("\n🚀 Running Smart Detection...")
print("Press 'C' to show electricity consumption report")
print("Press 'Q' to quit")

while True:
    success, img = cap.read()
    if not success: break

    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0,(416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>min_confidence and classes[class_id]=="person":
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

    new_relay={'relay1':False,'relay2':False,'relay3':False,'relay4':False}
    zone_text="No person"
    detected_in_any_zone = False
    active_zones = [False] * num_zones # Track which zones are active

    if person_boxes_filtered:
        # Iterate through all detected persons
        for x,y,w,h in person_boxes_filtered:
            distance=estimate_distance_from_yolo(w)
            current_person_zone_num = None

            # Check all defined zones to see if the person is within any of them
            for i, z in enumerate(zones):
                if (z['start']-margin) <= distance <= (z['end']+margin):
                    current_person_zone_num=i
                    active_zones[i] = True # Mark this zone as active
                    detected_in_any_zone = True
                    break
            
            # Draw bounding box and distance for each person
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, f"{int(distance)} cm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        if detected_in_any_zone:
            no_person_timer = None # Reset the no-person timer as a person is detected

            # Update relay states based on active zones
            for i in range(num_zones):
                if active_zones[i]:
                    if i == 0: # Zone 1
                        new_relay['relay1'] = True # Light for Zone 1
                        if fan_timer[0] is None: fan_timer[0] = time.time()
                        if (time.time() - fan_timer[0]) >= activation_delay: new_relay['relay3'] = True # Fan for Zone 1
                    elif i == 1: # Zone 2
                        new_relay['relay2'] = True # Light for Zone 2
                        if fan_timer[1] is None: fan_timer[1] = time.time()
                        if (time.time() - fan_timer[1]) >= activation_delay: new_relay['relay4'] = True # Fan for Zone 2
                else: # If a zone is not active, reset its fan timer
                    fan_timer[i] = None
            
            # Construct zone_text to show all active zones
            active_zone_names = [f"Zone {j+1}" for j, active in enumerate(active_zones) if active]
            if active_zone_names:
                zone_text = ", ".join(active_zone_names)
            else:
                zone_text = "No person"
     
    # If no person is detected in any defined zone, start/continue the no-person timer
    if not detected_in_any_zone:
        zone_text = "No person"
        if no_person_timer is None: no_person_timer = time.time()
        # If no person has been detected for 'turn_off_delay' seconds, turn off all relays
        if (time.time() - no_person_timer) >= turn_off_delay:
            fan_timer = [None] * num_zones # Reset all fan timers
            new_relay = {'relay1':False,'relay2':False,'relay3':False,'relay4':False} # Turn off all relays

    cv2.putText(img, zone_text, (30,60), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    
    # Display appliance status on screen with proper names
    relay_display_y_offset = 90
    appliance_names = {
        'relay1': 'Zone 1 Light',
        'relay2': 'Zone 2 Light', 
        'relay3': 'Zone 1 Fan',
        'relay4': 'Zone 2 Fan'
    }
    
    for j, (relay_name, state) in enumerate(new_relay.items()):
        status = "ON" if state else "OFF"
        color = (0,255,0) if state else (0,0,255)
        appliance_name = appliance_names[relay_name]
        cv2.putText(img, f"{appliance_name}: {status}", (30, relay_display_y_offset + j * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Update timestamps when relay states change
    for relay_name in new_relay:
        if new_relay[relay_name] != relay_state[relay_name]:
            update_appliance_timestamp(relay_name, new_relay[relay_name])
    
    # Send relay state to Arduino (active-low configuration)
    # A command byte is constructed where each bit corresponds to a relay.
    # If a relay should be OFF (False in new_relay), its corresponding bit is set to 1.
    # If a relay should be ON (True in new_relay), its corresponding bit is set to 0.
    if new_relay!=relay_state:
        command=0
        command |= (0 if new_relay['relay1'] else 1)<<0
        command |= (0 if new_relay['relay2'] else 1)<<1
        command |= (0 if new_relay['relay3'] else 1)<<2
        command |= (0 if new_relay['relay4'] else 1)<<3
        arduino.write(bytes([command]))
        relay_state=new_relay

    cv2.imshow("Smart Zone Detection", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c') or key == ord('C'):
        calculate_zone_consumption()

cap.release()
cv2.destroyAllWindows()
arduino.close()