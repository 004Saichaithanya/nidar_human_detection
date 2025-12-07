#!/usr/bin/env python3
import cv2
import time
import numpy as np
from ultralytics import YOLO
from pymavlink import mavutil

# -----------------------------
#  INITIALIZATION
# -----------------------------

print("\n🚀 Starting Person Detection System ...\n")

# Load YOLO Model
try:
    model = YOLO("best.onnx")     # Fast ONNX model (recommended)
    print("✔ Loaded ONNX model (fast mode)")
except:
    model = YOLO("best.pt")       # Fallback
    print("✔ Loaded PyTorch model (fallback mode)")

# Connect to Pixhawk
print("⌛ Waiting for Pixhawk heartbeat...")
master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=921600)
master.wait_heartbeat()
print("✔ Connected to Pixhawk")

# Open Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Camera not detected!")
    exit()

print("✔ Camera Stream Active")

# Resize for speed
INFER_W, INFER_H = 320, 240
SKIP_RATE = 2  # Process every 2nd frame

# Heatmap
heatmap = None
decay_rate = 0.92
frame_index = 0

# -----------------------------
#  MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Camera frame missing")
        break

    frame_index += 1
    if frame_index % SKIP_RATE != 0:
        continue

    # Resize for faster inference
    small = cv2.resize(frame, (INFER_W, INFER_H))

    # Heatmap init
    if heatmap is None:
        heatmap = np.zeros((INFER_H, INFER_W), dtype=np.float32)

    # YOLO Inference
    results = model(small, verbose=False)

    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if cls == 0 and conf >= 0.60:  # PERSON
                person_count += 1

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Draw detection box
                cv2.rectangle(small, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(small, f"Person {conf:.2f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Heatmap update
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(heatmap, (cx, cy), 12, 255, -1)

                # ------------------------------------------------------------------
                #  FETCH GPS FROM PIXHAWK
                # ------------------------------------------------------------------
                gps_msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)

                if gps_msg:
                    lat = gps_msg.lat / 1e7
                    lon = gps_msg.lon / 1e7
                    alt = gps_msg.alt / 1000.0

                    print(f"🟢 PERSON DETECTED | "
                          f"Lat: {lat:.7f}, Lon: {lon:.7f}, Alt: {alt:.1f} m | "
                          f"Conf: {conf:.2f}")

                    # TODO → Send custom MAVLink message here (if required)
                    # master.mav.my_custom_message_send(...)

    # Heatmap fade-out
    heatmap *= decay_rate
    heatmap_img = cv2.applyColorMap(cv2.convertScaleAbs(heatmap), cv2.COLORMAP_JET)

    # Overlay heatmap
    output = cv2.addWeighted(small, 0.7, heatmap_img, 0.3, 0)

    # Save last frame (for debugging on Pi when no display is available)
    cv2.imwrite("/home/pi/last_detection.jpg", output)

    # Show live feed if GUI exists
    try:
        cv2.imshow("Raspberry Pi YOLO Person Detection", output)
    except:
        pass  # Pi headless mode (no GUI)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
#  CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("\n🛑 Detection Stopped\n")
