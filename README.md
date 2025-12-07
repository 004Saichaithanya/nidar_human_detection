# 📘 **README — Raspberry Pi YOLOv8 Person Detection + MAVLink Integration**

This project enables **real-time human detection** on a Raspberry Pi using a **YOLOv8 ONNX model** and sends **GPS-tagged MAVLink messages** to Pixhawk for autonomous drone missions.

Compatible with:

* Raspberry Pi 4 / 5
* Pixhawk (ArduPilot / ArduPlane / ArduCopter)
* USB Camera / CSI Camera
* Python 3.9+

---

## 🚀 **1. Raspberry Pi Setup**

### 🔹 Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 🔹 Install Required Python Components

```bash
sudo apt install python3 python3-pip python3-venv -y
```

### 🔹 Install Camera Support

**USB Camera:**

```bash
sudo apt install v4l-utils -y
```

**CSI Camera:**

```bash
sudo apt install libcamera-apps -y
```

---

## 🎥 **2. Install OpenCV**

```bash
sudo apt install python3-opencv -y
```

Verify installation:

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

---

## 🤖 **3. Install AI + MAVLink Dependencies**

Install everything in one command:

```bash
pip3 install numpy ultralytics onnxruntime pymavlink mavproxy
```

---

## ⚡ **4. Export YOLOv8 Model to ONNX**

Run this on your **PC/Laptop**, not Raspberry Pi:

### Create a file `export_onnx.py`:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx")
```

Run:

```bash
python3 export_onnx.py
```

This generates:

```
best.onnx
```

Copy `best.onnx` to Raspberry Pi:

```bash
scp best.onnx pi@<pi-ip>:/home/pi/
```

---

## 🛰️ **5. Pixhawk → Raspberry Pi Connection**

### 🔹 Wiring (TELEM2 → USB-TTL → Raspberry Pi)

| Pixhawk TELEM2 | USB-TTL → Pi USB |
| -------------- | ---------------- |
| TX             | RX               |
| RX             | TX               |
| GND            | GND              |

**Baudrate must be:** `921600`

---

## 🧪 **6. Test MAVLink Connection**

Run:

```bash
mavproxy.py --master=/dev/ttyUSB0 --baudrate 921600
```

Expected:

```
Received HEARTBEAT
APM: ArduPlane ...
```

If this works → Pixhawk communication is OK.

---

## 🧭 **7. Test Pixhawk GPS Stream**

Create `test_gps.py`:

```python
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=921600)
master.wait_heartbeat()
print("Connected to Pixhawk")

while True:
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    print(msg.lat/1e7, msg.lon/1e7, msg.alt/1000)
```

Run:

```bash
python3 test_gps.py
```

---

## 👁️ **8. Run Final Detection Script**

Place your script as:

```
/home/pi/person_detect.py
```

Run:

```bash
python3 person_detect.py
```

Features:

* Real-time YOLOv8 inference
* Heatmap overlay
* GPS fusion
* Prints detection with Lat/Lon/Alt
* Saves latest frame at:
  `/home/pi/last_detection.jpg`

Expected output:

```
🟢 PERSON DETECTED | Lat: 17.49382, Lon: 78.39193, Alt: 4.2 m | Conf: 0.89
```

---

## 📡 **9. Sending Detection Data to Mission Planner**

Inside your detection code, add:

```python
master.mav.statustext_send(
    6,
    f"PERSON {lat:.6f},{lon:.6f}".encode()
)
```

This appears in **Mission Planner → Messages**.

---

## 🧩 **10. Full Package List (Summary)**

| Package        | Purpose                   |
| -------------- | ------------------------- |
| python3        | Interpreter               |
| pip3           | Package manager           |
| python3-opencv | Camera + image processing |
| v4l-utils      | USB camera                |
| libcamera-apps | CSI camera                |
| numpy          | Array operations          |
| ultralytics    | YOLOv8                    |
| onnxruntime    | Fast ONNX inference       |
| pymavlink      | MAVLink communication     |
| mavproxy       | MAVLink debugging         |

---

## 📝 **11. Checklist Before Flight**

### ✔ Pi detects camera

### ✔ Pixhawk heartbeat received

### ✔ GPS messages printing

### ✔ YOLO detects a person

### ✔ Printed output contains Lat/Lon

### ✔ Mission Planner receives MAVLink messages

### ✔ Heatmap and last_detection.jpg update

### ✔ Pi does not overheat (below 75°C recommended)

If all are ✔ → system is ready for on-drone testing.
