from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Export to ONNX
model.export(format="onnx")
