import cv2 as cv
import numpy as np
from ultralytics import YOLO

def start_detection(source='data/crowd_vid.mp4', output_file="output.mp4", skip_rate=3):
    """
    Real-time crowd detection with YOLOv8 + heatmap overlay.
    Shows live window and saves output to a video file.
    Optimized for Raspberry Pi (slightly faster style).
    """
    # Open video/camera
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Could not open video source: {source}")
        return

    # Load YOLO model
    try:
        model = YOLO('best.onnx')  # lightweight YOLO
        print('used onnx')
    except Exception:
        model = YOLO('best.pt')  # fallback
        print('used yolo')

    heatmap = None
    frame_count = 0

    # Get input FPS (webcam or video)
    input_fps = cap.get(cv.CAP_PROP_FPS)
    if input_fps == 0 or np.isnan(input_fps):
        input_fps = 20.0

    # Slightly faster video output (because of frame skipping)
    output_fps = input_fps / skip_rate

    # Prepare video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_file, fourcc, output_fps, (800, 600))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera disconnected.")
            break

        frame_count += 1
        if frame_count % skip_rate != 0:
            continue  # skip frames to reduce load

        # Resize for faster inference
        frame_resized = cv.resize(frame, (320, 240))
        h, w = frame_resized.shape[:2]
        if heatmap is None:
            heatmap = np.zeros((h, w), dtype=np.float32)

        # YOLO detection
        results = model(frame_resized, verbose=False)
        person_count = 0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i, c in enumerate(classes):
                if int(c) == 0:  # person
                    person_count += 1
                    x1, y1, x2, y2 = boxes[i]
                    cv.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv.putText(frame_resized, 'Person', (int(x1), int(y1)-5),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Update heatmap
                    cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                    cv.circle(heatmap, (cx, cy), 20, 255, -1)

        # Heatmap overlay
        heatmap_img = cv.applyColorMap(cv.convertScaleAbs(heatmap, alpha=0.3), cv.COLORMAP_JET)
        overlayed = cv.addWeighted(frame_resized, 0.7, heatmap_img, 0.3, 0)

        # Draw people count
        cv.putText(overlayed, f'People Count: {person_count}', (15, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Resize for display/output
        display_frame = cv.resize(overlayed, (800, 600))

        # Show live video
        cv.imshow("YOLO Crowd Detection", display_frame)

        # Save to video
        out.write(display_frame)

        # Exit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    print(f"✅ Output video saved as '{output_file}'")

# Example usage
start_detection(output_file="crowd_output.mp4")