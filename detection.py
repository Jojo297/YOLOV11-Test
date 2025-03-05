import cv2
import torch
from ultralytics import YOLO
import numpy as np
from flask import Flask
from flask_socketio import SocketIO

# Load YOLO model
model11 = YOLO("yolo11n.pt")

# Flask WebSocket server
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def process_webcam():
    cap = cv2.VideoCapture(0)  # 0 = default laptop webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model11(frame)  # Inference
        result = results[0]  # Get the first result object

        detections = []

        if result.boxes is not None:  # Check if there are any boxes
            for obj in result.boxes.data:
                x1, y1, x2, y2, conf, cls = obj.tolist()
                label = model11.names[int(cls)]
                detections.append({"label": label, "confidence": conf})

                # Draw bounding boxes
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (98, 136, 242), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Send detections via WebSocket
        socketio.emit("clove_detection", detections)

        # Display processed frame
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/")
def index():
    return "YOLO Server Running"

if __name__ == "__main__":
    socketio.start_background_task(process_webcam)
    socketio.run(app, host="0.0.0.0", port=5000)
