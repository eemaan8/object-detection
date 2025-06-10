from flask import Flask, request
from ultralytics import YOLO
import os
from collections import Counter

app = Flask(__name__)

# Load YOLOv8 model (use 'yolov8n.pt' for lighter model if needed)
model = YOLO("yolov8s.pt")

@app.route('/')
def home():
    return "ClearSight Object Detection API is running!"

@app.route('/detect', methods=['POST'])
def detect():
    # Check if image is sent in the request
    if 'image' not in request.files:
        return "No image file provided", 400

    image_file = request.files['image']
    if image_file.filename == '':
        return "Empty filename", 400

    # Save image temporarily
    temp_path = "temp.jpg"
    image_file.save(temp_path)

    # Run YOLO object detection
    results = model(temp_path)
    os.remove(temp_path)  # Clean up temp file

    # Process detections
    detected_classes = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    class_counts = Counter(detected_classes)

    if not class_counts:
        return "No objects detected"

    # Format output: "2 persons, 1 chair"
    readable_output = ", ".join(
        [f"{count} {name if count == 1 else name + 's'}" for name, count in class_counts.items()]
    )

    return f"Detected: {readable_output}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # For Render compatibility
    app.run(host="0.0.0.0", port=port)
