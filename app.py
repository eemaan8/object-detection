from flask import Flask, request
from ultralytics import YOLO
import os
from collections import Counter

app = Flask(__name__)

# Ensure model file exists
MODEL_PATH = "yolov8s.pt"
if not os.path.exists(MODEL_PATH):
    import requests
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    r = requests.get(url)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

@app.route('/')
def home():
    return "ClearSight Object Detection API is running!"

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image file provided", 400

    image_file = request.files['image']
    if image_file.filename == '':
        return "Empty filename", 400

    temp_path = "temp.jpg"
    image_file.save(temp_path)

    results = model(temp_path)
    os.remove(temp_path)

    detected_classes = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    class_counts = Counter(detected_classes)

    if not class_counts:
        return "No objects detected"

    readable_output = ", ".join(
        [f"{count} {name if count == 1 else name + 's'}" for name, count in class_counts.items()]
    )

    return f"Detected: {readable_output}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Compatible with Render
    app.run(host="0.0.0.0", port=port)
