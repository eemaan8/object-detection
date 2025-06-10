from flask import Flask, request
from ultralytics import YOLO
import os
from collections import Counter
import requests

app = Flask(__name__)

# Model settings
MODEL_PATH = "yolov8s.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv8s model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

# Load model
model = YOLO(MODEL_PATH)

@app.route('/')
def home():
    return "ClearSight Object Detection API is running!"

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image file provided", 400

    image = request.files['image']
    if image.filename == '':
        return "Empty filename", 400

    temp_path = "temp.jpg"
    image.save(temp_path)

    try:
        results = model(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return f"Detection error: {str(e)}", 500

    os.remove(temp_path)

    # Extract detected class names
    detected_classes = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names.get(class_id, f"class_{class_id}")
            detected_classes.append(class_name)

    if not detected_classes:
        return "No objects detected."

    # Count and format
    class_counts = Counter(detected_classes)
    output = ", ".join([
        f"{count} {name}" + ("s" if count > 1 else "")
        for name, count in class_counts.items()
    ])

    return f"Detected: {output}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
