from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import os
from PIL import Image
import uuid

# Initialize Flask app
app = Flask(__name__)

# Load the optimized YOLOv8n model
model = YOLO("yolov8n.pt")  # nano model is faster

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return "🚀 YOLOv8 Object Detection API is running!"

@app.route("/detect", methods=["POST"])
def detect_objects():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    # Perform object detection
    results = model(image_path, imgsz=320, conf=0.25)

    # Extract detection results
    result = results[0]
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        detections.append({
            "class": class_name,
            "confidence": round(confidence, 3)
        })

    # Optionally, return detections and image
    return jsonify({
        "detections": detections
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
