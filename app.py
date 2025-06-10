# from flask import Flask, request, jsonify
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import base64
# from PIL import Image
# import io

# app = Flask(__name__)
# model = YOLO("yolov8s.pt")  # Pre-trained COCO model

# def readb64(base64_string):
#     imgdata = base64.b64decode(base64_string)
#     image = Image.open(io.BytesIO(imgdata))
#     return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# @app.route('/detect', methods=['POST'])
# def detect():
#     data = request.json
#     image_b64 = data.get("image")
#     if not image_b64:
#         return jsonify({"error": "No image data provided"}), 400

#     img = readb64(image_b64)
#     results = model(img)[0]

#     detections = []
#     for box in results.boxes:
#         cls = int(box.cls[0])
#         conf = float(box.conf[0])
#         xyxy = box.xyxy[0].tolist()
#         detections.append({
#             "class": model.names[cls],
#             "confidence": round(conf, 2),
#             "bbox": [round(x) for x in xyxy]
#         })

#     return jsonify({"detections": detections})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)
from flask import Flask, request
from ultralytics import YOLO
import os
from collections import Counter

app = Flask(__name__)
model = YOLO("yolov8s.pt")  # You can change to yolov8n.pt if needed

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

    # Format like: "2 persons, 1 chair, 1 laptop"
    readable_output = ", ".join(
        [f"{count} {name if count == 1 else name + 's'}" for name, count in class_counts.items()]
    )

    return f"Detected {readable_output}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
