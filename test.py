import requests
import base64

# Load an image (place any test image in the same folder)
with open("test8.jpg", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode("utf-8")

# Send POST request
response = requests.post("http://127.0.0.1:10000/detect", json={"image": encoded})

# Print response
print(response.json())
