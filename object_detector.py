from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# Load model once, globally
model = YOLO("best_carries.pt")

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file", passes it
    through YOLOv8 object detection network and returns an array
    of bounding boxes.
    :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes, num_caries = detect_objects_on_image(buf.stream)
    return jsonify({"boxes": boxes, "num_caries": num_caries})

def detect_objects_on_image(buf):
    """
    Function receives an image,
    resizes it to 1080x1080,
    passes it through YOLOv8 neural network,
    and returns an array of detected objects
    and their bounding boxes.
    :param buf: Input image file stream
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    img = Image.open(buf).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (1080, 1080))  # Resize the image to 1080x1080
    img = Image.fromarray(img)
    results = model.predict(img)
    result = results[0]
    output = []
    num_caries = 0
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        if class_name.lower() == 'caries':
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            prob = round(box.conf[0].item(), 2)
            output.append([x1, y1, x2, y2, class_name, prob])
            num_caries += 1
    return output, num_caries
if __name__ == '__main__':
    app.run(debug=True)

serve(app, host='0.0.0.0', port=8080)
