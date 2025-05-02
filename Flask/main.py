from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import tempfile

app = Flask(__name__)

# Load YOLO model once at startup
MODEL_PATH = "yolov11_leafsnap.pt"
model = YOLO(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)
            image_path = temp_file.name

        # Run YOLO prediction
        results = model.predict(image_path)

        if not results or not results[0].boxes:
            return jsonify({'prediction': 'No leaf detected'}), 200

        # Extract prediction info
        box = results[0].boxes[0]
        cls_id = int(box.cls)
        predicted_class = model.names[cls_id]
        confidence = float(box.conf)

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
