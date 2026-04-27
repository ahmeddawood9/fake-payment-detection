import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from fake_screenshot_detector.fake_screenshot_detector_ocr import FakeScreenshotDetector

app = Flask(__name__)
CORS(app)

# Initialize the detector
detector = FakeScreenshotDetector()

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Save the file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Analyze the image
    result = detector.analyze(file_path)
    
    # Clean up the file
    os.remove(file_path)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
