import cv2
import asyncio
from fastanpr import FastANPR
from flask import Flask, jsonify, request
import numpy as np
import re
import base64
 
app = Flask(__name__)
 
# Initialize FastANPR once

fast_anpr = FastANPR()
 
# --- Utility Functions (adapted from Azure OCR code) ---

def preprocess_roi(roi):
    scale_factor = 2  # Example value, adjust as needed
    kernel_size = (3, 3)  # Example value, adjust as needed
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_roi, (int(gray_roi.shape[1] * scale_factor), int(gray_roi.shape[0] * scale_factor)))
    enhanced = cv2.equalizeHist(resized)
    kernel = np.ones(kernel_size, np.uint8)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morph
 
plate_pattern = re.compile(r'[A-Z]{2}\s?\d{2}[A-Z]?[A-Z]?\d{4}', re.IGNORECASE)

def filter_license_plate(text):
    combined_text = ' '.join(text)
    combined_text = re.sub(r'[^A-Za-z0-9]', '', combined_text)
    match = plate_pattern.search(combined_text)
    return match.group(0).upper() if match else None
 
def normalize_plate_text(plate_text):
    if plate_text:
        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', plate_text)
        return cleaned_text.upper()
    return None
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}
 
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')
 
# --- Modified ANPR Processing Function ---

async def run_fast_anpr_from_image_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    number_plates = await fast_anpr.run([img_rgb])

    if number_plates and number_plates[0]:
        best_plate = max(number_plates[0], key=lambda plate: plate.rec_conf)
        return {
            "detection_box": best_plate.det_box,
            "detection_confidence": best_plate.det_conf,
            "recognition_text": best_plate.rec_text,
            "recognition_polygon": best_plate.rec_poly,
            "recognition_confidence": best_plate.rec_conf
        }
    else:
        return None
 
# --- Modified API Route for POST Request ---

@app.route('/api/anpr', methods=['POST'])
def anpr_api_trigger_post():
    if 'file' not in request.files:
        return jsonify({'statusCode': 400, 'message': 'No file part in the request', 'vehicleNumber': ''}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'statusCode': 400, 'message': 'No selected file', 'vehicleNumber': ''}), 400
    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_fast_anpr_from_image_bytes(image_bytes))
            loop.close()

            if result:

                # Here you might want to apply the preprocessing and filtering steps

                # on the detected license plate region if FastANPR provides that info.

                # For now, we'll directly use the recognition text.

                normalized_plate = normalize_plate_text(result.get("recognition_text"))
                if normalized_plate and len(normalized_plate) >= 8:
                    return jsonify({
                        'statusCode': 200,
                        'message': 'License plate detected',
                        'vehicleNumber': normalized_plate
                    }), 200
                else:
                    return jsonify({
                        'statusCode': 204,
                        'message': 'No valid license plate found in the detected region',
                        'vehicleNumber': ''
                    }), 204
            else:
                return jsonify({
                    'statusCode': 201,
                    'message': 'No license plates detected in the image',
                    'vehicleNumber': ''
                }), 201
 
        except Exception as e:
            return jsonify({'statusCode': 500, 'message': f'Error processing image: {e}', 'vehicleNumber': ''}), 500
    else:
        return jsonify({'statusCode': 400, 'message': 'Invalid file format. Only JPEG and PNG are allowed', 'vehicleNumber': ''}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=9095) # Using the same port as Azure for consistency
 