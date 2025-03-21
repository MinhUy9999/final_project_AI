from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, decode_predictions
import requests
import base64

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Khởi tạo biến model là None, sẽ tải khi cần
model = None

# Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_HEADERS = {"Authorization": "Bearer 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"}

def load_model():
    """Tải mô hình MobileNet khi cần"""
    global model
    if model is None:
        print("Loading MobileNet with ImageNet weights...")
        model = MobileNet(weights='imagenet')

@app.route('/')
def serve_index():
    """Phục vụ index.html cho route gốc"""
    return send_from_directory('.', 'index.html')

@app.route('/cart.html')
def serve_cart():
    """Phục vụ cart.html"""
    return send_from_directory('.', 'cart.html')

@app.route('/assets/<path:path>')
def serve_static(path):
    """Phục vụ file tĩnh từ thư mục assets"""
    return send_from_directory('assets', path)

@app.route('/classify', methods=['POST'])
def classify_image():
    """Phân loại hình ảnh với MobileNet"""
    load_model()
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']
    try:
        img = tf.image.decode_image(image.read(), channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0
        prediction = model.predict(img[tf.newaxis, ...])
        decoded = decode_predictions(prediction, top=1)[0][0]
        
        category = decoded[1]
        confidence = float(decoded[2])
        description = f"Đây là một {category} chất lượng cao, được dự đoán với độ tin cậy {confidence:.2f}."
        
        image.seek(0)
        image_base64 = base64.b64encode(image.read()).decode('utf-8')
        
        print(f"Predicted: {category} (Confidence: {confidence:.2f})")
        
        del img
        del prediction
        
        return jsonify({
            'category': category,
            'confidence': confidence,
            'description': description,
            'image': f"data:image/jpeg;base64,{image_base64}"
        })
    except Exception as e:
        print(f"Error in classify: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reviews', methods=['POST'])
def analyze_reviews():
    """Phân tích cảm xúc của nhận xét bằng Hugging Face API"""
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    payload = {"inputs": text}
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        sentiment = max(result[0], key=lambda x: x['score'])['label']
        sentiment_text = 'Vui' if sentiment == 'POSITIVE' else 'Không vui'
        return jsonify({'sentiment': sentiment_text})
    except Exception as e:
        print(f"Error in reviews: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)