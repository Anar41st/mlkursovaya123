import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
from ultralytics import YOLO
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'image'
app.secret_key = 'your_secret_key'

model = load_model('models/carstf.h5')
model1 = load_model('models/TemyrCNN.h5')
model2 = YOLO('models/carsv8.pt')

classes = {
    0: 'Convertibel',
    1: 'Coupe',
    2: 'Hatchback',
    3: 'Sedan',
    4: 'SUV',
    5: 'Van'
}
classes2 = {
    0: 'Convertibel',
    1: 'Coupe',
    2: 'Hatchback',
    3: 'Sedan',
    4: 'SUV',
    5: 'Van'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict1', methods=['POST'])
def predict1():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет части файла'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Не выбран файл'})

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model.predict(x)
    class_idx = np.argmax(pred)

    class_name = classes[class_idx]

    result = {'class_name': class_name, 'probability': float(pred[0, class_idx])}
    return jsonify(result)

@app.route('/predict2', methods=['POST'])
def predict2():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет части файла'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Не выбран файл'})

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model1.predict(x)
    class_idx = np.argmax(pred)

    class_name = classes[class_idx]

    result = {'class_name': class_name, 'probability': float(pred[0, class_idx])}
    return jsonify(result)

@app.route('/predict3', methods=['POST'])
def predict3():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет части файла'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Не выбран файл'})

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    img = Image.open(img_path)
    results = model2(img)[0]
    boxes = results.boxes
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()

    detection_results = []
    for class_id, confidence, box in zip(class_ids, confidences, boxes):
        class_name = classes2[round(class_id)]
        x1, y1, x2, y2 = box.xyxy[0]
        detection_results.append({
            'class_name': class_name,
            'confidence': float(confidence),
            'box': {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            }
        })

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('arial.ttf', 15)
    for result in detection_results:
        class_name = result['class_name']
        confidence = result['confidence']
        box = result['box']
        draw.rectangle([(box['x1'], box['y1']), (box['x2'], box['y2'])], outline='red', width=2)
        draw.text((box['x1'], box['y1'] - 10), f'{class_name}: {confidence:.2f}', font=font, fill='red')

    upload_path = os.path.join('static/uploads', file.filename)
    img.save(upload_path)

    session['detection_results'] = detection_results
    session['file_name'] = file.filename

    return redirect(url_for('results'))

@app.route('/results')
def results():
    detection_results = session.get('detection_results', [])
    file_name = session.get('file_name', '')

    return render_template('results.html', detection_results=detection_results, file_name=file_name)

@app.route('/api/detect_base64', methods=['POST'])
def api_detect_base64():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Нет части изображения'}), 400

    image_data = data['image']
    image_data = image_data.replace('data:image/png;base64,', '')
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    results = model2(img)[0]
    boxes = results.boxes
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()

    detection_results = []
    for class_id, confidence, box in zip(class_ids, confidences, boxes):
        class_name = classes2[round(class_id)]
        x1, y1, x2, y2 = box.xyxy[0]
        detection_results.append({
            'class_name': class_name,
            'confidence': float(confidence),
            'box': {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            }
        })

    return jsonify(detection_results)

@app.route('/api/detect', methods=['POST'])
def api_detect_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет части файла'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Не выбран файл'}), 400

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    img = Image.open(img_path)
    results = model2(img)[0]
    boxes = results.boxes
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()

    detection_results = []
    for class_id, confidence, box in zip(class_ids, confidences, boxes):
        class_name = classes2[round(class_id)]
        x1, y1, x2, y2 = box.xyxy[0]
        detection_results.append({
            'class_name': class_name,
            'confidence': float(confidence),
            'box': {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            }
        })

    return jsonify(detection_results)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)