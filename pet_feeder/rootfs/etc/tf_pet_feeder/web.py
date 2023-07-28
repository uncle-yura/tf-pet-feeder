import base64

from .utils import create_train_data, get_image_base64, get_next_filename, train_model
from .constants import *
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

if os.path.isfile(MODEL_PATH):
    model = load_model(MODEL_PATH)

@app.context_processor
def inject_view_context():
    return dict(ingress_entry=os.environ.get('INGRESS_ENTRY'))

@app.route("/")
def index():
    return render_template('index.html', messages=[IMAGE_DIR, WORK_DIR, OUTPUT_DIR, MODEL_PATH])

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = image[Y_OFFSET : Y_OFFSET + HEIGHT, X_OFFSET : X_OFFSET + WIDTH]
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    predicted_class_name = CLASSES[predicted_class]

    response = {'predicted_class': predicted_class_name}

    return jsonify(response)

@app.route("/train", methods=['GET', 'POST'])
def train():
    message = ""

    if request.method == 'POST':
        value = request.form['button']
        try:
            if value == "create_data":
                result = create_train_data()
                message = f"Train images: {result.get('train_images_shape')},  Test images: {result.get('test_images_shape')}"
            elif value == "train_model":
                train_model()
                message = "Train completed"
            elif value == "test_model":
                test_images = np.load(TEST_IMAGES_PATH)
                test_labels = np.load(TEST_LABELS_PATH)

                test_loss, test_acc = model.evaluate(test_images, test_labels)
                message = f"Test accuracy: {test_acc}, Test loss: {test_loss}"
        except Exception as e:
            message = str(e)

        return jsonify({"message": message})

    return render_template('train.html', message=message)

@app.route('/sort', methods=['GET', 'POST'])
def sort():
    image_file = get_next_filename()

    if request.method == 'POST' and image_file:
        button_name = request.form['button']

        source_path = os.path.join(IMAGE_DIR, image_file)
        target_path = os.path.join(WORK_DIR, button_name, image_file)

        os.makedirs(os.path.join(WORK_DIR, button_name), exist_ok=True)
        os.rename(source_path, target_path)

        image_file = get_next_filename()

    image_base64 = get_image_base64(image_file)

    return render_template('sort.html', image_base64=image_base64, buttons=CLASSES)
