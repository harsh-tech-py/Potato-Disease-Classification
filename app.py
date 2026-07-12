from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# ---- Config (edit these if needed) ----
MODEL_PATH = "model/best_model.keras"
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 255

# ---- Load model once at startup ----
model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(file):
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img_array = prepare_image(file)
    predictions = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * float(np.max(predictions[0])), 2)

    return jsonify({
        'class': predicted_class,
        'confidence': confidence
    })


if __name__ == '__main__':
    app.run(debug=True)
