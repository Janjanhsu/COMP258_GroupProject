'''
pip install Flask tensorflow
python app.py
http://localhost:5000
'''
from flask import Flask, request, render_template
import base64
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('my_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    image = Image.open(file.stream)
    prediction = predict_image(image)
    print("Upload prediction: ", prediction)
    return f'Prediction: {prediction}'

@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    prediction = predict_image(image)
    print("Capture prediction: ", prediction)
    return f'Prediction: {prediction}'



def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((48, 48))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]

if __name__ == '__main__':
    app.run(debug=True)
