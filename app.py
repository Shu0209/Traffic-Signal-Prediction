from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

# Preprocessing Functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

# Class Names (43 classes - GTSRB)
def getClassName(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vechiles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vechiles',
        'Vechiles over 3.5 metric tons prohibited', 'No entry', 'General caution',
        'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve',
        'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
        'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
        'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
        'End of no passing', 'End of no passing by vechiles over 3.5 metric tons'
    ]
    return class_names[classNo]

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    return getClassName(classIndex)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)
        preds = model_predict(file_path, model)
        return preds
    return None

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(port=5001, debug=True)