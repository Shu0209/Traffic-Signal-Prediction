import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Camera Setup
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load Model
model = load_model("model.h5")

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

# Class Names
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

while True:
    success, imgOrignal = cap.read()
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # Predict
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        cv2.putText(imgOrignal, "CLASS: " + str(classIndex) + " " + getClassName(classIndex),
                    (120, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%",
                    (180, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(imgOrignal, "No Sign Detected", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    # cv2.imshow("Processed Image", img.reshape(32,32))  # optional

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()