import numpy as np
import cv2

def predict_image(model, path):
    img = cv2.imread(path)

    if img is None:
        print("Image not found!")
        return

    img = cv2.resize(img, (224,224)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    return "Disease Detected" if pred[0][0] > 0.5 else "Normal"