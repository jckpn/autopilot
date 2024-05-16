import os

import cv2
import imutils
import numpy as np
from tensorflow import keras

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Model:
    def __init__(self, model_name="autopilot.h5"):
        model_path = os.path.join(CURRENT_DIR, model_name)
        self.model = keras.models.load_model(model_path)
        self.model.summary()

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) / 255.0
        image = imutils.resize(image, width=80)
        image = image[int(image.shape[0] / 4) :, :, :]
        return image

    def predict(self, image):
        image = self.preprocess(image)
        angle, speed = self.model.predict(np.array([image]))[0]
        # Training data was normalised so convert back to car units
        angle = 80 * np.clip(angle, 0, 1) + 50
        speed = 35 * np.clip(speed, 0, 1)
        return angle, speed
