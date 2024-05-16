import os

import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Model:
    def __init__(self):
        speed_model_path = os.path.join(CURRENT_DIR, "speed_model")
        angle_model_path = os.path.join(CURRENT_DIR, "angle_model")
        self.speed_model = tf.keras.models.load_model(speed_model_path)
        self.angle_model = tf.keras.models.load_model(angle_model_path)

    def preprocess(self, image):
        im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(im, [100, 100])
        im = tf.expand_dims(im, axis=0)
        return im

    def predict(self, image):
        angles = np.arange(17) * 5 + 50
        image = self.preprocess(image)

        pred_speed = self.speed_model.predict(image)[0]
        speed = pred_speed[0].astype(int) * 35
        pred_angle = self.angle_model.predict(image)[0]
        angle = angles[np.argmax(pred_angle)]
        print("angle:", angle, "speed:", speed)

        return angle, speed
