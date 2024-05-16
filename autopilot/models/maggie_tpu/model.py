import os

import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# For guidance converting your model to TFLite, please refer to the README.


class Model:
    def __init__(self):
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"  # enable gpu

        model_path = os.path.join(CURRENT_DIR, "maggie_model.tflite")

        try:
            # attempt to load model on TPU
            print("Using TPU")
            self.interpreter = tf.lite.Interpreter(
                model_path,
                experimental_delegates=[
                    tf.lite.experimental.load_delegate("libedgetpu.so.1"),
                    # Works for Linux - use "libedgetpu.1.dylib" if testing on macOS
                ],
            )
            print(f"Using EdgeTPU for {model_path}")

        except Exception as e:
            # fallback to CPU if TPU not available
            print(f"Error loading EdgeTPU: {e}")
            self.interpreter = tf.lite.Interpreter(model_path)
            print(f"Using CPU for {model_path}")

        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.speed_output_index = self.interpreter.get_output_details()[0]["index"]
        self.angle_output_index = self.interpreter.get_output_details()[1]["index"]

    def preprocess(self, image):
        # im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(image, (120, 160))
        im = tf.divide(im, 255)  # Normalize, need to check if needed
        im = tf.expand_dims(im, axis=0)  # add batch dimension
        return im

    def predict(self, image):
        # Preprocess image and set as input tensor
        image = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details, image)

        # Run model and get output tensors
        self.interpreter.invoke()
        pred_speed = self.interpreter.get_tensor(self.speed_output_index)[0]
        pred_angle = self.interpreter.get_tensor(self.angle_output_index)[0]

        # Post-process outputs (model uses normalised data)
        speed = np.around(pred_speed[0]).astype(int) * 35
        angle = pred_angle[0] * 80 + 50

        return angle, speed
