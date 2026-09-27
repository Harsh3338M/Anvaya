"""
convert_tfjs.py

DEV-ONLY. Converts the trained gesture_cnn.h5 (Keras) model into the
TensorFlow.js format (model.json + weight shard .bin files) that
web_app/app.js loads at runtime.

Requires the `tensorflowjs` pip package (see requirements.txt).

After running this, per docs/GUIDE.md Step 1.5, copy the CONTENTS of
model_web/ into web_app/model/ (label_map.json should already be there
from preprocess.py).
"""

import os
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

THIS_DIR = os.path.dirname(__file__)
H5_PATH = os.path.join(THIS_DIR, "gesture_cnn.h5")
OUT_DIR = os.path.join(THIS_DIR, "model_web")


def main():
    if not os.path.exists(H5_PATH):
        raise RuntimeError(f"{H5_PATH} not found. Run train_cnn.py first.")

    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(H5_PATH)
    tfjs.converters.save_keras_model(model, OUT_DIR)

    print(f"Exported TensorFlow.js model to {OUT_DIR}")
    print("Next: copy its contents (and label_map.json) into web_app/model/")
    print("  cp -r model_web/* ../web_app/model/")
    print("  cp label_map.json ../web_app/model/   # if not already copied by preprocess.py")


if __name__ == "__main__":
    main()
