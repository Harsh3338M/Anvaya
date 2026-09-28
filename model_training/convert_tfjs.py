"""
convert_tfjs.py

DEV-ONLY. Converts the trained gesture_cnn.h5 into the TensorFlow.js
"layers-model" format (model.json + weight .bin shards) that web_app/app.js
loads in the browser.

Why this is hand-written instead of using the `tensorflowjs` pip package:
that package hard-depends on `tensorflow-decision-forests`, which has no
native Windows build, so it cannot be installed on Windows. The TF.js format
is simple (Keras topology JSON + raw float32 weights), so we write it directly.
The output was verified to give the same predictions as Keras (see docs/GUIDE.md).
"""

# --- IMPORTANT: must run BEFORE tensorflow is imported. ---
# TensorFlow 2.16+ defaults to Keras 3, whose saved models TensorFlow.js cannot
# load in the browser. This switches TensorFlow back to Keras 2 (tf_keras).
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import numpy as np
from tensorflow.keras.models import load_model

THIS_DIR = os.path.dirname(__file__)
H5_PATH = os.path.join(THIS_DIR, "gesture_cnn.h5")
OUT_DIR = os.path.join(THIS_DIR, "model_web")
SHARD_BYTES = 4 * 1024 * 1024  # browsers cache weight files best in <=4MB pieces


def export(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):  # clear stale shards from a previous export
        if old.endswith(".bin") or old == "model.json":
            os.remove(os.path.join(out_dir, old))

    # 1) Topology: Keras 2 JSON, rearranged into the layout TF.js expects.
    keras_json = json.loads(model.to_json())
    topology = {
        "keras_version": keras_json.get("keras_version", "2"),
        "backend": keras_json.get("backend", "tensorflow"),
        "model_config": {
            "class_name": keras_json["class_name"],
            "config": keras_json["config"],
        },
    }

    # 2) Weights: every layer weight as raw float32, described by name/shape.
    specs, blobs = [], []
    for w in model.weights:
        name = w.name.split(":")[0]  # "conv1d/kernel:0" -> "conv1d/kernel"
        arr = np.asarray(w.numpy(), dtype="<f4")
        specs.append({"name": name, "shape": list(arr.shape), "dtype": "float32"})
        blobs.append(arr.tobytes())
    data = b"".join(blobs)

    shards = [data[i:i + SHARD_BYTES] for i in range(0, len(data), SHARD_BYTES)] or [b""]
    paths = []
    for i, chunk in enumerate(shards, start=1):
        fname = f"group1-shard{i}of{len(shards)}.bin"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(chunk)
        paths.append(fname)

    manifest = {
        "format": "layers-model",
        "generatedBy": f"keras v{topology['keras_version']}",
        "convertedBy": "Anvaya convert_tfjs.py",
        "modelTopology": topology,
        "weightsManifest": [{"paths": paths, "weights": specs}],
    }
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(manifest, f)
    return paths


def main():
    if not os.path.exists(H5_PATH):
        raise RuntimeError(f"{H5_PATH} not found. Run train_cnn.py first.")
    model = load_model(H5_PATH)
    paths = export(model, OUT_DIR)
    print(f"Exported TensorFlow.js model to {OUT_DIR} ({len(paths)} weight file(s))")
    print("Next: copy its contents (and label_map.json) into web_app/model/")
    print("  Mac/Linux:  cp -r model_web/* ../web_app/model/")
    print("  Windows:    xcopy model_web\\* ..\\web_app\\model\\ /E /Y")


if __name__ == "__main__":
    main()