"""
train_cnn.py

DEV-ONLY. Trains the 1D-CNN described in docs/ANVAYA_PIPELINE.md (Section 4,
Phase II) on the preprocessed sequence dataset. Saves the best checkpoint as
gesture_cnn.h5 for convert_tfjs.py to pick up next.

This script is the ONLY place any "learning" happens in the whole project.
The browser app (web_app/) only ever loads the finished model and runs
inference -- see docs/GUIDE.md, "Common Questions".
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

THIS_DIR = os.path.dirname(__file__)
NPZ_PATH = os.path.join(THIS_DIR, "sequences_dataset.npz")
LABEL_MAP_PATH = os.path.join(THIS_DIR, "label_map.json")
MODEL_OUT_PATH = os.path.join(THIS_DIR, "gesture_cnn.h5")

SEQUENCE_LENGTH = 24
FRAME_VECTOR_SIZE = 128
EPOCHS = 100
BATCH_SIZE = 16


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, FRAME_VECTOR_SIZE)),

        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Collapses the time axis -- keeps the model small and less prone to
        # overfitting on a small, self-collected minor-project dataset.
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.exists(NPZ_PATH):
        raise RuntimeError(f"{NPZ_PATH} not found. Run preprocess.py first.")

    data = np.load(NPZ_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]}, "
          f"{num_classes} classes.")

    model = build_model(num_classes)
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_OUT_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=2,
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {val_acc:.3f}")
    if val_acc < 0.80:
        print(
            "NOTE: validation accuracy is below 80%. Per docs/GUIDE.md, the first thing "
            "to try is recording MORE and MORE VARIED samples per class before changing "
            "the model architecture -- a small/noisy dataset is the most common cause."
        )
    print(f"Saved best model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
