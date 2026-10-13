"""
train_cnn.py

DEV-ONLY. Trains the 1D-CNN described in docs/ANVAYA_PIPELINE.md (Section 4,
Phase II) on the preprocessed sequence dataset. Saves the best checkpoint as
gesture_cnn.h5 for convert_tfjs.py to pick up next.

After training it prints a PER-CLASS validation report (worst first, with what each
weak class gets mistaken for) -- use it, not just the single overall number, to see
which signs the model confuses. Training uses balanced class weights so classes with
few samples (e.g. INCLUDE words, ~40 windows) are not drowned out by classes with
many (e.g. Kaggle letters, ~200).

NOTE: the overall validation accuracy is optimistic. The checkpoint and early
stopping are both chosen using this same validation set, so it is not a clean test.
Live webcam testing is the honest measure (see docs/GUIDE_V2.md).

This script is the ONLY place any "learning" happens in the whole project.
The browser app (web_app/) only ever loads the finished model and runs
inference -- see docs/GUIDE.md, "Common Questions".
"""

# --- IMPORTANT: must run BEFORE tensorflow is imported. ---
# TensorFlow 2.16+ defaults to Keras 3, whose saved models TensorFlow.js cannot
# load in the browser. This switches TensorFlow back to Keras 2 (tf_keras).
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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


def balanced_class_weights(y_train):
    """weight_c = N / (K * n_c): each class contributes equally to the loss
    regardless of how many samples it has. Only for classes present in y_train."""
    classes, counts = np.unique(y_train, return_counts=True)
    total, k = len(y_train), len(classes)
    return {int(c): float(total / (k * n)) for c, n in zip(classes, counts)}


def report_per_class(model, X_val, y_val, label_map):
    """Per-class validation accuracy, worst first, plus the most common wrong guess."""
    if len(y_val) == 0:
        print("\nNo validation samples -- cannot report per-class accuracy.")
        return
    names = {i: n for n, i in label_map.items()}
    pred = model.predict(X_val, verbose=0).argmax(axis=1)

    rows = []
    for i in sorted(names):
        mask = (y_val == i)
        if not mask.any():
            continue
        acc = float((pred[mask] == i).mean())
        wrong = pred[mask][pred[mask] != i]
        confused = ""
        if wrong.size:
            top = int(np.bincount(wrong).argmax())
            confused = f"mostly mistaken for {names[top]} ({int((wrong == top).sum())}/{int(mask.sum())})"
        rows.append((acc, names[i], int(mask.sum()), confused))
    rows.sort(key=lambda r: (r[0], r[1]))

    print(f"\nSaved checkpoint -- overall validation accuracy: {(pred == y_val).mean():.3f} "
          f"on {len(y_val)} validation samples")
    print("Per-class validation accuracy (worst first):")
    for acc, name, n, confused in rows:
        print(f"  {name:<14}{acc * 100:5.1f}%  ({n} samples)  {confused}")
    unmeasured = [names[i] for i in sorted(names) if not (y_val == i).any()]
    if unmeasured:
        print(f"  (not measured -- no validation samples: {', '.join(unmeasured)})")


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
        class_weight=balanced_class_weights(y_train),
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

    # Evaluate the file that will actually be exported (best val_accuracy epoch),
    # not the in-memory model (early stopping restores the best val_loss epoch).
    report_per_class(models.load_model(MODEL_OUT_PATH), X_val, y_val, label_map)


if __name__ == "__main__":
    main()