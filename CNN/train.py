import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "lfw-deepfunneled"

MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
LOG_PATH = LOG_DIR / "training.csv"

IMAGE_SIZE = (250, 250)

CLASS_NAMES = [
    "Ariel_Sharon",
    "Colin_Powell",
    "Donald_Rumsfeld",
    "George_W_Bush",
    "Gerhard_Schroeder",
    "Hugo_Chavez",
    "Tony_Blair"
]

RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5


# ============================================================
# GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

print("TensorFlow:", tf.__version__)
print("GPU:", gpus[0] if gpus else "None")


# ============================================================
# LOAD DATA
# ============================================================

X, y = [], []

for class_index, class_name in enumerate(CLASS_NAMES):
    class_path = DATASET_PATH / class_name

    image_files = (
        list(class_path.glob("*.jpg")) +
        list(class_path.glob("*.jpeg")) +
        list(class_path.glob("*.png"))
    )

    print(f"{class_name:20s}: {len(image_files)} images")

    for image_file in image_files:
        image = Image.open(image_file).convert("RGB").resize(IMAGE_SIZE)
        X.append(np.asarray(image, dtype=np.float32))
        y.append(class_index)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

print("\nDataset:", X.shape, y.shape)


# ============================================================
# PREPROCESS
# ============================================================

X /= 255.0  # Normalized float32 images (N, 250, 250, 3)

print("CNN input:", X.shape)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_val
)

print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

y_train = keras.utils.to_categorical(y_train, len(CLASS_NAMES))
y_val = keras.utils.to_categorical(y_val, len(CLASS_NAMES))
y_test = keras.utils.to_categorical(y_test, len(CLASS_NAMES))


# ============================================================
# CNN MODEL
# Constraints enforced: exactly 2 Conv2D layers (32 filters, 3x3 kernel)
# and exactly 1 Dense output layer.
# ============================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
], name="data_augmentation")

model = keras.Sequential([
    layers.Input(shape=(250, 250, 3)),
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation="softmax")
])

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TEST LOGGER
# ============================================================

class TestLogger(keras.callbacks.Callback):
    def __init__(self, X_test, y_test, path):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.path = path

    def on_train_begin(self, logs=None):
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "test_loss",
                "test_accuracy"
            ])

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy = self.model.evaluate(
            self.X_test,
            self.y_test,
            verbose=0
        )

        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                logs["loss"],
                logs["accuracy"],
                logs["val_loss"],
                logs["val_accuracy"],
                test_loss,
                test_accuracy
            ])

        print(
            f" - test_loss: {test_loss:.4f}"
            f" - test_accuracy: {test_accuracy:.4f}"
        )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(CLASS_NAMES, f, indent=4)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    TestLogger(X_test, y_test, LOG_PATH)
]


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

start = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print(f"\nTraining time: {time.time() - start:.2f}s")
print(f"Epochs run: {len(history.history['loss'])}")
print(f"Model saved: {MODEL_PATH}")
print(f"Log saved: {LOG_PATH}")