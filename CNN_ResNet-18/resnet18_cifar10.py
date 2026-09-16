import csv
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers


# ============================================================
# SETTINGS & DIRECTORY CREATION
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (32, 32, 3)
NUM_CLASSES = 10
BATCH_SIZE = 128
INITIAL_LR = 0.001

# Check if running in 1-epoch demo mode
is_demo_mode = len(sys.argv) > 1 and sys.argv[1] in ["--demo", "demo", "1"]

if is_demo_mode:
    EPOCHS = 1
    MODEL_PATH = MODEL_DIR / "demo_resnet18.keras"
    print("\n[DEMO MODE] Running 1-epoch demonstration.")
    print(f"[DEMO MODE] Checkpoint redirected to: {MODEL_PATH}")
    print("[DEMO MODE] Your 94%+ model (best_resnet18.keras) is SAFE and untouched!\n")
else:
    EPOCHS = 28
    MODEL_PATH = MODEL_DIR / "best_resnet18.keras"


# ============================================================
# ENABLE MIXED PRECISION (NVIDIA A100 TENSOR CORES SPEEDUP)
# ============================================================

try:
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy set to 'mixed_float16'.")
except Exception as e:
    print(f"Warning: Mixed precision initialization skipped: {e}")


# ============================================================
# AUTOMATED DATASET DOWNLOADER & LOADER
# ============================================================

def download_and_extract_cifar10(data_dir: Path):
    """Downloads and extracts CIFAR-10 into local data/ directory if missing or incomplete."""
    cifar_extracted_path = data_dir / "cifar-10-batches-py"

    if cifar_extracted_path.exists():
        print(f"Dataset already extracted at: {cifar_extracted_path}")
        return

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    print(f"Downloading CIFAR-10 from {url} into {data_dir}...")

    tarball_path = data_dir / "cifar-10-python.tar.gz"
    if tarball_path.exists() and tarball_path.stat().st_size < 150 * 1024 * 1024:
        print("Removing incomplete tarball download...")
        tarball_path.unlink()

    if not tarball_path.exists():
        urllib.request.urlretrieve(url, tarball_path)
        print("Download completed.")

    print("Extracting dataset archive...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    print(f"Extraction completed: {cifar_extracted_path}")


download_and_extract_cifar10(DATA_DIR)

# Load dataset
os.environ["KERAS_HOME"] = str(DATA_DIR)
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Create a 10% validation split from training set (45,000 train, 5,000 val, 10,000 held-out test)
val_split_idx = int(len(X_train_full) * 0.9)
X_train, X_val = X_train_full[:val_split_idx], X_train_full[val_split_idx:]
y_train, y_val = y_train_full[:val_split_idx], y_train_full[val_split_idx:]

print(f"\nDataset loaded & segregated:")
print(f"  Train      : {X_train.shape}, {y_train.shape}")
print(f"  Validation : {X_val.shape}, {y_val.shape}")
print(f"  Held-out Test: {X_test.shape}, {y_test.shape}")


# ============================================================
# PREPROCESSING & ONE-HOT ENCODING
# ============================================================

X_train = X_train.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


# ============================================================
# HIGH-RESOLUTION EXECUTION TIMING CALLBACK
# ============================================================

class TimingCallback(keras.callbacks.Callback):
    """Tracks per-epoch execution duration and total wall-clock training time."""
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.epoch_times = []
        self.train_start_time = 0

    def on_train_begin(self, logs=None):
        self.train_start_time = time.perf_counter()
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "epoch_duration_sec",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy"
            ])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.perf_counter() - self.epoch_start
        self.epoch_times.append(duration)

        logs = logs or {}
        train_loss = logs.get("loss", 0.0)
        train_acc = logs.get("accuracy", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{duration:.2f}",
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}"
            ])

        print(
            f" - duration: {duration:.2f}s"
            f" - val_loss: {val_loss:.4f}"
            f" - val_accuracy: {val_acc:.4f}"
        )

    def on_train_end(self, logs=None):
        total_time = time.perf_counter() - self.train_start_time
        mins, secs = divmod(total_time, 60)
        avg_epoch = np.mean(self.epoch_times) if self.epoch_times else 0.0

        summary = {
            "total_seconds": round(total_time, 2),
            "formatted_time": f"{int(mins):02d}:{secs:05.2f}",
            "avg_epoch_seconds": round(avg_epoch, 2),
            "epochs_completed": len(self.epoch_times)
        }

        with open(SUMMARY_JSON_PATH, "w") as f:
            json.dump(summary, f, indent=4)

        print("\n" + "=" * 50)
        print("TRAINING TIMING SUMMARY")
        print("=" * 50)
        print(f"Total Training Time : {total_time:.2f} seconds ({summary['formatted_time']})")
        print(f"Average Epoch Time  : {avg_epoch:.2f} seconds")
        print(f"Epochs Completed    : {len(self.epoch_times)}")
        print("=" * 50 + "\n")


# ============================================================
# RESNET-18 MODEL ARCHITECTURE (CIFAR-10 ADAPTED)
# ============================================================

def basic_block(x, filters, stride=1):
    """Standard ResNet-18 Basic Residual Block with shortcut addition."""
    shortcut = x

    # First convolution in block
    x = layers.Conv2D(
        filters, kernel_size=3, strides=stride, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution in block
    x = layers.Conv2D(
        filters, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut if dimensions or filter counts change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, kernel_size=1, strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Residual Addition (Skip Connection)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet18(input_shape=(32, 32, 3), num_classes=10):
    """Builds full ResNet-18 model with residual skip connections."""
    inputs = layers.Input(shape=input_shape)

    # Data Augmentation Block
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")(inputs)

    # Stem: 3x3 Conv, stride 1, padding same (No Stem MaxPool for CIFAR-10 32x32)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding="same", use_bias=False)(data_aug)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1: 2 blocks, 64 filters, stride 1 (32x32)
    x = basic_block(x, filters=64, stride=1)
    x = basic_block(x, filters=64, stride=1)

    # Stage 2: 2 blocks, 128 filters, stride 2 (16x16)
    x = basic_block(x, filters=128, stride=2)
    x = basic_block(x, filters=128, stride=1)

    # Stage 3: 2 blocks, 256 filters, stride 2 (8x8)
    x = basic_block(x, filters=256, stride=2)
    x = basic_block(x, filters=256, stride=1)

    # Stage 4: 2 blocks, 512 filters, stride 2 (4x4)
    x = basic_block(x, filters=512, stride=2)
    x = basic_block(x, filters=512, stride=1)

    # Global Average Pooling & Output Dense Head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet18_CIFAR10")
    return model


# ============================================================
# MODEL COMPILED & TRAINED
# ============================================================

model = build_resnet18(input_shape=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()

total_steps = (len(X_train) // BATCH_SIZE) * EPOCHS

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=INITIAL_LR,
    decay_steps=total_steps,
    alpha=0.01
)

optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS & FIT
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    TimingCallback(LOG_CSV_PATH)
]

print("\nStarting ResNet-18 training on CIFAR-10...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ============================================================
# FINAL EVALUATION ON HELD-OUT UNSEEN TEST SET
# ============================================================

print("\nEvaluating best model on held-out unseen test set...")
best_model = keras.models.load_model(MODEL_PATH)
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=1)

print("\n" + "=" * 50)
print("FINAL TEST EVALUATION (HELD-OUT UNSEEN DATA)")
print("=" * 50)
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
if test_acc >= 0.94:
    print("SUCCESS: Target accuracy >= 94% achieved on unseen test data!")
print("=" * 50 + "\n")

print(f"Model saved: {MODEL_PATH}")
print(f"Logs saved : {LOG_CSV_PATH}")
