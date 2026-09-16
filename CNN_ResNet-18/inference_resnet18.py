import sys
from pathlib import Path

import numpy as np
from PIL import Image
import keras


# ============================================================
# SETTINGS & CLASS NAMES
# ============================================================

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_resnet18.keras"
DATA_DIR = BASE_DIR / "data"

IMAGE_SIZE = (32, 32)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# CHECK MODEL EXISTENCE
# ============================================================

if not MODEL_PATH.exists():
    print(f"Error: Trained model not found at {MODEL_PATH}")
    print("Please train the model first by running resnet18_cifar10.py!")
    sys.exit(1)

# Load saved ResNet-18 model
print(f"Loading pre-trained ResNet-18 model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)


# ============================================================
# MODE 1: FULL UNSEEN TEST SET EVALUATION (--eval / --test)
# ============================================================

if len(sys.argv) > 1 and sys.argv[1] in ["--eval", "--evaluate", "-e", "test"]:
    print("\nEvaluating pre-trained ResNet-18 on all 10,000 held-out test images...")
    import os
    os.environ["KERAS_HOME"] = str(DATA_DIR)
    (_, _), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_test_norm = X_test.astype(np.float32) / 255.0
    y_test_onehot = keras.utils.to_categorical(y_test, 10)

    test_loss, test_acc = model.evaluate(X_test_norm, y_test_onehot, verbose=1)

    print("\n" + "=" * 50)
    print("DEMO TEST EVALUATION (10,000 UNSEEN TEST IMAGES)")
    print("=" * 50)
    print(f"Unseen Test Loss    : {test_loss:.4f}")
    print(f"Unseen Test Accuracy: {test_acc * 100:.2f}%")
    if test_acc >= 0.94:
        print("RESULT: Target accuracy >= 94% met on unseen test dataset!")
    print("=" * 50 + "\n")
    sys.exit(0)


# ============================================================
# MODE 2: SINGLE IMAGE PREDICTION
# ============================================================

if len(sys.argv) > 1:
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    print(f"Loading image from file: {image_path}")
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_arr, axis=0)
else:
    print("No image file specified. Picking a random image from CIFAR-10 test set...")
    import os
    os.environ["KERAS_HOME"] = str(DATA_DIR)
    (_, _), (X_test, y_test) = keras.datasets.cifar10.load_data()
    sample_idx = np.random.randint(0, len(X_test))
    image_batch = np.expand_dims(X_test[sample_idx].astype(np.float32) / 255.0, axis=0)
    true_label = CIFAR10_CLASSES[y_test[sample_idx][0]]
    print(f"Random test image index: #{sample_idx} (Ground Truth: '{true_label}')")


# ============================================================
# RUN INFERENCE
# ============================================================

probabilities = model.predict(image_batch, verbose=0)[0]
prediction_idx = np.argmax(probabilities)
predicted_class = CIFAR10_CLASSES[prediction_idx]
confidence = probabilities[prediction_idx]

print("\n" + "=" * 40)
print("RESNET-18 INFERENCE PREDICTION")
print("=" * 40)
print(f"Predicted Class : {predicted_class}")
print(f"Confidence Score: {confidence * 100:.2f}%")
print("=" * 40)
print("\nClass Probabilities:")
for name, prob in zip(CIFAR10_CLASSES, probabilities):
    bar = "█" * int(prob * 30)
    print(f"  {name:12s}: {prob * 100:6.2f}% {bar}")
print("=" * 40 + "\n")
