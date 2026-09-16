import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = (250, 250)


# ============================================================
# CHECK INPUT ARGUMENTS AND FILES
# ============================================================

if len(sys.argv) < 2:
    print("Error: Missing image path argument.")
    print("Usage: python inference.py <path_to_image>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.exists():
    print(f"Error: Image file not found: {image_path}")
    sys.exit(1)

if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    print("Error: Model or class names file not found. Please run train.py first.")
    sys.exit(1)


# ============================================================
# LOAD MODEL + CLASS NAMES
# ============================================================

model = keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


# ============================================================
# LOAD + PREPROCESS IMAGE
# ============================================================

image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)

image = np.asarray(image, dtype=np.float32)
image /= 255.0

image = image[np.newaxis, ...]  # batch dimension (1, 250, 250, 3)


# ============================================================
# INFERENCE
# ============================================================

probabilities = model.predict(image, verbose=0)[0]

prediction = np.argmax(probabilities)

print("\nPrediction")
print("----------")
print("Image     :", image_path)
print("Class     :", class_names[prediction])
print("Confidence:", f"{probabilities[prediction]:.4f}")

print("\nProbabilities:")

for name, probability in zip(class_names, probabilities):
    print(f"{name:20s}: {probability:.4f}")