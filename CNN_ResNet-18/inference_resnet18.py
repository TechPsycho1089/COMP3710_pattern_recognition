import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# ============================================================
# SETTINGS & CLASS NAMES
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_resnet18.pth"

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# MODEL ARCHITECTURE DEFINITION
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR10, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers_list = []
        for s in strides:
            layers_list.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers_list)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Check Model Existence
if not MODEL_PATH.exists():
    print(f"Error: Pre-trained PyTorch model checkpoint not found at {MODEL_PATH}", flush=True)
    print("Please train the model first by running resnet18_cifar10.py or sbatch job_resnet18.sh!", flush=True)
    sys.exit(1)

print(f"Loading pre-trained PyTorch ResNet-18 model from: {MODEL_PATH}", flush=True)
model = ResNet18_CIFAR10().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

eval_transform = T.Compose([
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])


# ============================================================
# MODE 1: FULL UNSEEN TEST SET EVALUATION (--eval / default)
# ============================================================

def run_test_eval():
    print("\nEvaluating pre-trained PyTorch ResNet-18 on all 10,000 held-out test images (with TTA)...", flush=True)
    
    possible_paths = [
        DATA_DIR / "cifar-10-batches-py",
        Path.home() / ".keras" / "datasets" / "cifar-10-batches-py",
        Path.home() / "COMP3710_pattern_recognition" / "CNN_ResNet-18" / "data" / "cifar-10-batches-py",
        Path("/tmp/cifar-10-batches-py")
    ]

    dataset_dir = None
    for p in possible_paths:
        if p.exists() and (p / "test_batch").exists():
            dataset_dir = p
            break

    if dataset_dir is None:
        print("Error: CIFAR-10 test batch not found! Check dataset directory.", flush=True)
        sys.exit(1)

    test_path = dataset_dir / "test_batch"
    with open(test_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        X_test_raw = entry['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_test_raw = np.array(entry['labels'])

    t_X_test = torch.tensor(X_test_raw, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    t_y_test = torch.tensor(y_test_raw, dtype=torch.long).squeeze()

    batch_size = 128
    total_test = len(t_X_test)
    single_correct = 0
    tta_correct = 0
    start_infer_time = time.perf_counter()

    with torch.no_grad():
        for i in range(0, total_test, batch_size):
            inputs = t_X_test[i:i+batch_size].to(device)
            targets = t_y_test[i:i+batch_size].to(device)

            norm_orig = eval_transform(inputs)
            out_orig = torch.softmax(model(norm_orig), dim=1)

            norm_flip = eval_transform(torch.flip(inputs, dims=[3]))
            out_flip = torch.softmax(model(norm_flip), dim=1)

            out_tta = (out_orig + out_flip) / 2.0

            _, pred_single = out_orig.max(1)
            _, pred_tta = out_tta.max(1)

            single_correct += pred_single.eq(targets).sum().item()
            tta_correct += pred_tta.eq(targets).sum().item()

    infer_duration = time.perf_counter() - start_infer_time
    single_acc = (single_correct / total_test) * 100.0
    tta_acc = (tta_correct / total_test) * 100.0

    print("\n" + "=" * 50, flush=True)
    print("PYTORCH INFERENCE TEST EVALUATION (10,000 UNSEEN TEST IMAGES)", flush=True)
    print("=" * 50, flush=True)
    print(f"Total Test Samples       : {total_test}", flush=True)
    print(f"Inference Duration       : {infer_duration:.2f} seconds ({infer_duration * 1000 / total_test:.2f} ms/sample)", flush=True)
    print(f"Single-Pass Test Accuracy: {single_acc:.2f}%", flush=True)
    print(f"TTA (Flip) Test Accuracy : {tta_acc:.2f}%", flush=True)
    if tta_acc >= 94.0:
        print("RESULT: Target accuracy >= 94% met on unseen test dataset!", flush=True)
    print("=" * 50 + "\n", flush=True)


# If explicitly asking for single image or --eval, run appropriate mode
if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] in ["--eval", "--evaluate", "-e", "test"]):
    run_test_eval()
    sys.exit(0)


# ============================================================
# MODE 2: SINGLE IMAGE PREDICTION
# ============================================================

image_path = Path(sys.argv[1])
if not image_path.exists():
    print(f"Error: Image file not found at {image_path}", flush=True)
    sys.exit(1)

print(f"Loading image from file: {image_path}", flush=True)
image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
image_arr = np.asarray(image, dtype=np.float32) / 255.0
input_tensor = torch.tensor(image_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    norm_input = eval_transform(input_tensor)
    probabilities = torch.softmax(model(norm_input), dim=1)[0].cpu().numpy()

prediction_idx = np.argmax(probabilities)
predicted_class = CIFAR10_CLASSES[prediction_idx]
confidence = probabilities[prediction_idx]

print("\n" + "=" * 40, flush=True)
print("RESNET-18 PYTORCH INFERENCE PREDICTION", flush=True)
print("=" * 40, flush=True)
print(f"Predicted Class : {predicted_class}", flush=True)
print(f"Confidence Score: {confidence * 100:.2f}%", flush=True)
print("=" * 40, flush=True)
print("\nClass Probabilities:", flush=True)
for name, prob in zip(CIFAR10_CLASSES, probabilities):
    bar = "█" * int(prob * 30)
    print(f"  {name:12s}: {prob * 100:6.2f}% {bar}", flush=True)
print("=" * 40 + "\n", flush=True)
