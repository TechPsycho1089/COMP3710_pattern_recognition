import csv
import json
import os
import pickle
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


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

LOG_CSV_PATH = LOG_DIR / "training_log.csv"
SUMMARY_JSON_PATH = LOG_DIR / "timing_summary.json"

IMAGE_SIZE = (32, 32, 3)
NUM_CLASSES = 10
BATCH_SIZE = 128
MAX_LR = 0.14

# Check if running in 1-epoch demo mode
is_demo_mode = len(sys.argv) > 1 and sys.argv[1] in ["--demo", "demo", "1"]

if is_demo_mode:
    EPOCHS = 1
    MODEL_PATH = MODEL_DIR / "demo_resnet18.pth"
    print("\n[DEMO MODE] Running 1-epoch demonstration.", flush=True)
    print(f"[DEMO MODE] Checkpoint redirected to: {MODEL_PATH}", flush=True)
    print("[DEMO MODE] Your 94%+ model (best_resnet18.pth) is SAFE and untouched!\n", flush=True)
else:
    EPOCHS = 30
    MODEL_PATH = MODEL_DIR / "best_resnet18.pth"


# ============================================================
# DEVICE SELECTION (CUDA GPU vs CPU)
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using PyTorch Device: {device}", flush=True)


# ============================================================
# AUTOMATED DATASET DOWNLOADER & LOADER
# ============================================================

def download_and_extract_cifar10(data_dir: Path):
    """Downloads and extracts CIFAR-10 into local data/ directory if missing."""
    possible_paths = [
        data_dir / "cifar-10-batches-py",
        Path.home() / ".keras" / "datasets" / "cifar-10-batches-py",
        Path.home() / "COMP3710_pattern_recognition" / "CNN_ResNet-18" / "data" / "cifar-10-batches-py",
        Path("/tmp/cifar-10-batches-py")
    ]

    for p in possible_paths:
        if p.exists() and (p / "data_batch_1").exists():
            print(f"Dataset already present at: {p}", flush=True)
            return p

    cifar_extracted_path = data_dir / "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    print(f"Downloading CIFAR-10 from {url} into {data_dir}...", flush=True)

    tarball_path = data_dir / "cifar-10-python.tar.gz"
    if tarball_path.exists() and tarball_path.stat().st_size < 150 * 1024 * 1024:
        print("Removing incomplete tarball download...", flush=True)
        tarball_path.unlink()

    if not tarball_path.exists():
        try:
            urllib.request.urlretrieve(url, tarball_path)
            print("Download completed.", flush=True)
        except Exception as err:
            print(f"Error downloading dataset: {err}", flush=True)
            print("If Rangpur HPC compute nodes lack internet access, download dataset on login node first!", flush=True)
            raise err

    print("Extracting dataset archive...", flush=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    print(f"Extraction completed: {cifar_extracted_path}", flush=True)
    return cifar_extracted_path


cifar_dir = download_and_extract_cifar10(DATA_DIR)


def load_cifar10_from_dir(cifar_dir: Path):
    """Loads CIFAR-10 raw numpy arrays directly from pickle batch files."""
    train_x, train_y = [], []
    for i in range(1, 6):
        batch_path = cifar_dir / f"data_batch_{i}"
        with open(batch_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            train_x.append(entry['data'])
            train_y.extend(entry['labels'])

    train_x = np.vstack(train_x).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_y = np.array(train_y)

    test_path = cifar_dir / "test_batch"
    with open(test_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        test_x = entry['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_y = np.array(entry['labels'])

    return (train_x, train_y), (test_x, test_y)


(X_train_full, y_train_full), (X_test, y_test) = load_cifar10_from_dir(cifar_dir)

# 3-Way Segregation: 45,000 Train, 5,000 Validation, 10,000 Unseen Held-out Test
val_split = 45000
X_train, X_val = X_train_full[:val_split], X_train_full[val_split:]
y_train, y_val = y_train_full[:val_split], y_train_full[val_split:]

# Convert to PyTorch Tensors (NCHW Format: [Batch, Channels, Height, Width], range [0, 1])
t_X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
t_y_train = torch.tensor(y_train, dtype=torch.long).squeeze()

t_X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
t_y_val = torch.tensor(y_val, dtype=torch.long).squeeze()

t_X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
t_y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(t_X_test, t_y_test), batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataset loaded & segregated:")
print(f"  Train Set      : {t_X_train.shape}, {t_y_train.shape}")
print(f"  Validation Set : {t_X_val.shape}, {t_y_val.shape}")
print(f"  Held-out Test  : {t_X_test.shape}, {t_y_test.shape}\n")


# ============================================================
# FAST PYTORCH GPU DATA AUGMENTATION & CUTOUT
# ============================================================

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4, padding_mode='reflect'),
    T.RandomHorizontalFlip(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.5, scale=(0.1, 0.25), ratio=(1.0, 1.0), value=0)  # Cutout
])

eval_transform = T.Compose([
    T.Normalize(mean, std)
])


# ============================================================
# PYTORCH RESNET-18 ARCHITECTURE (CIFAR-10 ADAPTED)
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

        # CIFAR-10 Stem: 3x3 Conv, Stride 1, Padding 1 (NO Stem MaxPool!)
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


model = ResNet18_CIFAR10().to(device)


# ============================================================
# OPTIMIZER, SCHEDULER & PYTORCH AMP
# ============================================================

optimizer = optim.SGD(
    model.parameters(),
    lr=MAX_LR,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.2,
    div_factor=25.0,
    final_div_factor=1000.0
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

use_amp = (device.type == 'cuda')
scaler = torch.amp.GradScaler('cuda') if use_amp else None


# ============================================================
# EXECUTION TIMING & LOGGING TRACKER
# ============================================================

epoch_times = []
train_start_time = time.perf_counter()

with open(LOG_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "epoch_duration_sec",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy"
    ])

print("\nStarting ResNet-18 DAWNBench PyTorch GPU Training...\n")

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.perf_counter()

    # Training Pass
    model.train()
    train_loss, train_correct, total_train = 0.0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = train_transform(inputs)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    train_acc = train_correct / total_train
    avg_train_loss = train_loss / total_train

    # Validation Pass
    model.eval()
    val_loss, val_correct, total_val = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = eval_transform(inputs)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_val += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_acc = val_correct / total_val
    avg_val_loss = val_loss / total_val

    duration = time.perf_counter() - epoch_start
    epoch_times.append(duration)

    # Save Best Checkpoint
    saved_str = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        saved_str = f"[Saved best checkpoint to {MODEL_PATH}]"

    # Log to CSV
    with open(LOG_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{duration:.2f}",
            f"{avg_train_loss:.4f}",
            f"{train_acc:.4f}",
            f"{avg_val_loss:.4f}",
            f"{val_acc:.4f}"
        ])

    print(
        f"Epoch {epoch:02d}/{EPOCHS:02d} - duration: {duration:.2f}s"
        f" - loss: {avg_train_loss:.4f} - accuracy: {train_acc:.4f}"
        f" - val_loss: {avg_val_loss:.4f} - val_accuracy: {val_acc:.4f} {saved_str}",
        flush=True
    )


total_time = time.perf_counter() - train_start_time
mins, secs = divmod(total_time, 60)
avg_epoch = np.mean(epoch_times) if epoch_times else 0.0

summary = {
    "total_seconds": round(total_time, 2),
    "formatted_time": f"{int(mins):02d}:{secs:05.2f}",
    "avg_epoch_seconds": round(avg_epoch, 2),
    "epochs_completed": len(epoch_times)
}

with open(SUMMARY_JSON_PATH, "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 50, flush=True)
print("TRAINING TIMING SUMMARY", flush=True)
print("=" * 50, flush=True)
print(f"Total Training Time : {total_time:.2f} seconds ({summary['formatted_time']})", flush=True)
print(f"Average Epoch Time  : {avg_epoch:.2f} seconds", flush=True)
print(f"Epochs Completed    : {len(epoch_times)}", flush=True)
print("=" * 50 + "\n", flush=True)


# ============================================================
# FINAL EVALUATION ON HELD-OUT UNSEEN TEST SET (WITH TTA)
# ============================================================

print("\nEvaluating best saved checkpoint on 10,000 unseen test images with TTA...", flush=True)
best_model = ResNet18_CIFAR10().to(device)
best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
best_model.eval()

single_correct, tta_correct, total_test = 0, 0, 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Pass 1: Original test images
        norm_orig = eval_transform(inputs)
        out_orig = torch.softmax(best_model(norm_orig), dim=1)

        # Pass 2: Horizontally flipped test images (TTA)
        norm_flip = eval_transform(torch.flip(inputs, dims=[3]))
        out_flip = torch.softmax(best_model(norm_flip), dim=1)

        # Average probabilities for TTA
        out_tta = (out_orig + out_flip) / 2.0

        _, pred_single = out_orig.max(1)
        _, pred_tta = out_tta.max(1)

        total_test += targets.size(0)
        single_correct += pred_single.eq(targets).sum().item()
        tta_correct += pred_tta.eq(targets).sum().item()

single_acc = (single_correct / total_test) * 100.0
tta_acc = (tta_correct / total_test) * 100.0

print("\n" + "=" * 50, flush=True)
print("FINAL TEST EVALUATION (HELD-OUT UNSEEN DATA)", flush=True)
print("=" * 50, flush=True)
print(f"Single-Pass Test Accuracy: {single_acc:.2f}%", flush=True)
print(f"TTA (Flip) Test Accuracy : {tta_acc:.2f}%", flush=True)
if tta_acc >= 94.0:
    print("SUCCESS: Target accuracy >= 94% achieved on unseen test data!", flush=True)
print("=" * 50 + "\n", flush=True)

print(f"Model saved: {MODEL_PATH}", flush=True)
print(f"Logs saved : {LOG_CSV_PATH}", flush=True)
