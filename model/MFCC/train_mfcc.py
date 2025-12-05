# model/MFCC/train_noise_mfcc.py

import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from model.MFCC.model_drone_mfcc import DroneMultiClassifierMFCC
from model.dataset_drone import load_split, SEGMENT_SEC
from model.config import LR, EPOCHS, BATCH_SIZE, WEIGHT_DECAY, TARGET_SR
from model.augment import WaveAugmentV1   # 🔥 온라인 증강 사용
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1) Load dataset (7 classes)
# ============================================================
train_ds, val_ds, test_ds, label_map = load_split()
n_classes = len(label_map)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"\n📂 Loaded dataset {n_classes} classes:")
for cls, idx in label_map.items():
    print(f"  {idx}: {cls}")

# ============================================================
# 2) Class weights (segment-based duration)
# ============================================================
duration_sum = defaultdict(float)

print("\n⏳ Calculating class durations (segment-based)...")

for i in range(len(train_ds)):
    _, label = train_ds[i]             # (x, y)
    label_idx = int(label.item())
    duration_sum[label_idx] += SEGMENT_SEC  # segment = 2초

# weight = 1 / sqrt(total_seconds)
class_weights_list = []
for cls_idx in range(n_classes):
    tot = duration_sum[cls_idx]
    class_weights_list.append(1.0 if tot == 0 else 1.0 / np.sqrt(tot))

class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(DEVICE)

print("🎚 Class weights (duration–based):")
for cls, w in zip(label_map.keys(), class_weights_list):
    print(f"   {cls:10s}: {w:.5f}")

# ============================================================
# 3) Online Augment (노이즈 OFF or ON)
# ============================================================
# 📌 noise 증강 off → use_noise=False
augment = WaveAugmentV1(sample_rate=TARGET_SR, use_noise=False)

# ============================================================
# 4) Model / Loss / Optimizer
# ============================================================
model = DroneMultiClassifierMFCC(n_classes).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

best_f1 = 0.0
os.makedirs("chk", exist_ok=True)

# ============================================================
# 5) Training Loop (Online Augmentation)
# ============================================================
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    progress = tqdm(train_loader, ncols=100, colour="cyan")
    progress.set_description(f"[Epoch {epoch:02d}]")

    for x, y in progress:
        x, y = x.to(DEVICE), y.to(DEVICE).long()

        # 🔥 배치 단위 온라인 증강
        with torch.no_grad():
            B = x.size(0)
            x_aug = torch.empty_like(x)
            for i in range(B):
                x_aug[i] = augment(x[i])
            x = x_aug

        optimizer.zero_grad()
        logits = model(x)        # MFCC는 model 내부에서 front-end 변환됨
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if progress.n > 0:
            avg_loss = running_loss / progress.n
        else:
            avg_loss = 0.0

        progress.set_postfix(loss=f"{avg_loss:.4f}")


    # ============================================================
    # Validation (증강 없음)
    # ============================================================
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            pred = torch.argmax(model(x), dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average="macro")
    scheduler.step(f1)

    print(f"[{epoch}] loss={running_loss/len(train_loader):.4f} | val_f1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(
            {"model": model.state_dict(), "label_map": label_map},
            "chk/mfcc_best.pt"
        )
        print("🔥 BEST model updated!")

print("\n🎉 MFCC Training Completed.")
