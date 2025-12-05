# model/train.py
import os, torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from model.dataset_drone import load_split, SEGMENT_SEC   # 🔹 SEGMENT_SEC 추가
from model.model_drone import DroneMultiClassifier
from model.config import EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, TARGET_SR
from model.augment import WaveAugmentV1   # 🔹 온라인 증강용

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# ✅ 데이터 로드
# ============================================================
train_ds, val_ds, test_ds, label_map = load_split()
n_classes = len(label_map)
print(f"\n📂 Loaded dataset with {n_classes} classes:")
for k, v in label_map.items():
    print(f"  {v}: {k}")

if len(train_ds) == 0:
    raise RuntimeError("🚨 Train dataset is empty!")

# ============================================================
# ✅ 클래스 분포 및 가중치 계산 (루트 스케일링, duration 기반)
#    → DroneDataset는 이미 2초 segment로 쪼개져 있으므로
#      각 segment 하나당 SEGMENT_SEC(=WIN_SEC) 초로 계산
# ============================================================
duration_sum = defaultdict(float)

print("\n⏳ Calculating class durations (segment-based) ...")

for i in range(len(train_ds)):
    _, label = train_ds[i]          # train_ds[i] = (x, y)
    if torch.is_tensor(label):
        label_idx = int(label.item())
    else:
        label_idx = int(label)
    duration_sum[label_idx] += SEGMENT_SEC   # segment 하나 = SEGMENT_SEC 초

# weight = 1 / sqrt(total_seconds)
class_weights_list = []
for cls_idx in range(n_classes):
    total_sec = duration_sum[cls_idx]
    if total_sec == 0:
        w = 1.0
    else:
        w = 1.0 / np.sqrt(total_sec)
    class_weights_list.append(w)

class_weights = torch.tensor(class_weights_list, dtype=torch.float).to(DEVICE)

print("🎚 Class weights (duration-based, segment-level):")
for cls, w in zip(label_map.keys(), class_weights_list):
    print(f"  {cls:12s}: {w:.6f}")

# ============================================================
# ✅ DataLoader
# ============================================================
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# ✅ Augment (온라인 증강, 노이즈 OFF)
# ============================================================
# use_noise=False → real noise 증강 비활성화, 나머지 gain/shift/stretch/time_mask만 사용
augment = WaveAugmentV1(
    sample_rate=TARGET_SR,
    use_augment=True,   # gain/shift/stretch/time_mask 활성화
    use_noise=False     # noise 증강 비활성화
)


# ============================================================
# ✅ 모델 / 손실함수 / 옵티마이저
# ============================================================
model = DroneMultiClassifier(n_classes=n_classes).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3,
)

best_f1 = 0.0
os.makedirs("chk", exist_ok=True)

# ============================================================
# ✅ 학습 루프
# ============================================================
for epoch in range(1, EPOCHS + 1):

    # -----------------------
    # 🔵 TRAIN
    # -----------------------
    model.train()
    running_loss = 0.0
    progress = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, colour='cyan')
    progress.set_description(f"[Epoch {epoch:02d}]")

    for batch_idx, (x, y) in progress:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 🔥 온라인 증강 (노이즈 OFF, batch 안 각 세그먼트에 독립적으로 적용)
        with torch.no_grad():
            B = x.size(0)
            x_aug = torch.empty_like(x)
            for i in range(B):
                x_aug[i] = augment(x[i])
            x = x_aug

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}")

    # -----------------------
    # 🔵 VALIDATION (증강 없음)
    # -----------------------
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

    current_lr = optimizer.param_groups[0]['lr']
    print(f"\n[{epoch:02d}] Epoch complete | avg_loss={avg_loss:.4f} | val_f1={f1:.4f} | lr={current_lr:.2e}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save({"model": model.state_dict(), "label_map": label_map}, "chk/best.pt")
        print(f"  ✅ New best model saved (F1={best_f1:.4f})")

print("\n🎉 Training complete.")
print("Final label map:", label_map)
