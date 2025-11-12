# model/train.py
import os, torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.dataset_drone import load_split
from model.model_drone import DroneMultiClassifier
from model.config import EPOCHS, BATCH_SIZE, DEVICE, LR, WEIGHT_DECAY

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
# ✅ 클래스 분포 및 가중치 계산 (루트 스케일링)
# ============================================================
train_labels = [int(y.item()) if torch.is_tensor(y) else int(y)
                for _, y in [train_ds[i] for i in range(len(train_ds))]]
cnt = Counter(train_labels)
print("\n📊 Class distribution:", dict(cnt))

class_weights_list = [1.0 / (cnt[i] ** 0.5) if cnt[i] > 0 else 1.0
                      for i in range(n_classes)]
class_weights = torch.tensor(class_weights_list, dtype=torch.float).to(DEVICE)
print(f"🧮 Class weights (sqrt scale): {class_weights.tolist()}")

# ============================================================
# ✅ DataLoader (shuffle 기반)
# ============================================================
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# ✅ 모델, 손실함수, 옵티마이저, 스케줄러
# ============================================================
model = DroneMultiClassifier(n_classes=n_classes).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

best_f1 = 0.0
os.makedirs("chk", exist_ok=True)

# ============================================================
# ✅ 학습 루프
# ============================================================
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    progress = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, colour='cyan')
    progress.set_description(f"[Epoch {epoch:02d}]")

    for batch_idx, (x, y) in progress:
        x, y = x.to(DEVICE), y.to(DEVICE).long()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}")

    # -----------------------
    # 🔹 검증 단계
    # -----------------------
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            pred = torch.argmax(model(x), dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average='macro')
    scheduler.step(f1)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"\n[{epoch:02d}] Epoch complete | avg_loss={avg_loss:.4f} | val_f1={f1:.4f} | lr={current_lr:.2e}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save({"model": model.state_dict(), "label_map": label_map}, "chk/best.pt")
        print(f"  ✅ New best model saved (F1={best_f1:.4f})")

print("\n✅ Training complete.")
print("Final label map:", label_map)

# ============================================================
# ✅ classification report + confusion matrix
# ============================================================
print("\nValidation Report:")
print(classification_report(y_true, y_pred, target_names=list(label_map.keys()), digits=4))

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_map))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.keys()))
disp.plot(cmap="Blues", xticks_rotation=45, colorbar=True)
plt.title("Validation Confusion Matrix")
plt.tight_layout()
plt.savefig("chk/confusion_matrix.png")
plt.show()
