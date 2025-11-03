import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from .dataset_drone import load_split
from .model_drone import DroneClassifier
from .config import LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE, NUM_WORKERS, DEVICE
import os

# ======================
# ⚙️ 데이터셋 로드
# ======================
train_ds, val_ds, test_ds = load_split(
    balance_neg_to_pos=True,   # 👈 드론 수만큼 비드론 언더샘플링
    # max_neg=2000,            # 👈 (옵션) 비드론을 최대 2000개로 제한
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
device = DEVICE

# ======================
# ⚙️ 모델 정의
# ======================
model = DroneClassifier().to(device)

# ======================
# ⚙️ pos_weight 자동 계산
# ======================
def count_labels(subset):
    cnt_pos = cnt_neg = 0
    for idx in subset.indices:
        _, lbl = subset.dataset.data[idx]
        if lbl == 1:
            cnt_pos += 1
        else:
            cnt_neg += 1
    return cnt_pos, cnt_neg

n_pos, n_neg = count_labels(train_ds)
ratio = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
print(f"[ClassRatio] pos={n_pos}, neg={n_neg}, pos_weight={ratio:.3f}")

pos_weight = torch.tensor([1.5], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ======================
# ⚙️ 학습 루프
# ======================
best_f1 = 0.0
os.makedirs("chk", exist_ok=True)       # ✅ chk 폴더 자동 생성
save_path = "chk/best_model.pt"         # ✅ 평가 코드와 동일하게 설정

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}] Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred.squeeze(), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ======================
    # ⚙️ Validation
    # ======================
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = torch.sigmoid(model(x)).squeeze()
            preds.extend((out > 0.3).int().cpu().numpy())
            trues.extend(y.int().cpu().numpy())

    val_f1 = f1_score(trues, preds, zero_division=0)
    print(f"[Epoch {epoch}/{EPOCHS}] loss={avg_loss:.4f} | val_F1={val_f1:.4f}")

    # ✅ F1 최고값일 때만 저장
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), save_path)
        print(f"  ✅ New best model saved (F1={val_f1:.4f})")

print(f"\n🎯 Training complete! Best F1={best_f1:.4f} | Saved: {save_path}")
