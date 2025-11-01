# model/train.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from model.dataset_drone import load_split
from model.model_drone import DroneClassifier
from model.config import LR, EPOCHS, BATCH_SIZE
# ======================
# ⚙️ 데이터셋 로드
# ======================
train_ds, val_ds, test_ds = load_split()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🟢 Using device: {device}")

# ======================
# ⚙️ 모델/손실/옵티마이저
# ======================
model = DroneClassifier().to(device)

# 클래스 불균형 보정: pos_weight = N_neg / N_pos
pos_weight = torch.tensor([12.0], device=device)  # 드론 1.1k vs 비드론 10k 기준
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# 🧠 학습 루프
# ======================
best_f1 = 0.0
patience, patience_limit = 0, 5
os.makedirs("chk", exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logit = model(x)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ======================
    # 🧩 검증 단계
    # ======================
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy().flatten()
            pred = (prob >= 0.5).astype(int)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred)

    f1 = f1_score(y_true, y_pred)
    print(f"[Epoch {epoch}/{EPOCHS}] loss={avg_loss:.4f} | val_F1={f1:.4f}")

    # ======================
    # 💾 Best model 저장
    # ======================
    if f1 > best_f1:
        best_f1 = f1
        patience = 0
        torch.save({"model": model.state_dict()}, "chk/best.pt")
        print(f"  ✅ New best model saved (F1={best_f1:.4f})")
    else:
        patience += 1
        if patience >= patience_limit:
            print("⏹️ Early stopping triggered.")
            break

# 최종 모델 저장 (마지막 상태)
torch.save({"model": model.state_dict()}, "chk/last.pt")
print("✅ Training complete. Best model saved to chk/best.pt")
