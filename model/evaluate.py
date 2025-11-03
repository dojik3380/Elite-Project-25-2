import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
)
from model.dataset_drone import load_split
from model.model_drone import DroneClassifier
from torch.utils.data import DataLoader
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--val", action="store_true", help="Use validation set instead of test set")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds, val_ds, test_ds = load_split()
dataset = val_ds if args.val else test_ds
loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = DroneClassifier().to(device)
ckpt = torch.load("chk/best_model.pt", map_location=device)
model.load_state_dict(ckpt)
model.eval()

# ----------------------------
# ✅ 결과 저장용 리스트
# ----------------------------
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logit = model(x)
        prob = torch.sigmoid(logit).cpu().numpy().flatten()
        pred = (prob > 0.59).astype(int)  # 기본 threshold=0.5
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred)
        y_prob.extend(prob)

# ----------------------------
# ✅ 기본 지표 계산
# ----------------------------
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
auc = roc_auc_score(y_true, y_prob)
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# ----------------------------
# ✅ 최적 threshold 탐색
# ----------------------------
precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n🔍 Best threshold = {best_threshold:.3f} (F1={best_f1:.3f})")
