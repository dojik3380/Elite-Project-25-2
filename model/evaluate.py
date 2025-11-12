# model/evaluate.py
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from model.dataset_drone import load_split
from model.model_drone import DroneMultiClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="test")
parser.add_argument("--ckpt", default="chk/best.pt")
parser.add_argument("--bs", type=int, default=64)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터셋 로드 (라벨맵은 폴더명 기반)
train_ds, val_ds, test_ds, ds_label_map = load_split()  # <-- 반환 4개 맞음
dataset = val_ds if args.split == "val" else test_ds
loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4)

# 체크포인트 라벨맵 로드 (학습 당시의 클래스 순서)
ckpt = torch.load(args.ckpt, map_location=device)
ckpt_label_map = ckpt["label_map"]  # {'airplane':0, 'drone':1, ...}
idx_to_label = {v: k for k, v in ckpt_label_map.items()}
n_classes = len(ckpt_label_map)

# 데이터셋 라벨 -> 체크포인트 라벨 인덱스 매핑 (이름(lower) 기준)
# ds_label_map: {'airplane':0, 'drone':1, ...}
ds_idx_to_ckpt_idx = np.zeros(len(ds_label_map), dtype=np.int64)
for name, ds_idx in ds_label_map.items():
    if name in ckpt_label_map:
        ds_idx_to_ckpt_idx[ds_idx] = ckpt_label_map[name]
    else:
        raise ValueError(f"Class name mismatch: '{name}' not found in checkpoint label_map")

# 모델 준비
model = DroneMultiClassifier(n_classes=n_classes).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)            # [B, 1, T]
        # y: 데이터셋 인덱스 → 체크포인트 인덱스로 변환
        y_ckpt = torch.from_numpy(ds_idx_to_ckpt_idx[y.numpy()]).to(device)

        logit = model(x)            # [B, C]
        pred = torch.argmax(logit, dim=1)

        y_true.extend(y_ckpt.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
print(f"Accuracy: {acc:.4f}  Macro-F1: {f1_macro:.4f}")

# 상세 리포트
target_names = [idx_to_label[i] for i in range(n_classes)]
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
