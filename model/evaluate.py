# model/evaluate.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay # 코드 중간에 import 되는 것을 위로 올림

# 추가된 라이브러리 (SNR 평가용)
import soundfile as sf
from pathlib import Path
import random
import librosa # resample용

from model.dataset_drone import load_split
from model.model_drone import DroneMultiClassifier

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="test")
parser.add_argument("--ckpt", default="chk/best.pt")
parser.add_argument("--bs", type=int, default=64)
args = parser.parse_args()

# ============================================================
# ✅ 데이터 로드
# ============================================================
train_ds, val_ds, test_ds, ds_label_map = load_split()
dataset = val_ds if args.split == "val" else test_ds
loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4)

# ============================================================
# ✅ 체크포인트 라벨맵 로드
# ============================================================
ckpt = torch.load(args.ckpt, map_location=DEVICE)
ckpt_label_map = ckpt["label_map"]
idx_to_label = {v: k for k, v in ckpt_label_map.items()}
n_classes = len(ckpt_label_map)

# dataset 라벨 -> ckpt 라벨 매핑
ds_idx_to_ckpt_idx = np.zeros(len(ds_label_map), dtype=np.int64)
for name, ds_idx in ds_label_map.items():
    if name not in ckpt_label_map:
        raise ValueError(f"Class mismatch: '{name}' not in checkpoint")
    ds_idx_to_ckpt_idx[ds_idx] = ckpt_label_map[name]

# ============================================================
# ✅ 모델 로드
# ============================================================
model = DroneMultiClassifier(n_classes=n_classes).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

# ============================================================
# ✅ 기본 예측 수행 (Clean Data)
# ============================================================
all_logits = []
all_targets = []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        y_ckpt = torch.from_numpy(ds_idx_to_ckpt_idx[y.numpy()]).to(DEVICE)

        logit = model(x)
        all_logits.append(logit.cpu().numpy())
        all_targets.append(y_ckpt.cpu().numpy())

all_logits = np.concatenate(all_logits, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

y_pred = np.argmax(all_logits, axis=1)

# ============================================================
# 📊 기본 지표 출력
# ============================================================
acc = accuracy_score(all_targets, y_pred)
f1_macro = f1_score(all_targets, y_pred, average="macro")
target_names = [idx_to_label[i] for i in range(n_classes)]

print(f"Accuracy: {acc:.4f}  Macro-F1: {f1_macro:.4f}")
print("\nClassification Report:")
print(classification_report(all_targets, y_pred, target_names=target_names, digits=4))

# ============================================================
# 🔵 혼동행렬 (최종 베스트 모델)
# ============================================================
cm = confusion_matrix(all_targets, y_pred, labels=list(range(n_classes)))
plt.figure(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45, colorbar=True)
plt.title("Final Confusion Matrix (Best Model)")
plt.tight_layout()
plt.savefig("chk/final_confusion_matrix.png")
plt.close()

# ============================================================
# 🔴 AUC ROC 커브 (최종 베스트 모델)
# ============================================================
y_onehot = label_binarize(all_targets, classes=list(range(n_classes)))

plt.figure(figsize=(10, 8))
for cls in range(n_classes):
    fpr, tpr, _ = roc_curve(y_onehot[:, cls], all_logits[:, cls])
    cls_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{target_names[cls]} (AUC={cls_auc:.3f})")

plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Best Model)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("chk/final_roc_curve.png")
plt.close()

print("\n📁 Saved:")
print(" - chk/final_confusion_matrix.png")
print(" - chk/final_roc_curve.png")


# ============================================================
# (추가) SNR별 Class-wise F1 생성 (clean evaluate에서도 수행)
# ============================================================

TARGET_SR = 16000
WIN_SEC = 2.0
WIN_LEN = int(TARGET_SR * WIN_SEC)

# noise 폴더
NOISE_DIR = Path("datasets/DroneUnified/noise")
noise_files = list(NOISE_DIR.rglob("*.wav"))

if len(noise_files) == 0:
    print("⚠ Warning: No noise files found. SNR evaluation skipped.")
else:
    print("\n🔍 Running SNR evaluation for class-wise F1...")

    snr_list = [-5, 0, 5, 10, 20, "clean"]
    class_f1_dict = {}

    def add_noise_snr(clean, noise, snr_db):
        if snr_db == "clean":
            return clean
        rms_clean = np.sqrt(np.mean(clean**2) + 1e-12)
        rms_noise = np.sqrt(np.mean(noise**2) + 1e-12)
        desired = rms_clean / (10 ** (snr_db / 20))
        scaled = noise * (desired / rms_noise)
        return np.clip(clean + scaled, -1, 1).astype(np.float32)

    # ============================
    #  SNR LOOP
    # ============================
    for snr in snr_list:
        print(f"  → Evaluating SNR={snr}")

        y_true_all_snr = []
        y_pred_all_snr = []

        for x, y in loader:
            wav = x.squeeze(1).cpu().numpy()

            # 🔥 batch=1 shape fix
            if wav.ndim == 1:
                wav = wav.reshape(1, -1)

            y = ds_idx_to_ckpt_idx[y.numpy()]

            mixed_list = []

            for wav_i in wav:
                if snr == "clean":
                    mixed = wav_i
                else:
                    # Load one random noise wav
                    nf = random.choice(noise_files)
                    n, sr = sf.read(nf)

                    # channel average
                    if n.ndim > 1:
                        n = n.mean(axis=1)

                    # SR mismatch → resample
                    if sr != TARGET_SR:
                        n = librosa.resample(n.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

                    # ensure length
                    if len(n) < WIN_LEN:
                        reps = int(np.ceil(WIN_LEN / len(n)))
                        n = np.tile(n, reps)

                    st = random.randint(0, len(n) - WIN_LEN)
                    noise_seg = n[st:st+WIN_LEN].astype(np.float32)

                    mixed = add_noise_snr(wav_i, noise_seg, snr)

                mixed_list.append(mixed)

            mixed_tensor = torch.tensor(np.array(mixed_list)).unsqueeze(1).float().to(DEVICE)

            with torch.no_grad():
                logits = model(mixed_tensor)
                pred = torch.argmax(logits, dim=1)

            y_true_all_snr.extend(y.tolist())
            y_pred_all_snr.extend(pred.cpu().numpy().tolist())

        # 🔥 class-wise F1 저장
        class_f1 = f1_score(y_true_all_snr, y_pred_all_snr, average=None)
        class_f1_dict[snr] = class_f1

    # ============================================================
    #  SNR Class-wise F1 Plot (등간격 수정 버전)
    # ============================================================
    
    # x축 좌표를 0, 1, 2... 인덱스로 설정
    x_indices = range(len(snr_list))
    xtick_labels = ["-5", "0", "5", "10", "20", "Clean"]

    plt.figure(figsize=(10, 6))
    for cls_idx, cls_name in enumerate(target_names):
        plt.plot(
            x_indices,  # 실제 값이 아닌 인덱스를 x좌표로 사용
            [class_f1_dict[s][cls_idx] for s in snr_list],
            marker='o', linewidth=2, label=cls_name
        )

    plt.xticks(x_indices, xtick_labels) # 인덱스 위치에 라벨 매핑
    plt.ylim(0, 1.05)
    plt.xlabel("SNR (dB)")
    plt.ylabel("F1-score")
    plt.title("Class-wise F1 vs SNR : Clean Model (Noise → Clean)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig("chk/classwise_f1_snr_evaluate.png", dpi=200)
    print("\n📈 Saved: chk/classwise_f1_snr_evaluate.png")