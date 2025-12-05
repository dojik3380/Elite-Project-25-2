# model/evaluate_noise.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import random
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report, f1_score
)
from torch.utils.data import DataLoader

from model.dataset_drone import load_split
from model.model_drone import DroneMultiClassifier
from model.config import TARGET_SR, WIN_SEC

DEVICE=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1) NoisePool (평가 전용): noise 파일 뒤 30% → random 2초 crop
# ============================================================
BASE = Path(__file__).resolve().parents[1]
NOISE_ROOT = BASE / "datasets" / "DroneUnified" / "noise"

SEG_LEN = int(TARGET_SR * WIN_SEC)

# noise 파일 수집
noise_files = []
for cat in NOISE_ROOT.iterdir():
    if cat.is_dir():
        noise_files += list(cat.glob("*.wav")) + list(cat.glob("*.WAV"))

if len(noise_files) == 0:
    raise RuntimeError(f"❌ noise 파일 없음: {NOISE_ROOT}")


def load_noise_file(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


NOISE_BANK = [load_noise_file(p) for p in noise_files]


def get_noise_segment_eval():
    import random
    wav = random.choice(NOISE_BANK)
    L = len(wav)

    if L < SEG_LEN:
        reps = int(np.ceil(SEG_LEN / L))
        return np.tile(wav, reps)[:SEG_LEN]

    start = np.random.randint(0, L - SEG_LEN)
    return wav[start:start + SEG_LEN]


# ============================================================
# 2) SNR mixing
# ============================================================
def add_noise_snr(clean, noise_seg, snr_db):
    if snr_db == "clean":
        return clean

    cp = np.mean(clean ** 2) + 1e-9
    npow = np.mean(noise_seg ** 2) + 1e-9
    target = cp / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target / npow)

    mixed = clean + noise_seg * scale
    return np.clip(mixed, -1.0, 1.0)


# ============================================================
# 3) Load dataset (clean)
# ============================================================
train_ds, val_ds, test_ds, label_map = load_split()
n_classes = len(label_map)
idx_to_label = {v: k for k, v in label_map.items()}

loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print(f"📂 Loaded CLEAN dataset for noise evaluation")
print(f"Test samples: {len(test_ds)}")


# ============================================================
# 4) Load noise-trained model
# ============================================================
model_path = "chk/best_noise_val.pt"
ckpt = torch.load(model_path, map_location=DEVICE)

model = DroneMultiClassifier(n_classes).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()


# ============================================================
# 5) Evaluation per SNR
# ============================================================
snr_list = [-5, 0, 5, 10, 20, 'clean']

results_f1 = {}
macro_auc = {}
y_true_by_snr = {}
y_pred_by_snr = {}
reports_by_snr = {}
class_f1_dict = {}

os.makedirs("chk/noise_eval", exist_ok=True)

for snr in snr_list:

    print(f"\n🔊 Evaluating at SNR={snr} dB")

    y_true, y_pred, y_score = [], [], []

    for x, y in loader:
        clean = x.squeeze().numpy().astype(np.float32)

        # NoisePool에서 즉석으로 noise crop 생성
        noise_seg = get_noise_segment_eval()

        noisy = add_noise_snr(clean, noise_seg, snr)

        noisy_tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(noisy_tensor)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(prob))
        y_true.append(int(y))
        y_pred.append(pred)
        y_score.append(prob)

    y_score = np.array(y_score)
    y_true_by_snr[snr] = y_true
    y_pred_by_snr[snr] = y_pred

    # classification report
   # === 전체 macro F1 출력
# === 전체 macro F1 출력
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nSNR {snr} → Macro-F1 = {f1:.4f}", flush=True)

# === classification report 출력
    report_text = classification_report(
        y_true, y_pred,
        target_names=list(label_map.keys()),
        digits=4
    )
    print(report_text, flush=True)

    reports_by_snr[snr] = report_text   # ← report가 아니라 report_text 저장


    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(label_map.keys()))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix (SNR={snr} dB)")
    plt.tight_layout()
    plt.savefig(f"chk/noise_eval/confmat_{snr}.png")
    plt.close()

    # class-wise AUC
    auc_list = []
    for c in range(n_classes):
        y_true_bin = (np.array(y_true) == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, c])
        auc_list.append(auc(fpr, tpr))

    macro_auc[snr] = float(np.mean(auc_list))
    results_f1[snr] = f1

    # === Class-wise F1 저장 ===
    class_f1 = f1_score(y_true, y_pred, average=None)
    class_f1_dict[snr] = class_f1


# ============================================================
# 6) CLEAN model evaluation
# ============================================================
print("\n============================")
print("🔍 CLEAN model evaluation")
print("============================")

clean_ckpt = torch.load("chk/best.pt", map_location=DEVICE)
clean_model = DroneMultiClassifier(n_classes).to(DEVICE)
clean_model.load_state_dict(clean_ckpt["model"])
clean_model.eval()

clean_y_true, clean_y_pred, clean_y_score = [], [], []

for x, y in loader:
    x = x.to(DEVICE)
    with torch.no_grad():
        logits = clean_model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(prob)

    clean_y_true.append(int(y))
    clean_y_pred.append(pred)
    clean_y_score.append(prob)

clean_y_true_arr = np.array(clean_y_true)
clean_score_arr = np.array(clean_y_score)

clean_macro_f1 = f1_score(clean_y_true, clean_y_pred, average="macro")
print(f"[CLEAN MODEL] Macro-F1 = {clean_macro_f1:.4f}")

# ============================================================
# 7) Plot graphs (Macro-F1, Class-wise, Macro-AUC)
# ============================================================

# Macro-F1
plt.figure()
plt.plot(snr_list, [results_f1[s] for s in snr_list], marker="o", label="Noise-trained")
plt.axhline(y=clean_macro_f1, color="red", linestyle="--", label="Clean-trained")
plt.xlabel("SNR (dB)")
plt.ylabel("Macro-F1")
plt.title("Macro-F1 vs SNR")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("chk/noise_eval/macro_f1_vs_snr_compare.png")
plt.close()

print("\n🎉 All evaluation completed & plots saved at chk/noise_eval/")

# ============================================================
# (추가) Class-wise F1 vs SNR 그래프 생성
# ============================================================

class_names = ["Airplane", "Bicycles", "Cars", "Drone", "Helicopter", "Motorcycles", "Train"]

snr_values = list(class_f1_dict.keys())  # [-5,0,5,10,20,"clean"]

# 🔥 x축을 index 기반으로 바꿈
snr_for_plot = np.arange(len(snr_values))  # → [0,1,2,3,4,5]

plt.figure(figsize=(10,6))

for cls_idx, cls_name in enumerate(class_names):
    cls_f1_list = [class_f1_dict[s][cls_idx] for s in snr_values]

    plt.plot(
        snr_for_plot, cls_f1_list, marker='o', linewidth=2, label=cls_name
    )

plt.xticks(
    snr_for_plot,
    ["-5", "0", "5", "10", "20", "Clean"]   # 라벨만 표시
)

plt.ylim(0.5, 1.05)
plt.xlabel("SNR (dB)")
plt.ylabel("F1-score")
plt.title("Class-wise F1 vs SNR : LEAF Noise Model (Noise → Clean)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("chk/noise_eval/classwise_noise_f1_by_snr.png", dpi=200)

print("📊 Saved classwise_noise_f1_by_snr.png")


