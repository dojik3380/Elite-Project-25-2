# model/eval_clean_with_noise.py

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model.dataset_drone import load_split
from model.model_drone import DroneMultiClassifier
from model.config import TARGET_SR, WIN_SEC, DEVICE


# ============================================================
# 1) 데이터셋 로드 (clean)
# ============================================================
train_ds, val_ds, test_ds, label_map = load_split()
idx_to_label = {v: k for k, v in label_map.items()}
n_classes = len(label_map)

loader = DataLoader(test_ds, batch_size=1, shuffle=False)


# ============================================================
# 2) 노이즈 wav 불러오기
# ============================================================
NOISE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "DroneUnified" / "noise"
noise_files = list(NOISE_DIR.glob("*.wav")) + list(NOISE_DIR.glob("*.WAV"))

if len(noise_files) == 0:
    raise RuntimeError("❌ noise wav 없음: datasets/DroneUnified/noise/ 폴더에 넣어야 함")

noise, sr = sf.read(noise_files[0])
if noise.ndim > 1:
    noise = noise.mean(axis=1)

if sr != TARGET_SR:
    import librosa
    noise = librosa.resample(noise, sr, TARGET_SR)

TARGET_LEN = int(TARGET_SR * WIN_SEC)
if len(noise) < TARGET_LEN:
    reps = int(np.ceil(TARGET_LEN / len(noise)))
    noise = np.tile(noise, reps)
noise = noise[:TARGET_LEN]


# ============================================================
# 3) SNR 섞기 함수
# ============================================================
def add_noise_snr(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2) + 1e-9
    noise_power = np.mean(noise ** 2) + 1e-9

    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)

    noisy = clean + noise * scale
    return np.clip(noisy, -1.0, 1.0)


# ============================================================
# 4) 클린 모델 로드
# ============================================================
ckpt_path = "chk/best.pt"
ckpt = torch.load(ckpt_path, map_location=DEVICE)

model = DroneMultiClassifier(n_classes=n_classes).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"\n🎯 Loaded CLEAN model: {ckpt_path}\n")


# ============================================================
# 5) 평가 SNR 목록
# ============================================================
snr_list = [-5, 0, 5, 10, 20]
results = {}


# ============================================================
# 6) SNR별 성능 평가
# ============================================================
for snr in snr_list:
    print(f"\n🔊 Evaluating CLEAN model at SNR {snr} dB")

    y_true, y_pred = [], []

    for x, y in loader:
        wav = x.squeeze(0).squeeze(0).numpy()

        noisy = add_noise_snr(wav, noise, snr)
        noisy_tensor = torch.tensor(noisy).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logit = model(noisy_tensor)
            pred = torch.argmax(logit, dim=1).item()

        y_true.append(y.item())
        y_pred.append(pred)

    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro-F1 @ {snr} dB = {f1:.4f}")

    print(classification_report(y_true, y_pred, target_names=list(label_map.keys()), digits=4))

    results[snr] = f1


# ============================================================
# 7) F1 vs SNR 그래프 저장
# ============================================================
plt.figure(figsize=(7, 5))
plt.plot(snr_list, [results[s] for s in snr_list], marker="o")
plt.title("CLEAN Model: Macro-F1 vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Macro-F1")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("chk/clean_model_noise_eval_f1.png")
plt.close()

print("\n📁 Saved graph: chk/clean_model_noise_eval_f1.png")
