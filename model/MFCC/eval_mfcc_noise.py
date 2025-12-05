# model/eval_mfcc_clean.py

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from model.dataset_drone import load_split
from model.MFCC.model_drone_mfcc import DroneMultiClassifierMFCC
from model.config import TARGET_SR, WIN_SEC
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1) NoisePool 평가용 (augment.py에서 가져온 논리)
# ============================================================

NOISE_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "DroneUnified" / "noise"
SEG_LEN = int(TARGET_SR * WIN_SEC)


def load_noise_file(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


# 폴더 내부 모든 noise 파일 로드
noise_files = []
for cat in NOISE_ROOT.iterdir():
    if cat.is_dir():
        noise_files += list(cat.glob("*.wav")) + list(cat.glob("*.WAV"))

if len(noise_files) == 0:
    raise RuntimeError(f"❌ noise 파일 없음: {NOISE_ROOT}")

# noise 파일들 메모리에 올림
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
# 2) SNR Mixing
# ============================================================
def mix_snr(clean, noise_seg, snr_db):
    if snr_db == "clean":
        return clean

    cp = np.mean(clean**2) + 1e-9
    npow = np.mean(noise_seg**2) + 1e-9
    target = cp / (10 ** (snr_db / 10))
    scale = np.sqrt(target / npow)

    mixed = clean + noise_seg * scale
    return np.clip(mixed, -1.0, 1.0)


# ============================================================
# 3) 데이터셋 로드 (clean test only)
# ============================================================
_, _, test_ds, label_map = load_split()
loader = DataLoader(test_ds, batch_size=1, shuffle=False)

n_classes = len(label_map)
print(f"📂 Evaluating MFCC model with {n_classes} classes")


# ============================================================
# 4) 모델 로드 (noise-trained MFCC)
# ============================================================
ckpt = torch.load("chk/mfcc_best_noise.pt", map_location=DEVICE)

model = DroneMultiClassifierMFCC(n_classes).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()


# ============================================================
# 5) 평가 루프
# ============================================================
SNR_LIST = ["clean", -5, 0, 5, 10, 20]

for snr in SNR_LIST:
    print(f"\n🔊 Evaluating at SNR={snr}")
    y_true, y_pred = [], []

    for x, y in loader:
        clean = x.squeeze().numpy().astype(np.float32)

        if snr == "clean":
            noisy = clean
        else:
            noise_seg = get_noise_segment_eval()
            noisy = mix_snr(clean, noise_seg, snr)

        noisy_tensor = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(noisy_tensor)
            pred = torch.argmax(logits, dim=1).item()

        y_true.append(int(y))
        y_pred.append(pred)

    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"🎯 Macro-F1 @ SNR={snr}: {f1:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=list(label_map.keys()),
        digits=4
    ))
