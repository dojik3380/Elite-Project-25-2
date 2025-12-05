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
# 🔥 1) NoisePool (augment.py 내용을 평가용으로 가져온 버전)
#     augment.py 안의 noise 관련 기능을 그대로 옮겨옴
# ============================================================

NOISE_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "DroneUnified" / "noise"
SEG_LEN = int(TARGET_SR * WIN_SEC)

# noise 파일 수집
noise_files = []
for cat in ["기상 수정", "총 수정"]:
    d = NOISE_ROOT / cat
    if d.exists():
        noise_files += list(d.glob("*.wav")) + list(d.glob("*.WAV"))

if len(noise_files) == 0:
    raise RuntimeError(f"❌ 평가용 noise 파일 없음: {NOISE_ROOT}")


def load_noise_wav(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


# noise 파일 전체 로딩해 캐싱
NOISE_BANK = [load_noise_wav(p) for p in noise_files]


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
# 🔥 2) SNR mixing
# ============================================================
def mix_snr(clean, noise_seg, snr_db):
    if snr_db == "clean":
        return clean

    cp = np.mean(clean ** 2) + 1e-9
    npow = np.mean(noise_seg ** 2) + 1e-9
    target = cp / (10 ** (snr_db / 10))
    scale = np.sqrt(target / npow)

    mixed = clean + noise_seg * scale
    return np.clip(mixed, -1.0, 1.0)


# ============================================================
# 🔥 3) Load test dataset
# ============================================================
_, _, test_ds, label_map = load_split()
loader = DataLoader(test_ds, batch_size=1, shuffle=False)


# ============================================================
# 🔥 4) Load MFCC noise-trained model
# ============================================================
ckpt = torch.load("chk/mfcc_best.pt", map_location=DEVICE)
model = DroneMultiClassifierMFCC(len(label_map)).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()


# ============================================================
# 🔥 5) Evaluation
# ============================================================
snr_list = ["clean", -5, 0, 5, 10, 20]

for snr in snr_list:
    y_true, y_pred = [], []

    print(f"\n🔊 Evaluating MFCC model @ SNR={snr}")

    for x, y in loader:
        clean = x.squeeze().numpy().astype(np.float32)   # clean wav

        if snr == "clean":
            noisy = clean
        else:
            noise_seg = get_noise_segment_eval()         # 🔥 2초 random noise
            noisy = mix_snr(clean, noise_seg, snr)

        noisy_tensor = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = torch.argmax(model(noisy_tensor), dim=1).item()

        y_true.append(int(y))
        y_pred.append(pred)

    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nSNR {snr} → Macro-F1 = {f1:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=list(label_map.keys()),
        digits=4
    ))

