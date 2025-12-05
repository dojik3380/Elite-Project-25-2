# model/eval_compare_frontends.py

import torch
import numpy as np
import soundfile as sf
import random
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# LEAF & MFCC 모델
from model.model_drone import DroneMultiClassifier
from model.MFCC.model_drone_mfcc import DroneMultiClassifierMFCC

# dataset
from model.dataset_drone import load_split
from model.config import TARGET_SR, WIN_SEC
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1) NoisePool 로드 (평가 전용) - 파일 뒤 30%에서 랜덤 2초
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
NOISE_ROOT = ROOT / "datasets" / "DroneUnified" / "noise" / "testingnoise"
SEG_LEN = int(TARGET_SR * WIN_SEC)

# noise 파일들 수집
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


# NoiseBank 생성
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
def mix_snr(clean, noise_seg, snr_db):
    if snr_db == "clean":
        return clean

    cp = np.mean(clean ** 2) + 1e-9
    npow = np.mean(noise_seg ** 2) + 1e-9
    target = cp / (10 ** (snr_db / 10))
    scale = np.sqrt(target / npow)

    return np.clip(clean + noise_seg * scale, -1.0, 1.0)


# ============================================================
# 3) Load 4 checkpoints (LEAF_clean / LEAF_noise / MFCC_clean / MFCC_noise)
# ============================================================
ckpt_paths = {
    "LEAF_clean": ROOT / "chk" / "best.pt",
    "LEAF_noise": ROOT / "chk" / "best_noise_val.pt",
    "MFCC_clean": ROOT / "chk" / "mfcc_best.pt",
    "MFCC_noise": ROOT / "chk" / "mfcc_best_noise.pt",
}

models, label_maps = {}, {}

for name, ckpt_path in ckpt_paths.items():
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    label_map = ckpt["label_map"]
    n_classes = len(label_map)

    if "MFCC" in name:
        model = DroneMultiClassifierMFCC(n_classes).to(DEVICE)
    else:
        model = DroneMultiClassifier(n_classes).to(DEVICE)

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    models[name] = model
    label_maps[name] = label_map


print("\n🎯 Loaded models:", list(models.keys()))


# ============================================================
# 4) Dataset 로드 (test only)
# ============================================================
_, _, test_ds, ds_label_map = load_split()
loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# dataset idx → label name 매핑
ds_idx_to_name = {v: k for k, v in ds_label_map.items()}


# ============================================================
# 5) Evaluation Loop
# ============================================================
SNR_LIST = [-5, 0, 5, 10, 20, "clean"]
results = {name: [] for name in models.keys()}

for name, model in models.items():
    print(f"\n🔥 Evaluating model: {name}")

    label_map_ckpt = label_maps[name]
    # mapping: ds_label_idx → ckpt_label_idx
    ds_to_ckpt = np.zeros(len(ds_label_map), dtype=np.int64)
    for cls_name, ds_idx in ds_label_map.items():
        ds_to_ckpt[ds_idx] = label_map_ckpt[cls_name]

    for snr in SNR_LIST:
        y_true, y_pred = [], []

        for x, y in loader:
            clean = x.squeeze().numpy().astype(np.float32)
            """
            rms = np.sqrt(np.mean(clean ** 2) + 1e-12)
            if rms < 0.015:  # threshold
                target_rms = 0.03
                max_auto_boost_gain = 6.0
                gain = min(target_rms / max(rms, 1e-8), max_auto_boost_gain)
                clean = np.clip(clean * gain, -1.0, 1.0) 
            """
            if snr == "clean":
                noisy = clean
            else:
                noise_seg = get_noise_segment_eval()
                noisy = mix_snr(clean, noise_seg, snr)

            noisy_tensor = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_idx = torch.argmax(model(noisy_tensor), dim=1).item()

            y_true.append(int(ds_to_ckpt[y.item()]))
            y_pred.append(pred_idx)

        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"  SNR {snr}: F1 = {f1:.4f}")
        results[name].append(f1)


# ============================================================
# 6) 그래프 저장
# ============================================================
plt.figure(figsize=(9, 6))

styles = {
    "LEAF_clean": ("blue", "--"),
    "LEAF_noise": ("blue", "-"),
    "MFCC_clean": ("red", "--"),
    "MFCC_noise": ("red", "-"),
}

for name, f1_list in results.items():
    color, linestyle = styles[name]
    plt.plot(SNR_LIST, f1_list, label=name, color=color, linestyle=linestyle,
             marker="o", linewidth=2)

plt.xlabel("SNR (dB) - -5 → 0 → 5 → 10 → 20 → clean(∞)")
plt.ylabel("Macro-F1")
plt.title("LEAF vs MFCC | Clean vs Noise-Trained Robustness Comparison")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

SAVE_PATH = ROOT / "chk" / "frontend_compare.png"
plt.savefig(SAVE_PATH)
plt.close()

print(f"\n📁 Saved comparison graph:")
print(SAVE_PATH)
print("✅ Evaluation Complete.")
