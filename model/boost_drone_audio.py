# model/boost_drone_audio.py
import soundfile as sf, numpy as np
from pathlib import Path

DRONE_DIR = Path("/home/elite/DeepLearningProject/EliteProject/datasets/DroneUnified/Drone")
TARGET_RMS = 0.02    # 목표 음압 (대략 -34 ~ -30 dBFS 정도)
MAX_GAIN   = 6.0     # 과도 증폭 제한 (배)
COUNT = 0

for wav_path in DRONE_DIR.glob("*.wav"):
    name = wav_path.stem
    # 한글 파일만 우선 대상으로 삼되, RMS가 이미 충분하면 스킵
    if not any('\uac00' <= ch <= '\ud7a3' for ch in name):
        continue
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)

    rms = np.sqrt(np.mean(wav**2) + 1e-12)
    if rms >= TARGET_RMS:  # 이미 충분히 큰 파일은 건드리지 않음
        continue

    gain = min(MAX_GAIN, TARGET_RMS / max(rms, 1e-8))
    boosted = np.clip(wav * gain, -1.0, 1.0)
    sf.write(wav_path, boosted, sr)
    COUNT += 1
    print(f"🔊 boosted: {wav_path.name} (rms {rms:.4f} → target {TARGET_RMS:.4f}, ×{gain:.2f})")

print(f"\n✅ {COUNT}개 파일 볼륨 보정 완료.")
