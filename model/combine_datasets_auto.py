import os
import torch
import soundfile as sf
from torch.utils.data import Dataset
from pathlib import Path

# ==========================================
# ✅ Drone Dataset Loader (local folder only)
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "DroneUnified"
DRONE_DIR = BASE_DIR / "drone"
NODRONE_DIR = BASE_DIR / "no_drone"

# 폴더 자동 생성
for d in [BASE_DIR, DRONE_DIR, NODRONE_DIR]:
    os.makedirs(d, exist_ok=True)

# 데이터셋 로드 클래스
class DroneDataset(Dataset):
    def __init__(self, transform=None, target_sr=16000, duration=3.0):
        self.drone_files = list(DRONE_DIR.glob("*.wav"))
        self.no_drone_files = list(NODRONE_DIR.glob("*.wav"))
        self.files = [(f, 1) for f in self.drone_files] + [(f, 0) for f in self.no_drone_files]

        self.transform = transform
        self.target_sr = target_sr
        self.duration = duration
        self.samples = int(target_sr * duration)

        print(f"[Dataset] drone: {len(self.drone_files)} files, no_drone: {len(self.no_drone_files)} files")
        if len(self.files) == 0:
            print("⚠️  Warning: dataset folder is empty. Please upload .wav files before training.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav).float()

        # 패딩 or 자르기
        if len(wav) < self.samples:
            wav = torch.cat([wav, torch.zeros(self.samples - len(wav))])
        elif len(wav) > self.samples:
            wav = wav[:self.samples]

        if self.transform:
            wav = self.transform(wav)

        return wav, torch.tensor(label, dtype=torch.float32)