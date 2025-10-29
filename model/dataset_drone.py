
import sys
import os
from types import ModuleType

# ✅ 환경변수 먼저 설정
os.environ["HF_AUDIO_BACKEND_FORCED"] = "soundfile"
os.environ["HF_DISABLE_AUDIO_DECODING_BACKEND"] = "torchcodec"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .config import TARGET_SR, WIN_SEC, BATCH_SIZE, NUM_WORKERS
import librosa
import soundfile as sf

# 이제 datasets를 import
from datasets import load_dataset

WIN_SAMPLES = int(TARGET_SR * WIN_SEC)

def load_split(seed=42, test_size=0.2, val_size=0.5):
    """
    HuggingFace datasets 로드 시에 torchcodec을 전혀 사용하지 않음.
    """
    print("Loading dataset from HuggingFace...")
    ds = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        split="train",
    )
    
    print(f"Dataset loaded: {len(ds)} samples")
    labels = np.array([int(x["label"]) for x in ds])
    idx = np.arange(len(ds))
    
    tr, tmp = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    val, te = train_test_split(tmp, test_size=val_size, stratify=labels[tmp], random_state=seed)
    
    print(f"Split: train={len(tr)}, val={len(val)}, test={len(te)}")
    return ds.select(tr), ds.select(val), ds.select(te)

class WindowedAudio(Dataset):
    def __init__(self, hf_ds, train=True, augment=True):
        self.ds = hf_ds
        self.train = train
        self.augment = (augment and train)

    def __len__(self):
        return len(self.ds)

    def _rand_window(self, w):
        n = len(w)
        if n >= WIN_SAMPLES:
            st = random.randint(0, n - WIN_SAMPLES) if self.train else 0
            return w[st:st + WIN_SAMPLES]
        pad = WIN_SAMPLES - n
        return np.pad(w, (pad // 2, pad - pad // 2), mode="constant")
        
    def _augment(self, w):
        if random.random() < 0.5:
            gain = 10 ** (random.uniform(-3, 3) / 20.0)
            w = w * gain
        if random.random() < 0.5:
            shift = random.randint(-int(0.2 * TARGET_SR), int(0.2 * TARGET_SR))
            w = np.roll(w, shift)
        if random.random() < 0.3:
            w = w + np.random.randn(len(w)) * random.uniform(0.001, 0.01)
        return w
    def __getitem__(self, i):
        ex = self.ds[i]
        audio = ex["audio"]
        if isinstance(audio, dict) and "array" in audio:
            w = np.array(audio["array"], dtype=np.float32)
            sr = audio.get("sampling_rate", TARGET_SR)
        else:
            print(f"[SKIP] No audio data at index {i}")
            w = np.zeros(WIN_SAMPLES, dtype=np.float32)
            sr = TARGET_SR

        if sr != TARGET_SR:
            w = librosa.resample(w, orig_sr=sr, target_sr=TARGET_SR)

        w = w.astype(np.float32)
        w = self._rand_window(w)
        if self.augment:
            w = self._augment(w)

    # ✅ 모든 오디오 길이 맞추기
        if len(w) < WIN_SAMPLES:
            pad = WIN_SAMPLES - len(w)
            w = np.pad(w, (pad // 2, pad - pad // 2), mode="constant")
        elif len(w) > WIN_SAMPLES:
            w = w[:WIN_SAMPLES]

        assert len(w) == WIN_SAMPLES, f"Unexpected length: {len(w)}"

        y = float(int(ex["label"]))
        wav = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0).float()  # [1, T]
        if wav.shape[1] != WIN_SAMPLES:
            pad = WIN_SAMPLES - wav.shape[1]
            if pad > 0:
                wav = torch.nn.functional.pad(wav, (0, pad))
        return wav, torch.tensor(y, dtype=torch.float32)

def make_loader(hf_ds, train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True):
    ds = WindowedAudio(hf_ds, train=train, augment=True)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=(shuffle and train),
        num_workers=num_workers, 
        pin_memory=True
    )