
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

    # HF가 soundfile로 로드하면 array + sampling_rate 줌
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

        y = float(int(ex["label"]))
        return torch.from_numpy(w), torch.tensor(y, dtype=torch.float32)

    # def __getitem__(self, i):
    #     ex = self.ds[i]
        
    #     # ✅ 오디오 데이터 추출
    #     audio_data = ex["audio"]
        
    #     # ✅ DummyTorchCodec이 반환된 경우 -> path로 직접 로드
    #     if isinstance(audio_data, DummyTorchCodec):
    #         # torchcodec이 차단되었으므로, 원본 경로에서 직접 로드
    #         # HuggingFace 데이터셋의 파일 경로는 ex에서 직접 접근
    #         try:
    #             # 데이터셋 캐시 경로를 통해 파일 찾기
    #             if hasattr(self.ds, '_data_files') and self.ds._data_files:
    #                 # 데이터 파일 경로가 있는 경우
    #                 base_path = self.ds._data_files[0]['filename']
    #                 print(f"[Info] Using fallback: loading from dataset cache")
                
    #             # ex 자체에 숨겨진 경로가 있을 수 있음
    #             if '_data_files' in dir(ex):
    #                 audio_path = ex['_data_files']
    #             else:
    #                 # 최후의 수단: 인덱스 i로 원본 데이터에 접근
    #                 print(f"[Warning] DummyTorchCodec detected at index {i}, trying raw access...")
    #                 # 원본 데이터셋을 다시 로드 (비효율적이지만 동작함)
    #                 raw_ex = self.ds.data.to_pydict()
    #                 audio_path = raw_ex['audio'][i]['path']
                
    #             w, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    #         except Exception as e:
    #             print(f"[Error] Failed to handle DummyTorchCodec at index {i}: {e}")
    #             # 빈 오디오 생성
    #             w = np.zeros(WIN_SAMPLES, dtype=np.float32)
        
    #     # dict 형태인 경우
    #     elif isinstance(audio_data, dict):
    #         # path가 있는 경우
    #         if "path" in audio_data and audio_data["path"]:
    #             audio_path = audio_data["path"]
    #             try:
    #                 w, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    #             except Exception as e:
    #                 print(f"[Warning] Failed to load {audio_path}: {e}")
    #                 w = np.zeros(WIN_SAMPLES, dtype=np.float32)
            
    #         # array가 직접 있는 경우
    #         elif "array" in audio_data:
    #             w = np.array(audio_data["array"], dtype=np.float32)
    #             # 스테레오인 경우 모노로 변환
    #             if len(w.shape) > 1:
    #                 w = w.mean(axis=1)
    #             # 리샘플링 필요 여부 확인
    #             sr = audio_data.get("sampling_rate", TARGET_SR)
    #             if sr != TARGET_SR:
    #                 w = librosa.resample(w, orig_sr=sr, target_sr=TARGET_SR)
            
    #         # bytes 데이터인 경우
    #         elif "bytes" in audio_data:
    #             import io
    #             audio_bytes = audio_data["bytes"]
    #             try:
    #                 w, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    #                 if len(w.shape) > 1:
    #                     w = w.mean(axis=1)
    #                 if sr != TARGET_SR:
    #                     w = librosa.resample(w, orig_sr=sr, target_sr=TARGET_SR)
    #             except Exception as e:
    #                 print(f"[Warning] Failed to decode bytes: {e}")
    #                 w = np.zeros(WIN_SAMPLES, dtype=np.float32)
            
    #         else:
    #             print(f"[Warning] Unknown dict audio format at index {i}: {audio_data.keys()}")
    #             w = np.zeros(WIN_SAMPLES, dtype=np.float32)
        
    #     # array 형태로 바로 온 경우
    #     elif isinstance(audio_data, np.ndarray):
    #         w = audio_data.astype(np.float32)
        
    #     else:
    #         print(f"[Error] Unexpected audio type at index {i}: {type(audio_data)}")
    #         # 빈 오디오로 대체
    #         w = np.zeros(WIN_SAMPLES, dtype=np.float32)
        
    #     # 전처리
    #     w = w.astype(np.float32)
    #     w = self._rand_window(w)
        
    #     if self.augment:
    #         w = self._augment(w)
        
    #     y = float(int(ex["label"]))
    #     return torch.from_numpy(w), torch.tensor(y, dtype=torch.float32)

def make_loader(hf_ds, train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True):
    ds = WindowedAudio(hf_ds, train=train, augment=True)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=(shuffle and train),
        num_workers=num_workers, 
        pin_memory=True
    )