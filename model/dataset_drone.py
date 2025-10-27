import random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
from sklearn.model_selection import train_test_split
from .config import TARGET_SR, WIN_SEC, BATCH_SIZE, NUM_WORKERS

WIN_SAMPLES = int(TARGET_SR * WIN_SEC)

def load_split(seed=42, test_size=0.2, val_size=0.5):
    ds = load_dataset("geronimobasso/drone-audio-detection-samples", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    labels = np.array([int(x["label"]) for x in ds])
    idx = np.arange(len(ds))
    tr, tmp = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    val, te = train_test_split(tmp, test_size=val_size, stratify=labels[tmp], random_state=seed)
    return ds.select(tr), ds.select(val), ds.select(te)

class WindowedAudio(Dataset):
    def __init__(self, hf_ds, train=True, augment=True):
        self.ds, self.train, self.augment = hf_ds, train, (augment and train)

    def __len__(self): return len(self.ds)

    def _rand_window(self, w):
        n = len(w)
        if n >= WIN_SAMPLES:
            st = random.randint(0, n - WIN_SAMPLES) if self.train else 0
            return w[st:st+WIN_SAMPLES]
        pad = WIN_SAMPLES - n
        return np.pad(w, (pad//2, pad - pad//2), mode="constant")

    def _augment(self, w):
        if random.random() < 0.5:
            gain = 10 ** (random.uniform(-3, 3) / 20.0)
            w = w * gain
        if random.random() < 0.5:
            shift = random.randint(-int(0.2*TARGET_SR), int(0.2*TARGET_SR))
            w = np.roll(w, shift)
        if random.random() < 0.3:
            w = w + np.random.randn(len(w)) * random.uniform(0.001, 0.01)
        return w

    def __getitem__(self, i):
        ex = self.ds[i]
        w = ex["audio"]["array"].astype(np.float32)
        w = self._rand_window(w)
        if self.augment: w = self._augment(w)
        y = float(int(ex["label"]))
        return torch.from_numpy(w), torch.tensor(y, dtype=torch.float32)
# model/dataset_drone.py (하단의 make_loader만 수정)
from .config import BATCH_SIZE, NUM_WORKERS

def make_loader(hf_ds, train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True):
    ds = WindowedAudio(hf_ds, train=train, augment=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=(shuffle and train),
                      num_workers=num_workers, pin_memory=True)
