import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import soundfile as sf
from .config import WIN_SEC

# --- 기본 설정 ---
BASE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "DroneUnified"
TARGET_SR = 16000
SEGMENT_SEC = int(WIN_SEC)
TARGET_LEN = int(TARGET_SR * SEGMENT_SEC)

# --- 클래스 순서 (항상 7개 고정) ---
CLASS_ORDER = ["Airplane", "Bicycles", "Cars", "Drone", "Helicopter", "Motorcycles", "Train"]

def canonical_name(name: str) -> str:
    name = name.strip().lower()
    fixes = {
        "airplane": "Airplane", "airplanes": "Airplane",
        "bicycle": "Bicycles", "bicycles": "Bicycles",
        "car": "Cars", "cars": "Cars",
        "drone": "Drone",
        "helicopter": "Helicopter",
        "train": "Train",
        "motorcycle": "Motorcycles", "motorcycles": "Motorcycles",
    }
    return fixes.get(name, name.capitalize())


def _collect_audio_paths(cls_dir: Path):
    paths = set()
    paths.update(cls_dir.glob("*.wav"))
    paths.update(cls_dir.glob("*.WAV"))
    return sorted(paths)


# ============================================================
# 🔥 2초 segment 단위로 쪼개는 Dataset
# ============================================================
class DroneDataset(Dataset):
    def __init__(self, base_dir=BASE_DIR, transform=None, exclude_no_drone=True):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.segments = []   # (segment_np_array, label)
        self.label_map = {cls: i for i, cls in enumerate(CLASS_ORDER)}

        # --- 폴더 순회 ---
        raw_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        for d in raw_dirs:
            cname = canonical_name(d.name)

            if exclude_no_drone and cname.lower() == "no_drone":
                continue
            if cname not in self.label_map:
                print(f"⚠️ Unknown folder ignored: {d.name}")
                continue

            label_idx = self.label_map[cname]

            for wav_path in _collect_audio_paths(d):
                try:
                    wav, sr = sf.read(wav_path)
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    if sr != TARGET_SR:
                        import librosa
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
                except Exception as e:
                    print(f"⚠️ Read error {wav_path}: {e}")
                    continue

                total_len = len(wav)
                if total_len == 0:
                    continue

                # ============================================================
                # 🔥 여기서 실제로 2초 단위로 쪼갬 (손실 없음)
                # ============================================================
                num_segments = (total_len // TARGET_LEN)
                if num_segments == 0:
                    # 2초보다 짧은 파일은 pad해서 1개 segment로 만듬
                    padded = np.pad(wav, (0, TARGET_LEN - total_len))
                    self.segments.append((padded.astype(np.float32), label_idx))
                else:
                    # 2초씩 반복으로 추출 (손실 없음)
                    for i in range(num_segments):
                        seg = wav[i * TARGET_LEN : (i + 1) * TARGET_LEN]
                        self.segments.append((seg.astype(np.float32), label_idx))

        # --- 통계 출력 ---
        print(f"📂 Total segments: {len(self.segments)}")
        for cls, idx in self.label_map.items():
            cnt = len([1 for _, y in self.segments if y == idx])
            print(f"   {idx}: {cls} = {cnt} segments")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        wav, label = self.segments[idx]
        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y


# ============================================================
# 🔥 segment 기반 split
# ============================================================
from sklearn.model_selection import train_test_split

def load_split(base_dir=BASE_DIR, seed=42):
    ds = DroneDataset(base_dir)
    labels = np.array([label for _, label in ds.segments])

    train_idx, temp_idx = train_test_split(
        np.arange(len(ds)), test_size=0.2, stratify=labels, random_state=seed
    )

    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    test_ds = torch.utils.data.Subset(ds, test_idx)

    return train_ds, val_ds, test_ds, ds.label_map
