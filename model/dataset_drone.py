import torch
from torch.utils.data import Dataset, random_split
from pathlib import Path
import numpy as np
import soundfile as sf

# --- 기본 설정 ---
BASE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "DroneUnified"
TARGET_SR = 16000
SEGMENT_SEC = 1
TARGET_LEN = TARGET_SR * SEGMENT_SEC

# --- 클래스 순서 (항상 7개 고정) ---
CLASS_ORDER = ["Airplane", "Bicycles", "Cars", "Drone", "Helicopter", "Motorcycles", "Train"]

def canonical_name(name: str) -> str:
    """폴더 이름 변형 방지 및 정규화 (대소문자/복수형 통합)"""
    name = name.strip().lower()
    fixes = {
        "airplane": "Airplane",
        "airplanes": "Airplane",
        "bicycle": "Bicycles",
        "bicycles": "Bicycles",
        "car": "Cars",
        "cars": "Cars",
        "drone": "Drone",
        "helicopter": "Helicopter",
        "train": "Train",
        "motorcycle": "Motorcycles",
        "motorcycles": "Motorcycles",
    }
    return fixes.get(name, name.capitalize())


def _collect_audio_paths(cls_dir: Path):
    """해당 클래스 폴더의 .wav 파일 리스트"""
    paths = set()
    paths.update(cls_dir.glob("*.wav"))
    paths.update(cls_dir.glob("*.WAV"))
    return sorted(paths)


class DroneDataset(Dataset):
    def __init__(self, base_dir=BASE_DIR, transform=None, exclude_no_drone=True):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.files = []

        # --- label_map 고정 생성 (CLASS_ORDER 기준) ---
        self.label_map = {cls: i for i, cls in enumerate(CLASS_ORDER)}

        # --- 실제 존재하는 폴더 탐색 ---
        raw_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        for d in raw_dirs:
            cname = canonical_name(d.name)
            if exclude_no_drone and cname.lower() == "no_drone":
                continue

            if cname not in self.label_map:
                print(f"⚠️ Unknown folder ignored: {d.name}")
                continue

            idx = self.label_map[cname]
            for w in _collect_audio_paths(d):
                self.files.append((w, idx))

        # --- 통계 출력 ---
        print(f"📂 Found {len(self.files)} files in {len(CLASS_ORDER)} classes:")
        for cname, idx in self.label_map.items():
            cnt = len([f for f, i in self.files if i == idx])
            print(f"   {idx}: {cname} ({cnt} files)")

        # --- 안전검증 (라벨 범위 점검) ---
        for path, label in self.files:
            if label < 0 or label >= len(CLASS_ORDER):
                raise ValueError(f"❌ Invalid label {label} for {path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        try:
            wav, sr = sf.read(path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if len(wav) == 0:
                raise ValueError("empty audio")
            if sr != TARGET_SR:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        except Exception as e:
            print(f"⚠️ Read error {path}: {e}")
            wav = np.zeros(TARGET_LEN, dtype=np.float32)

        # --- 1초 세그먼트로 정규화 ---
        if len(wav) > TARGET_LEN:
            start = np.random.randint(0, len(wav) - TARGET_LEN)
            wav = wav[start:start + TARGET_LEN]
        elif len(wav) < TARGET_LEN:
            wav = np.pad(wav, (0, TARGET_LEN - len(wav)))

        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y


def load_split(base_dir=BASE_DIR, seed=42):
    """데이터셋 분할"""
    ds = DroneDataset(base_dir)
    n = len(ds)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    torch.manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
    print(f"📊 Split → train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds, ds.label_map


if __name__ == "__main__":
    train_ds, val_ds, test_ds, label_map = load_split()
    print("✅ Dataset ready. Classes:", label_map)
