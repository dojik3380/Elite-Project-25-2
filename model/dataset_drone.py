# model/dataset_drone.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import os, requests, zipfile, tarfile, io
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "DroneUnified"
DRONE_DIR = BASE_DIR / "drone"
NODRONE_DIR = BASE_DIR / "no_drone"

# 🧩 자동 다운로드 함수
def download_and_prepare_datasets():
    os.makedirs(DRONE_DIR, exist_ok=True)
    os.makedirs(NODRONE_DIR, exist_ok=True)

    sources = [
        {
            "name": "DroneNoise",
            "url": "https://salford.figshare.com/ndownloader/files/44140669",  # 실제 zip 파일
            "type": "zip"
        },
        {
            "name": "DroneAudioDataset_Sara",
            "url": "https://saraalemadi.com/wp-content/uploads/2019/01/Drone-Audio-Dataset.zip",
            "type": "zip"
        },
        {
            "name": "Kaggle_YehielLevi",
            "url": "https://github.com/yehiellevi/Drone-Sound-Detection/raw/main/data/drone_sound_dataset.zip",
            "type": "zip"
        }
    ]

    for src in sources:
        name = src["name"]
        url = src["url"]
        dst = BASE_DIR / f"{name}.zip"
        if not dst.exists():
            print(f"⬇️  Downloading {name} dataset ...")
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Downloaded {dst.name}")

                # 압축 해제
                with zipfile.ZipFile(dst, "r") as zip_ref:
                    zip_ref.extractall(BASE_DIR / name)
                    print(f"📦 Extracted {name}")
            except Exception as e:
                print(f"❌ Failed to download {name}: {e}")

    # 간단히 모든 폴더를 통합
    for root, _, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith(".wav"):
                src_path = os.path.join(root, f)
                # 분류 폴더명 판단
                if "no" in f.lower() or "background" in f.lower():
                    dst_path = NODRONE_DIR / f
                else:
                    dst_path = DRONE_DIR / f
                try:
                    os.rename(src_path, dst_path)
                except Exception:
                    pass

    print("[✅] All datasets prepared under DroneUnified/")

# 폴더 비어있으면 자동 다운로드 수행
if not any(DRONE_DIR.glob("*.wav")) and not any(NODRONE_DIR.glob("*.wav")):
    download_and_prepare_datasets()


for d in [BASE_DIR, DRONE_DIR, NODRONE_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        print(f"[Info] Created folder: {d}")

# 🔹 폴더 내 파일 개수 확인
def check_dataset_files():
    count_drone = len([f for f in os.listdir(DRONE_DIR) if f.endswith(".wav")])
    count_no_drone = len([f for f in os.listdir(NODRONE_DIR) if f.endswith(".wav")])
    print(f"[Dataset] drone: {count_drone} files, no_drone: {count_no_drone} files")
    if count_drone == 0 or count_no_drone == 0:
        print("⚠️  Warning: dataset folder is empty. Please upload .wav files before training.")
    return count_drone + count_no_drone

check_dataset_files()


class DroneDataset(Dataset):
    """
    DroneUnified 통합 데이터셋용 클래스.
    구조:
        datasets/DroneUnified/
            ├── drone/
            └── no_drone/
    """
    def __init__(self, root_dir="datasets/DroneUnified", target_sr=16000, max_sec=4):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.max_len = target_sr * max_sec
        self.data = []

        # 드론 / 노드론 파일 스캔
        for label_name in ["drone", "no_drone"]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder):
                print(f"[Warning] Missing folder: {folder}")
                continue

            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    path = os.path.join(folder, file)
                    label = 1 if label_name == "drone" else 0
                    self.data.append((path, label))

        print(f"✅ DroneUnified Dataset Loaded ({len(self.data)} files)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            wav, sr = sf.read(path)
        except Exception as e:
            print(f"[Error] {path}: {e}")
            wav = torch.zeros(self.max_len)
            sr = self.target_sr

        # mono 변환
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # 리샘플링
        if sr != self.target_sr:
            wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=self.target_sr)

        # pad or crop
        wav = torch.tensor(wav[:self.max_len], dtype=torch.float32)
        if len(wav) < self.max_len:
            wav = F.pad(wav, (0, self.max_len - len(wav)))

        return wav.unsqueeze(0), torch.tensor(label, dtype=torch.float32)

from sklearn.model_selection import train_test_split

def load_split(root_dir="datasets/DroneUnified", test_size=0.2, val_size=0.1, seed=42):
    full_ds = DroneDataset(root_dir=root_dir)
    indices = list(range(len(full_ds)))
    labels = [lbl for _, lbl in full_ds.data]

    # Train / temp split
    train_idx, temp_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed
    )

    # Val / test split from temp
    val_rel_size = val_size / test_size
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=val_rel_size, stratify=[labels[i] for i in temp_idx], random_state=seed
    )

    def subset(dataset, indices):
        return torch.utils.data.Subset(dataset, indices)

    tr_ds = subset(full_ds, train_idx)
    va_ds = subset(full_ds, val_idx)
    te_ds = subset(full_ds, test_idx)

    print(f"[Split] Train={len(tr_ds)}, Val={len(va_ds)}, Test={len(te_ds)}")
    return tr_ds, va_ds, te_ds
