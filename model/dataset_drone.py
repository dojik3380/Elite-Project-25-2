import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import os, random
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "DroneUnified"
DRONE_DIR = BASE_DIR / "drone"
NODRONE_DIR = BASE_DIR / "no_drone"


# ==============================
# 🧩 DroneUnified Dataset Class
# ==============================
class DroneDataset(Dataset):
    """
    DroneUnified 통합 데이터셋용 클래스.
    구조:
        datasets/DroneUnified/
            ├── drone/
            └── no_drone/
    """
    def __init__(
        self,
        root_dir="datasets/DroneUnified",
        target_sr=16000,
        max_sec=4,
        balance_neg_to_pos=False,  # 👈 비드론을 드론 수로 언더샘플
        max_neg=None,              # 👈 비드론 최대 개수 제한(옵션)
        seed=42
    ):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.max_len = target_sr * max_sec
        self.data = []

        drone_list, no_drone_list = [], []

        # 데이터 폴더 스캔
        for label_name in ["drone", "no_drone"]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder):
                print(f"[Warning] Missing folder: {folder}")
                continue

            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    path = os.path.join(folder, file)
                    label = 1 if label_name == "drone" else 0
                    if label == 1:
                        drone_list.append((path, label))
                    else:
                        no_drone_list.append((path, label))

        # ===== 🔻 언더샘플링/상한 제한 추가 🔻 =====
        rng = random.Random(seed)
        if balance_neg_to_pos:
            target = len(drone_list)
            pick = min(len(no_drone_list), target)
            no_drone_list = rng.sample(no_drone_list, pick)
            print(f"[Balance] no_drone 언더샘플링: {len(no_drone_list)} (드론 {len(drone_list)}개에 맞춤)")

        if max_neg is not None:
            pick = min(len(no_drone_list), int(max_neg))
            no_drone_list = rng.sample(no_drone_list, pick)
            print(f"[Cap] no_drone 상한 적용: {pick}개 제한")

        self.data = drone_list + no_drone_list
        rng.shuffle(self.data)
        print(f"✅ Dataset Loaded: total={len(self.data)}, drone={len(drone_list)}, no_drone={len(no_drone_list)}")
        # ===== 🔺 언더샘플링 추가 끝 🔺 =====

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        wav, sr = sf.read(path)

        # resample & mono
        if len(wav.shape) > 1:
            wav = librosa.to_mono(wav.T)
        if sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)

        # pad or crop
        if len(wav) < self.max_len:
            wav = F.pad(torch.tensor(wav), (0, self.max_len - len(wav)))
        else:
            wav = torch.tensor(wav[: self.max_len])

        return wav.float(), torch.tensor(label, dtype=torch.float32)


# ==============================
# ⚙️ Split Loader
# ==============================
def load_split(
    root_dir="datasets/DroneUnified",
    test_size=0.2,
    val_size=0.1,
    seed=42,
    balance_neg_to_pos=False,  # 👈 추가
    max_neg=None               # 👈 추가
):
    full_ds = DroneDataset(
        root_dir=root_dir,
        balance_neg_to_pos=balance_neg_to_pos,
        max_neg=max_neg,
        seed=seed,
    )

    indices = list(range(len(full_ds)))
    labels = [lbl for _, lbl in full_ds.data]

    # Train / temp split
    train_idx, temp_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed
    )

    # Val / Test split
    val_rel_size = val_size / test_size
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=val_rel_size,
        stratify=[labels[i] for i in temp_idx], random_state=seed
    )

    def subset(dataset, indices):
        return torch.utils.data.Subset(dataset, indices)
    tr_ds = subset(full_ds, train_idx)
    va_ds = subset(full_ds, val_idx)
    te_ds = subset(full_ds, test_idx)

    print(f"[Split] Train={len(tr_ds)}, Val={len(va_ds)}, Test={len(te_ds)}")
    return tr_ds, va_ds, te_ds
