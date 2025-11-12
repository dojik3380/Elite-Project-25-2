# model/check_dataset.py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.dataset_drone import load_split
from model.config import BATCH_SIZE, TARGET_SR

# 1️⃣ 데이터 로드
train_ds, val_ds, test_ds, label_map = load_split()
print("\n✅ Dataset loaded successfully!")
print("Classes:", label_map)
print(f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# 2️⃣ DataLoader 생성
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 3️⃣ 첫 번째 배치 꺼내보기
batch = next(iter(loader))
x, y = batch
print("\n🧩 Batch shape:", x.shape)      # [B, 1, 16000] 예상
print("🧩 Labels:", y.tolist()[:10])

# 4️⃣ 파형 하나 시각화
wav = x[0].squeeze().numpy()
plt.figure(figsize=(10, 3))
plt.plot(wav)
plt.title(f"Example waveform (label={y[0].item()})")
plt.xlabel("Samples"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
