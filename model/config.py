TARGET_SR = 16000
WIN_SEC = 1.0
N_FILTERS = 64           # 64~128 권장
FILTER_SIZE = 401        # ≈25 ms @16kHz
STRIDE = 160             # =10 ms hop
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
NUM_WORKERS = 6

import torch

# ===========================
# ⚙️ 기본 설정 (추가)
# ===========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
