TARGET_SR = 16000
WIN_SEC = 2.0
N_FILTERS = 64           # 64~128 권장
FILTER_SIZE = 401        # ≈25 ms @16kHz
STRIDE = 160             # =10 ms hop
BATCH_SIZE = 64
EPOCHS = 70
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
NUM_WORKERS = 6

