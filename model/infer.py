import numpy as np, torch, soundfile as sf
from .model_drone import DroneClassifier
from .config import TARGET_SR, WIN_SEC

# ✅ 전역 변수로 모델 한 번만 초기화
_cached_model = None
_cached_device = None

def predict_file(path, ckpt="chk/best.pt", agg="max", thresh=0.5, device=None):
    global _cached_model, _cached_device

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 모델 최초 1회만 로드
    if _cached_model is None:
        model = DroneClassifier().to(device)
        ckpt_obj = torch.load(ckpt, map_location=device)
        state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
        model.load_state_dict(state)
        model.eval()
        _cached_model = model
        _cached_device = device
        print(f">>> model loaded once from {ckpt}")
    else:
        model = _cached_model
        device = _cached_device

    # 🔹 오디오 로드 및 전처리
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=TARGET_SR)
    wav = wav.astype(np.float32)

    win = int(TARGET_SR * WIN_SEC)
    hop = int(TARGET_SR * 1.0)
    probs = []

    with torch.no_grad():
        for start in range(0, len(wav) - win + 1, hop):
            end = start + win
            chunk = wav[start:end]
            if len(chunk) < win:
                pad = np.zeros(win, dtype=chunk.dtype)
                pad[:len(chunk)] = chunk
                chunk = pad
            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    if not probs:
        probs = [0.0]

    if agg == "mean":
        p = float(np.mean(probs))
    elif agg == "logit_mean":
        eps = 1e-6
        lg = [np.log(x + eps) - np.log(1 - x + eps) for x in probs]
        p = 1 / (1 + np.exp(-np.mean(lg)))
    else:
        p = max(probs)

    return p, p >= thresh
