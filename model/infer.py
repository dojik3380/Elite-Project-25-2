import numpy as np, torch, soundfile as sf
from .model_drone import DroneClassifier
from .config import TARGET_SR, WIN_SEC
from EliteProject.leaf_pytorch.frontend_helper import Leaf

def predict_file(path, ckpt="chk/best.pt", agg="max", thresh=0.5, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = sf.read(path)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=TARGET_SR)
    wav = wav.astype(np.float32)

    model = DroneClassifier().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

    win = int(TARGET_SR * WIN_SEC)
    hop = int(TARGET_SR * 1.0)
    probs = []
    with torch.no_grad():
        for s in range(0, max(1, len(wav)-win+1), hop):
            chunk = wav[s:s+win]
            if len(chunk) < win:
                pad = win - len(chunk)
                chunk = np.pad(chunk, (pad//2, pad - pad//2))
            p = torch.sigmoid(model(torch.from_numpy(chunk).unsqueeze(0).to(device))).item()
            probs.append(p)
    if not probs: probs=[0.0]
    if agg == "mean":
        p = float(np.mean(probs))
    elif agg == "logit_mean":
        eps=1e-6; lg=[np.log(x+eps)-np.log(1-x+eps) for x in probs]; p=1/(1+np.exp(-np.mean(lg)))
    else:
        p = max(probs)
    return p, p >= thresh
