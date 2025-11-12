# model/infer.py
import torch, librosa, soundfile as sf, numpy as np
from model.model_drone import DroneMultiClassifier
from model.config import TARGET_SR, WIN_SEC

def predict_file(path, ckpt="chk/best.pt", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load audio
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=TARGET_SR)
    wav = wav.astype(np.float32)

    # Load model + label map
    ckpt_obj = torch.load(ckpt, map_location=device)
    label_map = ckpt_obj["label_map"]
    idx_to_label = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    model = DroneMultiClassifier(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt_obj["model"])
    model.eval()

    # Segment audio into 1-sec windows
    win = int(TARGET_SR * WIN_SEC)
    hop = win  # no overlap
    probs_all = []

    with torch.no_grad():
        for start in range(0, len(wav) - win + 1, hop):
            seg = wav[start:start+win]
            if len(seg) < win:
                pad = np.zeros(win, dtype=np.float32)
                pad[:len(seg)] = seg
                seg = pad
            x = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(x)
            prob = torch.softmax(logit, dim=1).cpu().numpy().flatten()
            probs_all.append(prob)

    if len(probs_all) == 0:
        return None

    # 평균 확률 (파일 단위)
    mean_prob = np.mean(probs_all, axis=0)
    pred_idx = int(np.argmax(mean_prob))
    pred_label = idx_to_label[pred_idx]
    confidence = float(mean_prob[pred_idx])

    print(f"\n🎧 File: {path}")
    print("📊 Class probabilities:")
    for k, v in idx_to_label.items():
        print(f"  {v:<12}: {mean_prob[k]:.3f}")
    print(f"\n🟢 Prediction: {pred_label} ({confidence*100:.1f}%)")

    return pred_label, confidence, mean_prob
