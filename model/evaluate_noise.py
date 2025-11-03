# model/evaluate_noise.py
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from model.infer import predict_file
from model.config import TARGET_SR

# ============================================================
# 🔹 정확한 SNR 계산으로 노이즈 추가
# ============================================================
def add_noise_snr(wav, snr_db):
    wav = wav.astype(np.float32)
    rms_signal = np.sqrt(np.mean(wav ** 2)) + 1e-12
    noise = np.random.randn(len(wav)).astype(np.float32)
    rms_noise = np.sqrt(np.mean(noise ** 2)) + 1e-12
    desired_rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise *= (desired_rms_noise / rms_noise) * 0.75   
    noisy = wav + noise
    noisy = np.clip(noisy, -1.0, 1.0)
    return noisy



# ============================================================
# 🔹 SNR별 정량 지표 계산 (F1, AUC 등)
# ============================================================
def compute_metrics_by_snr(drone_probs_dict, nodrone_probs_dict, threshold=0.58):
    snr_vals = sorted(drone_probs_dict.keys(), reverse=True)
    print("\n🔎 Metrics by SNR (thr=%.3f):" % threshold)
    for snr in snr_vals:
        p_d = np.array(drone_probs_dict[snr] or [0.0])
        p_n = np.array(nodrone_probs_dict[snr] or [0.0])
        y_prob = np.concatenate([p_d, p_n])
        y_true = np.concatenate([np.ones_like(p_d), np.zeros_like(p_n)])
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        print(
            f"  SNR {snr:>3} dB | Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}"
        )


# ============================================================
# 🔹 폴더 단위 평균 노이즈 강건성 평가
# ============================================================
def evaluate_noise_dataset(
    drone_dir="datasets/DroneUnified/drone",
    nodrone_dir="datasets/DroneUnified/no_drone",
    ckpt="chk/best_model.pt",
    limit=300,
):
    """드론/비드론 폴더 각각 limit개만 랜덤으로 샘플링하여 SNR별 평균 확률 비교 그래프 출력"""
    import random

    snr_list = [20, 10, 0, -5]
    drone_avg = {snr: [] for snr in snr_list}
    nodrone_avg = {snr: [] for snr in snr_list}

    # ----------------------------------------------------------
    # 내부 함수: 한 폴더에 대해 SNR별 예측 확률 저장
    # ----------------------------------------------------------
    def process_folder(folder, store_dict, label_name, k=None):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]
        if k is not None and len(files) > k:
            random.shuffle(files)
            files = files[:k]
        print(f"\n🎧 Evaluating {label_name} folder ({len(files)} samples)")
        for f in tqdm(files, desc=f"{label_name}"):
            try:
                wav, sr = sf.read(f)
            except:
                continue
            if sr != TARGET_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            for snr in snr_list:
                noisy = add_noise_snr(wav, snr)
                temp = f"temp_{label_name}_{snr}.wav"
                sf.write(temp, noisy, TARGET_SR)
                prob, _ = predict_file(temp, ckpt=ckpt)
                store_dict[snr].append(prob)

    # ----------------------------------------------------------
    # 드론/비드론 폴더 각각 평가
    # ----------------------------------------------------------
    process_folder(drone_dir, drone_avg, "drone", k=limit)
    process_folder(nodrone_dir, nodrone_avg, "no_drone", k=limit)

    # ----------------------------------------------------------
    # 평균 계산 및 시각화
    # ----------------------------------------------------------
    snr_vals = sorted(drone_avg.keys(), reverse=True)
    drone_means = [np.mean(drone_avg[s]) if drone_avg[s] else 0 for s in snr_vals]
    nodrone_means = [np.mean(nodrone_avg[s]) if nodrone_avg[s] else 0 for s in snr_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(snr_vals, drone_means, marker="o", label=f"Drone (avg {limit})")
    plt.plot(snr_vals, nodrone_means, marker="s", label=f"No-Drone (avg {limit})")
    plt.title("Noise Robustness Comparison (SNR vs Mean Probability)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Predicted Probability")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("noise_robustness_compare.png")
    plt.show()

    # ----------------------------------------------------------
    # 평균 확률 표 출력
    # ----------------------------------------------------------
    print("\n📊 Average Predicted Probabilities by SNR:")
    for snr in snr_vals:
        d = np.mean(drone_avg[snr]) if drone_avg[snr] else 0
        n = np.mean(nodrone_avg[snr]) if nodrone_avg[snr] else 0
        print(f"  SNR {snr:>3} dB | Drone={d:.3f} | NoDrone={n:.3f}")

    # ----------------------------------------------------------
    # SNR별 F1/AUC 등 지표 계산
    # ----------------------------------------------------------
    compute_metrics_by_snr(drone_avg, nodrone_avg, threshold=0.58)


# ============================================================
# 🔹 단일 오디오 테스트용 (원본 유지)
# ============================================================
def evaluate_noise_robustness(audio_path, ckpt="chk/best_model.pt"):
    """단일 오디오 파일에 대해 SNR별 탐지 확률 평가"""
    wav, sr = sf.read(audio_path)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    snr_list = [20, 10, 0, -5]
    probs = []

    print(f"\n🎧 Evaluating Noise Robustness for: {audio_path}")
    for snr in snr_list:
        noisy = add_noise_snr(wav, snr)
        temp_path = f"temp_snr{snr}.wav"
        sf.write(temp_path, noisy, TARGET_SR)
        prob, pred = predict_file(temp_path, ckpt=ckpt)
        probs.append(prob)
        print(f"  SNR {snr:>3} dB → 확률 {prob:.3f} | 판정: {'드론' if pred else '비드론'}")

    plt.figure(figsize=(7, 5))
    plt.plot(snr_list, probs, marker="o", label="Single File")
    plt.title("Noise Robustness Evaluation (SNR vs Probability)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Predicted Drone Probability")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("noise_robustness_curve.png")
    plt.show()


# ============================================================
# 🔹 실행부
# ============================================================
if __name__ == "__main__":
    # 평균 강건성 비교 (드론 vs 비드론)
    evaluate_noise_dataset(
        drone_dir="datasets/DroneUnified/drone",
        nodrone_dir="datasets/DroneUnified/no_drone",
        ckpt="chk/best_model.pt",
        limit=300,  # 속도 조절 가능 (100~1000)
    )

    # # 단일 파일 테스트 하고 싶으면 아래 주석 해제
    # test_audio = "datasets/DroneUnified/drone/B_S2_D1_067-bebop_000_.wav"
    # evaluate_noise_robustness(test_audio)
