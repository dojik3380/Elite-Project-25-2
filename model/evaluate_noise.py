# model/evaluate_noise.py
import numpy as np
import torch
import librosa, soundfile as sf
import matplotlib.pyplot as plt
from model.infer import predict_file
from model.config import TARGET_SR

def add_noise_snr(wav, snr_db):
    """주어진 오디오에 목표 SNR(dB)만큼 잡음을 추가"""
    rms_signal = np.sqrt(np.mean(wav**2))
    noise = np.random.randn(len(wav))
    rms_noise = np.sqrt(np.mean(noise**2))
    noise_scaled = noise * (rms_signal / (10**(snr_db / 20) * rms_noise))
    return wav + noise_scaled

def evaluate_noise_robustness(audio_path, ckpt="chk/drone_model.pt"):
    """입력 오디오에 SNR별 노이즈 추가 후 탐지 확률 평가"""
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

    # 그래프 출력
    plt.figure(figsize=(7,5))
    plt.plot(snr_list, probs, marker='o', label='LEAF model')
    plt.title("Noise Robustness Evaluation (SNR vs Probability)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Predicted Drone Probability")
    plt.gca().invert_xaxis()  # SNR 낮을수록 오른쪽으로
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("noise_robustness_curve.png")
    plt.show()

if __name__ == "__main__":
    # 사용 예시: python model/evaluate_noise.py
    test_audio = "datasets/DroneUnified/drone/example.wav"  # 👈 여기에 네 오디오 파일 경로 입력
    evaluate_noise_robustness(test_audio)
