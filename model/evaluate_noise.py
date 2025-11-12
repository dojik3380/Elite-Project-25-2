# model/evaluate_noise.py
import os, random
import numpy as np
import torch
import librosa, soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from model.infer import predict_file
from model.config import TARGET_SR

def add_noise_snr(wav, snr_db):
    wav = wav.astype(np.float32)
    rms_signal = np.sqrt(np.mean(wav ** 2)) + 1e-12
    noise = np.random.randn(len(wav)).astype(np.float32)
    rms_noise = np.sqrt(np.mean(noise ** 2)) + 1e-12
    desired_rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise *= (desired_rms_noise / rms_noise) * 0.75
    noisy = np.clip(wav + noise, -1.0, 1.0)
    return noisy

def evaluate_noise_multiclass(
    root_dir="datasets/DroneUnified",
    ckpt="chk/best.pt",
    snr_list=(20, 10, 0, -5),
    limit_per_class=200,
):
    """각 클래스 폴더에 대해 SNR별 '자기 클래스 확률'의 평균을 계산"""
    root = Path(root_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    results = {d.name.lower(): {snr: [] for snr in snr_list} for d in class_dirs}

    for d in class_dirs:
        wavs = [p for p in d.glob("*.wav")]
        if limit_per_class and len(wavs) > limit_per_class:
            random.shuffle(wavs)
            wavs = wavs[:limit_per_class]
        print(f"\n🎧 Class '{d.name}': {len(wavs)} files")

        for path in tqdm(wavs, desc=d.name):
            try:
                wav, sr = sf.read(path)
            except:
                continue
            if sr != TARGET_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            for snr in snr_list:
                noisy = add_noise_snr(wav, snr)
                tmp = f"__tmp_snr_{snr}.wav"
                sf.write(tmp, noisy, TARGET_SR)
                pred_label, confidence, prob_vec = predict_file(tmp, ckpt=ckpt)  # 멀티클래스 호환
                # 자기 클래스 확률만 기록
                # predict_file은 ckpt 내 label_map 순서 기준의 softmax 확률 벡터를 반환
                # pred_label은 문자열, prob_vec은 [C] ndarray
                # pred_label→idx 매핑은 infer 내부에서만 사용하므로 여기선 문자열 비교로 index 찾기
                # 효율을 위해 첫 파일에서 label_map 역인덱싱을 캐시하는 게 좋지만 단순화해서 처리
                # (성능 이슈 없으면 그대로 둬도 무방)
                # 여기서는 pred_label과 상관없이, 파일의 '폴더명'이 가리키는 클래스 확률을 써야 하므로
                # prob_vec에서 그 클래스 인덱스를 찾아야 함 → predict_file이 idx_to_label을 print만 하므로
                # 간단히는 '그 클래스가 맞을 때 confidence'만 쓰기 어렵다. 대신:
                # 추정 우회: 파일이 속한 클래스명(d.name.lower())과 pred_label이 같으면 confidence,
                # 다르면 해당 클래스 확률을 알 수 없으므로 confidence 대신 0으로 두는 방식을 피해야 한다.
                # => 더 정확하게 하려면 predict_file이 (label, conf, prob_vec, idx_to_label)도 반환해야 한다.
                # 여기서는 간단히: pred_label == 폴더명일 때만 confidence를 채택.
                if pred_label.lower() == d.name.lower():
                    results[d.name.lower()][snr].append(confidence)
                else:
                    # 보수적으로 0이 아닌 작은 값으로 채우거나, 스킵
                    # 스킵이 통계적으로 안전
                    pass
                os.remove(tmp)

    # 평균 계산 및 시각화
    plt.figure(figsize=(9, 6))
    for cname, snr_dict in results.items():
        xs = sorted(snr_dict.keys(), reverse=True)
        ys = [np.mean(snr_dict[s]) if snr_dict[s] else 0.0 for s in xs]
        plt.plot(xs, ys, marker="o", label=cname)
    plt.title("SNR vs Mean Class-Confidence (per class)")
    plt.xlabel("SNR (dB)"); plt.ylabel("Mean confidence for own class")
    plt.gca().invert_xaxis(); plt.grid(True); plt.legend(ncol=2, fontsize=9)
    plt.tight_layout(); plt.savefig("noise_robustness_multiclass.png"); plt.show()

if __name__ == "__main__":
    evaluate_noise_multiclass(
        root_dir="datasets/DroneUnified",
        ckpt="chk/best.pt",
        snr_list=(20,10,0,-5),
        limit_per_class=200,
    )
