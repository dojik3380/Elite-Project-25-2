# ===============================
# WaveAugmentV1 (증강 온오프 기능 강화)
# ===============================

import random
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path


class WaveAugmentV1:
    def __init__(
        self,
        sample_rate: int = 16000,

        # ----- 확률 파라미터 -----
        p_gain: float = 0.8,
        p_shift: float = 0.7,
        p_noise: float = 0.7,
        p_stretch: float = 0.3,
        p_time_mask: float = 0.4,

        # ----- 파라미터 -----
        gain_min: float = 0.7,
        gain_max: float = 1.3,
        shift_max_sec: float = 0.20,
        stretch_min: float = 0.95,
        stretch_max: float = 1.05,
        time_mask_max_ratio: float = 0.15,
        min_rms_for_gain: float = 0.01,   # 이 값보다 작으면 올려줌
        max_gain_for_boost: float = 10.0, # 너무 과도하게 키우지 않도록 상한

        # ----- noise -----
        noise_dirs=None,

        # ----- 증강 전체 on/off -----
        use_augment: bool = True,   # 🔥 noise 제외 전체 증강 ON/OFF
        use_noise: bool = True,     # 🔥 noise 증강 ON/OFF

        # ----- 개별 증강까지 on/off -----
        enable_gain: bool = True,
        enable_shift: bool = True,
        enable_stretch: bool = True,
        enable_mask: bool = True,


    ):
        self.sample_rate = sample_rate

        # 확률
        self.p_gain = p_gain
        self.p_shift = p_shift
        self.p_noise = p_noise
        self.p_stretch = p_stretch
        self.p_time_mask = p_time_mask

        # 파라미터
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.shift_max_sec = shift_max_sec
        self.stretch_min = stretch_min
        self.stretch_max = stretch_max
        self.time_mask_max_ratio = time_mask_max_ratio
        self.min_rms_for_gain = min_rms_for_gain
        self.max_gain_for_boost = max_gain_for_boost

        # 전체 증강 ON/OFF
        self.use_augment = use_augment     # 🔥 noise 제외 전체
        self.use_noise = use_noise         # 🔥 noise 만 ON/OFF

        # 개별 증강 ON/OFF
        self.enable_gain = enable_gain
        self.enable_shift = enable_shift
        self.enable_stretch = enable_stretch
        self.enable_mask = enable_mask

        self.boost_rms_threshold = 0.015   # 너무 작은 음원 기준 (조정 가능)
        self.target_rms = 0.03             # 목표 RMS (-32 dBFS 수준)
        self.max_auto_boost_gain = 5.0     # 너무 과도한 증폭 방지

        # ----------------------------------------------------
        # Real noise 파일 로드
        # ----------------------------------------------------
        if noise_dirs is None:
            root = Path(__file__).resolve().parents[1] / "datasets" / "DroneUnified" / "noise"
            noise_dirs = [root / "기상 수정", root / "총 수정"]

        self.noise_files = []
        for d in noise_dirs:
            d = Path(d)
            if d.exists():
                self.noise_files += list(d.glob("*.wav"))

        print(f"🔉 Loaded {len(self.noise_files)} real noise wav files.")

    # ===========================
    # 유틸
    # ===========================
    def _rand(self):
        return random.random()

    def _rms(self, x, eps=1e-8):
        return torch.sqrt(torch.mean(x * x) + eps)

    # 스위치 설정 메서드
    def set_noise(self, flag: bool):
        self.use_noise = bool(flag)

    def set_augment(self, flag: bool):
        """noise를 제외한 모든 증강 on/off"""
        self.use_augment = bool(flag)

    # ===========================
    # Gain
    # ===========================
    def random_gain(self, x):

        rms = torch.sqrt(torch.mean(x**2) + 1e-12)
        if rms < self.boost_rms_threshold:
            # 목표 RMS까지 증폭
            gain = self.target_rms / rms.clamp(min=1e-8)
            gain = min(gain.item(), self.max_auto_boost_gain)
            x = torch.clamp(x * gain, -1.0, 1.0)
            
        if not self.enable_gain:
            return x
        if self._rand() > self.p_gain:
            return x

        g = random.uniform(self.gain_min, self.gain_max)
        return torch.clamp(x * g, -1.0, 1.0)

    # ===========================
    # Stretch
    # ===========================
    def time_stretch(self, x):
        if not self.enable_stretch:
            return x
        if self._rand() > self.p_stretch:
            return x

        _, T = x.shape
        factor = random.uniform(self.stretch_min, self.stretch_max)
        new_len = int(T * factor)

        x_ = x.unsqueeze(0)
        y = F.interpolate(x_, size=new_len, mode='linear', align_corners=False).squeeze(0)

        if new_len > T:
            st = random.randint(0, new_len - T)
            return y[:, st:st + T]
        else:
            return F.pad(y, (0, T - new_len))

    # ===========================
    # Shift
    # ===========================
    def time_shift(self, x):
        if not self.enable_shift:
            return x
        if self._rand() > self.p_shift:
            return x

        _, T = x.shape
        max_s = int(self.shift_max_sec * self.sample_rate)
        s = random.randint(-max_s, max_s)

        if s > 0:
            pad = torch.zeros((1, s), device=x.device)
            return torch.cat([pad, x[:, :-s]], dim=1)
        elif s < 0:
            s = -s
            pad = torch.zeros((1, s), device=x.device)
            return torch.cat([x[:, s:], pad], dim=1)
        return x

    # ===========================
    # Noise
    # ===========================
    def random_noise_crop(self, target_len):
        if len(self.noise_files) == 0:
            return None

        path = random.choice(self.noise_files)
        wav, sr = sf.read(path)

        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)

        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)

        if len(wav) <= target_len:
            out = np.zeros(target_len, dtype=np.float32)
            out[:len(wav)] = wav
            return torch.tensor(out)

        st = random.randint(0, len(wav) - target_len)
        return torch.tensor(wav[st:st + target_len])

    def mix_real_noise(self, x):
        if not self.use_noise:
            return x
        if self._rand() > self.p_noise:
            return x

        _, T = x.shape
        noise = self.random_noise_crop(T)
        if noise is None:
            return x

        noise = noise.to(x.device).unsqueeze(0)

        snr_db = random.uniform(-5, 20)
        rms_signal = self._rms(x)
        rms_noise = self._rms(noise)

        desired_noise_rms = rms_signal / (10 ** (snr_db / 20.0))
        noise = noise * (desired_noise_rms / (rms_noise + 1e-8))

        return torch.clamp(x + noise, -1.0, 1.0)

    # ===========================
    # Time Mask
    # ===========================
    def time_mask(self, x):
        if not self.enable_mask:
            return x
        if self._rand() > self.p_time_mask:
            return x

        _, T = x.shape
        max_len = int(T * self.time_mask_max_ratio)
        ml = random.randint(1, max_len)
        st = random.randint(0, T - ml)
        x[:, st:st + ml] = 0
        return x

    # ===========================
    # MAIN AUG PIPELINE
    # ===========================
    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        # ---- noise 제외 전체 증강 ----
        if self.use_augment:
            x = self.random_gain(x)
            x = self.time_stretch(x)
            x = self.time_shift(x)
            x = self.time_mask(x)

        # ---- noise 증강 ----
        x = self.mix_real_noise(x)

        return x