# model/MFCC/model_drone_mfcc.py

import torch
import torch.nn as nn
import torchaudio

from model.config import TARGET_SR, N_FILTERS
from model.model_drone import NiNBlock2D, DenseBlock2D, Transition2D


class MFCCFrontend(nn.Module):
    """
    입력: wav [B, 1, T] 또는 [B, T]
    출력: MFCC [B, F, T']
    """
    def __init__(self, sample_rate: int = TARGET_SR,
                 n_mfcc: int = 40,
                 n_mels: int = N_FILTERS):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 512,
                "hop_length": 160,   # 10 ms @16kHz
                "n_mels": n_mels,
                "center": True,
            }
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, 1, T] or [B, T]
        return: [B, n_mfcc, T']
        """
        x = wav
        if x.dim() == 3:
            # [B, 1, T] -> [B, T]
            x = x.squeeze(1)
        # torchaudio MFCC expects [B, T]
        feat = self.mfcc(x)  # [B, n_mfcc, T']
        return feat


class DroneMultiClassifierMFCC(nn.Module):
    """
    LEAF 버전과 백본 구조 100% 동일.
    차이점은 frontend만 MFCC로 교체.
    """
    def __init__(self,
                 n_classes: int,
                 sample_rate: int = TARGET_SR,
                 n_mfcc: int = 40):
        super().__init__()

        # ---------------- Frontend : MFCC ----------------
        self.frontend = MFCCFrontend(sample_rate=sample_rate,
                                     n_mfcc=n_mfcc,
                                     n_mels=N_FILTERS)    # Leaf의 n_filters와 맞춤

        # ---------------- Stage 1 : NiN(7x7) ----------------
        C1 = 64
        self.stage1 = nn.Sequential(
            NiNBlock2D(1, C1 // 2, C1, 7),  # in=1, out=64
            nn.MaxPool2d(2, 2),
        )

        # ---------------- Stage 2 : NiN(5x5) ----------------
        C2 = 128
        self.stage2 = nn.Sequential(
            NiNBlock2D(C1, C2 // 2, C2, 5),  # 64 -> 128
            nn.MaxPool2d(2, 2),
        )

        # ---------------- Stage 3 : Dense-BC(k=12, L=6) + Transition(θ=0.5) ----------------
        self.dense3 = DenseBlock2D(C2, grth=12, L=6, bf=4)
        C3_dense = self.dense3.out_channels          # 128 + 6*12 = 200
        self.trans3 = Transition2D(C3_dense, theta=0.5)  # 200 -> 100
        C3 = self.trans3.out_channels                # 100

        # ---------------- Stage 4 : Dense-BC(k=16, L=12) ----------------
        self.dense4 = DenseBlock2D(C3, grth=16, L=12, bf=4)
        C4 = self.dense4.out_channels                # 100 + 12*16 = 292

        # ---------------- Head : GlobalAvgPool + FC ----------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(C4, n_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, 1, T]
        1) MFCC frontend → [B, F, T']
        2) 2D CNN 백본 → [B, C4]
        3) FC → [B, n_classes]
        """
        feat = self.frontend(wav)        # [B, F, T']
        x = feat.unsqueeze(1)            # [B, 1, F, T']

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)

        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, C4]
        x = self.dropout(x)
        return self.fc(x)
