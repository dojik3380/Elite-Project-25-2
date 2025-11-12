# model/model_drone.py
import torch
import torch.nn as nn
from leaf_pytorch.frontend import Leaf
from .config import TARGET_SR, N_FILTERS


class ConvBlock1d(nn.Module):
    """기본 1D Conv 블록: Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=1,
                      padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DroneMultiClassifier(nn.Module):
    """LEAF (with PCEN) + 1D CNN 멀티클래스 오디오 분류기"""
    def __init__(self, n_classes, sample_rate=TARGET_SR, n_filters=N_FILTERS):
        super().__init__()

        # ✅ LEAF 프런트엔드 (이미 PCEN 내장됨)
        self.frontend = Leaf(
            n_filters=n_filters,
            sample_rate=sample_rate,
            pcen_compression=True     # PCENLayer 활성화
        )

        # ✅ 혼합 풀링 CNN 백본
        self.backbone = nn.Sequential(
            ConvBlock1d(n_filters, 128, k=5),
            nn.MaxPool1d(2),
            ConvBlock1d(128, 256, k=5),
            nn.AvgPool1d(2),
            ConvBlock1d(256, 256, k=3),
            nn.AdaptiveAvgPool1d(1),
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, wav):
        feat = self.frontend(wav)            # [B, F, T′]
        x = self.backbone(feat).squeeze(-1)  # [B, 256]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
