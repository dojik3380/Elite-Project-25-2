import torch
import torch.nn as nn
from leaf_pytorch.frontend import Leaf
from .config import TARGET_SR, N_FILTERS


# ============================================================
#  NiN 블록 (2D) : 1x1 → kxk → 1x1
# ============================================================
class NiNBlock2D(nn.Module):
    """NiN 블록: 1x1 → kxk → 1x1 Conv2d"""
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, k: int):
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
#  DenseNet-BC 블록 (2D)
#   - Bottleneck: 1x1 (bf * k) → 3x3 (k)
#   - Transition: θ-compression + AvgPool2d(2,2)
# ============================================================
class DenseLayer2D(nn.Module):
    """DenseNet-BC 스타일 레이어: BN-ReLU-1x1- BN-ReLU-3x3 → concat"""
    def __init__(self, in_ch: int, grth: int, bf: int = 4):
        super().__init__()
        mid_ch = bf * grth
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, grth, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        new_feat = self.layer(x)
        return torch.cat([x, new_feat], dim=1)


class DenseBlock2D(nn.Module):
    """L개의 DenseLayer2D를 쌓는 블록."""
    def __init__(self, in_ch: int, grth: int, L: int, bf: int = 4):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(L):
            layers.append(DenseLayer2D(ch, grth, bf=bf))
            ch += grth
        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.block(x)


class Transition2D(nn.Module):
    """DenseNet-BC transition layer: θ-compression + AvgPool2d(2,2)"""
    def __init__(self, in_ch: int, theta: float = 0.5):
        super().__init__()
        out_ch = int(in_ch * theta)
        self.out_channels = out_ch
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
#  Drone 모델: LEAF + NiN×2 + DenseNet-BC×2 + FC
# ============================================================
class DroneMultiClassifier(nn.Module):
    """LEAF + [NiN(7x7) → NiN(5x5) → Dense-BC(k=12,L=6) → Dense-BC(k=16,L=12)] + FC"""
    def __init__(
        self,
        n_classes: int,
        sample_rate: int = TARGET_SR,
        n_filters: int = N_FILTERS,
    ):
        super().__init__()

        # LEAF 프론트엔드
        self.frontend = Leaf(
            n_filters=n_filters,
            sample_rate=sample_rate,
            pcen_compression=True,
        )

        # ---------------- Stage 1 : NiN(7x7) ----------------
        C1 = 64
        self.stage1 = nn.Sequential(
            NiNBlock2D(1, C1 // 2, C1, 7),  # 1x1→7x7→1x1, out=64
            nn.MaxPool2d(2, 2),
        )

        # ---------------- Stage 2 : NiN(5x5) ----------------
        C2 = 128  # 두 번째 NiN 출력 채널을 128로 설정
        self.stage2 = nn.Sequential(
            NiNBlock2D(C1, C2 // 2, C2, 5),  # 64→128
            nn.MaxPool2d(2, 2),
        )

        # ---------------- Stage 3 : Dense-BC(k=12,L=6) + Transition(θ=0.5) ----------------
        self.dense3 = DenseBlock2D(C2, grth=12, L=6, bf=4)
        C3_dense = self.dense3.out_channels  # 128 + 6*12 = 200
        self.trans3 = Transition2D(C3_dense, theta=0.5)  # 200→100, + AvgPool(2,2)
        C3 = self.trans3.out_channels  # 100

        # ---------------- Stage 4 : Dense-BC(k=16,L=12) ----------------
        self.dense4 = DenseBlock2D(C3, grth=16, L=12, bf=4)
        C4 = self.dense4.out_channels  # 100 + 12*16 = 292

        # ---------------- Head : GlobalAvgPool + FC ----------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(C4, n_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # LEAF: [B, T] → [B, F, T']
        feat = self.frontend(wav)
        x = feat.unsqueeze(1)  # [B, 1, F, T']

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)

        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, C4]
        x = self.dropout(x)
        return self.fc(x)
