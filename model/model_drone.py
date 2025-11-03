import torch
import torch.nn as nn
from .config import TARGET_SR, N_FILTERS, FILTER_SIZE, STRIDE

# 루트에 leaf_pytorch가 패키지로 있으므로 그냥 import 가능
from leaf_pytorch.frontend import Leaf

class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class AttnPool1d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim//2), nn.Tanh(), nn.Linear(in_dim//2, 1)
        )
    def forward(self, x):        # [B, T, C]
        a = torch.softmax(self.attn(x), dim=1)
        return torch.sum(a * x, dim=1)  # [B, C]

class DroneClassifier(nn.Module):
    def __init__(self,
                sample_rate=TARGET_SR,
                n_filters=N_FILTERS,
                filter_size=FILTER_SIZE,
                stride=STRIDE,
                pcen=True):
        super().__init__()
        # LEAF 프론트엔드 (Conv1D)
        self.frontend = Leaf(
            sample_rate=sample_rate,
            n_filters=n_filters,
            window_len=25.0,
            window_stride=10.0,
            pcen_compression=pcen
        )
        # Conv1D 기반 백본
        self.backbone = nn.Sequential(
            ConvBlock1d(n_filters, 128, 3, 1, 1),
            nn.MaxPool1d(2),
            ConvBlock1d(128, 256, 3, 1, 1),
            nn.MaxPool1d(2),
            ConvBlock1d(256, 256, 3, 1, 1),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.4)
        self.head = nn.Linear(256, 1)
        print(">>> model using:", self.backbone)


    def forward(self, wav):
        if wav.dim() == 3 and wav.size(1) != 1:
            wav = wav.mean(dim=1, keepdim=True)  # 여러 채널이면 모노로 변환
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)  
            
        feat = self.frontend(wav)      # [B, F, T]
        x = self.backbone(feat)        # [B, C, T']
        x = self.pool(x).squeeze(-1)   # [B, C]
        x = self.drop(x)
        return self.head(x)            # ✅ [B, 1] 유지



        
    