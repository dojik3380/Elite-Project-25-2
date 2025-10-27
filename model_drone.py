import torch
import torch.nn as nn
from .config import TARGET_SR, N_FILTERS, FILTER_SIZE, STRIDE

# 루트에 leaf_pytorch가 패키지로 있으므로 그냥 import 가능
from leaf_pytorch import Leaf   # leaf_pytorch/__init__.py 에서 export

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=(3,3), s=(1,1), p=(1,1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

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
        self.frontend = Leaf(
            sample_rate=sample_rate,
            n_filters=n_filters,
            filter_size=filter_size,
            stride=stride,
            learnable_filters=True,
            pooling_type="gaussian",
            pcen_compression=pcen
        )
        base = 64
        self.backbone = nn.Sequential(
            ConvBlock(1, base), nn.MaxPool2d((2,2)),
            ConvBlock(base, base*2), nn.MaxPool2d((2,2)),
            ConvBlock(base*2, base*4), nn.MaxPool2d((2,2)),
            ConvBlock(base*4, base*4),
        )
        self.proj = nn.Conv2d(base*4, base*4, kernel_size=1)
        self.pool = AttnPool1d(base*4)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(base*4, 1)

    def forward(self, wav):            # wav: [B, T]
        feat = self.frontend(wav)      # [B, F, M]
        x = feat.unsqueeze(1)          # [B, 1, F, M]
        x = self.backbone(x)           # [B, C, F', M']
        x = self.proj(x)               # [B, C, F', M']
        x = x.mean(dim=2)              # F' 평균 → [B, C, M']
        x = x.transpose(1, 2)          # [B, M', C]
        x = self.pool(x)               # [B, C]
        x = self.drop(x)
        return self.head(x).squeeze(-1)  # logits [B]
