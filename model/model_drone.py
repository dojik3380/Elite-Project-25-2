import torch
import torch.nn as nn
from .config import TARGET_SR, N_FILTERS, FILTER_SIZE, STRIDE

# 루트에 leaf_pytorch가 패키지로 있으므로 그냥 import 가능
from EliteProject.leaf_pytorch.frontend_helper import Leaf

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
            window_len=25.0,      # ms (≈ filter_size 401)
            window_stride=10.0,
            # learnable_filters=True,
            # pooling_type="gaussian",
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

    def forward(self, wav):
        wav = wav.float()  # ✅ 명시적으로 float32 변환
        
        feat = self.frontend(wav)  # [B, F, M]
        feat = feat.float()  # ✅ frontend 출력도 float32로
        
        x = feat.unsqueeze(1)  # [B, 1, F, M]
        x = self.backbone(x)   # [B, C, F', M']
        x = self.proj(x)       # [B, C, F', M']
        x = x.mean(dim=2)      # F' 평균 → [B, C, M']
        x = x.transpose(1, 2)  # [B, M', C]
        x = self.pool(x)       # [B, C]
        x = self.drop(x)
        return self.head(x).squeeze(-1)  # logits [B]
    
    #     # --- 입력 텐서 형태 강제 및 방어 코드 ---
    # # 1) numpy -> tensor 등 예외 케이스 방지
    #     if not isinstance(wav, torch.Tensor):
    #         wav = torch.tensor(wav)

    # # 2) 배치가 없는 1차원 waveform이 들어오는 경우 (예: single sample)
    # #    -> [T]  또는 [1, T] 가능성 처리
    #     if wav.dim() == 1:
    #     # single example: [T] -> [1, 1, T]
    #         wav = wav.unsqueeze(0).unsqueeze(1)
    #     elif wav.dim() == 2:
    #     # [B, T] -> [B, 1, T]
    #         wav = wav.unsqueeze(1)
    #     elif wav.dim() == 3:
    #     # [B, C, T] 인 경우: C가 1인지 확인
    #         if wav.size(1) != 1:
    #         # 만약 실수로 [B, F, T] (이미 frontend output) 같은 것을 넣었다면
    #         # 사용자 의도에 따라 처리해야 함. 기본적으로 에러를 명확히 던짐.
    #             raise RuntimeError(f"Expected waveform with 1 channel (B,1,T), but got {tuple(wav.shape)}. "
    #                            "If this is already frontend output, call model with raw waveform (B,T) instead.")
    #     else:
    #         raise RuntimeError(f"Unsupported input wav ndim: {wav.dim()}. Expected 1/2/3 dims.")

    # # 3) 타입/디바이스 정리 (옵션)
    #     if wav.dtype not in (torch.float32, torch.float64):
    #         wav = wav.float()
    # # move to same device as model parameters if not already
    #     device = next(self.parameters()).device
    #     if wav.device != device:
    #         wav = wav.to(device)
        
    #                                    # wav: [B, T]
    #     feat = self.frontend(wav)      # [B, F, M]
    #     x = feat.unsqueeze(1)          # [B, 1, F, M]
    #     x = self.backbone(x)           # [B, C, F', M']
    #     x = self.proj(x)               # [B, C, F', M']
    #     x = x.mean(dim=2)              # F' 평균 → [B, C, M']
    #     x = x.transpose(1, 2)          # [B, M', C]
    #     x = self.pool(x)               # [B, C]
    #     x = self.drop(x)
    #     return self.head(x).squeeze(-1)  # logits [B]
