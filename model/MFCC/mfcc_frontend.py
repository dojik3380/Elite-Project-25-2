# model/mfcc_frontend.py

import torch
import torchaudio
import torch.nn as nn

class MFCCFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=40, n_fft=400, hop_length=160):
        super().__init__()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mfcc,
                "center": True,
                "power": 2.0,
            }
        )

    def forward(self, wav):
        # wav: [B, 1, T]
        wav = wav.squeeze(1)           # [B, T]
        feat = self.mfcc(wav)          # [B, n_mfcc, T']
        return feat
