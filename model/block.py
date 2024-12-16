import torch
import torch.nn as nn
from .ssm import SelectiveSSM


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_ff, dropout=0.1):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Damn never heard of this activation function before until now 😅 (ref for myself: https://medium.com/@shauryagoel/gelu-gaussian-error-linear-unit-4ec59fb2e47c) 
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # make sure of proper normalization and residual connections 😇
        x = x + self.ssm(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
