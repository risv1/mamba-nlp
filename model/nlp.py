import torch
import torch.nn as nn
from .block import MambaBlock
from .encode import PositionalEncoding


class MambaForNLP(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_state=16, d_ff=1024, 
                 num_layers=6, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout) # positional encoding for awareness of token position (refer to encode.py)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)  # input (batch_size, seq_len, d_model) 🥱
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        x = self.fc(x)
        return x