import torch
import torch.nn as nn


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Here are the params for equation thingy with accordance to state-space dynamics (Heres ref for nerds 🤓: https://en.wikipedia.org/wiki/State-space_representation)
        self.A = nn.Parameter(torch.randn(1, d_state, d_state))
        self.B = nn.Parameter(torch.randn(1, d_state, d_model))
        self.C = nn.Parameter(torch.randn(1, d_model, d_state))
        self.D = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        h = torch.zeros(batch_size, 1, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            xt = x[:, t:t+1, :]  # This is the current input (batch_size, 1, d_model) 🐱

            # Update state with correct dimensions
            # h: (batch_size, 1, d_state)
            # A: (1, d_state, d_state)
            # B: (1, d_state, d_model)
            h = torch.bmm(h, self.A) + torch.bmm(xt, self.B.transpose(1, 2))

            # Output computation 🗣️
            # C: (1, d_model, d_state)
            yt = torch.bmm(h, self.C.transpose(1, 2))
            yt = yt + self.D(xt)

            outputs.append(yt)

        # Stack outputs along sequence dimension
        output = torch.cat(outputs, dim=1)
        return self.dropout(output)
