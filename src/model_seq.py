"""
Sequence-based models (RNN / GRU / LSTM) for lung cancer risk prediction.

We treat the 15 tabular features as a short 1D sequence of length 15 with input_dim=1.
Each model returns logits for 2 classes (NO / YES), same as the MLPs.
"""

import torch
import torch.nn as nn


class LungCancerRNN(nn.Module):
    """Basic RNN-based classifier."""

    def __init__(self, input_size: int = 15, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # We reshape features from (batch, 15) -> (batch, seq_len=15, input_dim=1)
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 15) -> (batch, 15, 1)
        x = x.unsqueeze(-1)
        out, h_n = self.rnn(x)
        # Use last hidden state from last layer: (num_layers, batch, hidden) -> (batch, hidden)
        h_last = h_n[-1]
        logits = self.fc(h_last)
        return logits


class LungCancerGRU(nn.Module):
    """GRU-based classifier."""

    def __init__(self, input_size: int = 15, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out, h_n = self.gru(x)
        h_last = h_n[-1]
        logits = self.fc(h_last)
        return logits


class LungCancerLSTM(nn.Module):
    """LSTM-based classifier."""

    def __init__(self, input_size: int = 15, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        logits = self.fc(h_last)
        return logits

