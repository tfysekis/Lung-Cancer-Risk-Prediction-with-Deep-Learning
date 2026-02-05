"""
Deep learning model architectures for lung cancer risk prediction.
Seven architectures chosen for thesis: standard, simple, minimal, deep, ELU, LayerNorm, residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LungCancerNet(nn.Module):
    """
    Standard MLP: 3 hidden layers (128, 64, 32), ReLU, BatchNorm, Dropout.
    Good baseline for thesis comparison.
    """
    def __init__(self, input_size=15, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(LungCancerNet, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LungCancerNetSimple(nn.Module):
    """Simple 2-layer network (64, 32). Lightweight baseline."""
    def __init__(self, input_size=15, hidden_size=64, dropout_rate=0.2):
        super(LungCancerNetSimple, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size // 2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


class LungCancerNetMinimal(nn.Module):
    """Minimal 1-layer network (32 units). Simplest baseline."""
    def __init__(self, input_size=15, hidden_size=32, dropout_rate=0.2):
        super(LungCancerNetMinimal, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


class LungCancerNetDeep(nn.Module):
    """Deeper MLP with 5 hidden layers (64 each). Tests effect of depth."""
    def __init__(self, input_size=15, hidden_size=64, num_layers=5, dropout_rate=0.3):
        super(LungCancerNetDeep, self).__init__()
        layers = []
        prev_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LungCancerNetELU(nn.Module):
    """Same as Standard but ELU activation instead of ReLU."""
    def __init__(self, input_size=15, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(LungCancerNetELU, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LungCancerNetLayerNorm(nn.Module):
    """Standard architecture with Layer Normalization instead of BatchNorm."""
    def __init__(self, input_size=15, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(LungCancerNetLayerNorm, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LungCancerNetResidual(nn.Module):
    """MLP with residual (skip) connections. Representative of modern design."""
    def __init__(self, input_size=15, hidden_size=128, dropout_rate=0.3):
        super(LungCancerNetResidual, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 2)
        self.skip = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        residual = out
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = out + self.skip(residual)
        out = F.relu(self.bn3(self.fc3(out)))
        return self.fc4(out)
