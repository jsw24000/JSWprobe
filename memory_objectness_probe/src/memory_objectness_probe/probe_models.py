from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShallowMLPProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_probe(model_name: str, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1) -> nn.Module:
    if model_name == "linear":
        return LinearProbe(input_dim, num_classes)
    if model_name == "mlp":
        return ShallowMLPProbe(input_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown probe model: {model_name}")
