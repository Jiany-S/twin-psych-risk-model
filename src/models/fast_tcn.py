"""Small causal TCN for fast physiological-state detection."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class FastTCN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        channels = input_channels
        for layer_idx in range(layers):
            dilation = 2**layer_idx
            blocks.extend([CausalConv1d(channels, hidden_channels, kernel_size, dilation=dilation), nn.ReLU(), nn.Dropout(dropout)])
            channels = hidden_channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.net(x)
        return self.head(encoded[:, :, -1]).squeeze(-1)

    def predict_proba_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_bytes(model: nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))
