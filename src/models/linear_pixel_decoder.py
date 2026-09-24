"""Map CNN latents to RGB pixels with a single linear layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearPixelDecoder(nn.Module):
    """Map CNN latents to RGB pixels with a single linear layer."""

    def __init__(self, latent_dim: int, out_h: int, out_w: int):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.head = nn.Linear(latent_dim, 3 * out_h * out_w)
        nn.init.zeros_(self.head.bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: (B, T, D) -> pixels (B, T, 3, H, W), bounded to [-0.5, 0.5]."""
        pixels = 0.5 * torch.tanh(self.head(latents))
        return pixels.view(latents.shape[0], latents.shape[1], 3, self.out_h, self.out_w)
