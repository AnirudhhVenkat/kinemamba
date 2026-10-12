"""Predict time-to-collision from Mamba latent predictions."""

from __future__ import annotations

import torch
import torch.nn as nn


class TTCPredictor(nn.Module):
    """MLP head: predicted CNN latent(s) -> scalar obs_ttc.

    Training uses ``forward`` on the full teacher-forced sequence ``(B, T, D)``.
    Autoregressive rollout calls ``predict_next`` once per step with ``(B, D)``.
    """

    def __init__(self, mamba_dim: int=1024, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(mamba_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, pred_latents: torch.Tensor) -> torch.Tensor:
        """Apply head on the last dim. ``(B, T, D) -> (B, T)`` or ``(B, D) -> (B,)``."""
        return self.head(pred_latents).squeeze(-1)

    def predict_next(self, pred_latents: torch.Tensor) -> torch.Tensor:
        """TTC for one step: ``(B, D) -> (B,)``. If given ``(B, T, D)``, use the last timestep only."""
        if pred_latents.ndim == 2:
            return self.head(pred_latents).squeeze(-1)
        return self.forward(pred_latents)[:, -1]
