"""Predict time-to-collision from Mamba latent predictions."""

from __future__ import annotations

import torch
import torch.nn as nn


class TTCPredictor(nn.Module):
    """Predict obs_ttc from Mamba latent predictions (one TTC per predicted latent)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, pred_latents: torch.Tensor) -> torch.Tensor:
        """pred_latents (B, T, D) -> TTC (B, T)."""
        return self.head(pred_latents).squeeze(-1)

    def predict_next(self, pred_latents: torch.Tensor) -> torch.Tensor:
        """Next-step TTC from final predicted latent; (B, L, D) or (B, D) -> (B,)."""
        if pred_latents.ndim == 2:
            return self.head(pred_latents).squeeze(-1)
        return self.forward(pred_latents)[:, -1]
