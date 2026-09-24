"""CNN latents + actions -> Mamba2 dynamics -> linear pixel decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .linear_pixel_decoder import LinearPixelDecoder
from .mamba2_latent_dynamics import Mamba2LatentDynamics


class LatentWorldModel(nn.Module):
    """CNN latents + actions -> Mamba2 dynamics -> linear pixel decoder."""

    def __init__(
        self,
        latent_dim: int,
        out_h: int,
        out_w: int,
        num_actions: int = 5,
        action_embed_dim: int = 32,
        mamba_dim: int = 512,
        n_layers: int = 4,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dynamics = Mamba2LatentDynamics(
            latent_dim=latent_dim,
            num_actions=num_actions,
            action_embed_dim=action_embed_dim,
            mamba_dim=mamba_dim,
            n_layers=n_layers,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )
        self.decoder = LinearPixelDecoder(latent_dim, out_h, out_w)

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full episode latents/actions (B, T, ...) -> (pred_latents, pred_pixels) each (B, T-1, ...)."""
        context_latents = latents[:, :-1]
        context_actions = actions[:, :-1]
        pred_latents = self.dynamics(context_latents, context_actions)
        pred_pixels = self.decoder(pred_latents)
        return pred_latents, pred_pixels

    def predict_next(
        self,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one step from context z_0..z_{t-1} and matching actions; returns (z_t_hat, x_t_hat)."""
        pred_latents = self.dynamics(context_latents, context_actions)
        pred_pixels = self.decoder(pred_latents[:, -1:])
        return pred_latents[:, -1], pred_pixels[:, 0]

    @torch.no_grad()
    def rollout(
        self,
        context_latents: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        step_head=None,
        step_callback=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Stateful AR rollout: prefill context once, then step SSM state.

        ``step_head`` runs per step during the rollout; pixels are decoded in a
        single batched pass once stepping is done.

        Returns (pred_latents, pred_pixels, step_outputs).
        """
        pred_latents, step_outputs = self.dynamics.rollout(
            context_latents,
            actions,
            horizon,
            step_head=step_head,
            step_callback=step_callback,
        )
        pred_pixels = self.decoder(pred_latents)
        return pred_latents, pred_pixels, step_outputs
