"""CNN latents + actions -> Mamba2 dynamics -> pixel decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder_decoder import Decoder
from .mamba2_latent_dynamics import Mamba2LatentDynamics
from typing import List

class LatentWorldModel(nn.Module):
    """CNN latents + actions -> Mamba2 dynamics -> CNN decoder."""

    def __init__(
        self,
        latent_dim: int,
        out_h: int,
        out_w: int,
        num_actions: int = 5,
        action_embed_dim: int = 32,
        mamba_dim: int = 1024,
        n_layers: int = 4,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
        depths: List[int]=[2,2,2,2], 
        channels: List[int]=[128,192,256,256], 
        attn_depths: List[bool]=[False,False,False,False]
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_h = out_h
        self.out_w = out_w
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
        self.decoder = Decoder(depths,channels,attn_depths)
        self.channels = channels
        self.out_h = out_h
        self.out_w = out_w
        

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full episode latents/actions (B, T, ...) -> (pred_latents, pred_pixels) each (B, T-1, ...)."""
        context_latents = latents[:, :-1]
        context_actions = actions[:, :-1]
        h, pred_latents = self.dynamics(context_latents, context_actions)
        b,t,_ = pred_latents.shape
        c,height,w = self.channels[-1], self.out_h // (2 ** len(self.channels)), self.out_w // (2 ** len(self.channels))
        pred_pixels = self.decoder(pred_latents.contiguous().view(b*t,c,height,w))
        pred_pixels = pred_pixels.view(b,t,3,self.out_h,self.out_w)
        return h, pred_latents, pred_pixels

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
        b,t,_ = pred_latents.shape
        c,height,w = self.channels[-1], self.out_h // (2 ** len(self.channels)), self.out_w // (2 ** len(self.channels))
        pred_pixels = self.decoder(pred_latents.contiguous().view(b*t,c,height,w))
        pred_pixels = pred_pixels.view(b,t,3,self.out_h,self.out_w)
        return pred_latents, pred_pixels, step_outputs
