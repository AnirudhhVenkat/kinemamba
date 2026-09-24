"""Mamba2 stack that predicts the next CNN latent at each timestep."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "mamba2") not in sys.path:
    sys.path.insert(0, str(_ROOT / "mamba2"))

from mamba_ssm import Mamba2

try:
    import causal_conv1d  # noqa: F401
    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False


class Mamba2LatentDynamics(nn.Module):
    """Mamba2 stack that predicts the next CNN latent at each timestep."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int = 5,
        action_embed_dim: int = 32,
        mamba_dim: int = 512, #output dim D for mamba2 block
        n_layers: int = 4,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim #dim from CNN encoding 
        self.action_embed_dim = action_embed_dim 
        self.mamba_dim = mamba_dim # output dim D for mamba2 block
        self.n_layers = n_layers
        input_dim = latent_dim + action_embed_dim #sum action dims + cnn encoding dims
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj_in = nn.Linear(input_dim, mamba_dim) if mamba_dim != input_dim else nn.Identity() #encoding -> mamba_dim
        self.layer_norms = nn.ModuleList([nn.LayerNorm(mamba_dim) for _ in range(n_layers)])
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=mamba_dim,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                layer_idx=i,
                # Fused path needs causal-conv1d CUDA extension
                use_mem_eff_path=_HAS_CAUSAL_CONV1D,
            )
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(mamba_dim)
        self.proj_out = nn.Linear(mamba_dim, latent_dim) #mamba_dim to encoding

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def make_inference_params(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
    ):
        """Allocate conv/SSM states for stateful decode (prefill once, then step)."""
        from mamba_ssm.utils.generation import InferenceParams

        params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
        params.key_value_memory_dict = self.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype
        )
        return params

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        inference_params=None,
    ) -> torch.Tensor:
        """latents (B, T, D), actions (B, T) -> predicted z_{t+1} at each t, shape (B, T, D).

        Pass ``inference_params`` to fill/update SSM+conv state. With
        ``seqlen_offset == 0`` this prefills the cache over the full sequence;
        with ``seqlen_offset > 0`` each call should be T=1 and only steps the state.
        """
        action_feats = self.action_embed(actions)
        x = torch.cat([latents, action_feats], dim=-1)
        x = self.proj_in(self.input_norm(x)) #encdonig to mamba_dim
        for norm, layer in zip(self.layer_norms, self.layers): #loop through all mamba2 layers
            x = x + layer(norm(x), inference_params=inference_params)
        x = self.norm(x)
        return self.proj_out(x) #B,T,latent_dim

    @torch.no_grad()
    def rollout(
        self,
        context_latents: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        step_head=None,
        step_callback=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prefill on context, then AR-step ``horizon`` next latents via SSM state.

        Args:
            context_latents: (B, C, D) encoded GT context frames.
            actions: (B, C + horizon) actions; uses a_0..a_{C-1} in prefill and
                a_C..a_{C+horizon-2} while stepping.
            horizon: number of future latents to predict.
            step_head: optional callable applied to each predicted latent (B, D)
                as soon as it is produced, e.g. a TTC head that must emit a
                value every step rather than in one pass at the end.
            step_callback: optional no-arg callable invoked once per predicted
                step, for progress reporting.

        Returns:
            (pred_latents, step_outputs) where pred_latents is (B, horizon, D)
            for z_C .. z_{C+horizon-1}, and step_outputs stacks ``step_head``
            results along dim 1 (or None when no head is given).
        """
        batch, context_steps, _ = context_latents.shape

        max_seqlen = context_steps + horizon
        inference_params = self.make_inference_params(
            batch_size=batch,
            max_seqlen=max_seqlen,
            dtype=context_latents.dtype,
        )

        # Prefill: (z_0..z_{C-1}, a_0..a_{C-1}) -> preds; last is z_C.
        prefill_out = self.forward(
            context_latents,
            actions[:, :context_steps],
            inference_params=inference_params,
        )
        inference_params.seqlen_offset = context_steps

        z_hat = prefill_out[:, -1]
        preds = [z_hat]
        step_outputs = [step_head(z_hat)] if step_head is not None else None
        if step_callback is not None:
            step_callback()
        for step in range(1, horizon):
            action_t = actions[:, context_steps + step - 1]
            z_hat = self.forward(
                z_hat.unsqueeze(1),
                action_t.unsqueeze(1),
                inference_params=inference_params,
            )[:, 0]
            inference_params.seqlen_offset += 1
            preds.append(z_hat)
            if step_outputs is not None:
                step_outputs.append(step_head(z_hat))
            if step_callback is not None:
                step_callback()

        stacked_outputs = (
            torch.stack(step_outputs, dim=1) if step_outputs is not None else None
        )
        return torch.stack(preds, dim=1), stacked_outputs  # (B, T, D)
