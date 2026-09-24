# Adapted from facebookresearch/jepa src/models/predictor.py
# https://github.com/facebookresearch/jepa

import math
from functools import partial

import torch
import torch.nn as nn

from jepa.masks import apply_masks
from jepa.modules import Block
from jepa.pos_embs import get_2d_sincos_pos_embed_hw


def _trunc_normal_(tensor, mean=0.0, std=1.0):
    with torch.no_grad():
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2.0, b=2.0)


class SpatialPredictor(nn.Module):
    """JEPA predictor for rectangular spatial token grids."""

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        embed_dim: int,
        predictor_embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_mask_tokens: bool = True,
        num_mask_tokens: int = 1,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.embed_dim = embed_dim

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_tokens = None
        if use_mask_tokens:
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ])

        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim),
            requires_grad=False,
        )
        sincos = get_2d_sincos_pos_embed_hw(grid_h, grid_w, predictor_embed_dim)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, ctxt, masks_ctxt, masks_tgt, mask_index: int = 0):
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        B = len(ctxt) // len(masks_ctxt)
        x = self.predictor_embed(ctxt)
        _, n_ctxt, _ = x.shape

        ctxt_pos = self.predictor_pos_embed.repeat(B, 1, 1)
        x = x + apply_masks(ctxt_pos, masks_ctxt)

        if self.mask_tokens is None:
            raise NotImplementedError("mask tokens required for this predictor")
        mask_index = mask_index % len(self.mask_tokens)
        pred_tokens = self.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_tgt)

        pos_embs = apply_masks(self.predictor_pos_embed.repeat(B, 1, 1), masks_tgt)
        pred_tokens = pred_tokens + pos_embs

        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)

        x = self.predictor_norm(x)
        x = x[:, n_ctxt:]
        return self.predictor_proj(x)
