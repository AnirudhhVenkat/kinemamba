from functools import partial
import math
from typing import List, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


# Settings for GroupNorm and Attention

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

# Convs

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)

# GroupNorm

def normalize_image(x: torch.Tensor, *, input_uint8: bool = True) -> torch.Tensor:
    if input_uint8:
        return x.float() / 255.0 - 0.5
    return x.float()


class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


# Self Attention


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


# [Down|Up]sampling


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# Small Residual block


class SmallResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.f = nn.Sequential(GroupNorm(in_channels), nn.SiLU(inplace=True), Conv3x3(in_channels, out_channels))
        self.skip_projection = nn.Identity() if in_channels == out_channels else Conv1x1(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.skip_projection(x) + self.f(x)


# Residual block (GroupNorm, no [down|up]sampling, optional self-attention)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = GroupNorm(in_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = GroupNorm(out_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        x = x + r
        x = self.attn(x)
        return x


# Sequence of residual blocks (in_channels -> mid_channels -> ... -> mid_channels -> out_channels)


class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(self, x: Tensor, to_cat: Optional[List[Tensor]] = None) -> Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x)
            outputs.append(x)
        return x


# Encoder and Decoder


class Encoder(nn.Module):
    def __init__(self, depths: List[int]=[2,2,2,2], channels: List[int]=[128,192,256,256], attn_depths: List[bool]=[False,False,False,False]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self.channels = channels #to be used in utils.py
        self._num_down = len(channels)
        self.startconv = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1)
        d_blocks= []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    attn=attn_depths[i],
                )
            )

        self.d_blocks = nn.ModuleList(d_blocks)

        downsamples = [Downsample(c) for c in channels]
        
        self.downsamples = nn.ModuleList(downsamples)
      

    def forward(self, x: Tensor) -> Tensor:
        # Dataset frames are uint8; conv weights/bias are float32.
        x = normalize_image(x).float()
        x = self.startconv(x)

        for block, down in zip(self.d_blocks, self.downsamples):
            x = block(x)
            x = down(x)
            
        return x


class Decoder(nn.Module):
    def __init__(self, depths: List[int]=[2,2,2,2], channels: List[int]=[128,192,256,256], attn_depths: List[bool]=[False,False,False,False]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1
        self.endconv = nn.Conv2d(channels[0], 3, kernel_size=3, stride=1, padding=1)
        u_blocks = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[c2] * n,
                    list_out_channels=[c2] * (n-1) + [c1],
                    attn=attn_depths[i],
                )
            )
        
        self.u_blocks = nn.ModuleList(reversed(u_blocks))


        upsamples = [Upsample(c) for c in reversed(channels)]
        
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: Tensor) -> Tensor:

        for block, up in zip(self.u_blocks, self.upsamples):
            x_up = up(x)
            x = block(x_up)
            
        x = self.endconv(x)

        return 0.5 * torch.tanh(x)
        

'''
"""DreamerV3-style CNN encoder with JEPA training."""

from __future__ import annotations

import copy
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.loss import jepa_loss, update_target_encoder
from jepa.masks import SpatialMaskGenerator, apply_masks
from jepa.predictor import SpatialPredictor




def _get_act(name: str) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "mish":
        return nn.Mish(inplace=True)
    raise NotImplementedError(name)


def _pool2x2(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // 2, 2, w // 2, 2)
    return x.amax(dim=(3, 5))


class RMSNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.scale.view(1, -1, 1, 1)


class _ConvEncoder(nn.Module):
    """DreamerV3-style conv stack: normalize -> conv/RMS/act -> 2x2 pool."""

    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4),
        kernel: int = 5,
        act: str = "silu",
        strided: bool = False,
        outer: bool = False,
        input_uint8: bool = True,
    ):
        super().__init__()
        self.input_uint8 = input_uint8
        self.strided = strided
        self.outer = outer

        depths = tuple(depth * m for m in mults)
        self.out_channels = depths[-1]
        padding = kernel // 2

        convs, norms, acts = [], [], []
        ch_in = in_channels
        for i, ch_out in enumerate(depths):
            stride = 2 if strided and not (outer and i == 0) else 1
            convs.append(nn.Conv2d(ch_in, ch_out, kernel, stride=stride, padding=padding))
            norms.append(RMSNorm2d(ch_out))
            acts.append(_get_act(act))
            ch_in = ch_out
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.acts = nn.ModuleList(acts)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_image(x, input_uint8=self.input_uint8)

    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.preprocess(x)
        for i, (conv, norm, act) in enumerate(zip(self.convs, self.norms, self.acts)):
            x = conv(x)
            if not self.strided and not (self.outer and i == 0):
                x = _pool2x2(x)
            x = act(norm(x))
        return x

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encode_spatial(x)
        return feat.flatten(2).transpose(1, 2)


class CNNEncoder(nn.Module):
    """DreamerV3-style CNN encoder with JEPA training built in.

    Inference: ``forward(x)`` or ``encode_tokens(x)`` on a trained model.
    Training: ``training_step(batch)`` then ``update_target()``.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (64, 64),
        in_channels: int = 3,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4),
        kernel: int = 5,
        act: str = "silu",
        strided: bool = False,
        outer: bool = False,
        input_uint8: bool = True,
        predictor_embed_dim: int = 256,
        predictor_depth: int = 4,
        predictor_heads: int = 8,
        loss_exp: float = 2.0,
        reg_coeff: float = 1.0,
        ema_momentum: float = 0.996,
        mask_spatial_scale=(0.2, 0.8),
        mask_aspect_ratio=(0.3, 3.0),
        mask_num_blocks: int = 1,
    ):
        super().__init__()
        self.encoder = _ConvEncoder(
            in_channels=in_channels,
            depth=depth,
            mults=mults,
            kernel=kernel,
            act=act,
            strided=strided,
            outer=outer,
            input_uint8=input_uint8,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size[0], image_size[1])
            spatial = self.encoder.encode_spatial(dummy)
            _, embed_dim, grid_h, grid_w = spatial.shape

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.embed_dim = embed_dim
        self.loss_exp = loss_exp
        self.reg_coeff = reg_coeff
        self.ema_momentum = ema_momentum

        self.predictor = SpatialPredictor(
            grid_h=grid_h,
            grid_w=grid_w,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )
        self.mask_generator = SpatialMaskGenerator(
            grid_h=grid_h,
            grid_w=grid_w,
            spatial_scale=mask_spatial_scale,
            aspect_ratio=mask_aspect_ratio,
            num_blocks=mask_num_blocks,
        )

    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_spatial(x)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_tokens(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: single image ``(C, H, W)`` -> feature vector."""
        assert x.ndim == 3
        return self.encode_spatial(x).flatten(1).squeeze(0)

    @torch.no_grad()
    def _forward_target(self, images: torch.Tensor, masks_pred: Sequence[torch.Tensor]):
        h = self.target_encoder.encode_tokens(images)
        h = F.layer_norm(h, (h.size(-1),))
        return apply_masks(h, masks_pred, concat=False)

    def _forward_context(self, images: torch.Tensor, masks_enc, masks_pred):
        z = self.encode_tokens(images)
        z = apply_masks(z, masks_enc)
        return self.predictor(z, masks_enc, masks_pred)

    def training_step(self, images: torch.Tensor) -> dict:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        masks_enc, masks_pred = self.mask_generator(images.shape[0])
        masks_enc = masks_enc.to(images.device)
        masks_pred = masks_pred.to(images.device)
        targets = self._forward_target(images, masks_pred)
        preds = self._forward_context(images, masks_enc, masks_pred)
        total, loss_jepa, loss_reg = jepa_loss(
            preds, targets, loss_exp=self.loss_exp, reg_coeff=self.reg_coeff,
        )
        return {
            "loss": total,
            "loss_jepa": loss_jepa,
            "loss_reg": loss_reg,
            "masks_enc": masks_enc,
            "masks_pred": masks_pred,
        }

    @torch.no_grad()
    def update_target(self):
        update_target_encoder(self.encoder, self.target_encoder, self.ema_momentum)
'''