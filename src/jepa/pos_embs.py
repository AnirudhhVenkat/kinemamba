# Adapted from facebookresearch/jepa src/models/utils/pos_embs.py
# https://github.com/facebookresearch/jepa

import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed_hw(grid_h: int, grid_w: int, embed_dim: int) -> np.ndarray:
    """Sinusoidal pos embed for a rectangular H x W token grid."""
    grid_h_arr = np.arange(grid_h, dtype=float)
    grid_w_arr = np.arange(grid_w, dtype=float)
    grid_w_mesh, grid_h_mesh = np.meshgrid(grid_w_arr, grid_h_arr)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h_mesh)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w_mesh)
    return np.concatenate([emb_h, emb_w], axis=1)
