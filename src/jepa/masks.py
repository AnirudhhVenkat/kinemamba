# Adapted from facebookresearch/jepa src/masks/utils.py and src/masks/multiblock3d.py
# https://github.com/facebookresearch/jepa

import math

import torch


def apply_masks(x, masks, concat=True):
    """Gather token rows by index masks.

    :param x: [B, N, D]
    :param masks: list of [B, K] index tensors
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


class SpatialMaskGenerator:
    """2D block masks for CNN feature maps (duration=1)."""

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        spatial_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        num_blocks: int = 1,
        max_keep: int | None = None,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.spatial_scale = spatial_scale
        self.aspect_ratio = aspect_ratio
        self.num_blocks = num_blocks
        self.max_keep = max_keep
        self._step = 0

    def step(self):
        self._step += 1
        return self._step

    def _sample_block_size(self, generator: torch.Generator):
        rand = torch.rand(1, generator=generator).item()
        min_s, max_s = self.spatial_scale
        spatial_mask_scale = min_s + rand * (max_s - min_s)
        spatial_num_keep = int(self.grid_h * self.grid_w * spatial_mask_scale)

        rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = self.aspect_ratio
        ar = min_ar + rand * (max_ar - min_ar)

        h = int(round(math.sqrt(spatial_num_keep * ar)))
        w = int(round(math.sqrt(spatial_num_keep / ar)))
        h = min(max(h, 1), self.grid_h)
        w = min(max(w, 1), self.grid_w)
        return h, w

    def _sample_block_mask(self, block_hw):
        h, w = block_hw
        top = torch.randint(0, self.grid_h - h + 1, (1,))
        left = torch.randint(0, self.grid_w - w + 1, (1,))
        mask = torch.ones((self.grid_h, self.grid_w), dtype=torch.int32)
        mask[top:top + h, left:left + w] = 0
        return mask

    def __call__(self, batch_size: int):
        g = torch.Generator()
        g.manual_seed(self.step())
        block_hw = self._sample_block_size(g)

        masks_pred, masks_enc = [], []
        min_keep_enc = min_keep_pred = self.grid_h * self.grid_w

        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                mask_e = torch.ones((self.grid_h, self.grid_w), dtype=torch.int32)
                for _ in range(self.num_blocks):
                    mask_e *= self._sample_block_mask(block_hw)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze(-1)
                mask_e = torch.nonzero(mask_e).squeeze(-1)

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    masks_pred.append(mask_p)
                    masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        masks_pred = [m[:min_keep_pred] for m in masks_pred]
        masks_enc = [m[:min_keep_enc] for m in masks_enc]
        return torch.utils.data.default_collate(masks_enc), torch.utils.data.default_collate(masks_pred)
