"""Shared data loading, encoding, and checkpoint helpers for training/eval scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mask_utils import build_mask_from_fullres
from models.cnn_encoder import CNNEncoder
from models.latent_world_model import LatentWorldModel
from models.ttc_predictor import TTCPredictor


def hwc_to_chw(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame).permute(2, 0, 1).contiguous()


def resize_chw(
    image: torch.Tensor,
    out_h: int,
    out_w: int,
    mode: str = "bicubic",
) -> torch.Tensor:
    x = image.unsqueeze(0).float()
    kwargs = {"mode": mode, "align_corners": False}
    if mode in ("bilinear", "bicubic"):
        kwargs["antialias"] = True
    x = F.interpolate(x, size=(out_h, out_w), **kwargs)
    return x.squeeze(0).round().clamp(0, 255).to(torch.uint8)


def load_visuals_paths(data_dir: Path, split: str) -> list[Path]:
    paths = sorted((data_dir / split).glob("*_visuals.npz"))
    if not paths:
        raise FileNotFoundError(f"No *_visuals.npz files in {data_dir / split}")
    return paths


def visuals_path_to_data_path(visuals_path: Path) -> Path:
    return visuals_path.with_name(visuals_path.name.replace("_visuals.npz", "_data.csv"))


def load_episode_tabular(
    data_path: Path,
    max_frames: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Actions and obs_ttc aligned 1:1 with frames; each shape (T,)."""
    import pandas as pd

    df = pd.read_csv(data_path)
    if max_frames is not None:
        df = df.iloc[:max_frames]
    actions = torch.tensor(df["action"].to_numpy(), dtype=torch.long)
    ttc = torch.tensor(df["obs_ttc"].to_numpy(), dtype=torch.float32)
    return actions, ttc


class EpisodeDataset(Dataset):
    """Full episodes as variable-length frame sequences."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        out_h: int,
        out_w: int,
        interp_mode: str = "bicubic",
        max_episodes: int | None = None,
        max_frames: int | None = None,
        min_frames: int = 2,
    ):
        self.out_h = out_h
        self.out_w = out_w
        self.interp_mode = interp_mode
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.paths: list[Path] = []
        self.episode_lengths: list[int] = []

        paths = load_visuals_paths(data_dir, split)
        if max_episodes is not None:
            paths = paths[:max_episodes]

        for path in paths:
            with np.load(path) as data:
                num_frames = data["visuals"].shape[0]
            usable = num_frames
            if max_frames is not None:
                usable = min(usable, max_frames)
            if usable >= min_frames:
                self.paths.append(path)
                self.episode_lengths.append(usable)

        if not self.paths:
            raise ValueError(f"No episodes with >= {min_frames} frames found for split={split!r}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        with np.load(path) as data:
            visuals = data["visuals"]

        frames = []
        for frame_idx in range(visuals.shape[0]):
            chw = hwc_to_chw(visuals[frame_idx])
            frames.append(resize_chw(chw, self.out_h, self.out_w, mode=self.interp_mode))
            if self.max_frames is not None and len(frames) >= self.max_frames:
                break

        actions, ttc = load_episode_tabular(
            visuals_path_to_data_path(path),
            max_frames=self.max_frames,
        )
        if actions.shape[0] != len(frames) or ttc.shape[0] != len(frames):
            raise ValueError(
                f"Tabular/frame length mismatch for {path.name}: "
                f"{actions.shape[0]} actions, {ttc.shape[0]} ttc vs {len(frames)} frames"
            )
        return torch.stack(frames), actions, ttc


def collate_episodes(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack equal-length episodes: frames (B, T, C, H, W), actions (B, T), ttc (B, T), lengths (B,)."""
    frames = torch.stack([sample[0] for sample in batch], dim=0)
    actions = torch.stack([sample[1] for sample in batch], dim=0)
    ttc = torch.stack([sample[2] for sample in batch], dim=0)
    lengths = torch.full((len(batch),), frames.shape[1], dtype=torch.long)
    return frames, actions, ttc, lengths


def flatten_valid_frames(frames: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) -> (N, C, H, W) using per-episode lengths."""
    chunks = [frames[i, : int(length)] for i, length in enumerate(lengths.tolist())]
    return torch.cat(chunks, dim=0)


def encode_episode_frames(
    encoder: CNNEncoder,
    frames: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Encode each frame with the (trainable) CNN encoder -> (B, T, D)."""
    b, t, _, _, _ = frames.shape
    d = encoder.embed_dim * encoder.grid_h * encoder.grid_w
    latents = torch.zeros(b, t, d, device=frames.device, dtype=torch.float32)
    for frame_idx in range(int(lengths.max().item())):
        z = encoder.encode_spatial(frames[:, frame_idx]).flatten(1)
        latents[:, frame_idx] = z
    return latents


def build_encoder(image_size: tuple[int, int], device: torch.device) -> CNNEncoder:
    return CNNEncoder(image_size=image_size, input_uint8=True).to(device)


def load_encoder_checkpoint(
    encoder: CNNEncoder,
    checkpoint_path: Path | None,
    device: torch.device,
) -> None:
    if checkpoint_path is None or not checkpoint_path.exists():
        return
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "encoder" in ckpt:
        try:
            encoder.load_state_dict(ckpt["encoder"])
        except RuntimeError:
            encoder.encoder.load_state_dict(ckpt["encoder"])
    elif "model" in ckpt:
        encoder.load_state_dict(ckpt["model"])
    else:
        raise KeyError(f"No encoder/model weights in {checkpoint_path}")


def load_training_checkpoint(
    checkpoint_path: Path,
    encoder: CNNEncoder,
    world_model: LatentWorldModel,
    device: torch.device,
    ttc_predictor: TTCPredictor | None = None,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    if "world_model" in ckpt:
        world_model.load_state_dict(ckpt["world_model"])
    elif "model" in ckpt:
        world_model.load_state_dict(ckpt["model"])
    if ttc_predictor is not None and "ttc_predictor" in ckpt:
        ttc_predictor.load_state_dict(ckpt["ttc_predictor"])
    return ckpt


def next_step_mask(lengths: torch.Tensor, pred_steps: int, device: torch.device) -> torch.Tensor:
    """Valid next-step mask; shape (B, pred_steps)."""
    mask = torch.zeros(len(lengths), pred_steps, device=device)
    for i, length in enumerate(lengths.tolist()):
        valid = max(int(length) - 1, 0)
        if valid > 0:
            mask[i, :valid] = 1.0
    return mask


def pixel_weight_map(frames: torch.Tensor, gain: float) -> torch.Tensor:
    """Per-pixel loss weights 1 + gain*mask from uint8 frames (B, T, C, H, W) -> (B, T, 1, H, W)."""
    b, t, c, h, w = frames.shape
    flat = frames.reshape(b * t, c, h, w).float()
    mask = build_mask_from_fullres(flat / 127.5 - 1.0, h, w)
    return 1.0 + gain * mask.reshape(b, t, 1, h, w)


def save_training_checkpoint(
    path: Path,
    encoder: CNNEncoder,
    world_model: LatentWorldModel,
    ttc_predictor: TTCPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    phase: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "world_model": world_model.state_dict(),
        "ttc_predictor": ttc_predictor.state_dict(),
        "model": world_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if phase is not None:
        payload["phase"] = phase
    torch.save(payload, path)


def save_training_log(path: Path, args: argparse.Namespace, history: list | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if isinstance(history, list):
        payload: dict[str, Any] = {"args": args_dict, "epochs": history}
    else:
        payload = {"args": args_dict, **history}
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def format_param(value: float) -> str:
    text = f"{value:g}" if float(value).is_integer() else f"{value:.4g}"
    return text.replace(".", "p").replace("-", "m")


def run_output_name(args: argparse.Namespace, *, include_c: bool = True) -> str:
    """Directory name for this run, derived from key hyperparameters."""
    if args.run_name is not None:
        return args.run_name
    name = f"a{format_param(args.loss_a)}_b{format_param(args.loss_b)}"
    if include_c:
        name += f"_c{format_param(args.loss_c)}"
    if getattr(args, "pixel_mask_gain", 0.0) > 0.0:
        name += f"_m{format_param(args.pixel_mask_gain)}"
    return name


def resolve_output_paths(
    args: argparse.Namespace,
    *,
    include_c: bool = True,
) -> tuple[Path, Path]:
    """Place checkpoints/logs under checkpoint_dir/<run_name>/ so runs don't overwrite."""
    run_dir = args.checkpoint_dir / run_output_name(args, include_c=include_c)
    run_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = run_dir
    log_path = args.log_path if args.log_path is not None else (run_dir / "train_log.json")
    return run_dir, log_path
