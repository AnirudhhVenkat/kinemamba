#!/usr/bin/env python3
"""Joint training: JEPA CNN encoder + Mamba2 latent dynamics + linear pixel decoder + linear TTC predictor."""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SEED = 42
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "mamba2"))

from mask_utils import set_mask_params
from models.cnn_encoder import CNNEncoder, normalize_image
from models.latent_world_model import LatentWorldModel
from models.ttc_predictor import TTCPredictor
from utils import (
    EpisodeDataset,
    build_encoder,
    collate_episodes,
    encode_episode_frames,
    flatten_valid_frames,
    load_encoder_checkpoint,
    load_training_checkpoint,
    next_step_mask,
    pixel_weight_map,
    resolve_output_paths,
    save_training_checkpoint,
    save_training_log,
)


def combine_training_loss(
    jepa_loss: torch.Tensor,
    latent_loss: torch.Tensor,
    pixel_loss: torch.Tensor,
    ttc_loss: torch.Tensor,
    loss_a: float,
    loss_b: float,
    loss_c: float,
) -> torch.Tensor:
    """jepa + a*latent + b*pixel + c*ttc."""
    return jepa_loss + loss_a * latent_loss + loss_b * pixel_loss + loss_c * ttc_loss


def compute_batch_metrics(
    encoder: CNNEncoder,
    world_model: LatentWorldModel,
    ttc_predictor: TTCPredictor,
    frames: torch.Tensor,
    actions: torch.Tensor,
    ttc: torch.Tensor,
    lengths: torch.Tensor,
    *,
    run_jepa: bool = True,
    pixel_mask_gain: float = 0.0,
) -> dict[str, torch.Tensor]:
    jepa_out = None
    if run_jepa:
        flat_frames = flatten_valid_frames(frames, lengths)
        jepa_out = encoder.training_step(flat_frames)

    latents = encode_episode_frames(encoder, frames, lengths)
    pred_latents, pred_pixels = world_model(latents, actions)
    target_latents = latents[:, 1:]
    target_pixels = normalize_image(frames[:, 1:])
    target_ttc = ttc[:, 1:]

    mask = next_step_mask(lengths, pred_latents.shape[1], frames.device)
    latent_mask = mask.unsqueeze(-1)
    pixel_mask = mask.view(*mask.shape, 1, 1, 1)
    ttc_mask = mask

    latent_sq = (pred_latents - target_latents).pow(2) * latent_mask
    latent_denom = latent_mask.sum() * pred_latents.shape[-1]
    latent_loss = latent_sq.sum() / latent_denom.clamp_min(1.0)

    pixel_sq = (pred_pixels - target_pixels).pow(2) * pixel_mask
    channels = pred_pixels.shape[2]
    if pixel_mask_gain > 0.0:
        with torch.no_grad():
            weights = pixel_weight_map(frames[:, 1:], pixel_mask_gain)
        pixel_sq = pixel_sq * weights
        pixel_denom = (weights * pixel_mask).sum() * channels
    else:
        pixels_per_frame = channels * pred_pixels.shape[3] * pred_pixels.shape[4]
        pixel_denom = pixel_mask.sum() * pixels_per_frame
    pixel_loss = pixel_sq.sum() / pixel_denom.clamp_min(1.0)

    pred_ttc = ttc_predictor(pred_latents)
    ttc_sq = (pred_ttc - target_ttc).pow(2) * ttc_mask
    ttc_loss = ttc_sq.sum() / ttc_mask.sum().clamp_min(1.0)

    metrics = {
        "latent_loss": latent_loss,
        "pixel_loss": pixel_loss,
        "ttc_loss": ttc_loss,
    }
    if jepa_out is not None:
        metrics["jepa_loss"] = jepa_out["loss"]
        metrics["loss_jepa"] = jepa_out["loss_jepa"]
        metrics["loss_reg"] = jepa_out["loss_reg"]
    return metrics


def training_loss(
    encoder: CNNEncoder,
    world_model: LatentWorldModel,
    ttc_predictor: TTCPredictor,
    frames: torch.Tensor,
    actions: torch.Tensor,
    ttc: torch.Tensor,
    lengths: torch.Tensor,
    loss_a: float = 0.5,
    loss_b: float = 0.5,
    loss_c: float = 0.5,
    pixel_mask_gain: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """JEPA encoder loss + teacher-forced latent/pixel world-model loss + TTC on predicted latents."""
    metrics = compute_batch_metrics(
        encoder, world_model, ttc_predictor, frames, actions, ttc, lengths, run_jepa=True,
        pixel_mask_gain=pixel_mask_gain,
    )
    loss = combine_training_loss(
        metrics["jepa_loss"],
        metrics["latent_loss"],
        metrics["pixel_loss"],
        metrics["ttc_loss"],
        loss_a,
        loss_b,
        loss_c,
    )
    metrics["loss"] = loss
    return loss, metrics


@torch.no_grad()
def evaluate(
    encoder: CNNEncoder,
    world_model: LatentWorldModel,
    ttc_predictor: TTCPredictor,
    loader: DataLoader,
    device: torch.device,
    pixel_mask_gain: float = 0.0,
) -> dict[str, float]:
    encoder.eval()
    world_model.eval()
    ttc_predictor.eval()
    totals = {
        "pixel_ttc_loss": 0.0,
        "jepa_loss": 0.0,
        "loss_jepa": 0.0,
        "loss_reg": 0.0,
        "latent_loss": 0.0,
        "pixel_loss": 0.0,
        "ttc_loss": 0.0,
    }
    count = 0
    for frames, actions, ttc, lengths in loader:
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        ttc = ttc.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        metrics = compute_batch_metrics(
            encoder, world_model, ttc_predictor, frames, actions, ttc, lengths, run_jepa=True,
            pixel_mask_gain=pixel_mask_gain,
        )
        batch_size = frames.shape[0]
        pixel_ttc_loss = metrics["pixel_loss"] + metrics["ttc_loss"]
        totals["pixel_ttc_loss"] += pixel_ttc_loss.item() * batch_size
        for key in ("jepa_loss", "loss_jepa", "loss_reg", "latent_loss", "pixel_loss", "ttc_loss"):
            totals[key] += metrics[key].item() * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_one_epoch(
    encoder: CNNEncoder,
    world_model: LatentWorldModel,
    ttc_predictor: TTCPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_a: float,
    loss_b: float,
    loss_c: float,
    max_steps: int | None = None,
    grad_clip: float | None = 1.0,
    pixel_mask_gain: float = 0.0,
) -> dict[str, float]:
    encoder.train()
    world_model.train()
    ttc_predictor.train()
    totals = {
        "loss": 0.0,
        "jepa_loss": 0.0,
        "loss_jepa": 0.0,
        "loss_reg": 0.0,
        "latent_loss": 0.0,
        "pixel_loss": 0.0,
        "ttc_loss": 0.0,
    }
    count = 0

    steps = itertools.islice(loader, max_steps) if max_steps is not None else loader
    pbar = tqdm(steps, desc=f"epoch {epoch}", leave=False)
    for frames, actions, ttc, lengths in pbar:
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        ttc = ttc.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = training_loss(
            encoder, world_model, ttc_predictor, frames, actions, ttc, lengths,
            loss_a=loss_a, loss_b=loss_b, loss_c=loss_c,
            pixel_mask_gain=pixel_mask_gain,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}; try lowering --lr or --mamba-dim")
        loss.backward()
        if grad_clip is not None:
            params = [
                p for p in itertools.chain(
                    encoder.parameters(),
                    world_model.parameters(),
                    ttc_predictor.parameters(),
                )
                if p.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        encoder.update_target()

        batch_size = frames.shape[0]
        totals["loss"] += metrics["loss"].item() * batch_size
        for key in ("jepa_loss", "loss_jepa", "loss_reg", "latent_loss", "pixel_loss", "ttc_loss"):
            totals[key] += metrics[key].item() * batch_size
        count += batch_size
        pbar.set_postfix(
            loss=f"{metrics['loss'].item():.4f}",
            jepa=f"{metrics['jepa_loss'].item():.4f}",
            latent=f"{metrics['latent_loss'].item():.4f}",
            pixel=f"{metrics['pixel_loss'].item():.4f}",
            ttc=f"{metrics['ttc_loss'].item():.4f}",
        )

    return {key: value / max(count, 1) for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint JEPA encoder + Mamba2 latent dynamics + linear pixel decoder",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset_episodes")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints_mamba2")
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=None,
        help="Optional warm-start for encoder (JEPA weights); trained end-to-end if omitted",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume full joint checkpoint")
    parser.add_argument("--out-h", type=int, default=48)
    parser.add_argument("--out-w", type=int, default=320)
    parser.add_argument("--interp-mode", default="bicubic", choices=["bilinear", "bicubic", "nearest", "area"])
    parser.add_argument("--max-frames", type=int, default=None, help="Truncate each episode to this many frames")
    parser.add_argument("--batch-size", type=int, default=1, help="Episodes per batch (episodes must share the same length)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mamba-dim", type=int, default=512, help="Mamba hidden dim (CNN latents are projected down)")
    parser.add_argument(
        "--loss-a",
        type=float,
        default=1.0,
        help="Coefficient on latent loss: total = jepa + a*latent + b*pixel + c*ttc",
    )
    parser.add_argument(
        "--loss-b",
        type=float,
        default=1.0,
        help="Coefficient on pixel loss: total = jepa + a*latent + b*pixel + c*ttc",
    )
    parser.add_argument(
        "--loss-c",
        type=float,
        default=1.0,
        help="Coefficient on TTC loss: total = jepa + a*latent + b*pixel + c*ttc",
    )
    parser.add_argument(
        "--pixel-mask-gain",
        type=float,
        default=20.0,
        help="Weight pixel loss by 1 + gain*vehicle_mask; 0 disables (uniform pixel loss)",
    )
    parser.add_argument("--mask-w-blue", type=float, default=1.0, help="Mask weight for traffic (blue)")
    parser.add_argument("--mask-w-green", type=float, default=0.6, help="Mask weight for ego (green)")
    parser.add_argument("--mask-sigma-global-ratio", type=float, default=0.20)
    parser.add_argument("--mask-sigma-ego-scale", type=float, default=0.50)
    parser.add_argument("--mask-sigma-min-px", type=float, default=2.0)
    parser.add_argument("--ttc-hidden-dim", type=int, default=256, help="Hidden dim for TTCPredictor MLP")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--num-actions", type=int, default=5, help="Discrete action count (0..num_actions-1)")
    parser.add_argument("--action-embed-dim", type=int, default=32, help="Action embedding dim fused with latents")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override output subdirectory name (default: a{a}_b{b}_c{c})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, log_path = resolve_output_paths(args)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(args.device)
    image_size = (args.out_h, args.out_w)

    set_mask_params(
        w_blue=args.mask_w_blue,
        w_green=args.mask_w_green,
        sigma_global_ratio=args.mask_sigma_global_ratio,
        sigma_ego_scale=args.mask_sigma_ego_scale,
        sigma_min_px=args.mask_sigma_min_px,
        downsample_mode=args.interp_mode,
    )

    train_ds = EpisodeDataset(
        args.data_dir,
        "train",
        out_h=args.out_h,
        out_w=args.out_w,
        interp_mode=args.interp_mode,
        max_episodes=args.max_episodes,
        max_frames=args.max_frames,
    )
    val_ds = EpisodeDataset(
        args.data_dir,
        "test",
        out_h=args.out_h,
        out_w=args.out_w,
        interp_mode=args.interp_mode,
        max_episodes=args.max_episodes,
        max_frames=args.max_frames,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_episodes,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_episodes,
    )

    encoder = build_encoder(image_size, device)
    if args.resume is None and args.encoder_checkpoint is not None:
        load_encoder_checkpoint(encoder, args.encoder_checkpoint, device)

    latent_dim = encoder.embed_dim * encoder.grid_h * encoder.grid_w
    world_model = LatentWorldModel(
        latent_dim=latent_dim,
        out_h=args.out_h,
        out_w=args.out_w,
        num_actions=args.num_actions,
        action_embed_dim=args.action_embed_dim,
        mamba_dim=args.mamba_dim,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        headdim=args.headdim,
    ).to(device)
    ttc_predictor = TTCPredictor(latent_dim=latent_dim, hidden_dim=args.ttc_hidden_dim).to(device)

    trainable_params = [
        p for p in itertools.chain(
            encoder.parameters(),
            world_model.parameters(),
            ttc_predictor.parameters(),
        )
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    if args.resume is not None:
        ckpt = load_training_checkpoint(args.resume, encoder, world_model, device, ttc_predictor)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    print(
        f"device={device}  image_size={image_size}  latent_dim={latent_dim}  "
        f"mamba_dim={args.mamba_dim}  loss_a={args.loss_a}  loss_b={args.loss_b}  "
        f"loss_c={args.loss_c}  max_frames={args.max_frames}"
    )
    print(
        f"train episodes={len(train_ds)}  val episodes={len(val_ds)}  "
        f"batch_size={args.batch_size}"
    )
    print(f"output_dir={run_dir}  log_path={log_path}")

    history: list[dict] = []
    best_val_pixel_ttc = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            encoder, world_model, ttc_predictor, train_loader, optimizer, device, epoch,
            loss_a=args.loss_a,
            loss_b=args.loss_b,
            loss_c=args.loss_c,
            max_steps=args.max_steps, grad_clip=args.grad_clip,
            pixel_mask_gain=args.pixel_mask_gain,
        )
        elapsed = time.time() - t0

        record: dict = {
            "epoch": epoch,
            "time_s": round(elapsed, 2),
            "train": {k: round(v, 6) for k, v in train_metrics.items()},
        }
        log = (
            f"epoch {epoch}/{args.epochs}  "
            f"train={train_metrics['loss']:.4f}  "
            f"jepa={train_metrics['jepa_loss']:.4f}  "
            f"latent={train_metrics['latent_loss']:.4f}  "
            f"pixel={train_metrics['pixel_loss']:.4f}  "
            f"ttc={train_metrics['ttc_loss']:.4f}  "
            f"time={elapsed:.1f}s"
        )

        if epoch % args.val_every == 0:
            val_metrics = evaluate(
                encoder, world_model, ttc_predictor, val_loader, device,
                pixel_mask_gain=args.pixel_mask_gain,
            )
            record["val"] = {k: round(v, 6) for k, v in val_metrics.items()}
            log += (
                f"  val_pixel={val_metrics['pixel_loss']:.4f}  "
                f"val_ttc={val_metrics['ttc_loss']:.4f}  "
                f"val_pixel_ttc={val_metrics['pixel_ttc_loss']:.4f}  "
                f"val_jepa={val_metrics['jepa_loss']:.4f}  "
                f"val_latent={val_metrics['latent_loss']:.4f}"
            )
            val_pixel_ttc = val_metrics["pixel_ttc_loss"]
            record["val_pixel_ttc_sum"] = round(val_pixel_ttc, 6)
            if val_pixel_ttc < best_val_pixel_ttc:
                best_val_pixel_ttc = val_pixel_ttc
                record["best_val_pixel_ttc"] = True
                save_training_checkpoint(
                    args.checkpoint_dir / "best.pt",
                    encoder,
                    world_model,
                    ttc_predictor,
                    optimizer,
                    epoch,
                    args,
                )

        history.append(record)
        print(log)
        save_training_checkpoint(
            args.checkpoint_dir / "last.pt",
            encoder,
            world_model,
            ttc_predictor,
            optimizer,
            epoch,
            args,
        )

        if epoch % args.log_every == 0 or epoch == args.epochs:
            save_training_log(log_path, args, history)
            print(f"saved log -> {log_path}")


if __name__ == "__main__":
    main()
