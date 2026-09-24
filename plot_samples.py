#!/usr/bin/env python3
"""Plot Mamba2 val-set predictions: context frames vs predicted next frame."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "mamba2"))

from models.latent_world_model import LatentWorldModel
from models.cnn_encoder import normalize_image
from utils import (
    EpisodeDataset,
    build_encoder,
    encode_episode_frames,
    load_encoder_checkpoint,
    load_training_checkpoint,
)


def chw_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    """(C, H, W) uint8 or normalized float -> (H, W, C) uint8 for imshow."""
    if image.dtype == torch.uint8:
        x = image.float()
    else:
        x = (image.float() + 0.5).clamp(0.0, 1.0) * 255.0
    return x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict_next_frame(
    model: LatentWorldModel,
    encoder,
    frames: torch.Tensor,
    actions: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    """Predict frame at target_idx from prior frames/actions (normalized pixels)."""
    encoder.eval()
    model.eval()
    lengths = torch.tensor([frames.shape[0]], device=frames.device, dtype=torch.long)
    encodings = encode_episode_frames(encoder, frames.unsqueeze(0), lengths)
    context_latents = encodings[:, :target_idx]
    context_actions = actions[:target_idx].unsqueeze(0)
    _, pred_pixels = model.predict_next(context_latents, context_actions)
    return pred_pixels[0]


def load_models(
    checkpoint_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
    encoder_checkpoint: Path | None = None,
) -> tuple[torch.nn.Module, LatentWorldModel]:
    encoder = build_encoder(image_size, device)
    latent_dim = encoder.embed_dim * encoder.grid_h * encoder.grid_w
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    world_model = LatentWorldModel(
        latent_dim=latent_dim,
        out_h=image_size[0],
        out_w=image_size[1],
        num_actions=int(train_args.get("num_actions", 5)),
        action_embed_dim=int(train_args.get("action_embed_dim", 32)),
        mamba_dim=int(train_args.get("mamba_dim", 512)),
        n_layers=int(train_args.get("n_layers", 4)),
        d_state=int(train_args.get("d_state", 128)),
        expand=int(train_args.get("expand", 2)),
        headdim=int(train_args.get("headdim", 64)),
        chunk_size=int(train_args.get("chunk_size", 64)),
    ).to(device)

    if "encoder" in ckpt:
        load_training_checkpoint(checkpoint_path, encoder, world_model, device)
    else:
        world_model.load_state_dict(ckpt["model"])
        load_encoder_checkpoint(encoder, encoder_checkpoint or checkpoint_path, device)

    encoder.eval()
    world_model.eval()
    return encoder, world_model


def plot_sample(
    context_frames: torch.Tensor,
    pred_frame: torch.Tensor,
    target_frame: torch.Tensor,
    title: str,
    out_path: Path,
) -> None:
    n_context = context_frames.shape[0]
    n_cols = n_context + 2
    fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.5))
    if n_cols == 1:
        axes = [axes]

    for i in range(n_context):
        axes[i].imshow(chw_to_uint8_hwc(context_frames[i]))
        axes[i].set_title(f"t-{n_context - i}")
        axes[i].axis("off")

    axes[n_context].imshow(chw_to_uint8_hwc(pred_frame))
    axes[n_context].set_title("predicted")
    axes[n_context].axis("off")

    axes[n_context + 1].imshow(chw_to_uint8_hwc(target_frame))
    axes[n_context + 1].set_title("ground truth")
    axes[n_context + 1].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Mamba2 val predictions")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset_episodes")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints_mamba2" / "best.pt")
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=None,
        help="Separate encoder weights if checkpoint has world model only",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints_mamba2" / "samples")
    parser.add_argument("--split", default="test", help="Dataset split (val set uses 'test')")
    parser.add_argument("--out-h", type=int, default=32)
    parser.add_argument("--out-w", type=int, default=128)
    parser.add_argument("--interp-mode", default="bicubic")
    parser.add_argument("--context-frames", type=int, default=8, help="Past frames shown and fed as context")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    image_size = (args.out_h, args.out_w)

    dataset = EpisodeDataset(
        args.data_dir,
        args.split,
        out_h=args.out_h,
        out_w=args.out_w,
        interp_mode=args.interp_mode,
        min_frames=args.context_frames + 1,
    )

    encoder, model = load_models(
        args.checkpoint,
        image_size,
        device,
        encoder_checkpoint=args.encoder_checkpoint,
    )

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]

    print(f"loaded {args.checkpoint.name}  split={args.split}  context={args.context_frames}")
    print(f"saving {len(indices)} samples -> {args.out_dir}")

    for plot_idx, episode_idx in enumerate(indices):
        frames, actions, _ttc = dataset[episode_idx]
        frames = frames.to(device)
        actions = actions.to(device)
        target_idx = args.context_frames
        if frames.shape[0] <= target_idx:
            target_idx = frames.shape[0] - 1
        if target_idx < 1:
            continue

        context = frames[:target_idx]
        pred_norm = predict_next_frame(model, encoder, frames, actions, target_idx)
        target_norm = normalize_image(frames[target_idx])

        episode_name = dataset.paths[episode_idx].stem
        out_path = args.out_dir / f"sample_{plot_idx:02d}_{episode_name}_t{target_idx}.png"
        plot_sample(
            context_frames=context,
            pred_frame=pred_norm,
            target_frame=target_norm,
            title=f"{episode_name}  predict frame {target_idx} from {target_idx} context frames",
            out_path=out_path,
        )
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
