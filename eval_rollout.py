#!/usr/bin/env python3
"""Demo: load a checkpoint and roll out test episode(s) autoregressively.

Feeds `--context-steps` GT frames, then predicts `--horizon` future frames.
All episodes share the same fixed horizon. Defaults to one episode; raise
`--num-samples` to evaluate more.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")  # headless: plots are written to disk, never displayed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "mamba2"))

from models.cnn_encoder import normalize_image
from models.latent_world_model import LatentWorldModel
from models.ttc_predictor import TTCPredictor
from utils import (
    EpisodeDataset,
    build_encoder,
    encode_episode_frames,
    load_training_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-episode rollout demo on the test split")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints_mamba2" / "a1_b0p1_c0p1" / "best.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset_episodes")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-h", type=int, default=32)
    parser.add_argument("--out-w", type=int, default=128)
    parser.add_argument("--interp-mode", default="bicubic")
    parser.add_argument("--context-steps", type=int, default=20, help="GT frames fed as input")
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Number of future steps to predict (same for every episode)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of episodes to eval; 0 or negative means all eligible episodes",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Keep small for long rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path for metrics JSON (default: next to checkpoint)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Where to write rollout plots (default: <checkpoint-dir>/rollout_plots)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting the decoded frame sequence",
    )
    parser.add_argument(
        "--mosaic-cols",
        type=int,
        default=20,
        help="Columns in the all-frames mosaic",
    )
    return parser.parse_args()


def load_models(checkpoint_path: Path, image_size: tuple[int, int], device: torch.device):
    encoder = build_encoder(image_size, device) #initialize from build_encoder function in train_kinemamba, which just returns a CNNEncoder object
    latent_dim = encoder.embed_dim * encoder.grid_h * encoder.grid_w #output dimensions of the encoder
    world_model = LatentWorldModel(
        latent_dim=latent_dim,
        out_h=image_size[0],
        out_w=image_size[1],
    ).to(device) #initialize from LatentWorldModel class in models.py
    ttc_predictor = TTCPredictor(latent_dim=latent_dim).to(device) #initialize from TTCPredictor class in models.py

    ckpt = load_training_checkpoint(checkpoint_path, encoder, world_model, device, ttc_predictor) #load the checkpoint from the checkpoint_path
    train_args = ckpt.get("args") or {}
    encoder.eval() #turn off dropout and batch normalization for evaluation
    world_model.eval() #^^^
    ttc_predictor.eval() #^^^
    return encoder, world_model, ttc_predictor, train_args


def sample_episodes(
    dataset: EpisodeDataset,
    num_samples: int | None, #how many episodes to sample from the validation set
    context_steps: int,
    horizon: int,
    seed: int,
) -> list[int]:
    """Return episode indices long enough for context_steps + horizon."""
    need = context_steps + horizon
    valid = [i for i, length in enumerate(dataset.episode_lengths) if length >= need]
    if not valid:
        raise ValueError(f"No episodes with >= {need} frames")

    rng = random.Random(seed)
    if num_samples is None or num_samples <= 0 or num_samples >= len(valid):
        rng.shuffle(valid)
        return valid
    return rng.sample(valid, num_samples)


def load_window_batch(
    dataset: EpisodeDataset,
    episode_indices: list[int],
    context_steps: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load context+horizon windows for each episode -> (B, context+horizon, ...)."""
    need = context_steps + horizon
    frame_list, action_list, ttc_list = [], [], []
    for ep_idx in episode_indices:
        frames, actions, ttc = dataset[ep_idx]
        frame_list.append(frames[:need])
        action_list.append(actions[:need])
        ttc_list.append(ttc[:need])
    return (
        torch.stack(frame_list, dim=0),
        torch.stack(action_list, dim=0),
        torch.stack(ttc_list, dim=0),
    )


@torch.no_grad() #turn off gradient computation 
def rollout_batch(
    encoder,
    world_model: LatentWorldModel,
    ttc_predictor: TTCPredictor,
    frames: torch.Tensor,
    actions: torch.Tensor,
    ttc: torch.Tensor,
    context_steps: int,
    horizon: int,
    step_callback=None,
    return_traces: bool = False, #returns frames for plotting
) -> dict[str, torch.Tensor]:
    """Prefill Mamba on context, then step SSM state for each future frame.

    The full rollout runs first; ground-truth targets are encoded afterwards so
    scoring work never lands inside the timed rollout window.
    """
    device = frames.device
    b = frames.shape[0] #batch size
    lengths = torch.full((b,), context_steps, device=device, dtype=torch.long) #generates a list of length b, with context_steps at each entry, 64 bit ints 

    context_frames = frames[:, :context_steps]
    context_latents = encode_episode_frames(encoder, context_frames, lengths) #get context latents from the encoder, function is defined in train_kinemaba.py

    # actions[:, : context_steps + horizon] covers prefill + stepping inputs.
    # TTC is emitted per step during the rollout; pixels decode once at the end.
    pred_latents_t, pred_pixels, pred_ttc_t = world_model.rollout(
        context_latents,
        actions[:, : context_steps + horizon],
        horizon,
        step_head=ttc_predictor.predict_next,
        step_callback=step_callback,
    )

    # Targets only exist for scoring, so build them after the rollout finishes.
    target_pixels = normalize_image(frames[:, context_steps : context_steps + horizon])
    target_ttc = ttc[:, context_steps : context_steps + horizon]
    target_latents = encode_episode_frames(
        encoder,
        frames[:, context_steps : context_steps + horizon],
        torch.full((b,), horizon, device=device, dtype=torch.long),
    )

    pixel_sq = (pred_pixels - target_pixels).pow(2)
    pixel_mse_per_step = pixel_sq.mean(dim=(0, 2, 3, 4))
    pixel_mse = pixel_sq.mean()

    ttc_sq = (pred_ttc_t - target_ttc).pow(2)
    ttc_mse_per_step = ttc_sq.mean(dim=0)
    ttc_mse = ttc_sq.mean()

    latent_sq = (pred_latents_t - target_latents).pow(2)
    latent_mse_per_step = latent_sq.mean(dim=(0, 2))
    latent_mse = latent_sq.mean()

    out = {
        "pixel_mse": pixel_mse,
        "ttc_mse": ttc_mse,
        "latent_mse": latent_mse,
        "pixel_mse_per_step": pixel_mse_per_step,
        "ttc_mse_per_step": ttc_mse_per_step,
        "latent_mse_per_step": latent_mse_per_step,
        "horizon": torch.tensor(float(horizon), device=device),
        "n": torch.tensor(float(b), device=device),
    }
    if return_traces:
        # Decoded frames from the first batch element, for plotting.
        out["trace_pred_pixels"] = pred_pixels[0].cpu()
        out["trace_target_pixels"] = target_pixels[0].cpu()
    return out


class StepTimer:
    """Times each AR rollout step and drives the progress bar.

    The first interval of each rollout is the warmup: it covers the context
    encode plus a prefill over ``context_steps`` tokens plus the first predicted
    step, so it accounts for ``context_steps + 1`` timesteps rather than one.
    It is weighted accordingly in the average instead of being charged as a
    single step.

    On CUDA the step kernels are async, so we synchronize before reading the
    clock; otherwise every step would look near-instant and the tail would
    absorb all the time.
    """

    def __init__(self, progress, device: torch.device, context_steps: int):
        self.progress = progress
        self.sync = device.type == "cuda"
        self.warmup_steps = context_steps + 1
        self.total_s = 0.0
        self.total_steps = 0
        self.warmup_s = 0.0
        self.warmup_count = 0
        self._last: float | None = None
        self._first_of_rollout = False

    def start(self) -> None:
        """Begin a rollout; the next interval is the warmup (encode + prefill)."""
        if self.sync:
            torch.cuda.synchronize()
        self._last = time.perf_counter()
        self._first_of_rollout = True

    def __call__(self) -> None:
        if self.sync:
            torch.cuda.synchronize()
        now = time.perf_counter()
        steps = self.warmup_steps if self._first_of_rollout else 1
        if self._last is not None:
            elapsed = now - self._last
            self.total_s += elapsed
            self.total_steps += steps
            if self._first_of_rollout:
                self.warmup_s += elapsed
                self.warmup_count += 1
        self._first_of_rollout = False
        self._last = now
        self.progress.update(steps)
        if self.total_steps:
            self.progress.set_postfix_str(
                f"{1000.0 * self.total_s / self.total_steps:.2f} ms/step"
            )

    def summary(self) -> dict[str, float]:
        if not self.total_steps:
            return {}
        out = {
            "total_s": self.total_s,
            "steps_timed": self.total_steps,
            "mean_ms": 1000.0 * self.total_s / self.total_steps,
            "steps_per_s": self.total_steps / self.total_s if self.total_s > 0 else float("inf"),
            "includes_warmup": True,
            "warmup_steps_each": self.warmup_steps,
        }
        if self.warmup_count:
            warmup_steps = self.warmup_steps * self.warmup_count
            out["warmup_count"] = self.warmup_count
            out["warmup_mean_ms"] = 1000.0 * self.warmup_s / self.warmup_count
            out["warmup_mean_ms_per_step"] = 1000.0 * self.warmup_s / warmup_steps
            steady_steps = self.total_steps - warmup_steps
            if steady_steps > 0:
                steady_s = self.total_s - self.warmup_s
                out["mean_ms_excl_warmup"] = 1000.0 * steady_s / steady_steps
        return out


def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    """(C, H, W) uint8 or normalized float -> (H, W, C) uint8 for imshow."""
    #this is for going from model outputs to images for plotting
    if image.dtype == torch.uint8:
        x = image.float()
    else:
        x = (image.float() + 0.5).clamp(0.0, 1.0) * 255.0 #undo earlier operation of (x/255 - 0.5)
    return x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def save_frame_mosaic(frames: torch.Tensor, out_path: Path, cols: int) -> None:
    """Tile every frame of (T, C, H, W) into one image, row-major, with 1px gaps."""
    t = frames.shape[0]
    cols = max(1, min(cols, t))
    rows = (t + cols - 1) // cols
    tiles = [_to_uint8_hwc(frames[i]) for i in range(t)]
    h, w, c = tiles[0].shape
    gap = 1
    canvas = np.zeros((rows * (h + gap) - gap, cols * (w + gap) - gap, c), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, col = divmod(idx, cols)
        y, x = r * (h + gap), col * (w + gap)
        canvas[y : y + h, x : x + w] = tile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, canvas)


def save_comparison_mosaic(
    pred_pixels: torch.Tensor,
    target_pixels: torch.Tensor,
    out_path: Path,
    cols: int | None = None,
) -> None:
    """Two rows: predicted frames on top, ground-truth frames below (time left→right).

    ``cols`` is ignored; every step is placed in a single horizontal strip so the
    two rows stay aligned for direct comparison.
    """
    del cols  # kept for call-site compatibility
    t = pred_pixels.shape[0]
    pred_tiles = [_to_uint8_hwc(pred_pixels[i]) for i in range(t)]
    gt_tiles = [_to_uint8_hwc(target_pixels[i]) for i in range(t)]
    h, w, c = pred_tiles[0].shape
    gap = 1
    row_gap = 3
    width = t * (w + gap) - gap
    canvas = np.zeros((2 * h + row_gap, width, c), dtype=np.uint8)
    for i in range(t):
        x = i * (w + gap)
        canvas[0:h, x : x + w] = pred_tiles[i]
        canvas[h + row_gap : 2 * h + row_gap, x : x + w] = gt_tiles[i]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, canvas)


def _summarize_per_step(values: list[float], checkpoints: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(values)
    for step in checkpoints:
        if 1 <= step <= n:
            out[f"step_{step}"] = values[step - 1]
    if n:
        out["step_last"] = values[-1]
    return out


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError(f"--horizon must be a positive integer, got {args.horizon}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    image_size = (args.out_h, args.out_w)
    horizon = args.horizon
    min_frames = args.context_steps + horizon

    dataset = EpisodeDataset(
        args.data_dir,
        args.split,
        out_h=args.out_h,
        out_w=args.out_w,
        interp_mode=args.interp_mode,
        min_frames=min_frames,
    )
    episodes = sample_episodes(
        dataset, args.num_samples, args.context_steps, horizon, args.seed
    )

    encoder, world_model, ttc_predictor, train_args = load_models(
        args.checkpoint, image_size, device
    )

    print(
        f"checkpoint={args.checkpoint}  split={args.split}  "
        f"episodes={len(episodes)}  context={args.context_steps}  "
        f"horizon={horizon}"
    )
    print(f"device={device}  batch_size={args.batch_size}")

    sum_pixel = 0.0
    sum_ttc = 0.0
    sum_latent = 0.0
    sum_n = 0.0
    sum_pixel_step = torch.zeros(horizon)
    sum_ttc_step = torch.zeros(horizon)
    sum_latent_step = torch.zeros(horizon)

    num_batches = (len(episodes) + args.batch_size - 1) // args.batch_size
    total_steps = (args.context_steps + horizon) * num_batches
    progress = tqdm(total=total_steps, desc="rollout steps", unit="step")
    timer = StepTimer(progress, device, args.context_steps)
    want_traces = not args.no_plot
    traces: dict[str, torch.Tensor] | None = None

    for start in range(0, len(episodes), args.batch_size):
        batch_episodes = episodes[start : start + args.batch_size]
        frames, actions, ttc = load_window_batch(
            dataset, batch_episodes, args.context_steps, horizon
        )
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        ttc = ttc.to(device, non_blocking=True)

        timer.start()
        metrics = rollout_batch(
            encoder,
            world_model,
            ttc_predictor,
            frames,
            actions,
            ttc,
            args.context_steps,
            horizon,
            step_callback=timer,
            return_traces=want_traces and traces is None,
        )
        if want_traces and traces is None:
            # One episode for plots: first item of the first batch.
            traces = {
                k[len("trace_") :]: v
                for k, v in metrics.items()
                if k.startswith("trace_")
            }
            traces["episode"] = batch_episodes[0]

        n = metrics["n"].item()
        sum_n += n
        sum_pixel += metrics["pixel_mse"].item() * n
        sum_ttc += metrics["ttc_mse"].item() * n
        sum_latent += metrics["latent_mse"].item() * n
        sum_pixel_step += metrics["pixel_mse_per_step"].cpu() * n
        sum_ttc_step += metrics["ttc_mse_per_step"].cpu() * n
        sum_latent_step += metrics["latent_mse_per_step"].cpu() * n

    progress.close()
    timing = timer.summary()

    pixel_mse = sum_pixel / sum_n
    ttc_mse = sum_ttc / sum_n
    latent_mse = sum_latent / sum_n
    pixel_per_step = (sum_pixel_step / sum_n).tolist()
    ttc_per_step = (sum_ttc_step / sum_n).tolist()
    latent_per_step = (sum_latent_step / sum_n).tolist()

    step_checkpoints = [1, 5, 10, 20, 50, 100, 200, 500]
    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_samples": int(sum_n),
        "context_steps": args.context_steps,
        "horizon": horizon,
        "seed": args.seed,
        "decode_all_predicted_steps": True,
        "stateful_mamba_rollout": True,
        "step_timing": timing,
        "pixel_mse": pixel_mse,
        "ttc_mse": ttc_mse,
        "latent_mse": latent_mse,
        "pixel_ttc_mse": pixel_mse + ttc_mse,
        "pixel_mse_at": _summarize_per_step(pixel_per_step, step_checkpoints),
        "ttc_mse_at": _summarize_per_step(ttc_per_step, step_checkpoints),
        "latent_mse_at": _summarize_per_step(latent_per_step, step_checkpoints),
        "pixel_mse_per_step": pixel_per_step,
        "ttc_mse_per_step": ttc_per_step,
        "latent_mse_per_step": latent_per_step,
        "train_args": {
            k: train_args.get(k)
            for k in (
                "loss_a",
                "loss_b",
                "loss_c",
                "mamba_dim",
                "n_layers",
                "out_h",
                "out_w",
            )
            if isinstance(train_args, dict)
        },
    }

    print("\n=== rollout metrics (stateful Mamba AR, decode all predicted steps) ===")
    print(f"pixel_mse                     = {pixel_mse:.6f}")
    print(f"ttc_mse                       = {ttc_mse:.6f}")
    print(f"latent_mse                    = {latent_mse:.6f}")
    print(f"pixel+ttc                     = {pixel_mse + ttc_mse:.6f}")
    print("pixel_mse_at:", results["pixel_mse_at"])
    print("ttc_mse_at:  ", results["ttc_mse_at"])
    print("latent_mse_at:", results["latent_mse_at"])

    if timing:
        print("\n=== per-step rollout timing (warmup included) ===")
        print(f"total rollout time            = {timing['total_s']:.3f} s")
        print(f"steps timed                   = {int(timing['steps_timed'])}")
        print(f"avg                           = {timing['mean_ms']:.3f} ms/step")
        if "mean_ms_excl_warmup" in timing:
            print(f"avg (excl warmup)             = {timing['mean_ms_excl_warmup']:.3f} ms/step")
        if "warmup_mean_ms" in timing:
            print(
                f"warmup (encode+prefill+step1) = {timing['warmup_mean_ms']:.3f} ms"
                f"  x{int(timing['warmup_count'])}"
                f"  ({int(timing['warmup_steps_each'])} steps each,"
                f" {timing['warmup_mean_ms_per_step']:.3f} ms/step)"
            )
        print(f"throughput                    = {timing['steps_per_s']:.1f} steps/s")

    if traces is not None:
        plot_dir = args.plot_dir or (args.checkpoint.parent / "rollout_plots")
        episode_name = dataset.paths[traces["episode"]].stem
        pred_pixels = traces["pred_pixels"]
        target_pixels = traces["target_pixels"]

        horizon_len = pred_pixels.shape[0]
        print(f"\n=== plots ({episode_name}) ===")
        mosaic_path = plot_dir / f"{episode_name}_pred_all_frames.png"
        save_frame_mosaic(pred_pixels, mosaic_path, args.mosaic_cols)
        print(f"all {horizon_len} predicted frames -> {mosaic_path}")

        compare_path = plot_dir / f"{episode_name}_pred_vs_gt_all_frames.png"
        save_comparison_mosaic(pred_pixels, target_pixels, compare_path)
        print(f"pred row over GT row ({horizon_len} steps) -> {compare_path}")

    out_json = args.out_json
    if out_json is None:
        out_json = args.checkpoint.parent / "rollout_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()
