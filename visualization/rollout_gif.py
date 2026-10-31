#!/usr/bin/env python3
"""Turn the rollout mosaics written by `eval_rollout.py` into animated GIFs.

Reads every `<run>/rollout_plots/*_pred_vs_gt_all_frames.png` under
`--ckpt-root`, slices it back into per-step frames (layout from
`save_comparison_mosaic`: pred row, 3px gap, GT row; 1px gap between steps),
and writes:

    <run>/<out-subdir>/<episode>_pred_vs_gt.gif    one run: pred over GT
    <ckpt-root>/<episode>_all_runs.gif             GT on top, every run below
                                                   (only with --all-runs)

Example:
    python visualization/rollout_gif.py --fps 5 --scale 2
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = "_pred_vs_gt_all_frames.png"
GAP = 1  # px between steps in the mosaic
ROW_GAP = 3  # px between pred and GT rows in the mosaic
LABEL_W = 110  # px reserved on the left for row labels (before scaling)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate rollout mosaics as GIFs")
    parser.add_argument("--ckpt-root", type=Path, default=ROOT / "checkpoints_mamba2")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--scale", type=int, default=2, help="Integer upscale of each frame")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run folder names to include (default: every run with a mosaic)",
    )
    parser.add_argument(
        "--out-subdir",
        default="rollout_gifs",
        help="Folder inside each run directory to write the per-run GIFs to",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Also write one cross-run comparison GIF per episode under --ckpt-root",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def split_mosaic(path: Path, horizon: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Comparison mosaic -> (pred frames, GT frames), each a list of (H, W, 3) uint8."""
    canvas = np.asarray(Image.open(path).convert("RGB"))
    total_h, total_w, _ = canvas.shape
    h = (total_h - ROW_GAP) // 2
    w = (total_w + GAP) // horizon - GAP
    if horizon * (w + GAP) - GAP != total_w:
        raise ValueError(f"{path}: width {total_w} doesn't fit {horizon} steps")
    pred, gt = [], []
    for i in range(horizon):
        x = i * (w + GAP)
        pred.append(canvas[:h, x : x + w])
        gt.append(canvas[h + ROW_GAP : 2 * h + ROW_GAP, x : x + w])
    return pred, gt


def compose(rows: list[tuple[str, np.ndarray]], header: str, scale: int) -> Image.Image:
    """Stack labelled rows vertically under a one-line header."""
    h, w, _ = rows[0][1].shape
    header_h, row_gap = 16, 4
    height = header_h + len(rows) * (h + row_gap)
    img = Image.new("RGB", (LABEL_W + w, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), header, fill="black")
    for r, (label, frame) in enumerate(rows):
        y = header_h + r * (h + row_gap)
        img.paste(Image.fromarray(frame), (LABEL_W, y))
        draw.text((4, y + h // 2 - 6), label, fill="black")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def save_gif(frames: list[Image.Image], out_path: Path, fps: float) -> None:
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def make_run_gif(
    path: Path, horizon: int, context: int, out_subdir: str, fps: float, scale: int
) -> Path:
    """Write the pred-over-GT GIF for one mosaic; returns the output path."""
    run_dir = path.parent.parent
    episode = path.name[: -len(SUFFIX)]
    pred, gt = split_mosaic(path, horizon)
    frames = [
        compose(
            [("prediction", pred[i]), ("ground truth", gt[i])],
            f"{run_dir.name} | {episode} | t+{i + 1}/{horizon} (after {context} ctx)",
            scale,
        )
        for i in range(horizon)
    ]
    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episode}_pred_vs_gt.gif"
    save_gif(frames, out_path, fps)
    return out_path


def main() -> None:
    args = parse_args()
    mosaics = sorted(args.ckpt_root.glob(f"*/rollout_plots/*{SUFFIX}"))
    if args.runs:
        mosaics = [p for p in mosaics if p.parent.parent.name in args.runs]
    if not mosaics:
        raise SystemExit(f"No *{SUFFIX} found under {args.ckpt_root}")

    # run dir -> (horizon, context steps)
    meta = {}
    for run_dir in {p.parent.parent for p in mosaics}:
        info = json.loads((run_dir / "rollout_eval.json").read_text())
        meta[run_dir] = (int(info["horizon"]), int(info["context_steps"]))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                make_run_gif, path, *meta[path.parent.parent], args.out_subdir, args.fps, args.scale
            )
            for path in mosaics
        ]
        for fut in futures:
            print(fut.result())

    if not args.all_runs:
        return

    # One side-by-side GIF per episode across runs (GT taken from the first run;
    # every run shares the same episode/seed so the GT rows are identical).
    by_episode: dict[str, list[tuple[str, list, list]]] = {}
    for path in mosaics:
        pred, gt = split_mosaic(path, meta[path.parent.parent][0])
        episode = path.name[: -len(SUFFIX)]
        by_episode.setdefault(episode, []).append((path.parent.parent.name, pred, gt))
    for episode, runs in by_episode.items():
        if len(runs) < 2:
            continue
        horizon = min(len(r[1]) for r in runs)
        gt = runs[0][2]
        frames = [
            compose(
                [("ground truth", gt[i])] + [(name, pred[i]) for name, pred, _ in runs],
                f"{episode} | t+{i + 1}/{horizon}",
                args.scale,
            )
            for i in range(horizon)
        ]
        out_path = args.ckpt_root / f"{episode}_all_runs.gif"
        save_gif(frames, out_path, args.fps)
        print(f"all runs: {out_path}")


if __name__ == "__main__":
    main()
