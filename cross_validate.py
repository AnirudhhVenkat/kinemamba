#!/usr/bin/env python3
"""Run k-fold CV by calling train_kinemamba.py once per fold.

Typical usage:
  python src/make_cv_folds.py --n-folds 5 --seed 0
  python cross_validate.py --epochs 5 --pixel-mask-gain 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-fold CV wrapper around train_kinemamba.py")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset_episodes")
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=None,
        help="Folds JSON (default: <data-dir>/cv_folds.json)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Fold indices to run (default: all folds in the JSON)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints_mamba2_cv",
        help="Parent dir; each fold writes under <run-name>/fold{k}/",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Prefix for fold run dirs (default: from loss args like train_kinemamba)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print train commands without running them",
    )
    # Remaining argv after known flags are forwarded to train_kinemamba.py
    args, unknown = parser.parse_known_args()
    args.train_args = unknown
    return args


def _format_param(value: float) -> str:
    text = f"{value:g}" if float(value).is_integer() else f"{value:.4g}"
    return text.replace(".", "p").replace("-", "m")


def default_run_prefix(train_args: list[str]) -> str:
    """Best-effort name from forwarded --loss-a/b/c and --pixel-mask-gain."""

    def get_float(flag: str, default: float) -> float:
        if flag in train_args:
            return float(train_args[train_args.index(flag) + 1])
        return default

    a = get_float("--loss-a", 1.0)
    b = get_float("--loss-b", 1.0)
    c = get_float("--loss-c", 1.0)
    m = get_float("--pixel-mask-gain", 0.0)
    name = f"a{_format_param(a)}_b{_format_param(b)}_c{_format_param(c)}"
    if m > 0.0:
        name += f"_m{_format_param(m)}"
    return name


def main() -> None:
    args = parse_args()
    fold_file = args.fold_file or (args.data_dir / "cv_folds.json")
    if not fold_file.is_file():
        raise FileNotFoundError(
            f"Missing {fold_file}. Run: python src/make_cv_folds.py --data-dir {args.data_dir}"
        )

    with fold_file.open() as f:
        payload = json.load(f)
    n_folds = int(payload["n_folds"])
    fold_indices = args.folds if args.folds is not None else list(range(n_folds))
    run_prefix = args.run_name or default_run_prefix(args.train_args)

    print(f"fold_file={fold_file}  folds={fold_indices}  checkpoint_dir={args.checkpoint_dir}")
    print(f"forwarded train args: {args.train_args}")

    for k in fold_indices:
        run_name = f"{run_prefix}/fold{k}"
        cmd = [
            sys.executable,
            str(ROOT / "train_kinemamba.py"),
            "--data-dir",
            str(args.data_dir),
            "--fold-file",
            str(fold_file),
            "--fold-index",
            str(k),
            "--checkpoint-dir",
            str(args.checkpoint_dir),
            "--run-name",
            run_name,
            *args.train_args,
        ]
        print(f"\n=== fold {k}/{n_folds - 1} -> {run_name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    print("\nCV complete." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
