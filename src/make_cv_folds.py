"""Build k-fold episode splits from dataset_episodes/{train,test}.

Writes a JSON file of relative *_visuals.npz paths under data_dir. No files are
copied or moved — train_kinemamba / cross_validate read these path lists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import KFold


def collect_episode_paths(data_dir: Path) -> list[str]:
    """All episode visuals as paths relative to data_dir (train/ + test/)."""
    rels: list[str] = []
    for split in ("train", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*_visuals.npz")):
            rels.append(str(path.relative_to(data_dir)))
    if not rels:
        raise FileNotFoundError(f"No *_visuals.npz under {data_dir}/train or .../test")
    return rels


def make_folds(
    episode_rels: list[str],
    n_folds: int,
    seed: int,
) -> list[dict[str, list[str]]]:
    """Shuffle episodes and partition into n_folds; each fold holds out one part as val."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if len(episode_rels) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} episodes for {n_folds}-fold CV, got {len(episode_rels)}"
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[dict[str, list[str]]] = []
    for train_idx, val_idx in kf.split(episode_rels):
        folds.append(
            {
                "train": [episode_rels[i] for i in train_idx],
                "val": [episode_rels[i] for i in val_idx],
            }
        )
    return folds


def load_fold_paths(
    fold_file: Path,
    data_dir: Path,
    fold_index: int,
) -> tuple[list[Path], list[Path]]:
    """Resolve absolute train/val visuals paths for one fold."""
    with fold_file.open() as f:
        payload = json.load(f)
    folds = payload["folds"]
    if fold_index < 0 or fold_index >= len(folds):
        raise IndexError(f"fold_index={fold_index} out of range for {len(folds)} folds")
    fold = folds[fold_index]
    train_paths = [data_dir / p for p in fold["train"]]
    val_paths = [data_dir / p for p in fold["val"]]
    return train_paths, val_paths


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Write k-fold CV episode path lists")
    parser.add_argument("--data-dir", type=Path, default=root / "dataset_episodes")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: <data-dir>/cv_folds.json)",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out if args.out is not None else (args.data_dir / "cv_folds.json")
    episode_rels = collect_episode_paths(args.data_dir)
    folds = make_folds(episode_rels, args.n_folds, args.seed)
    payload = {
        "seed": args.seed,
        "n_folds": args.n_folds,
        "data_dir": str(args.data_dir),
        "n_episodes": len(episode_rels),
        "folds": folds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"wrote {out}  episodes={len(episode_rels)}  n_folds={args.n_folds}  "
        f"val sizes={[len(fold['val']) for fold in folds]}"
    )


if __name__ == "__main__":
    main()
