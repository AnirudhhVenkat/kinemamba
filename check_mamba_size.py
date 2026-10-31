import torch
import torch.nn as nn
import argparse
from src.models.mamba2_latent_dynamics import Mamba2LatentDynamics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count Mamba2LatentDynamics params")
    parser.add_argument("--out-h", type=int, default=48)
    parser.add_argument("--out-w", type=int, default=320)
    parser.add_argument("--channels", type=int, nargs="+", default=[128, 192, 256, 256])
    parser.add_argument("--mamba-dim", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    latent_predictor= Mamba2LatentDynamics(
        latent_dim = (args.out_h // 2**len(args.channels)) * (args.out_w // 2**len(args.channels)) * args.channels[-1],
        mamba_dim=args.mamba_dim,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        headdim=args.headdim,
        )

    param_count = sum(p.numel() for p in latent_predictor.parameters()) #numel is number of elements, so loop over param tensors and sum the num of elements 
    print(f"{latent_predictor}: {param_count}, {(param_count * 4)/(1024**2)}")


    total_bytes = param_count * 4
    mb = total_bytes / (1024 ** 2)

    print(f"Total MB: {mb}")


if __name__ == "__main__":
    main()
