# KineMamba pipeline

Encoder → Mamba2 dynamics → Decoder (+ TTC from Mamba hidden `h`). No UNet skips, no JEPA.

Teacher-forced forward with defaults: `48×320`, 4-level Enc/Dec (`depths=[2,2,2,2]`, `channels=[128,192,256,256]`), bottleneck `3×20×256` → `D=15360`, `mamba_dim=1024`.

```text
frames (B, T, 3, 48, 320)
            │
     Encoder (4× down)
            ▼
   z (B, T, D=15360)  ← flatten (B,T,256,3,20)      actions (B, T)
            │                                          │
       z[:, :-1]                                  embed → (B, T-1, 32)
            \                                         /
             concat (B, T-1, D+32)
                      │
               LayerNorm + proj_in
                      ▼
             h (B, T-1, 1024)  ← 4× Mamba2
               │            │
               │         proj_out
               │            ▼
               │     ẑ (B, T-1, D) ──┬── latent loss vs z[:, 1:]
               │                     │
               │                  Decoder (4× up)
               │                     ▼
               │         pixelŝ (B, T-1, 3, 48, 320) ── pixel loss
               │
          TTCPredictor(h)
                      ▼
               ttĉ (B, T-1) ── TTC loss vs ttc[:, 1:]
```

## Shapes

| Tensor             | Shape                  |
| ------------------ | ---------------------- |
| frames             | `(B, T, 3, 48, 320)`   |
| actions            | `(B, T)`               |
| latents `z`        | `(B, T, 15360)`        |
| encoder feature    | `(B, T, 256, 3, 20)`   |
| action embed       | `(B, T-1, 32)`         |
| concat             | `(B, T-1, 15392)`      |
| Mamba features `h` | `(B, T-1, 1024)`       |
| pred latents `ẑ`   | `(B, T-1, 15360)`      |
| pred pixels        | `(B, T-1, 3, 48, 320)` |
| pred TTC           | `(B, T-1)`             |

`D = (H / 2^L) * (W / 2^L) * C_L` with `L = len(channels)` downs (default `L=4` → `3×20×256`).

## Loss

Teacher-forced next-step targets (`z/pixels/ttc` at `t+1` from context at `t`):

```text
total = a * latent + b * pixel + c * ttc
```

- Pointwise error: **MAE** for latent / pixel / TTC
- Optional `--pixel-mask-gain`: weight pixels by `1 + gain * vehicle_mask` (0 = uniform)
- Report-only `pixel_mask_mae`: MAE on vehicle-masked pixels
- TTC head reads **`h`** (pre-`proj_out`), not latent `z`

Grid runs (`gridsearch_params.txt`) use `a=1`, `b=1`, vary mask gain → dirs like `a1_b1_c1_m50`.

## Modules

| Piece | Path | Role |
| ----- | ---- | ---- |
| Encoder / Decoder | `src/models/encoder_decoder.py` | ResNet pyramid; GroupNorm; no AdaGN/skips |
| Mamba2 dynamics | `src/models/mamba2_latent_dynamics.py` | `forward` → `(h, ẑ)`; stateful AR `rollout` |
| World model | `src/models/latent_world_model.py` | Dynamics + Decoder; reshape flat `ẑ` → `(C,H',W')` for decode |
| TTC | `src/models/ttc_predictor.py` | MLP on `mamba_dim` |

## Train / eval

- Train: `train_kinemamba.py` (or `sbatch gridsearch.sh`)
- Rollout eval: `eval_rollout.py` / `eval.sh` — context GT frames, then AR horizon; metrics in `<run>/rollout_eval.json`, plots under `<run>/rollout_plots/`
- Checkpoints must use the new Encoder (`startconv.*` keys). Old JEPA/CNNEncoder weights (`encoder.convs.*`, `predictor.*`) will not load.
