import torch
import torch.nn as nn

from src.models.encoder_decoder import Encoder, Decoder
from src.models.latent_world_model import LatentWorldModel
from src.models.ttc_predictor import TTCPredictor

encoder = Encoder()
world_model = LatentWorldModel(latent_dim = 15360, out_h = 48, out_w = 320, mamba_dim=1024)
ttc_predictor = TTCPredictor()

models = {"Encoder": encoder, "Latent_Predictor + Decoder": world_model, "TTC_Predictor": ttc_predictor}
total = 0
for model in models.keys():
    m = models[model]
    param_count = sum(p.numel() for p in m.parameters()) #numel is number of elements, so loop over param tensors and sum the num of elements 
    print(f"{model}: {param_count}, {(param_count * 4)/(1024**2)}")

    total+=param_count

total_bytes = total * 4
mb = total_bytes / (1024 ** 2)

print(f"Total MB: {mb}")



