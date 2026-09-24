"""JEPA components adapted from https://github.com/facebookresearch/jepa"""

from jepa.loss import jepa_loss, jepa_prediction_loss, predictor_variance_loss, update_target_encoder
from jepa.masks import SpatialMaskGenerator, apply_masks
from jepa.predictor import SpatialPredictor

__all__ = [
    "SpatialMaskGenerator",
    "SpatialPredictor",
    "apply_masks",
    "jepa_loss",
    "jepa_prediction_loss",
    "predictor_variance_loss",
    "update_target_encoder",
]
