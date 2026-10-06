"""Timestep extent surrogate, using raw physical depth on target cells only."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LossTerms:
    total: torch.Tensor
    residual_mse: torch.Tensor
    extent: torch.Tensor
    weighted_extent: torch.Tensor
    raw_depth: torch.Tensor
    soft_wet: torch.Tensor


def composite_loss(
    prediction,
    target,
    baseline_wse,
    terrain,
    target_depth,
    residual_mean,
    residual_scale,
    temperature_m,
    extent_weight,
    threshold_m=0.03,
) -> LossTerms:
    """No clipping of raw depth; all-dry timesteps use mean soft wetness.

    Inputs are one-dimensional arrays of AOI target cells, excluding halo nodes.
    T is in metres; the MSE is in standardized residual units. Lambda=0 returns
    exactly the original MSE tensor for backwards-compatible optimisation.
    """
    if temperature_m <= 0 or extent_weight < 0:
        raise ValueError("Temperature must be positive and extent weight nonnegative")
    if prediction.numel() == 0:
        raise ValueError("Loss requires at least one target cell")
    mse = F.mse_loss(prediction, target)
    residual = prediction * residual_scale + residual_mean
    raw_depth = baseline_wse + residual - terrain
    probability = torch.sigmoid((raw_depth - threshold_m) / temperature_m)
    wet = (target_depth >= threshold_m).to(prediction.dtype)
    intersection = (probability * wet).sum()
    union = (probability + wet - probability * wet).sum()
    overlap_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
    extent = torch.where(wet.sum() > 0, overlap_loss, probability.mean())
    weighted = extent_weight * extent
    total = mse if extent_weight == 0 else mse + weighted
    return LossTerms(total, mse, extent, weighted, raw_depth, probability)
