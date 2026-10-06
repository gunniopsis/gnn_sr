"""Streaming timestep and epoch diagnostics; never used for selection."""

import math

import torch

from .model import physical_outputs

STEP_FIELDS = (
    "residual_mse",
    "extent_loss",
    "weighted_extent_loss",
    "total_loss",
    "depth_squared_error_sum",
    "cell_count",
    "depth_rmse_m",
    "hard_csi",
    "hits",
    "false_alarms",
    "misses",
    "both_empty",
    "soft_wet_mean",
    "soft_saturated_fraction",
)
SUMMARY_FIELDS = (
    "residual_mse",
    "extent_loss",
    "weighted_extent_loss",
    "total_loss",
    "pooled_depth_rmse_m",
    "mean_timestep_depth_rmse_m",
    "mean_timestep_csi",
    "csi_valid_timesteps",
    "csi_both_empty_timesteps",
    "hits",
    "false_alarms",
    "misses",
    "soft_wet_mean",
    "soft_saturated_fraction",
)


class TimestepDiagnostics:
    def __init__(self, threshold_m, writer=None, epoch=0, phase="validation"):
        self.threshold_m = threshold_m
        self.writer = writer
        self.epoch = epoch
        self.phase = phase
        self.rows = 0
        self.sums = dict.fromkeys(STEP_FIELDS, 0.0)

    @torch.no_grad()
    def add(self, terms, target_depth, ref):
        # Same hard threshold as evaluation; it is not in the loss gradient path.
        _, predicted = physical_outputs(
            terms.raw_depth, torch.zeros_like(target_depth), self.threshold_m
        )
        true_wet = target_depth >= self.threshold_m
        pred_wet = predicted >= self.threshold_m
        hits = (true_wet & pred_wet).sum()
        false = (~true_wet & pred_wet).sum()
        misses = (true_wet & ~pred_wet).sum()
        sse = (predicted - target_depth).square().sum()
        probability = terms.soft_wet
        values = (
            torch.stack(
                [
                    terms.residual_mse,
                    terms.extent,
                    terms.weighted_extent,
                    terms.total,
                    sse,
                    hits,
                    false,
                    misses,
                    probability.mean(),
                    ((probability < 0.01) | (probability > 0.99)).float().mean(),
                ]
            )
            .detach()
            .cpu()
            .tolist()
        )
        mse, extent, weighted, total, error, hit, fp, fn, soft, saturation = values
        union = hit + fp + fn
        row = dict(
            zip(
                STEP_FIELDS,
                [
                    mse,
                    extent,
                    weighted,
                    total,
                    error,
                    target_depth.numel(),
                    math.sqrt(error / target_depth.numel()),
                    hit / union if union else None,
                    int(hit),
                    int(fp),
                    int(fn),
                    int(union == 0),
                    soft,
                    saturation,
                ],
                strict=True,
            )
        )
        self.rows += 1
        for key, value in row.items():
            if value is not None:
                self.sums[key] += value
        if self.writer is not None:
            self.writer.writerow(
                {
                    "epoch": self.epoch,
                    "phase": self.phase,
                    "event_index": ref.event_index,
                    "timestep": ref.timestep,
                    **row,
                }
            )

    def summary(self):
        if not self.rows:
            raise ValueError("No diagnostic timesteps")
        s = self.sums
        valid = self.rows - int(s["both_empty"])
        result = {
            key: s[key] / self.rows
            for key in (
                "residual_mse",
                "extent_loss",
                "weighted_extent_loss",
                "total_loss",
                "soft_wet_mean",
                "soft_saturated_fraction",
            )
        }
        result.update(
            pooled_depth_rmse_m=math.sqrt(s["depth_squared_error_sum"] / s["cell_count"]),
            mean_timestep_depth_rmse_m=s["depth_rmse_m"] / self.rows,
            mean_timestep_csi=s["hard_csi"] / valid if valid else None,
            csi_valid_timesteps=valid,
            csi_both_empty_timesteps=int(s["both_empty"]),
            hits=int(s["hits"]),
            false_alarms=int(s["false_alarms"]),
            misses=int(s["misses"]),
        )
        return result
