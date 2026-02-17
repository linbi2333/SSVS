#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loss.py — Loss functions and utilities for network training.

Includes:
  - get_cls_weights: compute balanced weights for classification loss based on sample counts.
  - DiceLoss: binary foreground/background Dice loss for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cls_weights(cls_counts: dict[int, int]) -> dict[int, float]:
    """
    Compute balanced class weights for cross-entropy loss.

    If all classes have equal counts, each weight = 1.0.
    Otherwise, linearly map sample counts in [min, max] to weights in [1.0, 2.0],
    then normalize so that sum(weights) = number of classes.

    Parameters
    ----------
    cls_counts : dict[int, int]
        Mapping from class ID to sample count.

    Returns
    -------
    cls_weights : dict[int, float]
        Normalized weights for each class.
    """
    cnt_vals = list(cls_counts.values())
    min_cnt, max_cnt = min(cnt_vals), max(cnt_vals)

    def _raw_weight(cnt: int) -> float:
        # Return 1.0 if all counts are equal
        if max_cnt == min_cnt:
            return 1.0
        # Linear interpolation: cnt=min -> 1.0, cnt=max -> 2.0
        return 1.0 + (cnt - min_cnt) / (max_cnt - min_cnt)

    raw_weights = {cls_id: _raw_weight(cnt) for cls_id, cnt in cls_counts.items()}

    # Normalize so that sum of weights equals number of classes
    num_classes = len(raw_weights)
    total = sum(raw_weights.values())
    cls_weights = {cls_id: w / total * num_classes for cls_id, w in raw_weights.items()}
    return cls_weights


class DiceLoss(nn.Module):
    """
    Binary Dice loss for segmentation.

    Computes Dice loss = 1 - Dice coefficient between predicted foreground and target mask.

    Parameters
    ----------
    smooth : float, optional
        Smoothing factor to avoid division by zero (default=1e-6).
    """
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits with shape [B, 2, H, W, D].
            Channel 0 = background, channel 1 = foreground.
        target : torch.Tensor
            Ground truth mask with shape [B, H, W, D], values 0 or >0.

        Returns
        -------
        torch.Tensor
            Scalar Dice loss (1 - mean Dice coefficient).
        """
        # Softmax over channel dimension, select foreground probability
        prob_fg = F.softmax(logits, dim=1)[:, 1]

        # Binary ground truth for foreground
        gt_fg = (target > 0).float()

        # Intersection and union for Dice
        intersection = (prob_fg * gt_fg).sum(dim=[1, 2, 3])
        cardinality = prob_fg.sum(dim=[1, 2, 3]) + gt_fg.sum(dim=[1, 2, 3])

        # Compute Dice coefficient
        dice_coef = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Dice loss = 1 - average Dice coefficient
        loss = 1.0 - dice_coef.mean()
        return loss



class RunningEMA:
    """简单的指数移动平均，用来平滑记录 loss 曲线。"""
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * x
        return self.value
