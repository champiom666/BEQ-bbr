"""Long-tail Aware Asymmetric Learning (LTAL) for multi-label BBR.

LTAL combines three ingredients to handle the severe long-tailed multi-label
distribution of Bodily Behaviour Recognition under the macro-mAP objective:

1. An asymmetric weighted objective (:class:`AsymmetricLoss`) that suppresses
   easy negatives while preserving gradients for rare positives.
2. Class-balanced positive reweighting based on the effective number of samples
   (:func:`compute_effective_number_weights`), applied only to the positive term.
3. Rare-positive oversampling at the data-loader level (see the training
   scripts), which raises the sampling probability of clips containing rare
   positive labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for imbalanced multi-label classification.

    The negative branch is focused more strongly than the positive branch, so
    easy negatives contribute less to the training signal.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction
        self.register_buffer("pos_weight", torch.empty(0), persistent=False)

    def set_pos_weight(self, pos_weight: torch.Tensor | np.ndarray | list[float] | None) -> None:
        if pos_weight is None:
            self.pos_weight = torch.empty(0, dtype=torch.float32)
            return
        weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        if weight.ndim != 1:
            raise ValueError(f"pos_weight must be 1D, got shape {tuple(weight.shape)}")
        self.pos_weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().to(logits.device)
        logits = logits.float()
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1.0 - probs_pos

        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        pos_loss = targets * torch.log(probs_pos.clamp(min=self.eps))
        neg_targets = 1.0 - targets
        neg_loss = neg_targets * torch.log(probs_neg.clamp(min=self.eps))

        if self.pos_weight.numel() > 0:
            weight = self.pos_weight.to(logits.device).view(1, -1)
            pos_loss = pos_loss * weight

        loss = pos_loss + neg_loss
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = probs_pos * targets + probs_neg * neg_targets
            gamma = self.gamma_pos * targets + self.gamma_neg * neg_targets
            loss = loss * torch.pow((1.0 - pt).clamp(min=0.0), gamma)

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_multilabel_loss(loss_cfg: dict[str, Any] | None = None) -> nn.Module:
    loss_cfg = loss_cfg or {}
    name = str(loss_cfg.get("name", "bce")).lower()
    if name in {"bce", "bcewithlogits", "bce_with_logits"}:
        return nn.BCEWithLogitsLoss()
    if name in {"asl", "asymmetric", "asymmetric_loss", "ltal"}:
        return AsymmetricLoss(
            gamma_pos=float(loss_cfg.get("gamma_pos", 0.0)),
            gamma_neg=float(loss_cfg.get("gamma_neg", 4.0)),
            clip=float(loss_cfg.get("clip", 0.05)),
            eps=float(loss_cfg.get("eps", 1e-8)),
            reduction=str(loss_cfg.get("reduction", "mean")),
        )
    raise ValueError(f"Unknown multi-label loss: {name}")


def compute_effective_number_weights(
    counts: np.ndarray,
    beta: float = 0.9999,
    strength: float = 1.0,
    min_weight: float | None = None,
    max_weight: float | None = None,
) -> np.ndarray:
    """Class-balanced positive weights based on the effective number of samples."""

    counts = np.asarray(counts, dtype=np.float64)
    counts = np.clip(counts, 1.0, None)
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")

    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / weights.mean()
    if strength != 1.0:
        weights = np.power(weights, float(strength))
        weights = weights / weights.mean()
    if min_weight is not None:
        weights = np.maximum(weights, float(min_weight))
    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))
    return weights.astype(np.float32)


def compute_rare_positive_sampling_weights(
    label_matrix: np.ndarray,
    class_weights: np.ndarray,
    strength: float = 1.0,
    cap: float = 5.0,
) -> np.ndarray:
    """Per-sample weights for rare-positive oversampling.

    For sample ``i`` the weight is ``1 + strength * min(cap, sum_{c: y_ic=1} a_c)``
    where ``a_c`` is the class-balanced weight of class ``c``. Clips that contain
    rare positive labels therefore receive a higher sampling probability, while
    background / no-positive clips keep weight ``1`` so the negative decision
    boundary is preserved.
    """

    labels = np.asarray(label_matrix, dtype=np.float64)
    class_weights = np.asarray(class_weights, dtype=np.float64).reshape(1, -1)
    positive_score = (labels * class_weights).sum(axis=1)
    positive_score = np.minimum(positive_score, float(cap))
    return (1.0 + float(strength) * positive_score).astype(np.float64)
