"""Local BBR evaluation helper.

This mirrors the official MultiMediate bodily-behaviour evaluator (category-wise
mean Average Precision) so the framework does not import the baseline directory
at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EVAL_CLASS_ORDER

try:
    from sklearn.metrics import average_precision_score as _sklearn_average_precision_score
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    _sklearn_average_precision_score = None


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary average precision with an sklearn-compatible fallback."""

    if _sklearn_average_precision_score is not None:
        return float(_sklearn_average_precision_score(y_true, y_score))

    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    positive_count = int(y_true.sum())
    if positive_count == 0:
        return 0.0

    # Sort by descending score, grouping equal thresholds like sklearn does.
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[order]
    y_score = y_score[order]

    threshold_idxs = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[threshold_idxs, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positive_count

    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate(annotation_file: str | Path, prediction_file: str | Path) -> dict[str, object]:
    annotations = pd.read_csv(annotation_file, index_col="sample_id").sort_values("sample_id")
    predictions = pd.read_csv(prediction_file, index_col="sample_id").sort_values("sample_id")
    if not np.all(annotations.index == predictions.index):
        raise ValueError("Indexes of annotation and prediction files do not agree.")

    missing = [name for name in EVAL_CLASS_ORDER if name not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction file missing behaviour columns: {missing}")

    scores = []
    for behaviour in EVAL_CLASS_ORDER:
        if annotations[behaviour].sum() == 0:
            score = 1
        else:
            score = average_precision_score(
                annotations[behaviour].values,
                predictions[behaviour].values,
            )
        scores.append(score)

    per_class_scores = pd.DataFrame(
        {"behaviour": EVAL_CLASS_ORDER, "score": scores}
    ).set_index("behaviour")
    return {
        "macro_average": float(np.mean(scores)),
        "per_class_scores": per_class_scores,
    }
