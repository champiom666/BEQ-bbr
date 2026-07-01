"""Utility helpers for training, inference, and CSV output."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import LABEL_COLUMNS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def write_prediction_csv(sample_ids: list[int], probs: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(np.asarray(sample_ids))
    sample_ids_arr = np.asarray(sample_ids)[order]
    probs = probs[order]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", *LABEL_COLUMNS])
        for sid, row in zip(sample_ids_arr, probs):
            writer.writerow([int(sid), *[float(x) for x in row]])


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
