"""Shared constants for the BEQ Bodily Behaviour Recognition framework."""

from __future__ import annotations

import os
from pathlib import Path


# beq/constants.py -> beq/ -> repository root
CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT

_DATA_ROOT = os.environ.get("BEQ_BBR_DATA_ROOT")
DATA_ROOT = Path(_DATA_ROOT).expanduser() if _DATA_ROOT else REPO_ROOT / "dataset" / "bbr"

SAMPLE_LIST_DIR = CODE_ROOT / "sample_lists"


def _clip_dir(split: str) -> Path:
    base = DATA_ROOT / f"clips_{split}"
    nested = base / f"clips_{split}"
    return nested if nested.exists() else base


CLIP_DIRS = {
    "train": _clip_dir("train"),
    "val": _clip_dir("val"),
    "test": _clip_dir("test"),
}

# Official CSV label order for the 14 bodily behaviour categories.
LABEL_COLUMNS = [
    "Settle",
    "Legs crossed",
    "Groom",
    "Hand-mouth",
    "Fold arms",
    "Leg movement",
    "Scratch",
    "Gesture",
    "Hand-face",
    "Adjusting clothing",
    "Fumble",
    "Shrug",
    "Stretching",
    "Smearing hands",
]

# The official evaluator iterates this order. CSV files may keep LABEL_COLUMNS;
# pandas column lookup makes both orders fine as long as all names exist.
EVAL_CLASS_ORDER = [
    "Hand-face",
    "Hand-mouth",
    "Gesture",
    "Fumble",
    "Scratch",
    "Stretching",
    "Smearing hands",
    "Shrug",
    "Adjusting clothing",
    "Groom",
    "Fold arms",
    "Leg movement",
    "Settle",
    "Legs crossed",
]

VIEW_SUFFIXES = ["", "1", "2"]
VIEW_NAMES = ["frontal", "left side", "right side"]
