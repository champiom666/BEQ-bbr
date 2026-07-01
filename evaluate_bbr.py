#!/usr/bin/env python3
"""Evaluate a BBR prediction CSV using the local macro-mAP evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from beq.constants import SAMPLE_LIST_DIR
from beq.evaluator import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default=str(SAMPLE_LIST_DIR / "val_samples.csv"))
    parser.add_argument("--prediction", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.annotation, args.prediction)
    print("macro_average:", results["macro_average"])
    print(results["per_class_scores"])


if __name__ == "__main__":
    main()
