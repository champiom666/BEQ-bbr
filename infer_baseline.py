#!/usr/bin/env python3
"""Run inference for the global-pooling LVLM-LoRA baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from beq.config import add_config_arg, load_config
from beq.data import BBRDataset, QwenVLVideoCollator, parse_view_spec
from beq.modeling import LVLMBodilyClassifier, get_hidden_size, load_vlm_and_processor
from beq.utils import move_to_device, write_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with adapter/ and classifier.pt.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # For inference, load base, then attach saved adapter if available.
    cfg.setdefault("model", {})["apply_lora"] = False
    vlm, processor = load_vlm_and_processor(cfg)
    checkpoint_dir = Path(args.checkpoint_dir)
    adapter_dir = checkpoint_dir / "adapter"
    if adapter_dir.exists():
        from peft import PeftModel

        vlm = PeftModel.from_pretrained(vlm, adapter_dir)

    hidden_dim = get_hidden_size(vlm)
    model = LVLMBodilyClassifier(
        vlm=vlm,
        hidden_dim=hidden_dim,
        pooling=cfg["model"].get("pooling", "last"),
        dropout=float(cfg["model"].get("head_dropout", 0.2)),
        head_hidden_dim=int(cfg["model"].get("head_hidden_dim", 0)),
    )
    model.load_classifier(checkpoint_dir)

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if not cfg["model"].get("device_map"):
        model.to(device)
    model.eval()

    use_views = parse_view_spec(cfg.get("video", {}).get("views"))
    ds = BBRDataset(args.split, use_views=use_views)
    collator = QwenVLVideoCollator(
        processor=processor,
        num_frames=int(cfg["video"].get("num_frames", 16)),
        fps=int(cfg["video"].get("fps", 8)),
        max_pixels=int(cfg["video"].get("max_pixels", 262144)),
        add_vision_id=bool(cfg["video"].get("add_vision_id", True)),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["eval"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg["eval"].get("num_workers", 0)),
        collate_fn=collator,
    )

    sample_ids: list[int] = []
    probs_all: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"infer {args.split}"):
        ids = batch.pop("sample_ids").cpu().tolist()
        batch.pop("labels", None)
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        probs = torch.sigmoid(outputs["logits"]).float().cpu().numpy()
        sample_ids.extend(ids)
        probs_all.append(probs)

    write_prediction_csv(sample_ids, np.concatenate(probs_all, axis=0), args.output_csv)
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
