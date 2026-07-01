#!/usr/bin/env python3
"""Run inference for a trained BEQ checkpoint."""

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
from beq.data import BBRDataset, BEQVideoCollator, parse_view_spec
from beq.decoder import BEQClassifier, build_behaviour_query_init
from beq.behaviour_descriptions import build_behaviour_texts
from beq.modeling import get_hidden_size, load_vlm_and_processor
from beq.utils import move_to_device, write_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval.batch_size.")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--views", default=None)
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.batch_size is not None:
        cfg["eval"]["batch_size"] = args.batch_size
    if args.num_frames is not None:
        cfg["video"]["num_frames"] = args.num_frames
    if args.max_pixels is not None:
        cfg["video"]["max_pixels"] = args.max_pixels
    if args.fps is not None:
        cfg["video"]["fps"] = args.fps
    if args.views is not None:
        cfg["video"]["views"] = args.views


def build_model(vlm, processor, cfg: dict) -> BEQClassifier:
    hidden_dim = get_hidden_size(vlm)
    beq_cfg = cfg.get("beq", {})
    behaviour_texts = build_behaviour_texts(str(beq_cfg.get("behaviour_text_style", "name_description")))
    query_init = None
    if str(beq_cfg.get("query_init", "semantic")).lower() == "semantic":
        query_init = build_behaviour_query_init(vlm, processor, behaviour_texts, hidden_dim)
    return BEQClassifier(
        vlm=vlm,
        hidden_dim=hidden_dim,
        num_classes=14,
        query_init=query_init,
        decoder_dim=int(beq_cfg.get("decoder_dim", 512)),
        num_cross_attn_layers=int(beq_cfg.get("num_cross_attn_layers", 1)),
        cross_attn_heads=int(beq_cfg.get("cross_attn_heads", 8)),
        dropout=float(beq_cfg.get("dropout", cfg.get("model", {}).get("head_dropout", 0.2))),
        ffn_mult=float(beq_cfg.get("ffn_mult", 4.0)),
        label_correlation_layers=int(beq_cfg.get("label_correlation_layers", 0)),
        label_correlation_heads=int(beq_cfg.get("label_correlation_heads", 2)),
        head_hidden_dim=int(beq_cfg.get("head_hidden_dim", 0)),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    cfg.setdefault("model", {})["apply_lora"] = False
    vlm, processor = load_vlm_and_processor(cfg)
    checkpoint_dir = Path(args.checkpoint_dir)
    adapter_dir = checkpoint_dir / "adapter"
    if adapter_dir.exists():
        from peft import PeftModel

        vlm = PeftModel.from_pretrained(vlm, adapter_dir)

    model = build_model(vlm, processor, cfg)
    model.load_beq_head(checkpoint_dir)

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if not cfg["model"].get("device_map"):
        model.to(device)
    model.eval()

    use_views = parse_view_spec(cfg.get("video", {}).get("views"))
    ds = BBRDataset(args.split, use_views=use_views)
    collator = BEQVideoCollator(
        processor=processor,
        num_frames=int(cfg["video"].get("num_frames", 16)),
        fps=int(cfg["video"].get("fps", 8)),
        max_pixels=int(cfg["video"].get("max_pixels", 262144)),
        add_vision_id=bool(cfg["video"].get("add_vision_id", True)),
        visual_token_candidates=(cfg.get("beq") or {}).get("visual_token_candidates"),
        fallback_to_attention_mask=bool((cfg.get("beq") or {}).get("fallback_to_attention_mask", True)),
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
