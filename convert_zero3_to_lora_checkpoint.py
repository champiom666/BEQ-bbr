#!/usr/bin/env python3
"""Export a DeepSpeed ZeRO-3 baseline checkpoint as a regular PEFT adapter.

This is for the global-pooling baseline (``train_baseline_deepspeed.py``), whose
trainable parameters are the LoRA tensors and the ``classifier`` head. The BEQ
DeepSpeed trainer already writes a portable ``adapter/`` + ``beq_head.pt`` next
to each checkpoint, so it does not need this script.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig
from safetensors.torch import save_file

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from beq.config import load_config, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config used for the checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing deepspeed_checkpoint/, or the deepspeed_checkpoint directory itself.",
    )
    parser.add_argument("--tag", default="best", help="DeepSpeed checkpoint tag to export.")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter/ and classifier.pt.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    return parser.parse_args()


def resolve_zero_root(path: str | Path) -> Path:
    checkpoint_dir = Path(path)
    ds_root = checkpoint_dir / "deepspeed_checkpoint"
    if ds_root.exists():
        return ds_root
    return checkpoint_dir


def build_lora_config(cfg: dict[str, Any]) -> LoraConfig:
    lora_cfg = cfg.get("lora", {})
    model_cfg = cfg.get("model", {})
    return LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg.get("target_modules"),
        modules_to_save=lora_cfg.get("modules_to_save", None),
        base_model_name_or_path=model_cfg.get("backbone_path"),
        inference_mode=True,
    )


def to_peft_adapter_key(key: str) -> str:
    """Match PEFT save_pretrained(): strip the active adapter name from LoRA keys."""
    key = key.removeprefix("vlm.")
    return re.sub(r"\.default(?=\.weight$)", "", key)


def export_checkpoint(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    zero_root = resolve_zero_root(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    adapter_dir.mkdir(parents=True)

    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        str(zero_root),
        tag=args.tag,
        exclude_frozen_parameters=True,
    )
    lora_state = {
        to_peft_adapter_key(key): tensor.detach().cpu().contiguous()
        for key, tensor in state_dict.items()
        if "lora_" in key
    }
    classifier_state = {
        key.removeprefix("classifier."): tensor.detach().cpu()
        for key, tensor in state_dict.items()
        if key.startswith("classifier.")
    }

    if not lora_state:
        raise RuntimeError("No LoRA tensors were found in the ZeRO checkpoint.")
    if not classifier_state:
        raise RuntimeError("No classifier tensors were found in the ZeRO checkpoint.")

    build_lora_config(cfg).save_pretrained(str(adapter_dir))
    save_file(lora_state, adapter_dir / "adapter_model.safetensors", metadata={"format": "pt"})
    torch.save(classifier_state, output_dir / "classifier.pt")
    save_config(cfg, output_dir / "config.yaml")

    print(f"wrote adapter: {adapter_dir}")
    print(f"wrote classifier: {output_dir / 'classifier.pt'}")
    print(f"LoRA tensors: {len(lora_state)}")
    print(f"classifier tensors: {len(classifier_state)}")


def main() -> None:
    export_checkpoint(parse_args())


if __name__ == "__main__":
    main()
