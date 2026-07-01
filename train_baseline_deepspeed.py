#!/usr/bin/env python3
"""Train the global-pooling LVLM-LoRA baseline with DeepSpeed (ZeRO-2/ZeRO-3)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers.integrations import HfDeepSpeedConfig

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from beq.config import add_config_arg, load_config, save_config
from beq.constants import LABEL_COLUMNS, SAMPLE_LIST_DIR
from beq.data import BBRDataset, QwenVLVideoCollator, parse_view_spec
from beq.evaluator import evaluate
from beq.losses import compute_effective_number_weights
from beq.modeling import LVLMBodilyClassifier, get_hidden_size, load_vlm_and_processor
from beq.utils import move_to_device, save_json, set_seed, write_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--deepspeed-config", type=str, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-GPU train.batch_size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override per-GPU eval.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override train.num_workers and eval.num_workers.")
    parser.add_argument("--eval-num-workers", type=int, default=None, help="Override eval.num_workers only.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--views", default=None, help="Override config video.views; use 0, 1, 2, or 012.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip validation prediction/evaluation after training.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by DeepSpeed launcher.")
    return parser.parse_args()


def distributed_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if distributed_is_ready() else 0


def get_world_size() -> int:
    return dist.get_world_size() if distributed_is_ready() else 1


def is_rank0() -> bool:
    return get_rank() == 0


def rank0_print(*args: object, **kwargs: object) -> None:
    if is_rank0():
        print(*args, **kwargs)


def barrier() -> None:
    if distributed_is_ready():
        dist.barrier()


def broadcast_bool(value: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if value else 0], device=device, dtype=torch.int64)
    if distributed_is_ready():
        dist.broadcast(flag, src=0)
    return bool(flag.item())


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        cfg["eval"]["batch_size"] = args.eval_batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
        cfg["eval"]["num_workers"] = args.num_workers
    if args.eval_num_workers is not None:
        cfg["eval"]["num_workers"] = args.eval_num_workers
    if args.gradient_accumulation_steps is not None:
        cfg["train"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.num_frames is not None:
        cfg["video"]["num_frames"] = args.num_frames
    if args.max_pixels is not None:
        cfg["video"]["max_pixels"] = args.max_pixels
    if args.fps is not None:
        cfg["video"]["fps"] = args.fps
    if args.views is not None:
        cfg["video"]["views"] = args.views
    if args.deepspeed_config is not None:
        cfg.setdefault("train", {})["deepspeed_config"] = args.deepspeed_config


def load_deepspeed_config(cfg: dict[str, Any]) -> dict[str, Any]:
    ds_path = cfg["train"].get("deepspeed_config") or cfg.get("deepspeed", {}).get("config")
    if not ds_path:
        raise ValueError("DeepSpeed training requires train.deepspeed_config or --deepspeed-config.")

    with Path(ds_path).open("r", encoding="utf-8") as f:
        ds_config = json.load(f)

    ds_config["train_micro_batch_size_per_gpu"] = int(cfg["train"].get("batch_size", 1))
    ds_config["gradient_accumulation_steps"] = int(cfg["train"].get("gradient_accumulation_steps", 1))
    ds_config["gradient_clipping"] = float(cfg["train"].get("grad_clip", ds_config.get("gradient_clipping", 1.0)))
    return ds_config


def prepare_deepspeed_model_config(cfg: dict[str, Any]) -> None:
    model_cfg = cfg["model"]
    if model_cfg.get("device_map"):
        rank0_print("[DeepSpeed] overriding model.device_map because DeepSpeed owns model partitioning.")
    if model_cfg.get("max_memory"):
        rank0_print("[DeepSpeed] ignoring model.max_memory because DeepSpeed owns model partitioning.")
    model_cfg["device_map"] = None
    model_cfg.pop("max_memory", None)


def save_eval_artifacts(
    results: dict[str, object],
    output_dir: str | Path,
    epoch: int | None = None,
) -> None:
    per_class = results["per_class_scores"]["score"].astype(float).to_dict()  # type: ignore[index, union-attr]
    payload = {
        "macro_average": float(results["macro_average"]),
        "epoch": epoch,
        "per_class_scores": per_class,
    }
    save_json(payload, Path(output_dir) / "metrics.json")


def configure_class_balanced_loss(model: LVLMBodilyClassifier, train_ds: BBRDataset | None, cfg: dict[str, Any]) -> None:
    loss_cfg = cfg.get("loss") or {}
    class_balance = loss_cfg.get("class_balance") or {}
    if not class_balance.get("enabled", False) or train_ds is None:
        return

    counts = train_ds.rows[LABEL_COLUMNS].sum(axis=0).to_numpy(dtype=np.float64)
    weights = compute_effective_number_weights(
        counts,
        beta=float(class_balance.get("beta", 0.9999)),
        strength=float(class_balance.get("strength", 1.0)),
        min_weight=class_balance.get("min_weight"),
        max_weight=class_balance.get("max_weight"),
    )
    model.set_loss_pos_weight(weights.tolist())
    table = pd.DataFrame({"label": LABEL_COLUMNS, "positive_count": counts.astype(int), "pos_weight": weights})
    rank0_print("class-balanced positive weights:")
    rank0_print(table.to_string(index=False))


@torch.no_grad()
def predict_distributed(model, loader: DataLoader, device: torch.device) -> tuple[list[int], np.ndarray]:
    model.eval()
    local_sample_ids: list[int] = []
    local_rows: list[list[float]] = []

    iterator = tqdm(loader, desc="predict", leave=False, disable=not is_rank0())
    for batch in iterator:
        ids = batch.pop("sample_ids").cpu().tolist()
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        probs = torch.sigmoid(outputs["logits"]).float().detach().cpu().numpy()
        local_sample_ids.extend(int(x) for x in ids)
        local_rows.extend(probs.tolist())

    local_payload = {"sample_ids": local_sample_ids, "rows": local_rows}
    if distributed_is_ready():
        gathered: list[dict[str, Any] | None] = [None for _ in range(get_world_size())]
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]

    if not is_rank0():
        return [], np.empty((0, len(LABEL_COLUMNS)), dtype=np.float32)

    merged: dict[int, list[float]] = {}
    for payload in gathered:
        if payload is None:
            continue
        for sid, row in zip(payload["sample_ids"], payload["rows"]):
            merged.setdefault(int(sid), row)

    sample_ids = sorted(merged)
    probs = np.asarray([merged[sid] for sid in sample_ids], dtype=np.float32)
    return sample_ids, probs


def save_deepspeed_checkpoint(model_engine, output_dir: str | Path, tag: str) -> None:
    checkpoint_root = Path(output_dir) / "deepspeed_checkpoint"
    model_engine.save_checkpoint(str(checkpoint_root), tag=tag)


def build_validation_annotation(
    output_dir: Path,
    val_ds: BBRDataset,
    max_eval_samples: int | None,
) -> Path:
    if max_eval_samples is None:
        return SAMPLE_LIST_DIR / "val_samples.csv"

    val_annotation_path = output_dir / f"val_annotations_first_{len(val_ds)}.csv"
    if is_rank0():
        val_sample_ids = set(val_ds.rows["sample_id"].astype(int).tolist())
        val_ann = pd.read_csv(SAMPLE_LIST_DIR / "val_samples.csv")
        val_ann = val_ann[val_ann["sample_id"].astype(int).isin(val_sample_ids)]
        val_ann = val_ann.sort_values("sample_id")
        val_ann.to_csv(val_annotation_path, index=False)
    barrier()
    return val_annotation_path


def main() -> None:
    args = parse_args()
    if args.eval_only and args.skip_eval:
        raise ValueError("--eval-only cannot be combined with --skip-eval")

    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    config_skip_eval = bool(cfg.get("train", {}).get("skip_eval", False)) and not args.eval_only
    skip_eval = bool(args.skip_eval or config_skip_eval)
    cfg.setdefault("train", {})["skip_eval"] = skip_eval
    prepare_deepspeed_model_config(cfg)
    ds_config = load_deepspeed_config(cfg)

    set_seed(int(cfg["train"].get("seed", 42)) + get_rank())

    output_dir = Path(args.output_dir or cfg["train"]["output_dir"])
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, output_dir / "config.yaml")
        save_json(ds_config, output_dir / "deepspeed_config.json")
        effective_batch = (
            int(cfg["train"].get("batch_size", 1))
            * int(cfg["train"].get("gradient_accumulation_steps", 1))
            * get_world_size()
        )
        print(
            "baseline deepspeed train settings:",
            f"world_size={get_world_size()}",
            f"micro_batch_per_gpu={cfg['train'].get('batch_size')}",
            f"grad_accum={cfg['train'].get('gradient_accumulation_steps')}",
            f"effective_batch={effective_batch}",
            f"num_frames={cfg['video'].get('num_frames')}",
            f"max_pixels={cfg['video'].get('max_pixels')}",
            f"views={cfg['video'].get('views')}",
            f"loss={(cfg.get('loss') or {}).get('name', 'bce')}",
            f"skip_eval={skip_eval}",
        )
    barrier()

    # Keep this object alive while loading the model so Transformers enables
    # ZeRO-aware construction instead of materializing the full model per rank.
    hf_ds_config = HfDeepSpeedConfig(ds_config)
    vlm, processor = load_vlm_and_processor(cfg)
    hidden_dim = get_hidden_size(vlm)
    model = LVLMBodilyClassifier(
        vlm=vlm,
        hidden_dim=hidden_dim,
        pooling=cfg["model"].get("pooling", "last"),
        dropout=float(cfg["model"].get("head_dropout", 0.2)),
        head_hidden_dim=int(cfg["model"].get("head_hidden_dim", 0)),
        loss_config=cfg.get("loss"),
    )
    # Prevent the config object from being optimized away before initialization.
    model._hf_ds_config = hf_ds_config  # type: ignore[attr-defined]

    use_views = parse_view_spec(cfg.get("video", {}).get("views"))
    train_ds = None
    if not args.eval_only:
        train_ds = BBRDataset("train", use_views=use_views)
        if args.max_train_samples is not None:
            train_ds.rows = train_ds.rows.iloc[: args.max_train_samples].reset_index(drop=True)

    collator = QwenVLVideoCollator(
        processor=processor,
        num_frames=int(cfg["video"].get("num_frames", 16)),
        fps=int(cfg["video"].get("fps", 8)),
        max_pixels=int(cfg["video"].get("max_pixels", 262144)),
        add_vision_id=bool(cfg["video"].get("add_vision_id", True)),
    )

    train_loader = None
    train_sampler = None
    if train_ds is not None:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"].get("batch_size", 1)),
            sampler=train_sampler,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            collate_fn=collator,
        )
    configure_class_balanced_loss(model, train_ds, cfg)

    val_loader = None
    val_annotation_path = SAMPLE_LIST_DIR / "val_samples.csv"
    if not skip_eval:
        val_ds = BBRDataset("val", use_views=use_views)
        if args.max_eval_samples is not None:
            val_ds.rows = val_ds.rows.iloc[: args.max_eval_samples].reset_index(drop=True)
        val_annotation_path = build_validation_annotation(output_dir, val_ds, args.max_eval_samples)
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["eval"].get("batch_size", 1)),
            sampler=val_sampler,
            num_workers=int(cfg["eval"].get("num_workers", 0)),
            collate_fn=collator,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.1)),
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=params,
        optimizer=optimizer,
        config=ds_config,
    )

    epochs = int(cfg["train"].get("epochs", 1))
    max_train_steps = args.max_train_steps
    global_step = 0
    best_score = -float("inf")
    best_epoch: int | None = None
    best_dir = output_dir / "best-checkpoint"
    last_results: dict[str, object] | None = None

    if not args.eval_only:
        assert train_loader is not None
        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model_engine.train()
            pbar = tqdm(
                train_loader,
                desc=f"train epoch {epoch + 1}/{epochs}",
                disable=not is_rank0(),
            )
            for batch in pbar:
                batch.pop("sample_ids", None)
                batch = move_to_device(batch, device)
                outputs = model_engine(**batch)
                loss = outputs["loss"]
                model_engine.backward(loss)
                will_update = model_engine.is_gradient_accumulation_boundary()
                model_engine.step()
                if will_update:
                    global_step = int(model_engine.global_steps)
                    if is_rank0():
                        pbar.set_postfix(loss=float(loss.detach().float().cpu()), step=global_step)
                    if max_train_steps is not None and global_step >= max_train_steps:
                        break

            if not skip_eval:
                assert val_loader is not None
                sample_ids, probs = predict_distributed(model_engine, val_loader, device)
                should_save_best = False
                if is_rank0():
                    if args.max_eval_samples is None:
                        pred_name = f"predictions_val_epoch_{epoch + 1:03d}.csv"
                    else:
                        pred_name = f"predictions_val_first_{len(sample_ids)}_epoch_{epoch + 1:03d}.csv"
                    pred_path = output_dir / pred_name
                    write_prediction_csv(sample_ids, probs, pred_path)

                    results = evaluate(str(val_annotation_path), str(pred_path))
                    last_results = results
                    cur_score = float(results["macro_average"])
                    print(f"epoch {epoch + 1} macro_average:", cur_score)
                    print(results["per_class_scores"])
                    if cur_score > best_score:
                        best_score = cur_score
                        best_epoch = epoch + 1
                        best_dir.mkdir(parents=True, exist_ok=True)
                        write_prediction_csv(sample_ids, probs, best_dir / "predictions_val.csv")
                        save_eval_artifacts(results, best_dir, epoch=best_epoch)
                        save_config(cfg, best_dir / "config.yaml")
                        should_save_best = True
                        print(f"new best checkpoint will be saved: {best_dir}")

                if broadcast_bool(should_save_best, device):
                    save_deepspeed_checkpoint(model_engine, best_dir, tag="best")
                barrier()

            if max_train_steps is not None and global_step >= max_train_steps:
                break

        save_deepspeed_checkpoint(model_engine, output_dir, tag="last")

        if skip_eval:
            rank0_print("skip_eval: saved DeepSpeed checkpoint without validation prediction/evaluation")
            return

        rank0_print(f"best_macro_average: {best_score} (epoch {best_epoch})")
        rank0_print(f"best_checkpoint: {best_dir / 'deepspeed_checkpoint'}")
        return

    if args.eval_only or last_results is None:
        assert val_loader is not None
        sample_ids, probs = predict_distributed(model_engine, val_loader, device)
        if is_rank0():
            pred_name = "predictions_val.csv" if args.max_eval_samples is None else f"predictions_val_first_{len(sample_ids)}.csv"
            pred_path = output_dir / pred_name
            write_prediction_csv(sample_ids, probs, pred_path)
            results = evaluate(str(val_annotation_path), str(pred_path))
            print("macro_average:", results["macro_average"])
            print(results["per_class_scores"])
        return


if __name__ == "__main__":
    main()
