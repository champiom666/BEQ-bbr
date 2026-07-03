#!/usr/bin/env python3
"""Train the Behaviour Evidence Querying (BEQ) decoder with DeepSpeed.

This is the multi-GPU trainer used for the 4x A800 setup reported in the paper.
Combine it with an ``asl`` loss + class-balanced reweighting + rare-positive
sampling to train the full "BEQ + LTAL" model.
"""

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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Sampler, Subset
from tqdm.auto import tqdm
from transformers.integrations import HfDeepSpeedConfig

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from beq.config import add_config_arg, load_config, save_config
from beq.constants import LABEL_COLUMNS, SAMPLE_LIST_DIR
from beq.data import BBRDataset, BEQVideoCollator, parse_view_spec
from beq.decoder import BEQClassifier, build_behaviour_query_init
from beq.behaviour_descriptions import build_behaviour_texts
from beq.evaluator import evaluate
from beq.losses import compute_effective_number_weights, compute_rare_positive_sampling_weights
from beq.modeling import get_hidden_size, load_vlm_and_processor
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
    parser.add_argument("--train-with-val", action="store_true", help="Train on train+val and disable validation eval.")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None, help="DeepSpeed checkpoint dir with deepspeed_checkpoint/.")
    parser.add_argument("--tag", default="best", help="DeepSpeed checkpoint tag for --eval-only.")
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


class DistributedWeightedRandomSampler(Sampler[int]):
    """Weighted sampler that draws one shared replacement sample per epoch."""

    def __init__(
        self,
        weights: np.ndarray,
        num_replicas: int,
        rank: int,
        seed: int = 42,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {tuple(self.weights.shape)}")
        if self.weights.numel() == 0:
            raise ValueError("weights must be non-empty")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(np.ceil(self.weights.numel() / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


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


def prepare_deepspeed_model_config(cfg: dict[str, Any], ds_config: dict[str, Any]) -> None:
    model_cfg = cfg["model"]
    if model_cfg.get("device_map"):
        rank0_print("[DeepSpeed] overriding model.device_map; DeepSpeed owns placement.")
    if model_cfg.get("max_memory"):
        rank0_print("[DeepSpeed] ignoring model.max_memory; DeepSpeed owns placement.")
    model_cfg["device_map"] = None
    model_cfg.pop("max_memory", None)
    # Training throughput wins over activation memory savings when memory allows.
    model_cfg["gradient_checkpointing"] = bool(model_cfg.get("gradient_checkpointing", False))


def save_eval_artifacts(results: dict[str, object], output_dir: str | Path, epoch: int | None = None) -> None:
    per_class = results["per_class_scores"]["score"].astype(float).to_dict()  # type: ignore[index, union-attr]
    payload = {
        "macro_average": float(results["macro_average"]),
        "epoch": epoch,
        "per_class_scores": per_class,
    }
    save_json(payload, Path(output_dir) / "metrics.json")


def build_training_dataset(
    use_views: list[str],
    include_val: bool,
    max_train_samples: int | None,
) -> tuple[Dataset, pd.DataFrame]:
    datasets = [BBRDataset("train", use_views=use_views)]
    if include_val:
        datasets.append(BBRDataset("val", use_views=use_views))

    rows = pd.concat([ds.rows for ds in datasets], ignore_index=True)
    dataset: Dataset
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    if max_train_samples is not None:
        limit = min(int(max_train_samples), len(dataset))
        dataset = Subset(dataset, range(limit))
        rows = rows.iloc[:limit].reset_index(drop=True)
    return dataset, rows


def configure_class_balanced_loss(
    model: BEQClassifier,
    train_rows: pd.DataFrame | None,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    loss_cfg = cfg.get("loss") or {}
    class_balance = loss_cfg.get("class_balance") or {}
    if not class_balance.get("enabled", False) or train_rows is None:
        return None

    counts = train_rows[LABEL_COLUMNS].sum(axis=0).to_numpy(dtype=np.float64)
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
    return weights


def build_train_sampler(
    train_ds: Dataset,
    train_rows: pd.DataFrame | None,
    class_weights: np.ndarray | None,
    cfg: dict[str, Any],
) -> Sampler[int] | None:
    if train_rows is None:
        return None

    loss_cfg = cfg.get("loss") or {}
    rps_cfg = loss_cfg.get("rare_positive_sampling") or {}
    if rps_cfg.get("enabled", False) and class_weights is not None:
        label_matrix = train_rows[LABEL_COLUMNS].to_numpy(dtype=np.float64)
        sample_weights = compute_rare_positive_sampling_weights(
            label_matrix,
            class_weights,
            strength=float(rps_cfg.get("strength", 1.0)),
            cap=float(rps_cfg.get("cap", 5.0)),
        )
        rank0_print(f"rare-positive sampling enabled: weight range [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        return DistributedWeightedRandomSampler(
            sample_weights,
            num_replicas=get_world_size(),
            rank=get_rank(),
            seed=int(cfg["train"].get("seed", 42)),
        )

    return DistributedSampler(
        train_ds,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=False,
    )


def build_model(vlm, processor, cfg: dict[str, Any]) -> tuple[BEQClassifier, list[str]]:
    hidden_dim = get_hidden_size(vlm)
    beq_cfg = cfg.get("beq", {})
    behaviour_texts = build_behaviour_texts(str(beq_cfg.get("behaviour_text_style", "name_description")))

    query_init = None
    if str(beq_cfg.get("query_init", "semantic")).lower() == "semantic":
        query_init = build_behaviour_query_init(vlm, processor, behaviour_texts, hidden_dim)
        rank0_print("behaviour query initialisation: enabled")
    else:
        rank0_print("behaviour query initialisation: disabled; using random learnable queries")

    model = BEQClassifier(
        vlm=vlm,
        hidden_dim=hidden_dim,
        num_classes=len(LABEL_COLUMNS),
        query_init=query_init,
        decoder_dim=int(beq_cfg.get("decoder_dim", 512)),
        num_cross_attn_layers=int(beq_cfg.get("num_cross_attn_layers", 1)),
        cross_attn_heads=int(beq_cfg.get("cross_attn_heads", 8)),
        dropout=float(beq_cfg.get("dropout", cfg.get("model", {}).get("head_dropout", 0.2))),
        ffn_mult=float(beq_cfg.get("ffn_mult", 4.0)),
        label_correlation_layers=int(beq_cfg.get("label_correlation_layers", 0)),
        label_correlation_heads=int(beq_cfg.get("label_correlation_heads", 2)),
        head_hidden_dim=int(beq_cfg.get("head_hidden_dim", 0)),
        loss_config=cfg.get("loss"),
    )
    return model, behaviour_texts


def save_adapter_and_head(model_engine, cfg: dict[str, Any], output_dir: str | Path, behaviour_texts: list[str]) -> None:
    """Save regular PEFT adapter + BEQ head from a DeepSpeed engine."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model_engine.module if hasattr(model_engine, "module") else model_engine
    model.vlm.save_pretrained(output_dir / "adapter")
    model.save_beq_head(output_dir, behaviour_texts=behaviour_texts)
    save_config(cfg, output_dir / "config.yaml")


def save_deepspeed_checkpoint(model_engine, output_dir: str | Path, tag: str) -> None:
    checkpoint_root = Path(output_dir) / "deepspeed_checkpoint"
    model_engine.save_checkpoint(str(checkpoint_root), tag=tag)


def resolve_checkpoint_root(path: str | Path) -> Path:
    checkpoint_dir = Path(path)
    ds_root = checkpoint_dir / "deepspeed_checkpoint"
    if ds_root.exists():
        return ds_root
    return checkpoint_dir


def build_validation_annotation(output_dir: Path, val_ds: BBRDataset, max_eval_samples: int | None) -> Path:
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


def main() -> None:
    args = parse_args()
    if args.eval_only and args.train_with_val:
        raise ValueError("--eval-only cannot be combined with --train-with-val")
    if args.eval_only and args.skip_eval:
        raise ValueError("--eval-only cannot be combined with --skip-eval")
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    config_skip_eval = bool(cfg.get("train", {}).get("skip_eval", False)) and not args.eval_only
    skip_eval = bool(args.skip_eval or config_skip_eval or args.train_with_val)
    cfg.setdefault("train", {})["train_with_val"] = bool(args.train_with_val)
    cfg["train"]["skip_eval"] = skip_eval
    ds_config = load_deepspeed_config(cfg)
    prepare_deepspeed_model_config(cfg, ds_config)
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
            "BEQ DeepSpeed train settings:",
            f"world_size={get_world_size()}",
            f"micro_batch_per_gpu={cfg['train'].get('batch_size')}",
            f"grad_accum={cfg['train'].get('gradient_accumulation_steps')}",
            f"effective_batch={effective_batch}",
            f"num_frames={cfg['video'].get('num_frames')}",
            f"max_pixels={cfg['video'].get('max_pixels')}",
            f"views={cfg['video'].get('views')}",
            f"query_init={(cfg.get('beq') or {}).get('query_init', 'semantic')}",
            f"cross_layers={(cfg.get('beq') or {}).get('num_cross_attn_layers', 1)}",
            f"label_corr_layers={(cfg.get('beq') or {}).get('label_correlation_layers', 0)}",
            f"loss={(cfg.get('loss') or {}).get('name', 'bce')}",
            f"train_with_val={args.train_with_val}",
            f"skip_eval={skip_eval}",
        )
    barrier()

    # Keep this object alive while loading so Transformers can construct under
    # DeepSpeed memory policy. ZeRO-2 still materializes full weights per rank,
    # while ZeRO-3 can shard initialization.
    hf_ds_config = HfDeepSpeedConfig(ds_config)
    vlm, processor = load_vlm_and_processor(cfg)
    model, behaviour_texts = build_model(vlm, processor, cfg)
    model._hf_ds_config = hf_ds_config  # type: ignore[attr-defined]

    use_views = parse_view_spec(cfg.get("video", {}).get("views"))
    train_ds = None
    train_rows = None
    if not args.eval_only:
        train_ds, train_rows = build_training_dataset(
            use_views=use_views,
            include_val=bool(args.train_with_val),
            max_train_samples=args.max_train_samples,
        )

    collator = BEQVideoCollator(
        processor=processor,
        num_frames=int(cfg["video"].get("num_frames", 16)),
        fps=int(cfg["video"].get("fps", 8)),
        max_pixels=int(cfg["video"].get("max_pixels", 262144)),
        add_vision_id=bool(cfg["video"].get("add_vision_id", True)),
        visual_token_candidates=(cfg.get("beq") or {}).get("visual_token_candidates"),
        fallback_to_attention_mask=bool((cfg.get("beq") or {}).get("fallback_to_attention_mask", True)),
    )

    class_weights = configure_class_balanced_loss(model, train_rows, cfg)

    train_loader = None
    train_sampler = None
    if train_ds is not None:
        train_sampler = build_train_sampler(train_ds, train_rows, class_weights, cfg)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"].get("batch_size", 1)),
            sampler=train_sampler,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            collate_fn=collator,
        )

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

    if args.eval_only:
        if not args.checkpoint_dir:
            raise ValueError("--eval-only requires --checkpoint-dir")
        checkpoint_root = resolve_checkpoint_root(args.checkpoint_dir)
        load_path, _ = model_engine.load_checkpoint(str(checkpoint_root), tag=args.tag, load_optimizer_states=False)
        if load_path is None:
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint: root={checkpoint_root}, tag={args.tag}")
        rank0_print(f"loaded checkpoint: {load_path}")

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
                    pred_name = (
                        f"predictions_val_epoch_{epoch + 1:03d}.csv"
                        if args.max_eval_samples is None
                        else f"predictions_val_first_{len(sample_ids)}_epoch_{epoch + 1:03d}.csv"
                    )
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
                    if is_rank0():
                        save_adapter_and_head(model_engine, cfg, best_dir, behaviour_texts)
                barrier()

            if max_train_steps is not None and global_step >= max_train_steps:
                break

        save_deepspeed_checkpoint(model_engine, output_dir, tag="last")
        if is_rank0():
            save_adapter_and_head(model_engine, cfg, output_dir, behaviour_texts)

        if skip_eval:
            if args.train_with_val:
                rank0_print("train_with_val: saved DeepSpeed checkpoint trained on train+val without validation prediction/evaluation")
            else:
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
