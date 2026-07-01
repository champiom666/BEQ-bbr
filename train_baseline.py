#!/usr/bin/env python3
"""Train the global-pooling LVLM-LoRA baseline for BBR.

This corresponds to the "Baseline" (mean pooling) rows in the paper. Combine it
with an ``asl`` loss and class-balanced reweighting to obtain the
"Baseline + LTAL" variant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from beq.config import add_config_arg, load_config, save_config
from beq.constants import LABEL_COLUMNS, SAMPLE_LIST_DIR
from beq.data import BBRDataset, QwenVLVideoCollator, parse_view_spec
from beq.evaluator import evaluate
from beq.losses import compute_effective_number_weights, compute_rare_positive_sampling_weights
from beq.modeling import LVLMBodilyClassifier, get_hidden_size, load_vlm_and_processor
from beq.utils import move_to_device, save_json, set_seed, write_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override eval.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override train.num_workers and eval.num_workers.")
    parser.add_argument("--eval-num-workers", type=int, default=None, help="Override eval.num_workers only.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--views", default=None, help="Override config video.views; use 0, 1, 2, or 012.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip validation prediction/evaluation after training.")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[int], np.ndarray]:
    model.eval()
    sample_ids: list[int] = []
    rows: list[np.ndarray] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        ids = batch.pop("sample_ids").cpu().tolist()
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        probs = torch.sigmoid(outputs["logits"]).float().cpu().numpy()
        sample_ids.extend(ids)
        rows.append(probs)
    return sample_ids, np.concatenate(rows, axis=0)


def save_checkpoint(model: LVLMBodilyClassifier, cfg: dict[str, Any], checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.vlm.save_pretrained(checkpoint_dir / "adapter")
    model.save_classifier(checkpoint_dir)
    save_config(cfg, checkpoint_dir / "config.yaml")


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


def configure_class_balanced_loss(model: LVLMBodilyClassifier, train_ds: BBRDataset | None, cfg: dict[str, Any]) -> np.ndarray | None:
    """Apply class-balanced positive reweighting and return the class weights."""
    loss_cfg = cfg.get("loss") or {}
    class_balance = loss_cfg.get("class_balance") or {}
    if not class_balance.get("enabled", False) or train_ds is None:
        return None

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
    print("class-balanced positive weights:")
    print(table.to_string(index=False))
    return weights


def build_train_sampler(
    train_ds: BBRDataset | None,
    class_weights: np.ndarray | None,
    cfg: dict[str, Any],
) -> WeightedRandomSampler | None:
    """Build a rare-positive WeightedRandomSampler when enabled in the config."""
    loss_cfg = cfg.get("loss") or {}
    rps_cfg = loss_cfg.get("rare_positive_sampling") or {}
    if not rps_cfg.get("enabled", False) or train_ds is None or class_weights is None:
        return None
    label_matrix = train_ds.rows[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    sample_weights = compute_rare_positive_sampling_weights(
        label_matrix,
        class_weights,
        strength=float(rps_cfg.get("strength", 1.0)),
        cap=float(rps_cfg.get("cap", 5.0)),
    )
    print(f"rare-positive sampling enabled: weight range [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
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
    if args.eval_only and args.skip_eval:
        raise ValueError("--eval-only cannot be combined with --skip-eval")
    config_skip_eval = bool(cfg.get("train", {}).get("skip_eval", False)) and not args.eval_only
    skip_eval = bool(args.skip_eval or config_skip_eval)
    cfg.setdefault("train", {})["skip_eval"] = skip_eval
    set_seed(int(cfg["train"].get("seed", 42)))

    output_dir = Path(args.output_dir or cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    print(
        "baseline train settings:",
        f"batch_size={cfg['train'].get('batch_size')}",
        f"grad_accum={cfg['train'].get('gradient_accumulation_steps')}",
        f"num_frames={cfg['video'].get('num_frames')}",
        f"max_pixels={cfg['video'].get('max_pixels')}",
        f"views={cfg['video'].get('views')}",
        f"loss={(cfg.get('loss') or {}).get('name', 'bce')}",
        f"skip_eval={skip_eval}",
    )

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

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if not cfg["model"].get("device_map"):
        model.to(device)

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
    class_weights = configure_class_balanced_loss(model, train_ds, cfg)
    train_loader = None
    if train_ds is not None:
        sampler = build_train_sampler(train_ds, class_weights, cfg)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"].get("batch_size", 1)),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            collate_fn=collator,
        )
    val_loader = None
    val_annotation_path = SAMPLE_LIST_DIR / "val_samples.csv"
    if not skip_eval:
        val_ds = BBRDataset("val", use_views=use_views)
        if args.max_eval_samples is not None:
            val_ds.rows = val_ds.rows.iloc[: args.max_eval_samples].reset_index(drop=True)
            val_annotation_path = output_dir / f"val_annotations_first_{len(val_ds)}.csv"
            val_sample_ids = set(val_ds.rows["sample_id"].astype(int).tolist())
            val_ann = pd.read_csv(SAMPLE_LIST_DIR / "val_samples.csv")
            val_ann = val_ann[val_ann["sample_id"].astype(int).isin(val_sample_ids)]
            val_ann = val_ann.sort_values("sample_id")
            val_ann.to_csv(val_annotation_path, index=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["eval"].get("batch_size", 1)),
            shuffle=False,
            num_workers=int(cfg["eval"].get("num_workers", 0)),
            collate_fn=collator,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.1)),
    )
    grad_accum = int(cfg["train"].get("gradient_accumulation_steps", 16))
    epochs = int(cfg["train"].get("epochs", 1))
    max_train_steps = args.max_train_steps

    global_step = 0
    best_score = -float("inf")
    best_epoch: int | None = None
    best_dir = output_dir / "best-checkpoint"
    last_results: dict[str, object] | None = None

    if not args.eval_only:
        assert train_loader is not None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{epochs}")
            for step, batch in enumerate(pbar):
                batch.pop("sample_ids", None)
                batch = move_to_device(batch, device)
                outputs = model(**batch)
                loss = outputs["loss"] / grad_accum
                loss.backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, float(cfg["train"].get("grad_clip", 1.0)))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    pbar.set_postfix(loss=float(loss.detach().cpu()) * grad_accum, step=global_step)
                    if max_train_steps is not None and global_step >= max_train_steps:
                        break

            if not skip_eval:
                assert val_loader is not None
                sample_ids, probs = predict(model, val_loader, device)
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
                    save_checkpoint(model, cfg, best_dir)
                    write_prediction_csv(sample_ids, probs, best_dir / "predictions_val.csv")
                    save_eval_artifacts(results, best_dir, epoch=best_epoch)
                    print(f"new best checkpoint saved: {best_dir}")
                model.train()

            if max_train_steps is not None and global_step >= max_train_steps:
                break

        save_checkpoint(model, cfg, output_dir)

    if skip_eval:
        print("skip_eval: saved checkpoint without validation prediction/evaluation")
        return

    if args.eval_only or last_results is None:
        assert val_loader is not None
        sample_ids, probs = predict(model, val_loader, device)
        pred_name = "predictions_val.csv" if args.max_eval_samples is None else f"predictions_val_first_{len(sample_ids)}.csv"
        pred_path = output_dir / pred_name
        write_prediction_csv(sample_ids, probs, pred_path)

        results = evaluate(str(val_annotation_path), str(pred_path))
        print("macro_average:", results["macro_average"])
        print(results["per_class_scores"])
        return

    print(f"best_macro_average: {best_score} (epoch {best_epoch})")
    print(f"best_checkpoint: {best_dir}")


if __name__ == "__main__":
    main()
