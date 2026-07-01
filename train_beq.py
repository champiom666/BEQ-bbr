#!/usr/bin/env python3
"""Train the Behaviour Evidence Querying (BEQ) decoder for BBR.

Combine BEQ with an ``asl`` loss + class-balanced reweighting (+ optional
rare-positive sampling) to obtain the full "BEQ + LTAL" model from the paper.
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm.auto import tqdm

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
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override eval.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--views", default=None, help="Override config video.views; use 0, 1, 2, or 012.")
    parser.add_argument("--train-with-val", action="store_true", help="Train on train+val and disable validation eval.")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"], help="Split to predict in --eval-only mode.")
    parser.add_argument("--checkpoint-dir", default=None, help="For --eval-only, load adapter/ and beq_head.pt.")
    return parser.parse_args()


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


@torch.no_grad()
def predict(model: BEQClassifier, loader: DataLoader, device: torch.device) -> tuple[list[int], np.ndarray]:
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


def save_checkpoint(
    model: BEQClassifier,
    cfg: dict[str, Any],
    checkpoint_dir: str | Path,
    behaviour_texts: list[str],
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.vlm.save_pretrained(checkpoint_dir / "adapter")
    model.save_beq_head(checkpoint_dir, behaviour_texts=behaviour_texts)
    save_config(cfg, checkpoint_dir / "config.yaml")


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


def configure_class_balanced_loss(model: BEQClassifier, train_rows: pd.DataFrame | None, cfg: dict[str, Any]) -> np.ndarray | None:
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
    print("class-balanced positive weights:")
    print(table.to_string(index=False))
    return weights


def build_train_sampler(
    train_rows: pd.DataFrame | None,
    class_weights: np.ndarray | None,
    cfg: dict[str, Any],
) -> WeightedRandomSampler | None:
    loss_cfg = cfg.get("loss") or {}
    rps_cfg = loss_cfg.get("rare_positive_sampling") or {}
    if not rps_cfg.get("enabled", False) or train_rows is None or class_weights is None:
        return None
    label_matrix = train_rows[LABEL_COLUMNS].to_numpy(dtype=np.float64)
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


def build_model(vlm, processor, cfg: dict[str, Any]) -> tuple[BEQClassifier, list[str]]:
    hidden_dim = get_hidden_size(vlm)
    beq_cfg = cfg.get("beq", {})
    behaviour_texts = build_behaviour_texts(str(beq_cfg.get("behaviour_text_style", "name_description")))

    query_init = None
    if str(beq_cfg.get("query_init", "semantic")).lower() == "semantic":
        query_init = build_behaviour_query_init(vlm, processor, behaviour_texts, hidden_dim)
        print("behaviour query initialisation: enabled")
    else:
        print("behaviour query initialisation: disabled; using random learnable queries")

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


def maybe_load_checkpoint(model: BEQClassifier, checkpoint_dir: str | Path | None) -> None:
    if checkpoint_dir is None:
        return
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    adapter_dir = checkpoint_dir / "adapter"
    if adapter_dir.exists():
        from peft import PeftModel

        model.vlm = PeftModel.from_pretrained(model.vlm, adapter_dir, is_trainable=False)
    model.load_beq_head(checkpoint_dir)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    if args.eval_only and args.train_with_val:
        raise ValueError("--eval-only cannot be combined with --train-with-val")
    if args.eval_only and args.skip_eval:
        raise ValueError("--eval-only cannot be combined with --skip-eval")
    if not args.eval_only and args.eval_split != "val":
        raise ValueError("--eval-split is only supported with --eval-only")
    config_skip_eval = bool(cfg.get("train", {}).get("skip_eval", False)) and not args.eval_only
    skip_eval = bool(args.skip_eval or config_skip_eval or args.train_with_val)
    cfg.setdefault("train", {})["train_with_val"] = bool(args.train_with_val)
    cfg["train"]["skip_eval"] = skip_eval
    set_seed(int(cfg["train"].get("seed", 42)))

    output_dir = Path(args.output_dir or cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    print(
        "BEQ train settings:",
        f"batch_size={cfg['train'].get('batch_size')}",
        f"grad_accum={cfg['train'].get('gradient_accumulation_steps')}",
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

    if args.eval_only:
        cfg.setdefault("model", {})["apply_lora"] = False
    vlm, processor = load_vlm_and_processor(cfg)
    model, behaviour_texts = build_model(vlm, processor, cfg)
    maybe_load_checkpoint(model, args.checkpoint_dir if args.eval_only else None)

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if not cfg["model"].get("device_map"):
        model.to(device)

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
    if train_ds is not None:
        sampler = build_train_sampler(train_rows, class_weights, cfg)
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
        eval_split = args.eval_split if args.eval_only else "val"
        val_ds = BBRDataset(eval_split, use_views=use_views)
        if args.max_eval_samples is not None:
            val_ds.rows = val_ds.rows.iloc[: args.max_eval_samples].reset_index(drop=True)
            if eval_split == "val":
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
    grad_accum = int(cfg["train"].get("gradient_accumulation_steps", 1))
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
                    save_checkpoint(model, cfg, best_dir, behaviour_texts)
                    write_prediction_csv(sample_ids, probs, best_dir / "predictions_val.csv")
                    save_eval_artifacts(results, best_dir, epoch=best_epoch)
                    print(f"new best checkpoint saved: {best_dir}")
                model.train()

            if max_train_steps is not None and global_step >= max_train_steps:
                break

        save_checkpoint(model, cfg, output_dir, behaviour_texts)

    if skip_eval:
        if args.train_with_val:
            print("train_with_val: saved checkpoint trained on train+val without validation prediction/evaluation")
        else:
            print("skip_eval: saved checkpoint without validation prediction/evaluation")
        return

    if args.eval_only or last_results is None:
        assert val_loader is not None
        sample_ids, probs = predict(model, val_loader, device)
        eval_split = args.eval_split if args.eval_only else "val"
        pred_name = (
            f"predictions_{eval_split}.csv"
            if args.max_eval_samples is None
            else f"predictions_{eval_split}_first_{len(sample_ids)}.csv"
        )
        pred_path = output_dir / pred_name
        write_prediction_csv(sample_ids, probs, pred_path)
        if eval_split == "val":
            results = evaluate(str(val_annotation_path), str(pred_path))
            print("macro_average:", results["macro_average"])
            print(results["per_class_scores"])
        else:
            print(f"wrote {pred_path} ({len(sample_ids)} rows)")
        return

    print(f"best_macro_average: {best_score} (epoch {best_epoch})")
    print(f"best_checkpoint: {best_dir}")


if __name__ == "__main__":
    main()
