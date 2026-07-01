# BEQ: Behaviour Evidence Querying for Bodily Behaviour Recognition

Reference implementation for **BEQ: Behaviour Evidence Querying with Long-tail
Aware Asymmetric Learning for Bodily Behaviour Recognition**.

This repository contains the cleaned code release for the MultiMediate
**Bodily Behaviour Recognition (BBR)** task on the MPIIGroupInteraction
dataset. BBR is a 14-class multi-label video classification task evaluated
with category-wise macro-mAP.

## Method

The framework adapts **Qwen3-VL-30B-A3B-Instruct** with LoRA and contains two
core components:

- **Behaviour Evidence Querying (BEQ)**: each behaviour category owns a
  semantic query, initialized from the Qwen token embeddings of the category
  description. The queries cross-attend to LVLM video tokens and retrieve
  category-specific evidence before the independent binary heads.
- **Long-tail Aware Asymmetric Learning (LTAL)**: asymmetric loss, effective
  number based positive reweighting, and rare-positive sampling for the
  long-tailed multi-label distribution.

## Repository Layout

```text
.
├── beq/                         # core package
│   ├── constants.py             # label order, paths, view definitions
│   ├── config.py                # YAML config helpers
│   ├── data.py                  # BBRDataset and Qwen-VL collators
│   ├── modeling.py              # LVLM baseline, LoRA, backbone loading
│   ├── decoder.py               # BEQ classifier and decoder
│   ├── behaviour_descriptions.py# semantic query descriptions
│   ├── losses.py                # ASL, class-balanced weights, sampling
│   ├── evaluator.py             # local macro-mAP evaluator
│   └── utils.py
├── configs/
│   ├── beq_ltal.yaml            # BEQ + LTAL, single-process/device_map config
│   ├── beq_ltal_deepspeed.yaml  # BEQ + LTAL, multi-GPU DeepSpeed config
│   ├── deepspeed_zero2_bf16.json
│   └── deepspeed_zero3_bf16.json
├── sample_lists/                # train/val sample ids and labels
├── train_beq.py
├── train_beq_deepspeed.py
├── train_baseline.py
├── train_baseline_deepspeed.py
├── infer_beq.py
├── infer_baseline.py
├── evaluate_bbr.py
└── convert_zero3_to_lora_checkpoint.py
```

The 14 labels follow the official CSV order: `Settle`, `Legs crossed`,
`Groom`, `Hand-mouth`, `Fold arms`, `Leg movement`, `Scratch`, `Gesture`,
`Hand-face`, `Adjusting clothing`, `Fumble`, `Shrug`, `Stretching`, and
`Smearing hands`.

## Data and Weights

This release contains code, configs, and sample-id/label lists only. It does
not include MPIIGroupInteraction video clips or Qwen/model checkpoints.

By default, the code looks for clips under:

```text
dataset/bbr/
├── clips_train/00001-video.mp4
├── clips_val/00001-video.mp4
└── clips_test/00001-video.mp4
```

The original challenge archive may contain an extra nested directory such as
`clips_train/clips_train/`; both layouts are supported. You can also point to
an external data location with `BEQ_BBR_DATA_ROOT`.

Set the Qwen checkpoint path either in the YAML files (`model.backbone_path`)
or with `BEQ_BACKBONE_PATH`.

## Installation

```bash
conda create -n beq-bbr python=3.10 -y
conda activate beq-bbr
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default wheel
is not suitable. `flash-attn` is optional; the provided configs use `sdpa`.

## Quick Start

Run commands from the repository root.

```bash
export BEQ_BACKBONE_PATH=/path/to/Qwen3-VL-30B-A3B-Instruct
export BEQ_BBR_DATA_ROOT=/path/to/dataset/bbr
```

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python train_beq.py \
  --config configs/beq_ltal.yaml \
  --max-train-steps 2 \
  --max-train-samples 8 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --num-frames 2 \
  --max-pixels 262144
```

Train BEQ + LTAL for 5 epochs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_beq.py --config configs/beq_ltal.yaml
```

DeepSpeed training for 5 epochs:

```bash
deepspeed --num_gpus=4 train_beq_deepspeed.py --config configs/beq_ltal_deepspeed.yaml
```

Test inference from the validation-selected checkpoint:

```bash
python infer_beq.py \
  --config configs/beq_ltal.yaml \
  --checkpoint-dir outputs/acm_mm_bbr/beq_ltal/best-checkpoint \
  --split test \
  --output-csv submissions/beq_ltal_test.csv
```

Validation is enabled by default and the best checkpoint is selected by validation mAP. To re-run validation only:

```bash
python train_beq.py \
  --config configs/beq_ltal.yaml \
  --eval-only \
  --checkpoint-dir outputs/acm_mm_bbr/beq_ltal/best-checkpoint

python evaluate_bbr.py --prediction outputs/acm_mm_bbr/beq_ltal/best-checkpoint/predictions_val.csv
```

See [RUNNING.md](RUNNING.md) for a more detailed running guide.

## Configs

This release keeps only the final BEQ + LTAL training configs:

| Config | Use |
| --- | --- |
| `configs/beq_ltal.yaml` | Single-process / `device_map` training and inference |
| `configs/beq_ltal_deepspeed.yaml` | Multi-GPU DeepSpeed training |

## Checkpoint Format

BEQ checkpoints contain:

- `adapter/`: PEFT LoRA adapter for the LVLM backbone
- `beq_head.pt`: BEQ decoder and classification head
- `config.yaml`: resolved training config
- `metrics.json` and `predictions_val.csv`: validation artifacts for the best checkpoint
