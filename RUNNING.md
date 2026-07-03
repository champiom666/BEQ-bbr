# Running Guide

This guide describes how to train BEQ and generate a BBR submission file. All commands are intended to be run from the repository root.

## 1. Environment Setup

Python 3.10 is recommended:

```bash
conda create -n beq-bbr python=3.10 -y
conda activate beq-bbr
pip install -r requirements.txt
```

If the default PyTorch wheel does not match your server's CUDA version, install the appropriate `torch` build first, then install the remaining packages from `requirements.txt`. `flash-attn` is optional; the configs use `sdpa` attention by default.

## 2. Model and Data Preparation

Set the video data and Qwen3-VL checkpoint paths with environment variables:

```bash
export BEQ_BACKBONE_PATH=/path/to/Qwen3-VL-30B-A3B-Instruct
export BEQ_BBR_DATA_ROOT=/path/to/dataset/bbr
```

You can also edit `model.backbone_path` directly in `configs/*.yaml`.

The data directory supports either of the following layouts:

```text
dataset/bbr/
├── clips_train/00001-video.mp4
├── clips_val/00001-video.mp4
└── clips_test/00001-video.mp4
```

If the official archive extracts to an extra nested directory, that layout is also supported:

```text
dataset/bbr/
├── clips_train/clips_train/00001-video.mp4
├── clips_val/clips_val/00001-video.mp4
└── clips_test/clips_test/00001-video.mp4
```

Video filenames use a zero-padded five-digit `sample_id` plus the camera-view suffix: the frontal view is `00001-video.mp4`, and the two side views are `00001-video1.mp4` and `00001-video2.mp4`. The default configs use only the frontal view, `views: [0]`.

## 3. Quick Smoke Test

Run a tiny job first to verify the environment, model path, and data path. Validation is enabled by default:

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

If successful, the resolved config and validation-selected checkpoint will be written under `outputs/acm_mm_bbr/beq_ltal/`.

## 4. Train BEQ + LTAL

Single-process / `device_map` training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_beq.py --config configs/beq_ltal.yaml
```

DeepSpeed training:

```bash
deepspeed --num_gpus=4 train_beq_deepspeed.py --config configs/beq_ltal_deepspeed.yaml
```

If GPU memory is insufficient, first reduce these parameters:

```bash
--batch-size 2 --gradient-accumulation-steps 8
```

## 5. Validation Evaluation

Validation is enabled by default (`skip_eval: false`), and the best checkpoint is selected by validation mAP. To re-run validation only, use:

```bash
python train_beq.py \
  --config configs/beq_ltal.yaml \
  --eval-only \
  --checkpoint-dir outputs/acm_mm_bbr/beq_ltal/best-checkpoint

python evaluate_bbr.py \
  --annotation sample_lists/val_samples.csv \
  --prediction outputs/acm_mm_bbr/beq_ltal/best-checkpoint/predictions_val.csv
```

## 6. Generate the Test Submission File

Run inference from the validation-selected BEQ checkpoint:

```bash
python infer_beq.py \
  --config configs/beq_ltal.yaml \
  --checkpoint-dir outputs/acm_mm_bbr/beq_ltal/best-checkpoint \
  --split test \
  --output-csv submissions/beq_ltal_test.csv
```

The output CSV follows the official 14-label column order. The first column is `sample_id`, followed by one probability column per class.

## 7. Checkpoint Layout

The best BEQ checkpoint is saved under the training output directory:

```text
outputs/acm_mm_bbr/beq_ltal/best-checkpoint/
├── adapter/       # LoRA adapter
├── beq_head.pt    # BEQ decoder and classification head
├── config.yaml
├── metrics.json
└── predictions_val.csv
```
