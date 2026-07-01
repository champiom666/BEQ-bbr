"""Dataset, prompt construction, and Qwen-VL collators for BBR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import CLIP_DIRS, LABEL_COLUMNS, SAMPLE_LIST_DIR, VIEW_NAMES


def sample_id_to_video_name(sample_id: int | str, view_suffix: str) -> str:
    return f"{int(sample_id):05d}-video{view_suffix}.mp4"


def parse_view_spec(views: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Convert view ids to BBR filename suffixes.

    View 0 is the frontal file ``xxxxx-video.mp4``; views 1/2 are
    ``xxxxx-video1.mp4`` and ``xxxxx-video2.mp4``.
    """
    if views is None:
        return [""]
    if isinstance(views, str):
        raw = [views] if views == "012" else [v.strip() for v in views.split(",")]
    else:
        raw = [str(v).strip() for v in views]

    if raw == ["012"]:
        raw = ["0", "1", "2"]

    view_map = {
        "0": "",
        "front": "",
        "frontal": "",
        "": "",
        "1": "1",
        "left": "1",
        "left_side": "1",
        "2": "2",
        "right": "2",
        "right_side": "2",
    }
    suffixes: list[str] = []
    for item in raw:
        if not item:
            continue
        key = item.lower().replace("-", "_").replace(" ", "_")
        if key not in view_map:
            raise ValueError(f"Unknown BBR view spec: {item!r}. Use 0, 1, 2, or 012.")
        suffix = view_map[key]
        if suffix not in suffixes:
            suffixes.append(suffix)
    return suffixes or [""]


def build_bbr_prompt(class_names: list[str] | None = None) -> str:
    classes = class_names or LABEL_COLUMNS
    label_lines = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(classes))
    return (
        "You are analyzing the bodily behavior of one annotated participant "
        "during a 2.13-second, 64-frame clip from a multi-person conversation.\n"
        "Use the provided camera view or synchronized camera views of the SAME "
        "participant to judge visible body motion and posture.\n\n"
        "Behavior categories are multi-label; any subset can be present, "
        "including none:\n"
        f"{label_lines}\n\n"
        "Determine which behaviors are present over the entire clip. "
        "Your fused visual-language representation will be consumed by a "
        "classification head, so answer briefly."
    )


@dataclass(frozen=True)
class BBRExample:
    sample_id: int
    video_paths: list[str]
    labels: torch.Tensor | None


class BBRDataset(Dataset):
    """Bodily Behaviour Recognition samples with configurable video views."""

    def __init__(
        self,
        split: str,
        sample_csv: str | Path | None = None,
        clip_dir: str | Path | None = None,
        use_views: list[str] | None = None,
        require_all_views: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.sample_csv = Path(sample_csv) if sample_csv else SAMPLE_LIST_DIR / f"{split}_samples.csv"
        self.clip_dir = Path(clip_dir) if clip_dir else CLIP_DIRS[split]
        self.use_views = use_views or [""]
        self.require_all_views = require_all_views

        if not self.sample_csv.exists():
            if split == "test":
                self.rows = self._build_test_rows_from_clips()
            else:
                raise FileNotFoundError(self.sample_csv)
        else:
            self.rows = pd.read_csv(self.sample_csv)

        if "sample_id" not in self.rows.columns:
            raise ValueError(f"{self.sample_csv} must contain sample_id")

        if split != "test":
            missing = [c for c in LABEL_COLUMNS if c not in self.rows.columns]
            if missing:
                raise ValueError(f"{self.sample_csv} missing label columns: {missing}")

        self.rows = self.rows.sort_values("sample_id").reset_index(drop=True)
        self._validate_or_filter_views()

    def _build_test_rows_from_clips(self) -> pd.DataFrame:
        if not self.clip_dir.exists():
            raise FileNotFoundError(self.clip_dir)
        sample_ids = set()
        for path in self.clip_dir.glob("*-video*.mp4"):
            prefix = path.name.split("-", 1)[0]
            if prefix.isdigit():
                sample_ids.add(int(prefix))
        return pd.DataFrame({"sample_id": sorted(sample_ids)})

    def _validate_or_filter_views(self) -> None:
        keep_indices: list[int] = []
        missing_count = 0
        for idx, row in self.rows.iterrows():
            sid = row["sample_id"]
            paths = [self.clip_dir / sample_id_to_video_name(sid, v) for v in self.use_views]
            missing = [p for p in paths if not p.exists()]
            if missing:
                missing_count += 1
                if self.require_all_views:
                    raise FileNotFoundError(f"Missing videos for sample_id={sid}: {missing[:3]}")
                continue
            keep_indices.append(idx)
        if len(keep_indices) != len(self.rows):
            self.rows = self.rows.iloc[keep_indices].reset_index(drop=True)
            print(f"[BBRDataset] filtered {missing_count} rows with missing videos")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> BBRExample:
        row = self.rows.iloc[idx]
        sid = int(row["sample_id"])
        video_paths = [str(self.clip_dir / sample_id_to_video_name(sid, v)) for v in self.use_views]
        labels = None
        if self.split != "test":
            labels = torch.tensor(row[LABEL_COLUMNS].astype(float).values, dtype=torch.float32)
        return BBRExample(sample_id=sid, video_paths=video_paths, labels=labels)


class QwenVLVideoCollator:
    """Build Qwen-VL style multi-video batches.

    The actual processor API differs slightly across Qwen-VL generations. This
    collator keeps the Qwen-specific calls isolated so training code stays clean.
    """

    def __init__(
        self,
        processor: Any,
        num_frames: int = 16,
        fps: int = 8,
        max_pixels: int = 262144,
        add_vision_id: bool = True,
    ) -> None:
        self.processor = processor
        self.num_frames = num_frames
        self.fps = fps
        self.max_pixels = max_pixels
        self.add_vision_id = add_vision_id
        self.prompt = build_bbr_prompt()

        try:
            from qwen_vl_utils import process_vision_info
            from qwen_vl_utils import vision_process
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise ImportError(
                "qwen_vl_utils is required for video collation. Install the "
                "Qwen-VL utility package that matches your Qwen3-VL checkpoint."
            ) from exc
        image_factor = 14 * getattr(vision_process, "SPATIAL_MERGE_SIZE", 2)
        required_video_tokens = int((self.max_pixels + image_factor * image_factor - 1) // (image_factor * image_factor))
        if required_video_tokens > getattr(vision_process, "VIDEO_MAX_TOKEN_NUM", 768):
            vision_process.VIDEO_MAX_TOKEN_NUM = required_video_tokens
        self.process_vision_info = process_vision_info

    def _messages(self, example: BBRExample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for view_name, video_path in zip(VIEW_NAMES, example.video_paths):
            content.append(
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": self.num_frames,
                    "max_pixels": self.max_pixels,
                }
            )
        content.append({"type": "text", "text": self.prompt})
        return [{"role": "user", "content": content}]

    def _split_video_metadata(
        self,
        video_inputs: Any,
    ) -> tuple[Any, list[dict[str, Any]] | None]:
        if video_inputs is None:
            return None, None

        videos = []
        metadata = []
        has_metadata = False
        for item in video_inputs:
            if isinstance(item, tuple) and len(item) == 2:
                video, video_metadata = item
                videos.append(video)
                metadata.append(video_metadata)
                has_metadata = True
            else:
                videos.append(item)

        return videos, metadata if has_metadata else None

    def __call__(self, examples: list[BBRExample]) -> dict[str, Any]:
        texts = []
        all_images = []
        all_videos = []
        all_video_metadata = []
        sample_ids = []
        labels = []

        for ex in examples:
            messages = self._messages(ex)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_vision_id=self.add_vision_id,
            )
            image_inputs, video_inputs = self.process_vision_info(
                messages,
                return_video_metadata=True,
            )
            video_inputs, video_metadata = self._split_video_metadata(video_inputs)
            texts.append(text)
            all_images.append(image_inputs)
            all_videos.append(video_inputs)
            if video_metadata is not None:
                all_video_metadata.append(video_metadata)
            sample_ids.append(ex.sample_id)
            if ex.labels is not None:
                labels.append(ex.labels)

        processor_kwargs: dict[str, Any] = {}
        if all_video_metadata:
            processor_kwargs["video_metadata"] = all_video_metadata
            processor_kwargs["return_metadata"] = True
            processor_kwargs["do_sample_frames"] = False

        inputs = self.processor(
            text=texts,
            images=all_images if any(v is not None for v in all_images) else None,
            videos=all_videos,
            padding=True,
            return_tensors="pt",
            **processor_kwargs,
        )
        inputs.pop("video_metadata", None)
        inputs["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        if labels:
            inputs["labels"] = torch.stack(labels)
        return inputs


DEFAULT_VISUAL_TOKEN_CANDIDATES = [
    "<|video_pad|>",
    "<|image_pad|>",
    "<|vision_pad|>",
    "<video>",
    "<image>",
]


def infer_visual_token_ids(processor: Any, candidates: list[str] | None = None) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return []

    unk_id = getattr(tokenizer, "unk_token_id", None)
    token_ids: list[int] = []
    for token in candidates or DEFAULT_VISUAL_TOKEN_CANDIDATES:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unk_id:
            continue
        if isinstance(token_id, int) and token_id >= 0 and token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


class BEQVideoCollator(QwenVLVideoCollator):
    """Qwen-VL video collator that also returns a visual-token mask.

    Qwen-VL processors usually expand videos into repeated visual placeholder
    tokens. The mask lets the BEQ decoder attend primarily to video tokens. If
    the tokenizer does not expose visual token ids, the model falls back to all
    non-padding tokens.
    """

    def __init__(
        self,
        *args: Any,
        visual_token_candidates: list[str] | None = None,
        fallback_to_attention_mask: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.visual_token_ids = infer_visual_token_ids(self.processor, visual_token_candidates)
        self.fallback_to_attention_mask = fallback_to_attention_mask

    def __call__(self, examples):
        inputs = super().__call__(examples)
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if input_ids is None:
            return inputs

        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.visual_token_ids:
            vision_mask |= input_ids.eq(int(token_id))

        if self.fallback_to_attention_mask and attention_mask is not None:
            empty_rows = ~vision_mask.any(dim=1)
            if empty_rows.any():
                vision_mask[empty_rows] = attention_mask[empty_rows].bool()

        inputs["vision_token_mask"] = vision_mask
        return inputs
