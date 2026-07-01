"""LVLM backbone, LoRA setup, and the global-pooling baseline classifier."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .losses import AsymmetricLoss, build_multilabel_loss


def forward_vlm_last_hidden(vlm: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
    """Run the LVLM and return the final hidden states used by the classifier."""
    outputs = vlm(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
        logits_to_keep=1,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    hidden = getattr(outputs, "last_hidden_state", None)
    return hidden if hidden is not None else outputs[0]


@contextmanager
def _temporarily_disable_tied_weight_keys(model_cls: type[nn.Module]):
    sentinel = object()
    old_tied_keys = model_cls.__dict__.get("_tied_weights_keys", sentinel)
    old_adjust_tied_keys = model_cls.__dict__.get("_adjust_tied_keys_with_tied_pointers", sentinel)

    def _skip_tied_pointer_adjust(self, missing_keys):
        return None

    model_cls._tied_weights_keys = None
    model_cls._adjust_tied_keys_with_tied_pointers = _skip_tied_pointer_adjust
    try:
        yield
    finally:
        if old_tied_keys is sentinel:
            delattr(model_cls, "_tied_weights_keys")
        else:
            model_cls._tied_weights_keys = old_tied_keys
        if old_adjust_tied_keys is sentinel:
            delattr(model_cls, "_adjust_tied_keys_with_tied_pointers")
        else:
            model_cls._adjust_tied_keys_with_tied_pointers = old_adjust_tied_keys


def get_hidden_size(vlm: nn.Module) -> int:
    config = getattr(vlm, "config", None)
    candidates = [
        ("hidden_size",),
        ("text_config", "hidden_size"),
        ("llm_config", "hidden_size"),
        ("language_config", "hidden_size"),
    ]
    for path in candidates:
        cur: Any = config
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if isinstance(cur, int):
            return cur
    raise ValueError("Could not infer hidden size from model config")


class LVLMBodilyClassifier(nn.Module):
    """LVLM backbone plus a 14-dimensional multi-label classifier (baseline).

    This is the global-pooling baseline from the paper: the LVLM video token
    representations are aggregated by mean/last pooling into a single clip-level
    vector and fed to a 14-way head. The BEQ decoder (:mod:`beq.decoder`)
    replaces this global aggregation with category-conditioned evidence queries.
    """

    def __init__(
        self,
        vlm: nn.Module,
        hidden_dim: int,
        num_classes: int = 14,
        pooling: str = "last",
        dropout: float = 0.2,
        head_hidden_dim: int = 0,
        loss_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vlm = vlm
        self.pooling = pooling
        if head_hidden_dim and head_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        self.loss_fn = build_multilabel_loss(loss_config)

    def set_loss_pos_weight(self, pos_weight: torch.Tensor | list[float] | None) -> None:
        if isinstance(self.loss_fn, AsymmetricLoss):
            self.loss_fn.set_pos_weight(pos_weight)
            return
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            self.loss_fn.pos_weight = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32)
            return
        raise TypeError(f"Loss does not support pos_weight: {type(self.loss_fn).__name__}")

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling == "last":
            if attention_mask is None:
                return hidden[:, -1, :]
            attention_mask = attention_mask.to(hidden.device)
            seq_lens = attention_mask.sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, seq_lens]
        if self.pooling == "mean":
            if attention_mask is None:
                return hidden.mean(dim=1)
            attention_mask = attention_mask.to(hidden.device)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        raise ValueError(f"Unknown pooling: {self.pooling}")

    def _align_classifier(self, feat: torch.Tensor) -> torch.Tensor:
        classifier_param = next(self.classifier.parameters())
        if classifier_param.device != feat.device:
            self.classifier.to(feat.device)
            classifier_param = next(self.classifier.parameters())
        if classifier_param.is_floating_point() and feat.dtype != classifier_param.dtype:
            feat = feat.to(dtype=classifier_param.dtype)
        return feat

    def forward(self, **inputs: Any) -> dict[str, torch.Tensor | None]:
        labels = inputs.pop("labels", None)
        inputs.pop("sample_ids", None)
        inputs.pop("vision_token_mask", None)
        attention_mask = inputs.get("attention_mask")

        hidden = forward_vlm_last_hidden(self.vlm, inputs)
        feat = self._pool(hidden, attention_mask)
        logits = self.classifier(self._align_classifier(feat))

        loss = None
        if labels is not None:
            if isinstance(self.loss_fn, nn.BCEWithLogitsLoss) and self.loss_fn.pos_weight is not None:
                self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)
            loss = self.loss_fn(logits.float(), labels.float().to(logits.device))
        return {"loss": loss, "logits": logits}

    def save_classifier(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.classifier.state_dict(), output_dir / "classifier.pt")

    def load_classifier(self, checkpoint_dir: str | Path, map_location: str = "cpu") -> None:
        state = torch.load(Path(checkpoint_dir) / "classifier.pt", map_location=map_location)
        self.classifier.load_state_dict(state)


def apply_lora(vlm: nn.Module, config: dict[str, Any]) -> nn.Module:
    """Apply LM + ViT LoRA using PEFT."""

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("Install peft to enable LoRA: pip install peft") from exc

    lora_cfg = config.get("lora", {})
    target_modules = lora_cfg.get(
        "target_modules",
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
        ],
    )
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=target_modules,
        modules_to_save=lora_cfg.get("modules_to_save", None),
    )
    model = get_peft_model(vlm, peft_cfg)
    if bool(lora_cfg.get("print_trainable_parameters", True)):
        model.print_trainable_parameters()
    return model


def load_vlm_and_processor(config: dict[str, Any]) -> tuple[nn.Module, Any]:
    """Load backbone and processor.

    The default class names match modern Qwen-VL checkpoints. Keep this function
    small so model-specific fixes are localized after the actual checkpoint is
    downloaded.
    """

    from transformers import AutoConfig, AutoProcessor

    model_cfg = config["model"]
    model_path = os.environ.get("BEQ_BACKBONE_PATH") or model_cfg["backbone_path"]
    model_class = model_cfg.get("model_class", "AutoModelForImageTextToText")
    dtype_name = model_cfg.get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)

    transformers_mod = __import__("transformers", fromlist=[model_class])
    model_cls = getattr(transformers_mod, model_class)

    hf_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if "tie_word_embeddings" in model_cfg:
        hf_config.tie_word_embeddings = bool(model_cfg["tie_word_embeddings"])

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    load_kwargs: dict[str, Any] = {
        "config": hf_config,
        "torch_dtype": dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
    }
    if model_cfg.get("device_map"):
        load_kwargs["device_map"] = model_cfg["device_map"]
    if model_cfg.get("max_memory"):
        load_kwargs["max_memory"] = model_cfg["max_memory"]
    if model_cfg.get("attn_implementation"):
        load_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    tied_context = (
        _temporarily_disable_tied_weight_keys(model_cls)
        if getattr(hf_config, "tie_word_embeddings", False) is False
        else nullcontext()
    )
    with tied_context:
        vlm = model_cls.from_pretrained(model_path, **load_kwargs)
    if bool(model_cfg.get("gradient_checkpointing", True)):
        vlm.gradient_checkpointing_enable()
    if bool(model_cfg.get("apply_lora", True)):
        vlm = apply_lora(vlm, config)
    return vlm, processor
