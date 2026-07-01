"""Behaviour Evidence Querying (BEQ) decoder for LVLM-based BBR.

BEQ replaces global video aggregation with category-conditioned evidence
retrieval: each of the 14 bodily behaviour categories is represented by a
semantic query that cross-attends to the LVLM video tokens and retrieves its own
relevant evidence. Each query is then scored by an independent binary head.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .constants import LABEL_COLUMNS
from .losses import AsymmetricLoss, build_multilabel_loss
from .modeling import forward_vlm_last_hidden


def _input_embeddings(vlm: nn.Module) -> nn.Module:
    if hasattr(vlm, "get_input_embeddings"):
        embeddings = vlm.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    base_model = getattr(vlm, "base_model", None)
    if base_model is not None and hasattr(base_model, "get_input_embeddings"):
        embeddings = base_model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    raise ValueError("Could not find input embeddings for behaviour query initialisation.")


def _maybe_gather_deepspeed_params(module: nn.Module):
    params = [param for param in module.parameters(recurse=False) if hasattr(param, "ds_id")]
    if not params:
        return nullcontext()
    try:
        import deepspeed
    except Exception:
        return nullcontext()
    return deepspeed.zero.GatheredParameters(params, modifier_rank=None)


@torch.no_grad()
def build_behaviour_query_init(
    vlm: nn.Module,
    processor: Any,
    behaviour_texts: list[str],
    hidden_dim: int,
) -> torch.Tensor:
    """Average Qwen token embeddings for behaviour descriptions."""
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None:
        raise ValueError("Processor does not expose a tokenizer.")
    encoded = tokenizer(
        behaviour_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    embeddings = _input_embeddings(vlm)
    emb_param = next(embeddings.parameters())
    input_ids = encoded["input_ids"].to(emb_param.device)
    mask = encoded["attention_mask"].to(emb_param.device).unsqueeze(-1)
    with _maybe_gather_deepspeed_params(embeddings):
        token_emb = embeddings(input_ids).float()
    query_init = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    if query_init.shape[-1] != hidden_dim:
        raise ValueError(f"Behaviour query dim {query_init.shape[-1]} != hidden dim {hidden_dim}")
    return query_init.cpu()


class IndependentBehaviourHeads(nn.Module):
    """Fourteen independent binary heads, one per behaviour query."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.2, head_hidden_dim: int = 0) -> None:
        super().__init__()
        heads = []
        for _ in range(num_classes):
            if head_hidden_dim and head_hidden_dim > 0:
                heads.append(
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, head_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(head_hidden_dim, 1),
                    )
                )
            else:
                heads.append(
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                )
        self.heads = nn.ModuleList(heads)

    def forward(self, behaviour_tokens: torch.Tensor) -> torch.Tensor:
        logits = [head(behaviour_tokens[:, idx, :]) for idx, head in enumerate(self.heads)]
        return torch.cat(logits, dim=-1)


class BEQClassifier(nn.Module):
    """Qwen3-VL backbone plus the Behaviour Evidence Querying decoder."""

    def __init__(
        self,
        vlm: nn.Module,
        hidden_dim: int,
        num_classes: int = 14,
        query_init: torch.Tensor | None = None,
        decoder_dim: int = 512,
        num_cross_attn_layers: int = 1,
        cross_attn_heads: int = 8,
        dropout: float = 0.2,
        ffn_mult: float = 4.0,
        label_correlation_layers: int = 0,
        label_correlation_heads: int = 2,
        head_hidden_dim: int = 0,
        loss_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vlm = vlm
        self.hidden_dim = int(hidden_dim)
        self.decoder_dim = int(decoder_dim)
        self.num_classes = int(num_classes)
        if query_init is None:
            query = torch.empty(num_classes, hidden_dim)
            nn.init.normal_(query, mean=0.0, std=0.02)
        else:
            if tuple(query_init.shape) != (num_classes, hidden_dim):
                raise ValueError(f"query_init must be {(num_classes, hidden_dim)}, got {tuple(query_init.shape)}")
            query = query_init.float().clone()
        # Behaviour-specific semantic queries (one per category).
        self.behaviour_queries = nn.Parameter(query)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.decoder_dim),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.decoder_dim),
        )

        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        ffn_dim = max(self.decoder_dim, int(self.decoder_dim * float(ffn_mult)))
        for _ in range(int(num_cross_attn_layers)):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.decoder_dim,
                    num_heads=int(cross_attn_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
            )
            self.cross_attn_norms.append(nn.LayerNorm(self.decoder_dim))
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, self.decoder_dim),
                    nn.Dropout(dropout),
                )
            )
            self.ffn_norms.append(nn.LayerNorm(self.decoder_dim))

        if int(label_correlation_layers) > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.decoder_dim,
                nhead=int(label_correlation_heads),
                dim_feedforward=ffn_dim,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.label_correlation = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(label_correlation_layers),
                norm=nn.LayerNorm(self.decoder_dim),
            )
        else:
            self.label_correlation = None

        self.heads = IndependentBehaviourHeads(
            hidden_dim=self.decoder_dim,
            num_classes=num_classes,
            dropout=float(dropout),
            head_hidden_dim=int(head_hidden_dim),
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

    def move_non_vlm_to(self, device: torch.device | str) -> None:
        for module in [
            self.query_proj,
            self.token_proj,
            self.cross_attn_layers,
            self.cross_attn_norms,
            self.ffn_layers,
            self.ffn_norms,
            self.heads,
            self.loss_fn,
        ]:
            module.to(device)
        if self.label_correlation is not None:
            self.label_correlation.to(device)
        self.behaviour_queries.data = self.behaviour_queries.data.to(device)

    def _decoder_dtype(self) -> torch.dtype:
        return self.behaviour_queries.dtype

    def _align_decoder(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.behaviour_queries.device != hidden.device:
            self.move_non_vlm_to(hidden.device)
        dtype = self._decoder_dtype()
        if hidden.dtype != dtype:
            hidden = hidden.to(dtype=dtype)
        return hidden

    def _visual_mask(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None, vision_token_mask: torch.Tensor | None) -> torch.Tensor | None:
        mask = vision_token_mask
        if mask is None:
            mask = attention_mask
        if mask is None:
            return None
        mask = mask.to(hidden.device).bool()
        if mask.shape[-1] != hidden.shape[1]:
            return attention_mask.to(hidden.device).bool() if attention_mask is not None else None
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any() and attention_mask is not None:
            mask[empty_rows] = attention_mask.to(hidden.device).bool()[empty_rows]
        return mask

    def forward(self, **inputs: Any) -> dict[str, torch.Tensor | None]:
        labels = inputs.pop("labels", None)
        inputs.pop("sample_ids", None)
        vision_token_mask = inputs.pop("vision_token_mask", None)
        attention_mask = inputs.get("attention_mask")

        hidden = forward_vlm_last_hidden(self.vlm, inputs)
        hidden = self._align_decoder(hidden)
        visual_mask = self._visual_mask(hidden, attention_mask, vision_token_mask)
        visual_hidden = self.token_proj(hidden)
        key_padding_mask = None if visual_mask is None else ~visual_mask

        batch_size = hidden.shape[0]
        behaviour_queries = self.query_proj(self.behaviour_queries)
        behaviour_tokens = behaviour_queries.unsqueeze(0).expand(batch_size, -1, -1)
        for attn, attn_norm, ffn, ffn_norm in zip(
            self.cross_attn_layers,
            self.cross_attn_norms,
            self.ffn_layers,
            self.ffn_norms,
        ):
            attended, _ = attn(
                query=behaviour_tokens,
                key=visual_hidden,
                value=visual_hidden,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            behaviour_tokens = attn_norm(behaviour_tokens + attended)
            behaviour_tokens = ffn_norm(behaviour_tokens + ffn(behaviour_tokens))

        if self.label_correlation is not None:
            behaviour_tokens = self.label_correlation(behaviour_tokens)

        logits = self.heads(behaviour_tokens)
        loss = None
        if labels is not None:
            if isinstance(self.loss_fn, nn.BCEWithLogitsLoss) and self.loss_fn.pos_weight is not None:
                self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)
            loss = self.loss_fn(logits.float(), labels.float().to(logits.device))
        return {"loss": loss, "logits": logits}

    def beq_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("vlm.")
        }

    def save_beq_head(self, output_dir: str | Path, behaviour_texts: list[str] | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.beq_state_dict(),
            "label_columns": LABEL_COLUMNS,
            "behaviour_texts": behaviour_texts,
        }
        torch.save(payload, output_dir / "beq_head.pt")

    def load_beq_head(self, checkpoint_dir: str | Path, map_location: str = "cpu") -> None:
        payload = torch.load(Path(checkpoint_dir) / "beq_head.pt", map_location=map_location)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("vlm.")]
        missing = [key for key in incompatible.missing_keys if not key.startswith("vlm.")]
        if unexpected or missing:
            raise RuntimeError(f"Could not load BEQ head: missing={missing}, unexpected={unexpected}")
