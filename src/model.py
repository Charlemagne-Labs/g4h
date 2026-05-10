"""Model wrapper for the Gemma 4 E4B classifier head.

Direct port from `gateguard-suite/gemma3_classifier_lora.py:54-109` with three
changes informed by Phase 1 (see `docs/phase1_smoke_test.md`):

  1. Use the lm_head bypass via `_get_inner_base_model` instead of
     `output_hidden_states=True`. Saves ~196 MB per forward in bf16 at T=384.
  2. Pool from the last position (`last_hidden[:, -1, :]`) instead of the
     gateguard `attention_mask.sum(dim=1) - 1` formula, which was the right
     formula for right-padding but the tokenizer is left-padded. The fix is
     correct for our left-padded inputs and gets us back the signal that was
     pooled from the pad region in the baseline.
  3. Drop the Gemma 4 vision/audio towers + multimodal embedders right after
     load so they don't sit in RAM. ~30-40% memory reduction. The composite
     `Gemma4Model.forward` would only access them with `pixel_values` /
     `audio_features` arguments, which we never pass — but freeing the params
     means PEFT/QLoRA don't need to manage them either.

Phase 1 confirmed: `Gemma4TextModel.forward(input_ids, attention_mask)` returns
`last_hidden_state` bit-exactly equal to
`Gemma4ForCausalLM(..., output_hidden_states=True).hidden_states[-1]`. So PLE
composes upstream of the final hidden state in a way that's transparent for
last-token-pooled classification.
"""
from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput

# Gemma 4 ships these multimodal siblings of `language_model` even on text-only
# checkpoints. We never call them; freeing them post-load cuts steady-state
# memory by ~30-40%.
_MULTIMODAL_ATTRS = ("vision_tower", "audio_tower", "embed_vision", "embed_audio")


def _get_inner_base_model(m: nn.Module) -> nn.Module:
    """Walk past PEFT wrappers, the CausalLM head, and any multimodal-style
    composite container to reach the text backbone (the module exposing
    `.layers` whose `forward()` returns `last_hidden_state`).

    Layouts handled:
      - PEFT:                     m.base_model.model -> ForCausalLM
      - Single-stack (Gemma 3):   ForCausalLM.model -> backbone with .layers
      - Multimodal (Gemma 4):     ForCausalLM.model -> Gemma4Model (composite)
                                  -> .language_model -> Gemma4TextModel with .layers
    """
    cur = m
    if hasattr(cur, "base_model") and hasattr(cur.base_model, "model"):
        cur = cur.base_model.model
    if hasattr(cur, "model"):
        cur = cur.model
    if not hasattr(cur, "layers"):
        for attr in ("language_model", "text_model"):
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
                break
    return cur


def load_text_only_gemma4(
    model_id: str,
    dtype: torch.dtype,
    device_map: dict | str = "auto",
    bnb_config=None,
) -> AutoModelForCausalLM:
    """Load a Gemma 4 multimodal checkpoint and free the vision + audio towers.

    Peak memory during load is the full model size; steady state after this
    function returns is text-only. On Colab + bnb 4-bit, the vision/audio
    weights still get quantized before being deleted — wasteful for that brief
    window but tolerable, and keeps the load path identical between Mac and
    Colab.
    """
    kwargs: dict = {"dtype": dtype, "device_map": device_map}
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False  # disable kv-cache for training

    composite = model.model if hasattr(model, "model") else model
    for attr in _MULTIMODAL_ATTRS:
        if hasattr(composite, attr):
            delattr(composite, attr)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


class CausalLMWithClassifier(nn.Module):
    """Gemma 4 text backbone + linear classification head over last-token pooling.

    Forward returns `SequenceClassifierOutput` (loss + logits) so it slots into
    HF Trainer or any standard training loop.
    """

    def __init__(
        self,
        base_lm: nn.Module,
        num_labels: int,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_lm = base_lm
        self.num_labels = num_labels
        self.label_smoothing = float(label_smoothing)
        text_cfg = getattr(base_lm.config, "text_config", base_lm.config)
        hidden = getattr(text_cfg, "hidden_size", None) or getattr(text_cfg, "hidden_dim")
        self.classifier = nn.Linear(hidden, num_labels)
        # buffer so it moves with .to(device) / .to(dtype)
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_unused,
    ) -> SequenceClassifierOutput:
        inner = _get_inner_base_model(self.base_lm)
        out = inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # Left-padded: the last real token is always at position -1.
        pooled = out.last_hidden_state[:, -1, :].to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            weight = self.class_weights.float() if self.class_weights is not None else None
            loss = F.cross_entropy(
                logits.float(),
                labels,
                weight=weight,
                label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
            )

        return SequenceClassifierOutput(loss=loss, logits=logits)
