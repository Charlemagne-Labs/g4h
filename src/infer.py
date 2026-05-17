"""Inference for a saved Gemma 4 classifier-head artifact.

Load a trained adapter + head from an artifact directory and run predictions.
Designed to be Mac-friendly (bf16, no bitsandbytes required) so the same
saved artifact can be evaluated locally on Apple Silicon after training on
Colab — `bnb_config=None` is the default and the model loads in bf16.

Public API:
  - load_for_inference(out_dir, device=None, bnb_config=None) -> InferenceBundle
  - predict_one(bundle, text) -> (label, scores_dict)

CLI:
  python -m src.infer --out-dir runs/gemma4-e4b-cls \\
      --text 'url:ip_hostname:{"hostname":"1.2.3.4"} security:no_https:{}'
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from src.model import CausalLMWithClassifier, load_text_only_gemma4


@dataclass
class InferenceBundle:
    model: CausalLMWithClassifier
    tokenizer: object
    id2label: dict[int, str]
    label2id: dict[str, int]
    max_length: int


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_for_inference(
    out_dir: str,
    device: torch.device | str | None = None,
    bnb_config=None,
) -> InferenceBundle:
    """Reconstruct a trained classifier from a saved artifact directory.

    The artifact directory must match `src.train._save_artifacts` output:
        out_dir/
          ├── base_lm/ (PEFT adapter)
          ├── classifier_head.pt
          ├── inference_config.json
          ├── label_map.json
          └── tokenizer files

    On Mac (no CUDA), pass bnb_config=None — the model loads in bf16 from
    the base checkpoint and gets the LoRA adapter applied on top.
    """
    with open(os.path.join(out_dir, "inference_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(out_dir, "label_map.json")) as f:
        lm = json.load(f)
    label2id = {k: int(v) for k, v in lm["label2id"].items()}
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    # Load tokenizer. The saved tokenizer files can break across transformers
    # versions (e.g. Gemma 4's `extra_special_tokens` field shifted from list
    # to dict between minor versions). The tokenizer is identical to the
    # base model's anyway since we didn't add tokens — fall back to base on
    # any load error.
    try:
        tokenizer = AutoTokenizer.from_pretrained(out_dir)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "tokenizer load from %s failed (%s); falling back to %s",
            out_dir, type(e).__name__, cfg["base_model_id"],
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_id"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dev = torch.device(device) if device else _pick_device()
    dtype = torch.bfloat16 if dev.type != "cpu" else torch.float32

    base = load_text_only_gemma4(
        cfg["base_model_id"],
        dtype=dtype,
        device_map={"": dev},
        bnb_config=bnb_config,
    )
    base = PeftModel.from_pretrained(base, os.path.join(out_dir, "base_lm"))

    model = CausalLMWithClassifier(base, num_labels=cfg["num_labels"])
    head_state = torch.load(
        os.path.join(out_dir, "classifier_head.pt"),
        map_location="cpu",
    )
    model.classifier.load_state_dict(head_state)
    # Match head dtype to base for inference-mode forward consistency
    base_dtype = next(p for p in model.base_lm.parameters()).dtype
    model.classifier.to(dtype=base_dtype, device=dev)
    model.to(dev)
    model.eval()

    return InferenceBundle(
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        label2id=label2id,
        max_length=cfg["max_length"],
    )


def predict_one(bundle: InferenceBundle, text: str) -> tuple[str, dict[str, float]]:
    """Predict a single text. Returns (top label, per-label softmax scores)."""
    enc = bundle.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=bundle.max_length,
    )
    device = next(bundle.model.classifier.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = bundle.model(**enc)
    logits = out.logits[0]
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    cls_id = int(probs.argmax())
    return bundle.id2label[cls_id], {bundle.id2label[i]: float(p) for i, p in enumerate(probs)}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Predict a label for one input string.")
    parser.add_argument("--out-dir", default="runs/gemma4-e4b-cls", help="Artifact directory")
    parser.add_argument("--text", required=True, help="Indicator string to classify")
    args = parser.parse_args()

    bundle = load_for_inference(args.out_dir)
    label, scores = predict_one(bundle, args.text)
    print(json.dumps({"text": args.text, "label": label, "scores": scores}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
