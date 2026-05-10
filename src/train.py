"""End-to-end training for the Gemma 4 E4B classifier head.

Port from `gateguard-suite/gemma3_classifier_lora.py` with:
  - QLoRA (bnb 4-bit nf4) support — pass `bnb_config` for cloud GPU runs.
  - Text-only model load (drops vision/audio towers post-load).
  - Pooling fix from `src/model.py` (last-position under left-padding).
  - Single-CSV input with stratified split done internally.
  - Save format matches gateguard's `save_artifacts` so artifacts are
    interchangeable: `adapter/` + `classifier_head.pt` + `label_map.json`
    + `inference_config.json`.

Defaults track the gateguard 270m-it baseline recipe (r=16, alpha=32, lr=2e-4,
all 7 LoRA targets, etc.). The hackathon goal isn't a strict head-to-head
comparison so these can be swept later if time permits.

Entrypoint: `run_training(...)`. Designed to be called from
`notebooks/02_train_colab.ipynb`.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, set_seed

from src.model import CausalLMWithClassifier, load_text_only_gemma4

DEFAULT_LABELS: tuple[str, ...] = ("allow", "warn", "block")
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


# ----------------------------- Data ----------------------------- #

class _TokenizedDataset(Dataset):
    """Pre-tokenized rows held as tensors. Small enough that we materialize."""

    def __init__(self, encodings: dict[str, torch.Tensor], labels: torch.Tensor):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def _stratified_split(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    by_label: dict = defaultdict(list)
    for i, lab in df[label_col].items():
        by_label[lab].append(i)
    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in by_label.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test : n_test + n_val])
        train_idx.extend(idxs[n_test + n_val :])
    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def _build_label_maps(labels: Sequence[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def _normalize_label(x, label2id: dict[str, int]) -> int:
    if isinstance(x, str):
        s = x.strip().lower()
        if s in label2id:
            return label2id[s]
        if s.isdigit() and int(s) in label2id.values():
            return int(s)
    elif isinstance(x, (int, np.integer)):
        if int(x) in label2id.values():
            return int(x)
    raise ValueError(f"Label {x!r} not in {list(label2id)} or 0..{len(label2id) - 1}")


def _compute_class_weights(train_label_ids: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized so the mean weight is 1."""
    counts = Counter(train_label_ids)
    total = sum(counts.values())
    raw = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    w = torch.tensor(raw, dtype=torch.float32)
    return w * (num_classes / w.sum())


def _tokenize(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


# ----------------------------- LoRA ----------------------------- #

def _attach_lora(
    base_lm: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
    is_kbit: bool,
) -> nn.Module:
    from peft import LoraConfig, get_peft_model
    if is_kbit:
        from peft import prepare_model_for_kbit_training
        base_lm = prepare_model_for_kbit_training(base_lm)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
    )
    return get_peft_model(base_lm, lora_cfg)


# ----------------------------- Save ----------------------------- #

def _save_artifacts(
    out_dir: str,
    model: CausalLMWithClassifier,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    *,
    model_id: str,
    max_length: int,
    is_lora: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)

    base_dir = os.path.join(out_dir, "base_lm")
    os.makedirs(base_dir, exist_ok=True)
    # PEFT save_pretrained writes adapter_config.json + adapter_model.{bin|safetensors}
    model.base_lm.save_pretrained(base_dir, safe_serialization=True)

    torch.save(model.classifier.state_dict(), os.path.join(out_dir, "classifier_head.pt"))

    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": {int(i): l for i, l in id2label.items()}}, f, indent=2)

    with open(os.path.join(out_dir, "inference_config.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "base_model_id": model_id,
            "max_length": max_length,
            "labels": list(label2id.keys()),
            "pad_token_id": int(tokenizer.pad_token_id),
            "model_type": "lora" if is_lora else "base",
            "is_merged": False,
            "hidden_size": model.classifier.in_features,
            "num_labels": model.classifier.out_features,
        }, f, indent=2)

    print(f"Saved artifacts to: {out_dir}")


# ----------------------------- Eval ----------------------------- #

def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = torch.nan_to_num(out.logits, nan=0.0, posinf=0.0, neginf=0.0)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            losses.append(out.loss.item() if out.loss is not None else float("nan"))
    model.train()
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    return {
        "eval_loss": float(np.nanmean(losses)) if losses else float("nan"),
        "eval_accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "eval_f1_macro": float(f1_score(y_true, y_pred, average="macro")) if y_true.size else 0.0,
    }


# ----------------------------- Main loop ----------------------------- #

@dataclass
class TrainConfig:
    model_id: str
    csv_path: str
    out_dir: str
    labels: tuple[str, ...] = DEFAULT_LABELS
    text_col: str = "text"
    label_col: str = "label"
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_length: int = 256
    epochs: int = 3
    batch: int = 8
    grad_accum: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    seed: int = 42


def run_training(cfg: TrainConfig, *, bnb_config=None) -> dict[str, float]:
    """End-to-end training. Returns final eval metrics on the held-out val split."""
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Device + dtype
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Data
    df = pd.read_csv(cfg.csv_path)
    if cfg.text_col not in df.columns or cfg.label_col not in df.columns:
        raise ValueError(f"CSV must contain {cfg.text_col!r} and {cfg.label_col!r}; got {list(df.columns)}")
    df = df[~df[cfg.text_col].isna()]
    df = df[df[cfg.text_col].astype(str).str.strip() != ""].reset_index(drop=True)

    label2id, id2label = _build_label_maps(cfg.labels)
    df["label_id"] = df[cfg.label_col].apply(lambda x: _normalize_label(x, label2id))

    train_df, val_df, test_df = _stratified_split(
        df, "label_id", cfg.val_ratio, cfg.test_ratio, cfg.seed
    )
    print(f"Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    print(f"Train label dist: {dict(Counter(train_df['label_id'].tolist()))}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_enc = _tokenize(tokenizer, train_df[cfg.text_col].tolist(), cfg.max_length)
    val_enc = _tokenize(tokenizer, val_df[cfg.text_col].tolist(), cfg.max_length)
    train_ds = _TokenizedDataset(train_enc, torch.tensor(train_df["label_id"].tolist(), dtype=torch.long))
    val_ds = _TokenizedDataset(val_enc, torch.tensor(val_df["label_id"].tolist(), dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False)

    # Model
    base_lm = load_text_only_gemma4(
        cfg.model_id,
        dtype=dtype,
        device_map={"": device},
        bnb_config=bnb_config,
    )
    base_lm.config.pad_token_id = tokenizer.pad_token_id
    base_lm = _attach_lora(
        base_lm,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        is_kbit=bnb_config is not None,
    )

    class_weights = None
    if cfg.use_class_weights:
        class_weights = _compute_class_weights(
            train_df["label_id"].tolist(), num_classes=len(label2id)
        ).to(device=device, dtype=torch.float32)

    model = CausalLMWithClassifier(
        base_lm=base_lm,
        num_labels=len(label2id),
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
    )
    # The classifier head should be in the same dtype as the base for stability
    base_dtype = next(p for p in model.base_lm.parameters() if p.requires_grad).dtype
    model.classifier.to(dtype=base_dtype, device=device)
    model.to(device)

    # Optim + schedule
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    num_steps = max(1, len(train_loader) // cfg.grad_accum) * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup_ratio * num_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # cosine decay to 0
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # Training loop
    model.train()
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        running_loss, running_n = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                print(f"[epoch {epoch} step {step}] non-finite loss: {loss.item()}; clamping")
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)
            (loss / cfg.grad_accum).backward()
            running_loss += loss.item()
            running_n += 1

            if (step + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=cfg.max_grad_norm,
                )
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    avg = running_loss / max(1, running_n)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"[epoch {epoch}] step {global_step} | lr {lr_now:.2e} | loss {avg:.4f}")
                    running_loss, running_n = 0.0, 0

        metrics = _evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] val: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # Final eval + save
    final_metrics = _evaluate(model, val_loader, device)
    _save_artifacts(
        cfg.out_dir,
        model,
        tokenizer,
        label2id,
        id2label,
        model_id=cfg.model_id,
        max_length=cfg.max_length,
        is_lora=True,
    )
    # Also persist the test split for downstream eval
    test_df.to_csv(os.path.join(cfg.out_dir, "test_split.csv"), index=False)
    return final_metrics


if __name__ == "__main__":
    raise SystemExit("Run via notebooks/02_train_colab.ipynb or import run_training directly.")
