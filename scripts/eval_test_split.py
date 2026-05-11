"""Full evaluation on the held-out test split, with a duplicate-vs-novel breakout.

The test split is saved alongside the trained model (`<out_dir>/test_split.csv`).
These rows were never seen during training. For an honest read on generalization,
this script also splits the test predictions into:

  - rows whose `text` field appears verbatim in the training CSV (memorization
    contributes to the score on these)
  - novel rows whose text does NOT appear in training (true generalization)

Outputs a confusion matrix, sklearn classification report, and the dup/novel
F1 breakdown side-by-side.

Usage:
    cd ~/CharlemagneLabs/g4h
    ./.venv/bin/python -m scripts.eval_test_split

Reads:
    runs/gemma4-e4b-cls/
        - the saved artifact directory (override with --out-dir)
        - test_split.csv (must exist in the artifact dir)
    data/latest_ft_train_data.csv
        - the source training CSV (override with --train-csv)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.infer import load_for_inference
from src.train import _TokenizedDataset, _tokenize


def _predict_batch(bundle, texts: list[str], batch_size: int = 8) -> np.ndarray:
    enc = _tokenize(bundle.tokenizer, texts, bundle.max_length)
    dummy_labels = torch.zeros(len(texts), dtype=torch.long)
    ds = _TokenizedDataset(enc, dummy_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    device = next(bundle.model.classifier.parameters()).device
    preds = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = bundle.model(**batch)
            logits = torch.nan_to_num(out.logits, nan=0.0, posinf=0.0, neginf=0.0)
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(preds)


def _report_block(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title} (n={len(y_true)})")
    print('=' * 60)
    if len(y_true) == 0:
        print("  (empty)")
        return
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"  order: {label_names}")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))))
    print("\nPer-class report:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    ))
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/gemma4-e4b-cls",
                        help="Artifact directory (must contain test_split.csv)")
    parser.add_argument("--train-csv", default="data/latest_ft_train_data.csv",
                        help="Training CSV — used to identify duplicate texts in test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-errors", action="store_true",
                        help="Write misclassified rows to <out_dir>/eval_errors.csv")
    args = parser.parse_args()

    test_csv = os.path.join(args.out_dir, "test_split.csv")
    if not os.path.exists(test_csv):
        print(f"ERROR: {test_csv} not found.", file=sys.stderr)
        return 1
    if not os.path.exists(args.train_csv):
        print(f"ERROR: {args.train_csv} not found.", file=sys.stderr)
        return 1

    print(f"Loading model from {args.out_dir}/ ...")
    bundle = load_for_inference(args.out_dir)
    print(f"  labels: {bundle.id2label}")
    print(f"  max_length: {bundle.max_length}")
    print(f"  device: {next(bundle.model.classifier.parameters()).device}")

    print(f"\nReading test split from {test_csv} ...")
    test_df = pd.read_csv(test_csv).reset_index(drop=True)
    print(f"  test rows: {len(test_df)}")

    print(f"\nReading training CSV from {args.train_csv} to identify duplicates ...")
    train_texts = set(pd.read_csv(args.train_csv)["text"].astype(str))
    test_df["is_dup"] = test_df["text"].astype(str).apply(lambda t: t in train_texts)
    n_dup = int(test_df["is_dup"].sum())
    print(f"  duplicates in test (text seen in train): {n_dup}/{len(test_df)}")

    label_names = [bundle.id2label[i] for i in range(len(bundle.id2label))]

    print(f"\nRunning predictions ...")
    y_pred = _predict_batch(bundle, test_df["text"].astype(str).tolist(), args.batch_size)
    y_true = test_df["label_id"].astype(int).to_numpy() if "label_id" in test_df.columns \
             else test_df["label"].astype(int).to_numpy()

    _report_block("OVERALL — all test rows", y_true, y_pred, label_names)

    dup_mask = test_df["is_dup"].to_numpy()
    _report_block("NOVEL — test rows whose text was NOT in training",
                  y_true[~dup_mask], y_pred[~dup_mask], label_names)
    _report_block("DUPLICATE — test rows whose text appeared in training",
                  y_true[dup_mask], y_pred[dup_mask], label_names)

    if args.save_errors:
        errors = test_df.copy()
        errors["pred_id"] = y_pred
        errors["pred_label"] = [bundle.id2label[int(p)] for p in y_pred]
        errors = errors[errors["label"].astype(int) != errors["pred_id"]]
        errors_path = os.path.join(args.out_dir, "eval_errors.csv")
        errors.to_csv(errors_path, index=False)
        print(f"\nWrote {len(errors)} misclassified rows to {errors_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
