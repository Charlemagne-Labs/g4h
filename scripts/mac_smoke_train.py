"""Mac smoke training run — proves the pipeline end-to-end on M-series Macs.

Not a real training run. Trains 1 epoch on a 100-row subset at small batch and
short max_length, in bf16 (no bitsandbytes — nf4 is CUDA-only). The goal is
"no exceptions, full forward + backward + save," not "good F1."

See docs/phase2_training.md (Path A) for what to expect.

Usage:
    cd ~/CharlemagneLabs/g4h
    ./.venv/bin/python -m scripts.mac_smoke_train

If you hit `RuntimeError: MPS does not support type ...`, retry with the MPS
fallback enabled:
    PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python -m scripts.mac_smoke_train
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from src.train import TrainConfig, run_training

SRC_CSV = "data/latest_ft_train_data.csv"
SUB_CSV = "data/_smoke_subset.csv"
OUT_DIR = "runs/_smoke"
N_ROWS = 100


def main() -> int:
    if not os.path.exists(SRC_CSV):
        print(f"ERROR: {SRC_CSV} not found. Copy it from gateguard-suite/data/.", file=sys.stderr)
        return 1

    pd.read_csv(SRC_CSV).sample(N_ROWS, random_state=0).to_csv(SUB_CSV, index=False)
    print(f"Wrote {N_ROWS}-row subset to {SUB_CSV}")

    cfg = TrainConfig(
        model_id="google/gemma-4-E4B-it",
        csv_path=SUB_CSV,
        out_dir=OUT_DIR,
        val_ratio=0.1,
        test_ratio=0.1,
        max_length=128,
        epochs=1,
        batch=2,
        grad_accum=1,
        lr=2e-4,
        lora_r=8,
        lora_alpha=16,
        use_class_weights=False,  # 80-row train split is too small to estimate
    )

    metrics = run_training(cfg, bnb_config=None)
    print()
    print("Smoke run finished. Final val metrics (numbers don't matter at this scale):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nArtifacts in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
