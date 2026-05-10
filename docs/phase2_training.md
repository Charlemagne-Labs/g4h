# Phase 2 — Training runbook

Walks through fine-tuning `google/gemma-4-E4B-it` on the gateguard phishing-indicator dataset with QLoRA. Two paths:

- **A. Mac smoke run** — local M-series Mac, bf16 (no bitsandbytes), tiny subset, ~5 min. Proves the pipeline end-to-end before you commit to a Colab run.
- **B. Colab full run** — CUDA + bnb 4-bit nf4, full dataset, 3 epochs, ~20–35 min on A100.

Same code path (`src/train.run_training`) for both — only `bnb_config` differs.

Tracks issue: [`gateguard-suite#73`](https://github.com/Charlemagne-Labs/gateguard-suite/issues/73), Phase 2.

---

## Goal

Produce a trained classifier-head adapter that:

1. Loads via `src/model.load_text_only_gemma4` (vision/audio towers dropped at load).
2. Predicts allow / warn / block (3 classes) on indicator-string inputs.
3. Saves in gateguard-compatible format (`adapter/`, `classifier_head.pt`, `inference_config.json`, `label_map.json`) so the artifact is interchangeable across the two repos.

Success criteria: training loss decreases monotonically (not strictly — but trends down across epochs); validation macro-F1 > 0.50 by epoch 3 (the dataset has 3 balanced classes and class-weighted CE loss; 0.50 is a generous floor — anything not learning will sit near 0.33).

---

## Prerequisites (both paths)

### 1. Hugging Face access

`google/gemma-4-E4B-it` is gated.

1. Visit https://huggingface.co/google/gemma-4-E4B-it → **Acknowledge license**.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Be ready to paste it when the notebook prompts (or run `hf auth login` ahead of time).

### 2. Dataset

Both paths need `data/latest_ft_train_data.csv` (3259 rows, columns `text`/`label`). It's already copied into `g4h/data/` locally — check with `ls data/`. For Colab, you'll re-upload (the file is gitignored and not pushed).

### 3. Phase 1 already passed

The training assumes `_get_inner_base_model` returns `Gemma4TextModel` and the unwrap is bit-exact equivalent to the fallback. If you haven't run `01_smoke_test.ipynb` yet, do that first — see [`docs/phase1_smoke_test.md`](phase1_smoke_test.md).

---

## Path A — Mac smoke run

### When to use this

To prove the code path runs end-to-end on your hardware before you commit Colab compute. **This is not a real training run.** The goal is "no exceptions for one full epoch on a tiny subset" — not "good F1." Don't tune from these numbers.

### What's different from the Colab path

- `bnb_config=None` (bitsandbytes nf4 doesn't work on MPS)
- Smaller everything: `epochs=1`, `batch=2`, `max_length=128`, train on a 100-row subset
- bf16 throughout (so memory is ~4× a 4-bit run, but our reduced settings keep it under the 36 GB ceiling on M4 Max)

### Run it

Create `scripts/mac_smoke_train.py` in the repo, then run with the venv we set up in Phase 0:

```python
# scripts/mac_smoke_train.py
import pandas as pd
from src.train import run_training, TrainConfig

# Subset to 100 rows for a fast smoke
src_csv = "data/latest_ft_train_data.csv"
sub_csv = "data/_smoke_subset.csv"
pd.read_csv(src_csv).sample(100, random_state=0).to_csv(sub_csv, index=False)

cfg = TrainConfig(
    model_id="google/gemma-4-E4B-it",
    csv_path=sub_csv,
    out_dir="runs/_smoke",
    val_ratio=0.1,
    test_ratio=0.1,
    max_length=128,
    epochs=1,
    batch=2,
    grad_accum=1,
    lr=2e-4,
    lora_r=8,           # smaller for smoke
    lora_alpha=16,
    use_class_weights=False,  # 80-row train split is too small to estimate weights
)

print(run_training(cfg, bnb_config=None))
```

```bash
cd ~/CharlemagneLabs/g4h
./.venv/bin/python -m scripts.mac_smoke_train
```

### Pass criteria

- Model loads. You see `Device: mps, dtype: torch.bfloat16`.
- After load, vision/audio towers print as deleted (no error).
- LoRA wraps without error. PEFT prints something like `trainable params: ...`.
- One forward + backward succeeds. Logs like `[epoch 1] step 10 | lr ...e-04 | loss ...` appear.
- Epoch ends. `[epoch 1] val: eval_loss=..., eval_accuracy=..., eval_f1_macro=...` prints.
- Final `Saved artifacts to: runs/_smoke` line appears.
- `runs/_smoke/` contains `base_lm/` (with adapter files), `classifier_head.pt`, `inference_config.json`, `label_map.json`, `test_split.csv`.

The actual loss / F1 numbers don't matter at this scale (80 train rows / 10 val). Just: did it run?

### Common Mac-specific failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: MPS does not support type X` | An op fell back to an unsupported MPS path | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` env var to fall back to CPU for that op. Run with `PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python -m scripts.mac_smoke_train` |
| OOM during load | Other MPS apps holding memory | Quit them. Worst case, reduce `max_length=64` and `batch=1` |
| `NotImplementedError` from bitsandbytes | You passed a non-None `bnb_config` | Don't — Mac path is `bnb_config=None` |
| Optimizer step very slow (>30 s) | MPS fused AdamW path missing | Tolerate it for the smoke run; this is one of the reasons we go to Colab for the real run |
| Loss is nan after the first step | bf16 attention numerical issue | Try `dtype=torch.float32` in `load_text_only_gemma4` (slow but stable). If still nan, paste the stack trace |

### When it passes

You've proven the code path on your hardware. **Don't keep training on Mac** — you've already paid the load time once, just go to Colab for the real run with the same code.

---

## Path B — Colab full run

### Open the notebook

Easiest: https://colab.research.google.com/github/Charlemagne-Labs/g4h/blob/main/notebooks/02_train_colab.ipynb opens directly. Save a copy to your Drive if you want changes to persist.

### Pick the runtime

Runtime → Change runtime type. In order of preference:

| GPU | VRAM | Verdict |
|---|---|---|
| **A100** | 40 GB | First choice. Fast, plenty of headroom. ~20 min for 3 epochs. |
| L4 | 24 GB | Solid. ~30 min. |
| **A10** | 24 GB | Solid. ~30 min. |
| T4 | 16 GB | Borderline. Text-only at 4-bit + LoRA + activations sits around 12–14 GB. Will work but no headroom for batch increases. |
| (CPU) | — | Don't. The notebook will refuse via the assert. |

### Run cells in order

| Cell | What it does | Expected duration |
|---|---|---|
| 1. Clone + install | `git clone` and `pip install -e ".[train]"` | ~1 min |
| 2. HF login | Interactive token paste | ~10 s |
| 3. Sanity checks | Print torch/transformers/peft/accelerate/bnb versions, GPU info | <1 s |
| 4. Get data | Manual upload (drag CSV into Colab Files panel) | ~2 s |
| 5. Verify data | Reads CSV, prints row count + label distribution | <1 s |
| 6. **Train** | The big one — model load + 3 epochs + save | **20–35 min on A100** |
| 7. Inspect artifacts | Walk `runs/gemma4-e4b-cls/` | <1 s |
| 8. Tarball | Compress run dir for download | ~10 s |

### Pass criteria for the training cell

- **Load**: prints `Device: cuda, dtype: torch.bfloat16`. Shortly after, the gemma weights download (16 GB, ~2–3 min on Colab's network). Then you'll see `Saved artifacts to:` only at the end.
- **First few steps log lines** look like:
  ```
  [epoch 1] step 10 | lr 5.42e-05 | loss 1.0934
  [epoch 1] step 20 | lr 1.08e-04 | loss 0.9423
  [epoch 1] step 30 | lr 1.62e-04 | loss 0.8156
  ```
  Loss should be **trending down** within the first 30 steps. If it's bouncing wildly or rising, something's wrong (see failure modes).
- **End of epoch 1**: `[epoch 1] val: eval_loss=0.8X, eval_accuracy=0.6X, eval_f1_macro=0.5X-0.6X` (rough — these are guesses based on dataset size; calibrate to what you actually see).
- **End of epoch 3**: macro-F1 has gone up vs epoch 1. If it's flat or decreased, we're either overfitting or the lr is too high — sweep.

### Recording results

Capture:
- Final `eval_loss`, `eval_accuracy`, `eval_f1_macro`
- Per-epoch progression (paste the three `[epoch N] val:` lines)
- Total training wall time
- Peak GPU memory (run `!nvidia-smi` in a Colab cell while training)

### Failure modes (Colab-specific)

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: 401` during model load | HF token missing / license not accepted | Re-run cell 2; check the license page |
| `bitsandbytes Library not found` | bnb didn't install (Colab CUDA mismatch) | `!pip install -U bitsandbytes` in a fresh cell |
| `KeyError: 'gemma4'` / `model type gemma4 ... not recognized` | Colab pre-installed `transformers` was older than 5.8 and pip's `>=` constraint didn't trigger upgrade | `!pip install -U "transformers>=5.8"`, restart runtime, retry |
| `RuntimeError: ... already a kernel registered ... bitsandbytes namespace` | bnb ops registered twice in one Python session (after pip-upgrade in same kernel) | Restart runtime — pip-installed version is on disk and a fresh interpreter loads only the new bnb |
| `OutOfMemoryError ... Tried to allocate 10.50 GiB` during `prepare_model_for_kbit_training` | PEFT helper tries to upcast Gemma 4's Per-Layer Embeddings (~2.6 B params) to fp32 — that single tensor is ~10.5 GB | We skip the helper and do the prep manually — already in `src/train.py` as of commit 67cfc00. If you see this on an older checkout: `git pull`. |
| `AttributeError: 'PeftModel' object has no attribute ...` | PEFT version mismatch with transformers | `!pip install -U "peft>=0.13"` |
| OOM during training | Batch too big for tier | Drop `cfg.batch=4, grad_accum=4` (effective batch stays 16) |
| Loss is nan after step 1 | bf16 numerical issue with QLoRA | Switch `bnb_4bit_compute_dtype=torch.float16` in the bnb config; or `lr=1e-4` |
| Loss bounces wildly, doesn't decrease | lr too high | `cfg.lr=1e-4` |
| Loss decreases to ~0 in <50 steps | Memorization (dataset too easy) or label leakage | Investigate before celebrating; check that train/val don't overlap |

### Download the tarball

Cell 8 writes `runs/gemma4-e4b-cls.tar.gz`. In the Colab Files panel, right-click → Download. Or:

```python
from google.colab import files
files.download(tarball)
```

Save it locally as e.g. `~/CharlemagneLabs/g4h/runs/gemma4-e4b-cls.tar.gz` for Phase 3 eval.

---

## What this unblocks

Successful Phase 2 means:

- A trained adapter + classifier head on disk (locally or in the tarball).
- Validated end-to-end pipeline (load → LoRA → train → save).
- A baseline F1 number for E4B-it on this task.

Phase 3 then fills in `src/infer.py` and an eval script that loads the saved artifacts, runs on the held-out test split, and produces a confusion matrix + per-class F1 + (optionally) per-indicator F1 to surface tail-data weaknesses.

---

## Sign-off checklist

Phase 2 is complete when all of these are checked:

- [ ] Mac smoke run passes (Path A) — code path verified on your hardware
- [ ] Colab training run completes without errors (Path B)
- [ ] Final `eval_f1_macro` recorded
- [ ] Per-epoch metrics recorded
- [ ] `runs/gemma4-e4b-cls.tar.gz` downloaded locally
- [ ] No surprises in the artifact directory layout

When done, ping me with the final metrics + any surprises and we'll move to Phase 3 (eval + demo).
