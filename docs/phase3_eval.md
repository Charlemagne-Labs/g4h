# Phase 3 — Evaluation runbook

Walks through unpacking the trained model artifact and running both the held-out test evaluation and ad-hoc single-input predictions on your Mac.

Phase 3 takes the artifact produced by Phase 2 (`runs/gemma4-e4b-cls.tar.gz` downloaded from Colab) and produces the headline number for the hackathon writeup.

Tracks issue: [`gateguard-suite#73`](https://github.com/Charlemagne-Labs/gateguard-suite/issues/73), Phase 3.

---

## Goal

Produce one defensible eval result on data the trained model has never seen:

1. Extract the saved adapter + classifier head to `runs/gemma4-e4b-cls/`.
2. Run `scripts/eval_test_split.py` on the held-out `test_split.csv` (the 324 rows the training and validation passes never saw).
3. Capture per-class metrics and confusion matrices, split into:
   - **OVERALL** — all test rows (the headline number)
   - **NOVEL** — test rows whose text is NOT identical to anything in the training CSV (true generalization)
   - **DUPLICATE** — test rows whose text appears verbatim in training (memorization contribution)

The NOVEL number is the one to lead with in the writeup.

---

## Prerequisites

- Phase 2 complete — you have `gemma4-e4b-cls.tar.gz` downloaded to your laptop (`~/Downloads/` or similar).
- Local g4h venv set up with `[dev]` extras (per the README — already done if you ran the Phase 1 smoke test).
- `data/latest_ft_train_data.csv` in place — used by the eval script to identify which test rows are duplicates of training rows.

No HF login or network access required for Phase 3 — the model loads entirely from disk.

---

## Step A — Unpack the tarball

```bash
cd ~/CharlemagneLabs/g4h
git pull              # make sure src/infer.py and scripts/eval_test_split.py are present

mkdir -p runs
mv ~/Downloads/gemma4-e4b-cls.tar.gz runs/
cd runs && tar -xzf gemma4-e4b-cls.tar.gz && cd ..

ls runs/gemma4-e4b-cls/
```

Expected directory contents:

```
runs/gemma4-e4b-cls/
├── base_lm/
│   ├── adapter_config.json     # PEFT LoRA config
│   ├── adapter_model.safetensors  # ~140 MB — trained adapter weights
│   └── README.md
├── classifier_head.pt          # 33 KB — nn.Linear(2560, 3) weights
├── inference_config.json       # base model id, max_length, label set, etc.
├── label_map.json              # {"label2id": {...}, "id2label": {...}}
├── test_split.csv              # ~46 KB — 324 held-out test rows
└── tokenizer.json (+ friends)  # tokenizer files for offline reload
```

If anything is missing, the tarball didn't pack cleanly — re-run cell 8 of `02_train_colab.ipynb` and re-download.

---

## Step B — Run the eval

```bash
./.venv/bin/python -m scripts.eval_test_split
```

This:
1. Loads the trained model on MPS in bf16 (no bitsandbytes — Mac runs straight bf16).
2. Reads `runs/gemma4-e4b-cls/test_split.csv` and `data/latest_ft_train_data.csv`.
3. Marks each test row as `is_dup` (text appears in training) or novel.
4. Predicts on all 324 rows.
5. Prints three blocks: OVERALL, NOVEL, DUPLICATE.

Expected wall time: ~1–2 minutes on M4 Max.

### Useful flags

- `--save-errors` writes misclassified rows to `runs/gemma4-e4b-cls/eval_errors.csv` for manual inspection
- `--batch-size N` (default 8) — bump if you have headroom, drop if you OOM
- `--out-dir <path>` — point at a different artifact directory (default `runs/gemma4-e4b-cls`)
- `--train-csv <path>` — point at a different training CSV for duplicate detection

### What to expect

Based on validation numbers (99.4% macro F1), the test split should show similar performance. The interesting split is **NOVEL vs DUPLICATE**:

- **DUPLICATE** rows are likely near-100% accurate (the model memorized those exact strings during training).
- **NOVEL** rows are the true generalization signal. If NOVEL F1 is still in the 99% range, the model is genuinely learning. If NOVEL F1 drops meaningfully (say below 95%), the headline number is inflated by memorization.

---

## Step C — Ad-hoc single-input prediction

```bash
./.venv/bin/python -m src.infer \
    --text 'url:ip_hostname:{"hostname":"45.141.87.195"} security:no_https:{"scheme":"http"}'
```

Prints JSON:

```json
{
  "text": "url:ip_hostname:...",
  "label": "block",
  "scores": {
    "allow": 0.0001,
    "warn": 0.0023,
    "block": 0.9976
  }
}
```

Useful for spot-checking specific patterns you care about. The input must be in the indicator-string format the model was trained on — `category:name:{"key":"value"}` separated by spaces.

You can also use `predict_one(bundle, text)` programmatically from a Python REPL or notebook:

```python
from src.infer import load_for_inference, predict_one
bundle = load_for_inference("runs/gemma4-e4b-cls")
label, scores = predict_one(bundle, 'url:ip_hostname:{"hostname":"1.2.3.4"}')
print(label, scores)
```

---

## Recording results

Capture for the writeup:

```
PHASE 3 EVAL RESULTS
Run date:               __________
Hardware:               M4 Max, MPS, bf16
Artifact:               runs/gemma4-e4b-cls (commit __________)

OVERALL (n=324)
  accuracy:             __________
  f1_macro:             __________
  Confusion matrix:
    [allow  warn  block]   <- pred
    [ ___   ___   ___ ] true=allow
    [ ___   ___   ___ ] true=warn
    [ ___   ___   ___ ] true=block

NOVEL (text not in train, n=____)
  accuracy:             __________
  f1_macro:             __________

DUPLICATE (text appeared in train, n=____)
  accuracy:             __________
  f1_macro:             __________

Notable errors:
  __________ (paste any rows from eval_errors.csv worth flagging)
```

---

## Sign-off checklist

Phase 3 is done when all of these are checked:

- [ ] Tarball extracted to `runs/gemma4-e4b-cls/` and all expected files are present
- [ ] `scripts.eval_test_split` runs cleanly on Mac (MPS, bf16, no errors)
- [ ] OVERALL accuracy + macro F1 recorded
- [ ] NOVEL accuracy + macro F1 recorded (the defensible generalization number)
- [ ] DUPLICATE counts recorded (memorization contribution)
- [ ] At least one ad-hoc prediction via `python -m src.infer --text "..."` works end-to-end
- [ ] Any notable errors from `eval_errors.csv` reviewed and noted in the writeup

When all check, this is the entire trained-model story for the hackathon submission: a Gemma 4 E4B-it classifier head fine-tuned with QLoRA, evaluated honestly on held-out data with the duplication caveat surfaced explicitly.

---

## What's intentionally NOT in this phase

Two pieces were considered and deliberately scoped out for the open-source submission:

- **Live OpenPhish/PhishTank demo** — would need a feature extractor to convert raw URLs into the indicator-string format the model expects. A clean-room minimal extractor is feasible (~100 LOC, public phishing heuristics only), but adding it is an extra surface to maintain and a deviation from "smallest end-to-end demo of the technique." Worth revisiting post-hackathon if the project continues.
- **gateguard's production `feature_extractors.py` port** — 5000+ lines of business logic; clearly IP-bound and incompatible with an open-source submission.

If a richer demo is needed later, both are tracked in `docs/phase3_eval.md` and the README's open-questions section.
