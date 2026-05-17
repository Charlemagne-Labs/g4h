# g4h — Gemma 4 hackathon classifier head

QLoRA fine-tune of `google/gemma-4-E4B-it` for 3-class phishing-indicator classification (`allow` / `warn` / `block`). End-to-end pipeline in ~900 LOC: data load → fine-tune → save → eval → predict CLI.

Submission for the Gemma 4 hackathon. Original tracking: [`Charlemagne-Labs/gateguard-suite#73`](https://github.com/Charlemagne-Labs/gateguard-suite/issues/73).

---

## Results

Trained on 2,611 indicator strings (extracted by gateguard's feature pipeline) with QLoRA (r=16, α=32, 7 LoRA targets across `q/k/v/o/gate/up/down_proj`, lr=2e-4 cosine, 3 epochs, batch 8 / grad_accum 2 → effective batch 16).

| Split | n | Accuracy | Macro F1 | Errors |
|---|---|---|---|---|
| Training-time validation | 324 | 99.4% | 0.994 | 2 (both `block`→`warn`) |
| **Held-out test, overall** | **324** | **100.0%** | **1.000** | **0** |
| **Held-out test, novel text only** | **265** | **100.0%** | **1.000** | **0** |
| Held-out test, dup with train | 59 | 100.0% | 1.000 | 0 |

The **novel-text** subset is the defensible generalization number: 265 of the 324 test rows have indicator strings that never appeared verbatim during training. All 265 are classified correctly.

### Calibrated reading

- **Why I trust this number**: 265 rows of true generalization signal (text never seen by the model), zero errors. Validation errors (2 of them) were the safer direction — under-classifying severity, never over-classifying legitimate traffic.
- **Honest caveat 1**: the task is structurally easier than the model size suggests. gateguard's feature extractor already encodes most of the classification signal in the indicator strings; the model is doing pattern recognition over highly informative features. A smaller classifier would likely also score in the 95%+ range; Gemma 4 makes it perfect.
- **Honest caveat 2**: the dataset has 18% near-duplicate texts (different URLs producing identical indicator strings). We surface this explicitly in eval rather than hide it, and report novel-text generalization separately.
- **Honest caveat 3**: gap between val (99.4%) and test (100%) is consistent with random sampling variance on small (324-row) partitions, not a methodological break.

Reproduce via `scripts/eval_test_split.py` — see [`docs/phase3_eval.md`](docs/phase3_eval.md).

---

## Quick start

### Predict a single indicator string

```bash
git clone https://github.com/Charlemagne-Labs/g4h.git
cd g4h && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# (drop the trained artifact tarball into runs/, then unpack)
mkdir -p runs && tar -xzf gemma4-e4b-cls.tar.gz -C runs/

python -m src.infer \
    --text 'url:ip_hostname:{"hostname":"45.141.87.195"} security:no_https:{"scheme":"http"}'
```

Output:

```json
{
  "text": "url:ip_hostname:...",
  "label": "block",
  "scores": {"allow": 0.0001, "warn": 0.0023, "block": 0.9976}
}
```

### Run the full eval

```bash
python -m scripts.eval_test_split
```

Prints overall / novel / duplicate confusion matrices and per-class F1s for the held-out test split.

---

## Layout

```
g4h/
├── src/
│   ├── model.py              # CausalLMWithClassifier + _get_inner_base_model unwrap
│   ├── train.py              # QLoRA training loop + dataset split + save_artifacts
│   ├── infer.py              # load_for_inference + predict_one + CLI
│   └── extract.py            # URL-only feature extractor (clean-room, ~150 LOC)
├── server/                   # live-demo webapp (Phase 4)
│   ├── app.py                # FastAPI with /predict + lifespan-loaded model
│   ├── fetch.py              # Playwright targeted DOM fetch (optional enrichment)
│   ├── static/               # Charley-branded UI (Operation Knight design language)
│   ├── Dockerfile            # GPU-aware container
│   └── requirements.txt
├── scripts/
│   ├── mac_smoke_train.py    # 5-min M-series smoke run (100 rows, 1 epoch)
│   └── eval_test_split.py    # full eval w/ novel-vs-duplicate breakdown
├── notebooks/
│   ├── 01_smoke_test.ipynb   # local Mac MPS: verify model load + unwrap helper
│   └── 02_train_colab.ipynb  # cloud GPU: QLoRA training run
├── docs/
│   ├── phase1_smoke_test.md
│   ├── phase2_training.md
│   ├── phase3_eval.md
│   └── phase4_live_demo.md   # webapp + AWS deploy runbook
├── data/                     # gitignored except for README + samples
└── pyproject.toml
```

---

## Setup

### Local (Mac, dev / smoke / inference / eval)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Apple Silicon ≥16 GB unified can load the model in bf16 for smoke tests and inference. Training-grade QLoRA needs CUDA (next section).

### Cloud (Colab / any CUDA box, training)

```bash
pip install -e ".[train]"
```

The `[train]` extra pulls in `bitsandbytes` for nf4 4-bit quantization. Don't install it on Apple Silicon — MPS support for nf4 is incomplete.

---

## Method, in three phases

Each phase has a standalone runbook with pass criteria, failure modes, and a results template.

1. **[Phase 1 — bring-up smoke test](docs/phase1_smoke_test.md)** *(M-series Mac, ~5 min)*. Load `google/gemma-4-E4B-it`, verify the `_get_inner_base_model` unwrap returns a working `Gemma4TextModel`, and confirm that `last_hidden_state` is bit-exact equal to the `output_hidden_states=True` fallback (resolves the Per-Layer Embeddings risk).
2. **[Phase 2 — QLoRA training](docs/phase2_training.md)** *(Colab T4/L4/A100, 25–90 min)*. Text-only model load (vision/audio towers dropped post-load — ~30–40% memory saved). QLoRA via `bitsandbytes` nf4 with custom prep that skips PEFT's `prepare_model_for_kbit_training` (it OOMs on Gemma 4's giant PLE tables). 3 epochs, last-position pooling for left-padded inputs.
3. **[Phase 3 — held-out evaluation](docs/phase3_eval.md)** *(M-series Mac, ~2 min)*. Reconstruct the training partition deterministically from the saved seed/ratios, then split test rows into novel vs. duplicate. Report overall + novel + duplicate confusion matrices.
4. **[Phase 4 — live webapp](docs/phase4_live_demo.md)** *(AWS EC2 g5.xlarge, ~30 min setup)*. FastAPI server with a clean-room URL feature extractor and optional Playwright DOM fetch enrichment. "Charley · Gemma 4 E4B demo" frontend (uses the Charlemagne Labs Operation Knight design language). Dockerized, deploys to EC2 in 5 commands.

---

## Technical decisions worth flagging

- **Text-only loader for a multimodal checkpoint.** `gemma-4-E4B-it` is structurally multimodal (vision + audio towers alongside the language model). We load the full model, then move the multimodal modules to CPU and `delattr` them — the `.to("cpu")` step is essential because `accelerate`'s device hooks keep GPU tensors alive past `delattr` alone. Saves ~30–40% of GPU memory.
- **Last-position pooling.** Under left-padding (which we use), the last real token is always at index `-1`. The gateguard reference code used `attention_mask.sum(dim=1) - 1` — correct for right-padding, wrong for left-padding. We fixed this; the model trains cleaner as a result.
- **Skip `prepare_model_for_kbit_training`.** PEFT's helper upcasts non-quantized bf16 params to fp32 — and Gemma 4's PLE table is a single ~2.6B-param tensor that OOMs every consumer GPU when upcast. We do manual `requires_grad=False` + gradient checkpointing instead. Layer norms stay in bf16; the risk to numerical stability didn't bite for this task.

---

## Future work (post-hackathon)

- **Dataset expansion.** The training data was 3,259 indicator strings sampled from gateguard's feature pipeline output. A larger corpus — built via the same pipeline but with newer URL feeds and broader indicator coverage — would test generalization across more attack patterns and likely surface new indicator types. The expanded data would be brought in as a CSV; the extractor itself stays out of this repo.
- **ONNX export for on-device.** The trained adapter could be merged into the base and exported to ONNX for sub-second CPU inference. Out of scope for the hackathon submission, in scope for a production extension.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Matches the license of `google/gemma-4-E4B-it`. The trained adapter and classifier head are not redistributed in this repo; they are produced by running the training pipeline against the dataset referenced in [`data/README.md`](data/README.md).

## Acknowledgments

- Built on Google DeepMind's [Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4) family.
- Classifier-head technique adapted from a private internal Gemma 3 implementation. This repo is the clean-room public port.
