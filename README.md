# g4h — Gemma 4 hackathon classifier head

QLoRA fine-tune of `google/gemma-4-E4B-it` for a 3-class URL classification task (`allow` / `warn` / `block`) over structured phishing-indicator inputs. End-to-end clean-room pipeline: data → fine-tune → save → eval → predict CLI → live webapp.

Submission for the Gemma 4 hackathon.

> **Live demo — paste a URL, get a classification**
> **[https://charley-g4demo.charlemagnelabs.ai](https://charley-g4demo.charlemagnelabs.ai)**
>
> Server-side feature extraction + optional Playwright DOM fetch + trained Gemma 4 inference. Toggle FETCH DOM to compare URL-only vs. with-DOM signal. Per-phase timing breakdown is shown in the response card (the red cell is the model inference latency). Running on a single AWS EC2 g5.xlarge (NVIDIA A10G).

---

## Results

Fine-tuned on **2,611 indicator strings** with QLoRA (`r=16`, α=32, LoRA on all seven projections `q/k/v/o/gate/up/down_proj`, lr=2e-4 cosine, 3 epochs, batch 8 / grad_accum 2 → effective batch 16, all on a Colab L4).

| Split | n | Accuracy | Macro F1 | Errors |
|---|---|---|---|---|
| Training-time validation | 324 | 99.4% | 0.994 | 2 (both `block`→`warn`) |
| **Held-out test, overall** | **324** | **100.0%** | **1.000** | **0** |
| **Held-out test, novel text only** | **265** | **100.0%** | **1.000** | **0** |
| Held-out test, duplicates of train | 59 | 100.0% | 1.000 | 0 |

The **novel-text** row is the defensible generalization number: 265 of the 324 held-out test rows have indicator strings that never appeared verbatim during training. All 265 are classified correctly.

### How to read this

- **What the number actually shows**: 265 rows of genuine generalization signal (no text overlap with training), zero errors. Validation errors observed during training (2 of them) were in the safer direction — under-classifying severity, never over-classifying legitimate traffic.
- **Caveat 1 — the task is structurally tractable.** The input is *not* raw URLs but pre-extracted indicator strings (e.g. `url:ip_hostname:{"hostname":"..."} security:no_https:{"scheme":"http"}`). Most of the classification signal is already encoded in the input format. A smaller classifier would likely also score in the high-90s; Gemma 4 makes it perfect.
- **Caveat 2 — dataset has 18% near-duplicate text.** Different URLs can produce identical indicator strings. We surface this explicitly in eval rather than hide it: the table above reports novel-text generalization separately so the headline isn't inflated by memorization.
- **Caveat 3 — small partitions.** The gap between val F1 (0.994) and test F1 (1.000) is within the range of random sampling variance on 324-row splits.

Reproduce: see [`docs/phase3_eval.md`](docs/phase3_eval.md) for the exact command.

---

## Try it

Easiest is the [live demo](https://charley-g4demo.charlemagnelabs.ai). To run locally:

```bash
git clone https://github.com/Charlemagne-Labs/g4h.git
cd g4h && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

You'll need a trained-model artifact (`runs/gemma4-e4b-cls/`) to run inference. **The artifact is not redistributed in this repo** — produce it yourself by running [`notebooks/02_train_colab.ipynb`](notebooks/02_train_colab.ipynb) on Colab (T4/L4/A100; ~25-90 min depending on tier). The notebook's outputs are committed so you can inspect the actual training run before deciding to spend the compute.

Once you have the artifact extracted at `runs/gemma4-e4b-cls/`:

```bash
# Predict a single indicator string
python -m src.infer --text \
  'url:ip_hostname:{"hostname":"45.141.87.195"} security:no_https:{"scheme":"http"}'

# Full eval on the held-out test split (with novel-vs-duplicate breakdown)
python -m scripts.eval_test_split
```

Or spin up the live webapp locally:

```bash
pip install -r server/requirements.txt
python -m playwright install chromium
./.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

---

## Method

Four phases, each with a standalone runbook (pass criteria, failure modes, results template):

1. **[Phase 1 — bring-up smoke test](docs/phase1_smoke_test.md)** *(local Mac, ~5 min).* Load `google/gemma-4-E4B-it`, verify the `_get_inner_base_model` traversal returns a working `Gemma4TextModel`, and confirm `last_hidden_state` is bit-exact equal to the `output_hidden_states=True` fallback (resolves the Per-Layer Embeddings risk).
2. **[Phase 2 — QLoRA training](docs/phase2_training.md)** *(Colab GPU, 25-90 min).* Text-only model load (vision + audio towers dropped post-load, ~30-40% memory saved). QLoRA via `bitsandbytes` nf4 with custom prep that skips PEFT's `prepare_model_for_kbit_training` (it OOMs on Gemma 4's giant Per-Layer Embeddings tables on consumer GPUs). 3 epochs, last-position pooling for left-padded inputs.
3. **[Phase 3 — held-out evaluation](docs/phase3_eval.md)** *(local Mac, ~2 min).* Reconstruct the training partition deterministically from the saved seed/ratios, split test rows into novel vs. duplicate, report overall + novel + duplicate confusion matrices.
4. **[Phase 4 — live webapp](docs/phase4_live_demo.md)** *(AWS EC2 g5.xlarge, deployed at [charley-g4demo.charlemagnelabs.ai](https://charley-g4demo.charlemagnelabs.ai)).* FastAPI server, clean-room URL feature extractor, optional Playwright DOM fetch enrichment. Caddy + Elastic IP + Route 53 + Let's Encrypt for HTTPS in front of the Dockerized FastAPI container.

---

## Repository layout

```
g4h/
├── src/
│   ├── model.py              CausalLMWithClassifier + _get_inner_base_model unwrap
│   ├── train.py              QLoRA training loop, dataset split, save_artifacts
│   ├── infer.py              load_for_inference + predict_one + CLI
│   └── extract.py            Clean-room URL feature extractor
├── server/                   Live-demo webapp (Phase 4)
│   ├── app.py                FastAPI with /predict + lifespan-loaded model
│   ├── fetch.py              Playwright targeted DOM fetch (optional enrichment)
│   ├── static/               Single-page UI (Charley · Gemma 4 E4B demo)
│   ├── Dockerfile            GPU-aware container
│   └── requirements.txt
├── scripts/
│   ├── mac_smoke_train.py    5-min M-series smoke run (100 rows, 1 epoch)
│   └── eval_test_split.py    Full eval with novel-vs-duplicate breakdown
├── notebooks/
│   ├── 01_smoke_test.ipynb   Phase 1: model load + unwrap helper verification
│   └── 02_train_colab.ipynb  Phase 2: full QLoRA training run (outputs committed)
├── docs/
│   ├── phase1_smoke_test.md
│   ├── phase2_training.md
│   ├── phase3_eval.md
│   └── phase4_live_demo.md
├── data/                     Dataset pointer (data files not committed)
└── pyproject.toml
```

---

## Technical decisions worth flagging

- **Text-only loader for a multimodal checkpoint.** `gemma-4-E4B-it` is structurally multimodal — its `Gemma4Model` has `vision_tower` and `audio_tower` siblings of `language_model`. We load the full model, then move the multimodal modules to CPU and `delattr` them. The `.to("cpu")` step is essential because `accelerate`'s device-mapping hooks keep GPU tensors alive past `delattr` alone, and the next `prepare_model_for_kbit_training` pass would OOM upcasting them. Saves ~30-40% of steady-state GPU memory.
- **Last-position pooling under left-padding.** Under left-padded inputs (which we use), the last real token is always at index `-1`. A common pooling formula `attention_mask.sum(dim=1) - 1` is correct for right-padding only — applied to left-padding, it points into the padding region for any sequence shorter than `max_length`. We pool from index `-1` directly.
- **Skip `prepare_model_for_kbit_training`.** PEFT's helper upcasts every non-quantized `bf16` parameter to `fp32`. Gemma 4's Per-Layer Embeddings tables are a single ~2.6 B-parameter tensor; the upcast OOMs every consumer GPU we tried. We do the parts that matter manually (`requires_grad=False` on base params + gradient checkpointing). Layer norms stay in `bf16`; the numerical-stability risk did not materialize for this task.
- **Tokenizer fallback in inference.** Saved tokenizer files can break across transformers minor versions (Gemma 4's `extra_special_tokens` field shifted between list and dict between point releases). The inference loader catches the exception and re-downloads the tokenizer from the base model — identical bytes, fresh schema.

---

## Future work

- **Dataset expansion.** Training used 3,259 indicator strings. A larger corpus from a richer extraction pipeline and broader URL feeds would test generalization across more attack patterns and surface tail indicator types the model is under-trained on. Bringing in the data is straightforward (it's just a CSV); the extractor itself would stay as a separate internal component.
- **Richer DOM extraction.** The live demo's Playwright fetcher implements ~6 DOM-based indicators (HSTS / CSP / clickjack-protection headers, login-form analysis, canonical link, trusted-CDN scripts). The training data has ~25 more DOM-derivable signals (page text classification, fingerprinting checks, etc.) that would require a deeper page-rendering pass — out of scope for the hackathon demo.
- **ONNX export for on-device inference.** The trained adapter could be merged into the base and exported to ONNX for sub-second CPU inference, removing the GPU-instance cost.
- **Per-request authentication.** The live demo is currently open; for any long-lived production use, adding an API key to the `/predict` endpoint is a ~5-line change.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Matches the license of `google/gemma-4-E4B-it`. The trained adapter and classifier head are not redistributed in this repo; they are produced by running the training pipeline against the dataset referenced in [`data/README.md`](data/README.md).

## Acknowledgments

- Built on Google DeepMind's [Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4) family. The model card was the source of truth for architecture details (PLE, layer counts, vocab size).
- Classifier-head wrapper pattern is a clean-room reimplementation informed by prior internal work on the same task with a Gemma 3 base. The `g4h` repo is the open-source port.
- Phishing-indicator format and the training dataset format originate from an internal feature-extraction pipeline at Charlemagne Labs. The pipeline itself is not part of this repo; only the indicator strings it produced (as flat CSV) were used as model inputs.
