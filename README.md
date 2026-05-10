# g4h — Gemma 4 hackathon classifier head

Clean-room port of gateguard's classifier-head technique from **Gemma 3 270M** to **Gemma 4 E4B**, built for the Gemma 4 hackathon. Tracks [Charlemagne-Labs/gateguard-suite#73](https://github.com/Charlemagne-Labs/gateguard-suite/issues/73).

The shape is data → fine-tune → save → run, with the smallest possible code surface (~300–500 LOC target, hard ceiling 1000). Production daemon scaffolding from gateguard-suite is intentionally excluded.

## Layout

```
g4h/
├── src/
│   ├── model.py              # CausalLMWithClassifier + _get_inner_base_model unwrap
│   ├── train.py              # data load, QLoRA config, training loop, save artifacts
│   ├── infer.py              # load_for_inference + predict_one CLI
│   └── export_onnx.py        # optional Phase 3
├── notebooks/
│   ├── 01_smoke_test.ipynb   # local M-series Mac (MPS, bf16, no bitsandbytes)
│   └── 02_train_colab.ipynb  # Colab / cloud GPU (QLoRA nf4 via bitsandbytes)
├── data/
│   └── README.md             # dataset pointers (deferred until smoke test passes)
└── pyproject.toml
```

## Compute split

Decided **dev on Mac, train on cloud**:

- **Local Apple Silicon** — Phase 1 smoke test in bf16 on MPS. No bitsandbytes (nf4 is CUDA-native; MPS support is partial). Use `notebooks/01_smoke_test.ipynb`.
- **Cloud GPU (Colab Pro / A10 / A100)** — Phase 2 actual fine-tuning with QLoRA nf4. Use `notebooks/02_train_colab.ipynb`.

## Setup

### Local (Mac, smoke test only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
jupyter lab notebooks/01_smoke_test.ipynb
```

### Cloud (Colab / CUDA, training)

```bash
pip install -e ".[train]"
```

The `[train]` extra pulls in `bitsandbytes`. Don't install it on Apple Silicon.

## Phases (per issue #73)

1. **Bring-up smoke test** — load Gemma 4 E4B, verify `_get_inner_base_model` unwrap works on the new architecture, probe Per-Layer Embeddings behavior. **Runbook: [`docs/phase1_smoke_test.md`](docs/phase1_smoke_test.md).**
2. **Port training** — `src/model.py` + `src/train.py`, QLoRA fine-tune on whichever dataset gets picked.
3. **Eval + demo** — `src/infer.py`, head-to-head F1 vs the 270M baseline, optional ONNX export.

## Source files being ported (in `Charlemagne-Labs/gateguard-suite` at `a9f69a2`)

| g4h file | Comes from |
|---|---|
| `src/model.py` | `gemma3_classifier_lora.py:54-109` (`CausalLMWithClassifier`) + `api_server/runners.py:156-175` (`_get_inner_base_model`, mandatory lm_head bypass) |
| `src/train.py` | `gemma3_classifier_lora.py:366-430` (`train_pure_torch`) + `:188-222` (`save_artifacts`) + `:113-160` (split / class-weight helpers) |
| `src/infer.py` | `gemma3_classifier_lora.py:224-298` (`load_for_inference`) + `:299-323` (`predict_one`) |
| `src/export_onnx.py` *(optional)* | branch `staford/onnx-model-v53` — export logic only, NOT the runtime backend |

## Open questions still to resolve

1. **Dataset** — deferred until Phase 1 is green.
2. **Demo surface** — possibly integrate with `Charlemagne-Labs/test_suite/phase2` (replace its `llm_verdicts.py` stub with a Gemma 4 call). Decide after training works.
3. **License file** — add `LICENSE` (Apache-2.0, matching Gemma's license) before flipping the repo public.

## Status

- [x] Repo scaffolded
- [x] **Phase 1 smoke test passes on M4 Max** (2026-05-10) — `Gemma4TextModel` unwrap works, `last_hidden_state` is bit-exact equal to `output_hidden_states=True` fallback (PLE risk resolved), `q/k/v/o_proj` confirmed for LoRA. Hidden size 2560, vocab 262144, 42 layers.
- [ ] Phase 2 QLoRA training run on Colab — first run on gateguard phishing data for baseline comparison
- [ ] Phase 3 eval + demo
