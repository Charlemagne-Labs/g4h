# Phase 1 — Smoke test runbook

A step-by-step guide for running `notebooks/01_smoke_test.ipynb` on your M4 Max and deciding whether the port is on track. Tracks issue [`gateguard-suite#73`](https://github.com/Charlemagne-Labs/gateguard-suite/issues/73), Phase 1.

The notebook has eight runnable steps. This doc is the layer on top: **what success looks like, what to record, and what to do when something fails.**

---

## Goal

Prove three things about Gemma 4 E4B before we write any porting code:

1. **It loads on your Mac.** bf16 on MPS, no quantization, no OOM.
2. **The lm_head bypass survives.** `_get_inner_base_model` from `gateguard-suite/api_server/runners.py:156-175` returns a working backbone module on Gemma 4. This is mandatory — vocab is still 262K, so the wasted lm_head logits tensor is the same memory waste at the same scale (~196 MB per forward in bf16 at T=384, see gateguard-suite#64).
3. **Per-Layer Embeddings (PLE) doesn't break the assumptions.** Gemma 4 introduces PLE; we need to confirm it doesn't cause `inner(...).last_hidden_state` to diverge from `model(..., output_hidden_states=True).hidden_states[-1]`. If it does, we'll need a different unwrap or a fallback.

If all three are green, Phase 2 (`src/model.py` and `src/train.py`) becomes a near-direct port from `gateguard-suite/gemma3_classifier_lora.py`. If any are red, we adjust the wrapper before training.

---

## Prerequisites

Run these once before opening the notebook.

### 1. Disk space (~18 GB free)

Gemma 4 E4B-it in bf16 is **~16 GB on disk** (observed 2026-05-10 — the model card's 9 GB figure is parameters-only and excludes safetensors framing + tokenizer + per-layer embeddings tables). The HF cache lands in `~/.cache/huggingface/hub/`.

```bash
df -h ~/.cache
```

### 2. Hugging Face access token

Gemma weights are gated. You need to:

1. Visit https://huggingface.co/google/gemma-4-E4B-it and click **Acknowledge license**.
2. Generate a read token at https://huggingface.co/settings/tokens.
3. Log in via CLI:
   ```bash
   ./.venv/bin/hf auth login
   # paste the token when prompted
   ```

### 3. Model ID — `google/gemma-4-E4B-it` (decided 2026-05-10)

The notebooks use **`google/gemma-4-E4B-it`** — the instruction-tuned variant, chosen to match the gateguard baseline (`gemma-3-270m-it`). Mirroring the baseline's variant means only **one** thing varies between baseline and new system: model size. If we used base, we'd be changing both model size and pretraining objective, making any F1 delta ambiguous.

For last-token-pooled classifier heads, the IT-vs-base difference is empirically small (<1 F1 in most papers) — apples-to-apples comparison wins.

Other variants in the family for reference:
- `google/gemma-4-E4B` — base (pretrained). Equivalent for classifier fine-tuning, worse for baseline comparability.
- `google/gemma-4-E2B-it` — smaller (2B effective). Fallback if E4B-it doesn't fit on Colab compute.
- `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` — bigger; out of scope for the hackathon.

### 4. Venv ready

Already done if you followed the README, but to confirm:

```bash
cd /Users/stafordtituss/CharlemagneLabs/g4h
./.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
# Expect: True
```

---

## Running the notebook

```bash
cd /Users/stafordtituss/CharlemagneLabs/g4h
source .venv/bin/activate
jupyter lab notebooks/01_smoke_test.ipynb
```

The kernel `Python (g4h)` should be pre-selected. Run cells top-to-bottom.

Don't run the whole notebook in one click — pause after each step and check the output against the expected values below.

---

## What to verify, step by step

### Step 1 — Environment check

**Cell prints:** Python version, torch / transformers versions, MPS availability.

**Pass criteria:**
- `MPS available: True`
- `MPS built: True`
- `Using device: mps`

**If MPS is False:** you're on Intel Mac or torch was installed without MPS. Reinstall torch from the official wheel: `./.venv/bin/pip install --upgrade --force-reinstall torch`.

### Step 2 — Load Gemma 4 E4B-it

**What happens:** Downloads ~16 GB of bf16 weights on first run. Subsequent runs use the cache.

**Pass criteria:**
- No `OSError` about gated repo (means HF auth worked)
- No `OutOfMemoryError`
- Prints `config class`, `text-tower cfg`, `hidden_size`, `vocab_size`, `num_layers`. The top-level config is **composite** (`Gemma4Config` with a nested `text_config`) — the load cell already knows to read text-tower hyperparameters from `model.config.text_config`. Expected: `vocab_size=262144`, `num_layers` around 42.

**If you get gated-repo error:** Step back to Prerequisites #2 and accept the license + add the token.

**If OOM during load:** Activity Monitor → check unified memory pressure. Close other GPU-using apps. If it persists, try `dtype=torch.float16` instead of `bfloat16` — slightly less memory, very slightly worse numerics. (Note: transformers 5.x renamed the kwarg from `torch_dtype` to `dtype`.)

**Record the printed values** in the [Results template](#results-template) at the bottom.

### Step 3 — Verify `_get_inner_base_model` unwrap

**Cell prints:** `inner type`, `inner module path`, `inner has forward`, `inner has layers`, `num layers`.

Gemma 4 is structurally multimodal even on text-only checkpoints — `google/gemma-4-E4B-it` ships with `vision_tower`, `audio_tower`, `embed_vision`, and `embed_audio` siblings of `language_model` inside its `Gemma4Model` container. The helper handles three layouts:

  1. **PEFT wrapper**: `m.base_model.model` → `ForCausalLM`
  2. **Single-stack (Gemma 3)**: `ForCausalLM.model` → backbone with `.layers` (one step)
  3. **Multimodal composite (Gemma 4)**: `ForCausalLM.model` → `Gemma4Model` → `.language_model` → `Gemma4TextModel` with `.layers` (two steps)

**Pass criteria for Gemma 4:**
- `inner type: Gemma4TextModel`
- `inner module path: transformers.models.gemma4.modeling_gemma4`
- `inner has forward: True`
- `inner has layers: True`
- `num layers: 42`

**If `inner type` is `Gemma4Model`** (one step short): the helper landed on the multimodal composite. Run `[(n, type(c).__name__) for n, c in inner.named_children()]` and confirm `language_model` is among them — if so, the helper update merged but you're on a stale notebook; pull and restart the kernel. If something else (`text_model`, etc.), tell me and I'll add it to the helper.

**If `inner type` is `Gemma4ForCausalLM`** (zero steps): same fix — pull and restart. The previous helper version stopped at `.model`.

**Phase 2 follow-up flagged:** carrying `vision_tower` + `audio_tower` + multimodal embedders in RAM is wasted memory for a text-only classifier. They show up as ~30–40% of the model's parameters (rough; measure in Phase 2). Options to investigate when we get there: (a) freeze them and let `accelerate` skip activation memory, (b) instantiate `Gemma4TextModel.from_pretrained(MODEL_ID)` directly via the text-only sub-config, (c) custom load that drops the unused state-dict keys.

### Step 4 — One forward pass

**Cell prints:** `last_hidden_state` shape, dtype, NaN check, mean magnitude, pooled shape.

**Pass criteria:**
- Shape: `(2, T, hidden_size)` where `T` is the padded length (small, ~16) and `hidden_size` matches what step 2 reported.
- `any NaN: False`
- `mean abs` is non-trivial — somewhere between **0.05 and 5.0** is expected. If it's `0.0` or `nan`, something is broken upstream.
- Pooled shape: `(2, hidden_size)`.

**If NaN or near-zero magnitude:** PLE is misbehaving in this path. Skip to step 5 and see if the fallback works — if the fallback is also bad, the model load is broken; if only step 4 is bad, the unwrap is hitting an unfinished module.

### Step 5 — Unwrap-vs-fallback diff (the critical PLE check)

**Cell prints:** `max abs diff between inner.last_hidden_state and hidden_states[-1]: <number>`

**Pass criteria:**
- `max abs diff < 1e-3` → **OK**, use the unwrap. This is the green light.

**Yellow (1e-3 to 1e-2):** Probably bf16 numerical noise, but worth a closer look. Re-run with float32 to rule out precision (`torch_dtype=torch.float32`, will be slower).

**Red (> 1e-2):** PLE is genuinely doing something the inner-model forward skips. Two options:
1. **Fall back to `output_hidden_states=True` and `hidden_states[-1]`.** Costs us the lm_head bypass benefit — we'll need `memory_guard`-style cleanups during training. Pragmatic for the hackathon.
2. **Find a deeper module that includes PLE.** Read the Gemma 4 model code in transformers to see where PLE is applied; it might be in `Gemma4Model.forward` and what we want is `Gemma4Model` itself, not `Gemma4Model.layers[...]` directly.

**This is the most important step.** Capture the diff value verbatim in the results template.

### Step 6 — LoRA target module probe

**Cell prints:** any module name in layer 0 containing `proj`.

**Pass criteria:** see all four of `q_proj`, `k_proj`, `v_proj`, `o_proj` in the output.

**If only some are present:** Gemma 4 may use fused QKV (`qkv_proj` instead of separate q/k/v). Update `LORA_TARGET_MODULES` in `02_train_colab.ipynb` and `src/train.py` accordingly. PEFT supports `qkv_proj` as a target — same wrapper, different target list.

**If `inner` doesn't have `.layers`:** print `inner` and find the right attribute (could be `.encoder.layers`, `.transformer.layers`, etc.).

### Steps 7 & 8 — Final acceptance and sign-off

The notebook's last markdown cell has a checklist. Match it against what you observed and copy results into the template below.

---

## Results template

Paste this into a working doc (or just into chat with me) after running the notebook. Each line should have either a value, "OK", or a brief note.

```
PHASE 1 SMOKE TEST RESULTS
Run date:           __________
Hardware:           M4 Max, 36 GB unified
Model ID used:      __________________________________
Notebook commit:    $(git rev-parse --short HEAD)

Step 1 — Environment
  MPS available:    __________
  Device used:      __________

Step 2 — Load
  hidden_size:      __________
  vocab_size:       __________
  num_layers:       __________
  Peak memory:      __________  # check Activity Monitor during load
  Load time:        __________

Step 3 — Unwrap
  inner type:       __________
  inner module:     __________

Step 4 — Forward
  last_hidden_state shape:  __________
  any NaN:          __________
  mean abs:         __________

Step 5 — PLE check (CRITICAL)
  max abs diff:     __________
  Verdict:          OK / yellow / red — __________

Step 6 — LoRA targets
  proj modules found: __________
  Will use:         q_proj/k_proj/v_proj/o_proj  OR  qkv_proj  OR  __________

Notes / surprises:
  __________
```

---

## Sign-off checklist

Phase 1 is complete when all of these are checked.

- [ ] Model loads on MPS in bf16 without OOM
- [ ] `_get_inner_base_model` returns a Gemma 4 backbone module (not the CausalLM wrapper)
- [ ] `inner(...).last_hidden_state` has the expected `(B, T, hidden_size)` shape, no NaNs, non-trivial magnitude
- [ ] **Diff vs. `output_hidden_states=True` fallback is `< 1e-3`**, OR the divergence is understood and we've decided on a fallback path
- [ ] LoRA target modules confirmed (either Gemma 3's `q/k/v/o_proj` set or Gemma 4's actual names)
- [ ] Results captured in the template above

---

## What this unblocks

Once Phase 1 is green, Phase 2 fills in:

- **`src/model.py`** — the `CausalLMWithClassifier` wrapper, with `_get_inner_base_model` (verified in step 3) baked in. Phase 1's outputs tell us:
  - Which inner module type to expect (step 3)
  - Whether to use the unwrap or the `output_hidden_states=True` fallback (step 5)
  - The exact `hidden_size` to use for the `nn.Linear` head (step 2)
- **`src/train.py`** — the QLoRA training loop. Phase 1 tells us:
  - Which LoRA target modules to wrap (step 6)
  - Whether MPS is involved at all in training (no — we're going to Colab — but the model wrapper code needs to load identically on both)
- **`notebooks/02_train_colab.ipynb`** — gets its LoRA target list and any model-ID adjustments from Phase 1.

---

## Failure-mode quick reference

| Symptom | Most likely cause | Fix |
|---|---|---|
| `OSError: 401` during `from_pretrained` | HF token missing / license not accepted | Prerequisites #2 |
| `OSError: 404` during `from_pretrained` | Wrong `MODEL_ID` | Prerequisites #3 — find the real ID |
| OOM during load | Other apps using GPU memory | Quit them; or use `float16` |
| `inner type` = `Gemma4ForCausalLM` | Helper didn't unwrap deep enough | Print `model`, add another `.model` step |
| `mean abs` is `nan` or `0.0` | PLE skipped, model not initialized | See step 4 troubleshooting |
| `max abs diff > 1e-2` (step 5) | PLE applied between unwrap point and CausalLM head | See step 5 troubleshooting — this is the real risk |
| No `q_proj` etc. | Gemma 4 uses fused QKV | Update LoRA targets to `qkv_proj` |

---

When you're done, ping me with the results template filled in and we'll move to Phase 2.
