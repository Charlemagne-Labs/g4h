"""
CausalLMWithClassifier wrapper for Gemma 4 E4B.

Port target: gateguard-suite/gemma3_classifier_lora.py:54-109 + the lm_head bypass
from gateguard-suite/api_server/runners.py:156-175.

The wrapper:
  1. Loads an HF causal LM
  2. Bypasses the lm_head with `_get_inner_base_model` (mandatory — Gemma 4 vocab
     is still 262K, so the wasted lm_head logits tensor is the same problem at the
     same scale; ~196 MB per forward at bf16 / T=384). Do NOT use
     `output_hidden_states=True`.
  3. Pulls the last-token pooled hidden state (left-padded input)
  4. Feeds it into nn.Linear(hidden_size, num_labels)

Phase 1 unknowns to verify in the smoke test before this gets fleshed out:
  - What does `type(model.model).__name__` return for Gemma 4 E4B?
  - Does Per-Layer Embeddings (PLE) interfere with `model.model(...)` returning
    a sane `last_hidden_state`? If yes, fall back to `output_hidden_states=True`
    and `hidden_states[-1]`.
  - What's the right LoRA target_modules set? Try Gemma 3's
    ["q_proj", "k_proj", "v_proj", "o_proj"] first.
"""

# TODO(phase-2): port CausalLMWithClassifier here.
