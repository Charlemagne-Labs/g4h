"""
Training entrypoint for the Gemma 4 E4B classifier head.

Port target:
  - Training loop: gateguard-suite/gemma3_classifier_lora.py:366-430 (train_pure_torch)
  - Save artifacts: gateguard-suite/gemma3_classifier_lora.py:188-222 (save_artifacts)
  - Split / class-weighting helpers: gateguard-suite/gemma3_classifier_lora.py:113-160

QLoRA config (per issue #73):
  BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
  )

LoRA hyperparameters:
  rank 8-16, alpha 16-32, lr 1e-4, 1-3 epochs, batch 4-8.

Cut from the gemma3 version unless the dataset demands it:
  - focal loss
  - class weighting

Save shape (matches the gemma3 inference_config.json schema so artifacts are
interchangeable across the two repos):
  out_dir/
    adapter/                 # PEFT save_pretrained
    classifier_head.pt       # nn.Linear weights for the head
    inference_config.json    # max_length, label2id, is_merged, hidden_size
    label_map.json
"""

# TODO(phase-2): port train_pure_torch + save_artifacts here.

if __name__ == "__main__":
    raise NotImplementedError("Phase 2: see notebooks/02_train_colab.ipynb")
