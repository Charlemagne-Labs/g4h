"""
Inference entrypoint for the Gemma 4 E4B classifier head.

Port target:
  - Loader: gateguard-suite/gemma3_classifier_lora.py:224-298 (load_for_inference)
  - Single-prediction CLI: gateguard-suite/gemma3_classifier_lora.py:299-323 (predict_one)

Loader must apply the same lm_head bypass via _get_inner_base_model that the
training wrapper uses (see src/model.py). The bypass is what keeps inference
memory honest at 262K vocab — ~196 MB per forward saved in bf16 at T=384.

Public API (target):
  load_for_inference(artifact_dir: str) -> ClassifierBundle
  predict_one(bundle: ClassifierBundle, text: str) -> dict
      # returns {"label": str, "scores": dict[str, float]}
"""

# TODO(phase-3): port load_for_inference + predict_one here.

if __name__ == "__main__":
    raise NotImplementedError("Phase 3: implement predict_one CLI")
