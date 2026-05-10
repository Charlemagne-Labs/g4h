# Data

Dataset choice is **deferred** until the Phase 1 smoke test passes (per issue #73 open question 1).

## Candidates

1. **Reuse gateguard phishing data** — fastest path. `gemma_v35_audit_results.csv` from `Charlemagne-Labs/gateguard-suite` gives a clean head-to-head F1 vs. the 270M baseline.
2. **Hackathon-aligned task** — long-context email-thread classification, multimodal screenshots, etc. More compelling demo, more dataset work.

Whatever lands here:
- A pointer in this README to the source-of-truth CSV / parquet location.
- A small (≤1 MB) sample committed under `data/sample/` so the smoke notebook is reproducible.
- Real datasets stay out of git — the `.gitignore` already excludes everything in `data/` except this README and `data/sample/`.
