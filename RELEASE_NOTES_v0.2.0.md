## sparkfit v0.2.0

Second release of **sparkfit**, the LLM memory capacity planner for the NVIDIA
DGX Spark (GB10). This release broadens model coverage and adds device
calibration.

### Highlights
- **Local config.json**: `sparkfit <path>` plans any model from a local file or
  directory, including multimodal configs nested under `text_config`. Ideal for
  models served from local paths such as vLLM.
- **MLA support (DeepSeek V2/V3/R1)**: models the compressed latent KV-cache,
  avoiding a roughly 50x overestimate, with DeepSeek fine-grained MoE parameter
  estimation and built-in `deepseek-v2-lite` and `deepseek-r1` entries.
- **Calibration**: `--efficiency` flag and `SPARKFIT_*` environment overrides to
  match the bandwidth roofline and reserves to a measured device.
- **Real-hardware validation**: on a DGX Spark, bandwidth efficiency about 0.85 to
  0.89, near-linear concurrency to 8 streams, and measurable contention when
  co-serving models.
- **Robustness**: `--total-mem` is validated; clearer `fit --concurrency-scan`
  message when nothing fits.

### Commands
`plan`, `advise`, `fit`, `scan`, `models`, plus a one-shot quick mode (pass a model
name, a partial name, a Hugging Face id, or a local config path). Every command
supports `--json`.

### Install
```bash
pipx install git+https://github.com/engineering87/sparkfit.git
sparkfit llama3.1-70b
```
Or clone and run the single file: `python src/sparkfit.py llama3.1-70b`.

### Quality
Linted with ruff, type-checked with mypy, 32 tests at about 80% coverage, CI on
Python 3.8 to 3.12.

### Notes
These are capacity-planning estimates, not measurements; cross-check with
`sparkfit scan` on the real machine. Auto-fetch needs internet; gated Hugging Face
models need `HF_TOKEN`. Hybrid linear-attention models (e.g. Qwen3.5) are not yet
modeled exactly on the KV-cache side.

**Full changelog**: https://github.com/engineering87/sparkfit/compare/v0.1.0...v0.2.0
