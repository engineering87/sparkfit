## sparkfit v0.3.0

This release sharpens sparkfit's accuracy for modern architectures and for
models served from local paths, validated against a real DGX Spark running
vLLM. **sparkfit** is an LLM memory capacity planner for the NVIDIA DGX Spark
(GB10): a 128 GB unified-memory, memory-bandwidth-bound machine (273 GB/s shared
between CPU and GPU).

### Highlights
- **Hybrid-attention KV-cache**: for models that interleave linear/SSM and
  full-attention layers (e.g. Qwen3.5), only the `full_attention` layers count
  toward the KV-cache. This corrects a large long-context overestimate (for the
  27B at 32k context, KV drops from ~8.6 GB to ~2.1 GB).
- **Exact on-disk weight size**: for local model directories sparkfit reads the
  real footprint from the safetensors index (or by summing the files), with a
  `--weights-gb` override. This avoids guessing for mixed-precision or quantized
  checkpoints (the 27B reads as ~35.9 GB rather than a ~24.4 GB estimate).
- **Sliding-window attention**: the resident KV-cache is capped at the window
  size for models that use it (Mistral, Gemma, and similar).
- **Vision-tower parameters** are now included in the memory footprint for
  multimodal models.

### Fixed
- Decode throughput now reflects concurrency: per-stream tok/s falls and
  aggregate tok/s scales as `-n` grows. Previously only the memory budget
  accounted for concurrent streams, so the reported speed stayed at the
  single-stream value regardless of `-n`.
- `--context` / `--batch` / `--concurrency` reject zero or negative values with
  a clear error instead of a nonsensical budget or a crash.
- Local-config models now show `[local]` in the header instead of `[built-in]`.
- A non-positive `--weights-gb` no longer crashes; it falls back to the estimate.
- Non-positive `--efficiency` / `--bandwidth` fall back to defaults instead of
  producing nonsensical speeds.
- Removed an unsupported `python_version` from the mypy config that broke recent
  mypy versions in CI.

### Install
```bash
pipx install git+https://github.com/engineering87/sparkfit.git
sparkfit llama3.1-70b
```
Or clone and run the single file: `python src/sparkfit.py llama3.1-70b`.

### Quality
Linted with ruff, type-checked with mypy, 57 tests, CI on Python 3.8 to 3.12.
Calibrated and validated on a real DGX Spark (Qwen3.5-27B FP8 under vLLM):
single-stream and concurrent decode within roughly 5% of measured throughput.

### Notes
These are capacity-planning estimates; cross-check with `sparkfit scan` on the
real machine. Auto-fetch needs internet; gated Hugging Face models need
`HF_TOKEN`.

**Full changelog**: https://github.com/engineering87/sparkfit/compare/v0.2.0...v0.3.0
