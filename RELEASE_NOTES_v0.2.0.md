## sparkfit v0.2.0

Broader model coverage, device calibration, and real-hardware validation.

### Added
- Local `config.json` support: `sparkfit <path>` plans any model from a local file
  or directory, including multimodal configs that nest fields under `text_config`.
  Ideal for models served from local paths (e.g. vLLM).
- Multi-head Latent Attention (MLA) for DeepSeek V2/V3/R1: the compressed latent
  KV-cache is modeled correctly, avoiding a large per-head overestimate. Parameter
  estimation now also covers DeepSeek fine-grained MoE. Built-in entries
  `deepseek-v2-lite` and `deepseek-r1`.
- `--efficiency` flag and `SPARKFIT_*` environment overrides to calibrate the
  bandwidth roofline and reserves to a measured device.

### Fixed
- `--total-mem` rejects zero or negative values with a clear message instead of
  crashing.

### Changed
- Clearer `fit --concurrency-scan` message when nothing fits.
- CI adds mypy type-checking and enforced coverage; expanded the test suite.

### Validated on a real DGX Spark
- Bandwidth efficiency about 0.85 to 0.89 (vs the conservative 0.70 default).
- Aggregate throughput scales near-linearly to 8 concurrent streams (within ~5%).
- Co-serving a second model cuts decode speed (shared 273 GB/s); model your share
  with `--bandwidth` or `--efficiency`.

**Full changelog**: https://github.com/engineering87/sparkfit/compare/v0.1.0...v0.2.0
