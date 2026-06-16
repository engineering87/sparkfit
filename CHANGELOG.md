# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `--total-mem` now rejects zero or negative values with a clear error message
  instead of raising an unhandled exception.

### Changed
- `fit --concurrency-scan` shows a clearer message when a model does not fit even
  at a single stream.
- CI now also runs mypy type-checking and enforces a minimum test coverage; the
  test suite was expanded with CLI integration tests.

## [0.1.0] - 2026-06-16

### Added
- Initial release.
- `plan`: unified-memory budget (weights, KV-cache, activations, OS reserve,
  CUDA/framework) plus a memory-bandwidth roofline estimate of decode tok/s.
- `advise`: recommends the highest-quality quantization that fits with a safe
  (10%) margin.
- `fit`: scans the built-in model database for what fits; `--concurrency-scan`
  finds the maximum number of parallel streams for a model.
- `scan`: reads live memory on the device (`nvidia-smi` and `/proc/meminfo`).
- `models`: lists the built-in model database.
- One-shot quick mode: pass just a model name (`sparkfit llama3.1-70b`), a partial
  name (`sparkfit "llama 70"`), or a Hugging Face repo id (auto-fetches
  `config.json` and estimates parameters).
- `--live` flag: plan against memory free right now instead of the theoretical
  total.
- Multi-head Latent Attention (MLA) support for DeepSeek V2/V3/R1: the KV-cache is
  modeled as a compressed latent (`kv_lora_rank + qk_rope_head_dim`) per layer,
  avoiding a large overestimation of the per-head formula.
- Parameter estimation for DeepSeek fine-grained MoE (routed and shared experts,
  `first_k_dense_replace` dense prefix) and MLA attention projections. Validated:
  DeepSeek-V2-Lite ~15.7B, DeepSeek-V3 ~671B total / ~37B active.
- Built-in catalog entries `deepseek-v2-lite` and `deepseek-r1`.
- Context-aware guidance and "hypothetical" labeling when a configuration does not
  fit.
- Packaged for pip and pipx with a `sparkfit` console entry point; also usable as
  a single standalone script.

[Unreleased]: https://github.com/engineering87/sparkfit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/engineering87/sparkfit/releases/tag/v0.1.0
