# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-15

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
- Context-aware guidance and "hypothetical" labeling when a configuration does not
  fit.
- Packaged for pip and pipx with a `sparkfit` console entry point; also usable as
  a single standalone script.

[Unreleased]: https://github.com/engineering87/sparkfit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/engineering87/sparkfit/releases/tag/v0.1.0
