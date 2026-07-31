# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-07-31

### Added
- Refreshed the built-in catalog with current models, with parameters derived from
  the real Hugging Face configs: `qwen3-8b/14b/32b`, the MoE `qwen3-30b-a3b` and
  `qwen3-235b-a22b`, `mistral-small-24b`, `phi-4`, and `deepseek-v3`.
- `--engine vllm|llama.cpp|ollama` on `plan`, `advise`, `fit`, `serve`, `cluster`
  and the quick report applies a per-engine bandwidth-efficiency preset for the run
  (an explicit `--efficiency` still wins).
- `scan` now reports GPU temperature and thermal-throttle status (read from
  `nvidia-smi`, best effort): a warm/hot verdict, and a warning when a hardware or
  software thermal slowdown is active, since throttling makes real decode run below
  sparkfit's estimate.
- `sparkfit doctor`: calibrate the tool to your device. From a measured single-stream
  decode tok/s it backs out the real bandwidth efficiency (`eff = tok_s * bytes_per_step
  / bandwidth`); `--engine vllm|llama.cpp|ollama` seeds a per-engine default; with no
  arguments it reports the device and the settings in effect. `--save` persists to a
  config file.
- Config file: commands now read defaults from `~/.sparkfitrc` (or `$SPARKFIT_CONFIG`),
  with precedence flag > environment variable > config file > built-in default, so a
  calibrated efficiency (and the other `SPARKFIT_*` knobs) can stick between runs.
- Cluster planner: `sparkfit cluster MODEL -N 2 ...` plans one model across several
  DGX Spark nodes. Weights, KV-cache and activations split evenly across the nodes
  (so models too big for one box can fit), the 200 Gb/s ConnectX fabric is surfaced
  as the second bottleneck, and decode throughput scales sub-linearly with node
  count (calibratable with `--scaling` / `--fabric-gbps`). Pipeline parallelism
  (default) reports aggregate throughput; tensor parallelism reports single-stream.

### Changed
- `cluster` now lets `--fabric-gbps` scale the multi-node throughput (a slower
  inter-node fabric lowers the speedup), instead of being informational only.
- `doctor` never persists a calibrated efficiency above 1.0, and the config file
  writer preserves comments and unrecognized keys instead of dropping them.

## [0.4.0] - 2026-07-15

### Added
- Packaging: tagged releases are built and published to PyPI automatically via
  GitHub Actions Trusted Publishing, so sparkfit can be installed with
  `pip install sparkfit` (or `pipx install sparkfit`).
- Co-serving planner: `sparkfit serve MODEL[:quant[:context]] ...` plans several
  models on one Spark at once. Their weights, KV-cache and activations sum against
  the unified 128 GB, and the shared 273 GB/s is split across the models decoding
  at the same time (worst case: each gets about its solo speed divided by the
  number active), so you can see both what co-resides and how much each slows down.
- Prefill / time-to-first-token roofline: `plan` and the quick report now estimate
  TTFT as `max(compute-bound, bandwidth-bound)`, so short prompts are limited by
  streaming the weights once and long prompts by compute. New `--prompt-tokens`
  (defaults to the context length), plus `--mfu` and `--compute-tflops` (with
  `SPARKFIT_MFU` / `SPARKFIT_TFLOPS`) to calibrate the compute side to a device.
- Quantization auto-detect: when a model's `config.json` carries a
  `quantization_config` (FP8, GPTQ, AWQ, bitsandbytes, compressed-tensors), the
  quick report uses the detected precision as the default quant instead of
  guessing, and labels it `(detected)`. Explicit `-q` still overrides it, and
  models without the field are unaffected.

## [0.3.0] - 2026-06-17

### Added
- Hybrid-attention KV-cache: for models that mix linear/SSM and full-attention
  layers (e.g. Qwen3.5), only the `full_attention` layers (via `layer_types` or
  `full_attention_interval`) count toward the KV-cache, fixing a large
  long-context overestimate.
- Real on-disk weight size for local model directories (read from the safetensors
  index or by summing the files), plus a `--weights-gb` override, for an exact
  footprint regardless of architecture or mixed precision.
- Vision-tower parameters are counted in the memory footprint for multimodal
  models.
- Sliding-window attention caps the resident KV-cache (Mistral, Gemma, ...).

### Fixed
- Decode throughput now accounts for concurrency in `plan`, `advise`, `fit` and
  the quick report: per-stream tok/s falls and aggregate tok/s scales as `-n`
  grows. Previously only the memory budget reflected concurrent streams, so the
  reported speed stayed at the single-stream value regardless of `-n`.
- `--context`, `--batch` and `--concurrency` reject zero or negative values with
  a clear error instead of producing a nonsensical budget (or, for `--batch 0`,
  crashing with a division error).
- Closed a file handle left open when reading the safetensors weight index.
- Local-config models showed `[built-in]` in the header instead of `[local]`.
- A non-positive `--weights-gb` no longer crashes; it falls back to the estimate.
- Non-positive `--efficiency` / `--bandwidth` fall back to defaults instead of
  producing nonsensical speeds.
- Removed an unsupported `python_version` from the mypy config that broke recent
  mypy versions in CI.

## [0.2.0] - 2026-06-16

### Added
- Local `config.json` support: `sparkfit <path>` reads a model config from any
  file or directory (handy for models served from local paths such as vLLM), with
  automatic flattening of multimodal `text_config`.
- `--efficiency` flag and `SPARKFIT_EFFICIENCY` / `SPARKFIT_BANDWIDTH` /
  `SPARKFIT_OS_RESERVE` / `SPARKFIT_FRAMEWORK` environment overrides, so estimates
  can be calibrated to a measured device.

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

[Unreleased]: https://github.com/engineering87/sparkfit/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/engineering87/sparkfit/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/engineering87/sparkfit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/engineering87/sparkfit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/engineering87/sparkfit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/engineering87/sparkfit/releases/tag/v0.1.0
