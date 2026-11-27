# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-11-27

### Added
- MoE batched decode models expert coverage: with B tokens per step, a model with E
  routed experts and top-k routing reads about `E * (1 - (1 - k/E)^B)` distinct
  experts, not just the active parameters. Batched MoE throughput was previously
  overstated several-fold (gpt-oss-120b, MXFP4, `--engine vllm -c 512 -n 64`:
  ~2,900 tok/s aggregate estimated before, ~260 now, versus ~680-700 measured
  counting input plus output tokens, ~350 output only).
  Built-in MoE models carry `n_experts`/`top_k`, Hugging Face configs provide
  them, and custom models take `--experts` and `--top-k`.
- Decode adds a fixed cost per step (kernel launches, sampling, scheduling, MoE
  routing) on top of streaming the bytes: 4 ms by default, `--step-overhead-ms` /
  `SPARKFIT_STEP_MS` / `step_ms` in the config, and an `--engine` preset (vLLM 4,
  llama.cpp 1, ollama 2). Small and few-active-parameter MoE models were estimated
  up to 2x too fast (qwen3-30b-a3b NVFP4 on vLLM: ~105 tok/s before, ~74 now,
  ~47-67 measured); 8B+ dense models barely move. `plan --json` reports the whole
  decode step as `decode_tok_s.step_time_ms`. `doctor --measured-tok-s`
  subtracts the overhead before backing out the efficiency (the README example
  now calibrates to ~0.92 instead of ~0.89) and saves both values together, so
  later runs reproduce the measurement; `doctor --engine --save` saves the preset
  pair (an explicit `--step-overhead-ms` wins).
- `doctor --measured-ttft SECONDS` (with `--model`, `--prompt-tokens`, `--engine`)
  backs out the prefill MFU from a measured time-to-first-token and can save it.
  It can be combined with `--measured-tok-s` in one run, and refuses to save an
  MFU measured at the weight-read bandwidth floor.
- Quantization auto-detect now recognizes the Blackwell-native 4-bit formats: MXFP4
  (`quant_method: "mxfp4"`, as shipped by gpt-oss) and NVFP4 (ModelOpt `quant_algo`,
  and compressed-tensors 4-bit float weights). ModelOpt FP8 is detected too.
- `serve` now honours `-b/--batch` and `-n/--concurrency` (streams per model, which
  grow each model's KV-cache and split its decode speed) and reports a per-model
  prefill / time-to-first-token estimate in `--json` (`prefill_ttft_s`).
- `fit --concurrency-scan` reports when it hits the 256-stream scan limit
  (`"capped": true` in JSON, `256+` in text).

### Changed
- Precedence of efficiency and step overhead is now flag > `SPARKFIT_*` > config
  file > `--engine` preset > default: a value calibrated with `doctor` is no
  longer silently replaced by the generic preset when `--engine` is passed.
- `serve` resolves each model's quant as per-model suffix > explicit `-q` >
  quant detected from the config > `q4_k_m` (an explicit `-q` used to lose to the
  detected one), matching `cluster`.
- `--experts`/`--top-k` must be given together, on a custom MoE (`--active` <
  `--params`); they were silently ignored otherwise.
- Quantization auto-detect also maps `fbgemm_fp8` to fp8 and 2/5/6-bit GPTQ-style
  configs to `q2_k`/`q5_0`/`q6_k`; a multimodal MoE's vision tower is no longer
  counted in the routed-expert pool.
- Prefill roofline recalibrated to measured GB10 compute: dense BF16 base 100
  TFLOPS (was the ~125 theoretical), FP4 speedup 3.5x (was 4x), no precision
  speedup on engines that dequantize (`--engine llama.cpp`/`ollama`), and half
  the MFU for MoE grouped GEMMs. Prefill was overstated up to ~7x for MoE models
  (gpt-oss-120b on llama.cpp: ~12,600 tok/s estimated before, ~1,400 now, ~1,700
  measured).
- A measured `--weights-gb` (or local model size) for a MoE now contributes only
  its streamed share to decode, instead of the whole model per step.
- `cluster` and `doctor` default to the quantization detected from the model's
  config (as `quick` and `serve` already did), falling back to `q4_k_m`.
- `cluster` caps the fabric factor at the 200 Gb/s reference, so N nodes never
  report more than N x one node, and rejects a non-positive `--fabric-gbps` or a
  `--scaling` outside 0..1.
- Numeric options are validated up front: `--efficiency` and `--mfu` must be in
  (0, 1]; `--os-reserve` and `--framework` >= 0; `--total-mem`, `--bandwidth`,
  `--compute-tflops` and `--weights-gb` > 0. Out-of-range values (including from
  `SPARKFIT_*` or the config file) now stop with a clear message instead of being
  silently replaced by the default.
- Live telemetry (`scan`, `--live`, `doctor`) converts nvidia-smi MiB and
  `/proc/meminfo` KiB to decimal GB, consistent with the rest of the tool
  (previously GiB were treated as GB, understating free memory by about 7%).
- Thermal-throttle status is read from `clocks_event_reasons.*` on newer drivers,
  falling back to the older `clocks_throttle_reasons.*`.
- With measured on-disk weights, the prefill estimate uses the detected
  quantization (e.g. FP8) for the compute speedup.
- Python 3.10+ is now required (3.8 and 3.9 are end-of-life); CI covers 3.10-3.14.
- Packaging: the version is read from `sparkfit.__version__` (single source), the
  license uses an SPDX expression, and the release workflow runs the tests and
  checks that the tag matches the package version before publishing.
- Security: CodeQL analysis (`security-extended` queries) of the Python code and of
  the GitHub workflows on every push and pull request to `main`/`release/**` and
  weekly, with a README badge.
- CI dependencies updated: `actions/checkout@v7`, `actions/setup-python@v7`,
  `actions/upload-artifact@v7`, `actions/download-artifact@v8` (Node 24; a hash
  mismatch on download now fails the release), and dev tooling floors raised to
  the tested versions (`pytest>=9`, `pytest-cov>=7`, `ruff>=0.17`, `mypy>=2`).
  Dependabot now proposes weekly grouped updates for actions and dev tools.
- Docs: the README covers `cluster`, quantization auto-detect, the `serve` spec
  format and option ranges, and fixes the calibration example (6.8 tok/s on
  35.9 GB is about 0.89 efficiency, not 0.85); SECURITY lists supported versions
  and what leaves the machine (the optional `HF_TOKEN`); `--help` shows more
  examples and the option ranges.

### Fixed
- Windows: redirecting or piping the output (`sparkfit llama3.1-8b > out.txt`) no
  longer crashes with `UnicodeEncodeError`; such streams are switched to UTF-8.
- `serve` accepts Windows paths (`C:\models\x[:quant[:ctx]]`) instead of splitting
  them at the drive letter, and Hugging Face URLs with a suffix
  (`https://huggingface.co/org/model:fp8`).
- `doctor` validates `SPARKFIT_EFFICIENCY`, `SPARKFIT_MFU` and `SPARKFIT_STEP_MS`
  like the other commands, and its settings report shows `--bandwidth` and
  `--step-overhead-ms` when given.
- The config file is re-read on every run, so `$SPARKFIT_CONFIG` set after import
  takes effect; the test suite no longer depends on the developer's `~/.sparkfitrc`.
- The built-in gpt-oss-20b and gpt-oss-120b default to their shipped MXFP4 format
  in quick mode, `serve`, `cluster` and `doctor` (shown as `mxfp4 (native)`);
  quick mode used to pick `q6_k` for gpt-oss-120b, about 96 GB instead of ~62 GB.
- The PyPI project page shows the logo and demo (README images use absolute URLs),
  and the source distribution includes `tests/conftest.py`, so its tests no longer
  read the developer's `~/.sparkfitrc`.

### Removed
- A stale copy of the source (`outputs_sparkfit_copy.py`) that had been committed
  by mistake.

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

[Unreleased]: https://github.com/engineering87/sparkfit/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/engineering87/sparkfit/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/engineering87/sparkfit/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/engineering87/sparkfit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/engineering87/sparkfit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/engineering87/sparkfit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/engineering87/sparkfit/releases/tag/v0.1.0
