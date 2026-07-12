## sparkfit v0.1.0

First public release. **sparkfit** is an LLM memory capacity planner for the
NVIDIA DGX Spark (GB10).

Unlike fit/no-fit catalogs, sparkfit treats the Spark as what it is: a 128 GB
unified-memory, **memory-bandwidth-bound** machine (273 GB/s shared between CPU
and GPU). It tells you not just whether a model fits, but how the memory is spent,
whether decoding will be fast, and how to make a model fit or scale.

### Highlights
- **Unified-memory budgeting**: weights, KV-cache, activations, CUDA/framework,
  and the OS/Grace reserve, against the real 128 GB.
- **Bandwidth-aware decode roofline**: estimates tokens/s, because on Spark "it
  fits" and "it is usable" are different things.
- **Quantization advisor and concurrency planner**: the highest-quality quant that
  fits with margin, and how many parallel streams fit at once.
- **MLA support (DeepSeek V2/V3/R1)**: models the compressed latent KV-cache,
  avoiding a roughly 50x overestimate of the per-head formula.
- **Hugging Face auto-fetch**: pass any repo id and parameters are estimated from
  `config.json`.
- **`--live`**: plan against the memory free right now, on the device.
- **Zero dependencies**, pure Python standard library, Python 3.8+.

### Commands
`plan`, `advise`, `fit`, `scan`, `models`, plus a one-shot quick mode (just pass a
model name). Every command supports `--json`.

### Install
```bash
pipx install git+https://github.com/engineering87/sparkfit.git
sparkfit llama3.1-70b
```
Or clone and run the single file: `python src/sparkfit.py llama3.1-70b`.

### Quality
Linted with ruff, type-checked with mypy, 24 tests at ~78% coverage, CI on Python
3.8 to 3.12.

### Notes
These are capacity-planning estimates, not measurements; cross-check with
`sparkfit scan` on the real machine. Auto-fetch needs internet; gated Hugging Face
models need `HF_TOKEN`.

**Full changelog**: https://github.com/engineering87/sparkfit/commits/v0.1.0
