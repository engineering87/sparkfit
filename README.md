<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="sparkfit" src="assets/logo.svg" width="360">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/engineering87/sparkfit/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/engineering87/sparkfit/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
  <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <a href="https://github.com/astral-sh/ruff"><img alt="Linting: Ruff" src="https://img.shields.io/badge/linting-ruff-261230?style=flat-square&logo=ruff&logoColor=white"></a>
  <img alt="Zero dependencies" src="https://img.shields.io/badge/dependencies-zero-success?style=flat-square">
</p>

**LLM memory capacity planner for the NVIDIA DGX Spark (GB10).**

Catalog tools answer "does this model fit on my hardware, yes or no?" on generic
GPUs. sparkfit treats the DGX Spark as what it is: a 128 GB unified-memory machine
that is, above all, memory-bandwidth bound (273 GB/s shared between CPU and GPU).
So it answers three questions a fit/no-fit catalog does not:

1. Unified-memory budgeting. How the 128 GB is really split between weights,
   KV-cache, activations, the CUDA/serving framework, and the slice you must leave
   for the Grace CPU side and the OS.
2. Will it actually be fast? A memory-bandwidth roofline estimate of decode
   throughput (tokens/s). On Spark, "it fits" and "it is usable" are two different
   things, and bandwidth, not capacity, is the limit.
3. How to make it fit or scale. A quantization advisor (the highest-quality format
   that fits with a safety margin) and a concurrency planner (how many parallel
   streams or model replicas fit at once).

Pure Python standard library. No dependencies.

## Install

With [pipx](https://pipx.pypa.io/) (recommended; isolated, global `sparkfit`
command):

```bash
pipx install git+https://github.com/engineering87/sparkfit.git
```

Or with pip:

```bash
pip install git+https://github.com/engineering87/sparkfit.git
```

Or clone and run the single file, no install needed (handy for pasting onto a
fresh DGX Spark):

```bash
git clone https://github.com/engineering87/sparkfit.git
cd sparkfit
python src/sparkfit.py llama3.1-70b  # or: ./sparkfit llama3.1-70b
```

## Quick start

Pass just a model name and sparkfit does the rest: it auto-selects the
highest-quality quantization that fits with margin, a typical context (8192), and
shows the budget, the verdict, estimated tok/s, and the alternatives.

```bash
sparkfit llama3.1-70b          # built-in catalog id
sparkfit "llama 70"            # partial match -> llama3.1-70b
sparkfit 72b                   # even a fragment works
sparkfit Qwen/Qwen2.5-7B       # Hugging Face id -> reads params online
```

```text
  NVIDIA DGX Spark (GB10)  |  llama3.1-70b  |  fp8 (auto)  |  ctx 8192
  71B params | 80 layers | hidden 8192 | kv_heads 8 | [built-in]

  Weights           70.6 GB  ██████████████████░░░░░░░░░░░░░░
  KV-cache           2.7 GB  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  Activations        0.3 GB  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  OS reserve         8.0 GB  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  CUDA/framework     2.0 GB  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  --------------------------------------------------------
  USED              83.6 GB  █████████████████████░░░░░░░░░░░ 65.3%
  of               128.0 GB

  ✓ FITS  | 44.4 GB headroom

  Decode speed (memory-bandwidth roofline)
       2.6 tok/s per stream
    feel: slow - bandwidth bound
```

Override any default: `sparkfit qwen2.5-72b -c 16384 -n 4 -q q5_k_m`.

### `--live`: plan against the memory free right now

Run on the Spark, `--live` budgets against currently-free memory (read from
`/proc/meminfo` and `nvidia-smi`) instead of the theoretical 128 GB, and waives
the OS reserve, since the free figure already excludes the OS and running
processes. Useful when you already have models or containers loaded and want to
know what you can still add.

```bash
sparkfit llama3.1-8b --live
sparkfit qwen2.5-32b --live
```

## Commands

| Command | What it does |
|---|---|
| *(none)* | Pass just a model and get a one-shot smart report with sensible defaults |
| `plan`   | Full unified-memory budget plus a decode-speed roofline for one configuration |
| `advise` | Recommends the highest-quality quant that fits with a 10% margin |
| `fit`    | Scans the model DB for what fits; `--concurrency-scan` finds max parallel streams |
| `scan`   | Reads live memory when run on the Spark |
| `models` | Lists the built-in model database |

Every command supports `--json` for machine-readable output (CI, dashboards,
scripting).

### Key flags

`-q/--quant` (default `q4_k_m` for `plan`, `advise`, `fit`; auto in quick mode),
`-c/--context`, `-b/--batch`, `-n/--concurrency`,
`--kv-dtype {fp16,fp8,int8,q4}`, `--os-reserve` (GB, default 8),
`--framework` (GB, default 2), `--total-mem`, `--bandwidth`, `--live`, `--json`.
For models not in the DB: `--params --active --layers --hidden --kv-heads --head-dim`.

## Methodology

All assumptions are explicit constants near the top of `src/sparkfit.py` and can be
overridden from the command line.

- Weights = `params * bits_per_weight / 8`. The `QUANT_BITS` table includes the
  real overhead of GGUF formats (for example `q4_k_m` is about 4.85 bits, not
  4.0).
- KV-cache per token = `2 * layers * kv_heads * head_dim * dtype_bytes` (models
  GQA via `kv_heads`); total = times context times streams.
- Decode speed (roofline): token generation is memory-bound. Each step streams the
  weights once (active weights only, for MoE) plus the whole KV-cache, so
  `tok/s ~= bandwidth * efficiency / bytes_per_step`, with a default 70% of
  273 GB/s. Prefill/compute-bound time is not modeled (Spark is fast at prefill;
  the practical limit is decode).
- OS and framework reserve: on unified memory the CPU and GPU draw from the same
  pool, so by default 8 GB (OS/DGX plus the Grace side) and 2 GB (CUDA context
  plus serving framework) are reserved. Both are adjustable; `--live` sets the OS
  reserve to 0 because the live-free figure already excludes it.
- Hugging Face auto-fetch reads `config.json` and estimates parameters from the
  architecture (embeddings, attention, MLP/MoE). Validated against real configs:
  Qwen2.5-7B gives 7.62B (actual 7.61B), Mixtral-8x7B gives 46.7B total and 12.9B
  active.

These are capacity-planning estimates, not measurements. For exact numbers,
cross-check with `sparkfit scan` on the real machine.

## DGX Spark (GB10) reference

128 GB unified LPDDR5x, 273 GB/s shared CPU+GPU bandwidth, Blackwell GPU with FP4
(about 1 PFLOP / 1000 TOPS), 20-core Grace Arm CPU. The 273 GB/s bandwidth is the
dominant constraint for token generation.

## Roadmap

- Local cache of fetched Hugging Face configs.
- Expanded built-in model database.
- Optional Go single-binary port.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).
Issues and pull requests are welcome.

## License

[MIT](LICENSE).
