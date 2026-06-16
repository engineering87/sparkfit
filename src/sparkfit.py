#!/usr/bin/env python3
"""
sparkfit - DGX Spark capacity planner for LLM inference.

Unlike "does it fit?" catalog tools, sparkfit treats the DGX Spark for what it
really is: a 128 GB unified-memory, bandwidth-bound machine (273 GB/s shared
between CPU and GPU). So it answers three questions catalog tools don't:

  1. UNIFIED BUDGET  - how the 128 GB is really split between weights, KV-cache,
                       activations, CUDA/framework and the OS reserve you must
                       leave for the Grace CPU side.
  2. WILL IT BE FAST - a memory-bandwidth roofline that estimates decode tok/s,
                       because on Spark "it fits" and "it's usable" are different
                       things.
  3. HOW TO MAKE IT FIT / SCALE - a quantization advisor and a concurrency planner
                       (how many parallel streams / model replicas fit at once).

Pure standard-library Python 3.8+. No external dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Constants & hardware profile
# ---------------------------------------------------------------------------

GB = 1_000_000_000  # decimal GB, matches NVIDIA's "128 GB" and common LLM rules

# DGX Spark (GB10 Grace Blackwell) reference profile. Override via flags.
SPARK: dict = {
    "name": "NVIDIA DGX Spark (GB10)",
    "total_mem_gb": 128.0,      # advertised unified LPDDR5x
    "bandwidth_gbps": 273.0,    # 273 GB/s shared CPU+GPU, the real bottleneck
    "fp4_tops": 1000.0,         # ~1 PFLOP FP4 (sparse)
}

# Default chunks of memory you cannot use for a model on a unified system.
DEFAULT_OS_RESERVE_GB = 8.0     # DGX OS + Grace CPU side working set
DEFAULT_FRAMEWORK_GB = 2.0      # CUDA context + serving framework overhead
BW_EFFICIENCY = 0.70            # fraction of peak bandwidth actually reached

# Thresholds (named so they are not magic numbers scattered through the code).
BAR_WARN_FRAC = 0.75            # utilization at/above which a bar turns yellow
BAR_CRIT_FRAC = 0.95            # utilization at/above which a bar turns red
SAFE_MARGIN = 0.10              # min free fraction the advisor treats as "safe"
INTERACTIVE_TOK_S = 20.0        # decode speed at/above which it feels interactive
USABLE_TOK_S = 8.0              # decode speed at/above which it is still usable

# Bits-per-weight for common quantization formats (incl. real GGUF overhead).
QUANT_BITS = {
    "fp16": 16.0, "bf16": 16.0, "fp32": 32.0,
    "fp8": 8.0, "int8": 8.0, "q8_0": 8.5,
    "q6_k": 6.56, "q5_k_m": 5.5, "q5_0": 5.5,
    "q4_k_m": 4.85, "q4_0": 4.5,
    "nvfp4": 4.25, "fp4": 4.25, "mxfp4": 4.25,
    "q3_k_m": 3.9, "q2_k": 3.35,
}

# Quant ladder from heaviest to lightest (for the advisor).
QUANT_LADDER = ["fp16", "fp8", "q6_k", "q5_k_m", "q4_k_m", "nvfp4", "q3_k_m", "q2_k"]

# KV-cache element bytes by cache dtype.
KV_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "q4": 0.5}

# Built-in model database. Architecture fields drive accurate KV-cache math.
# total_b / active_b in billions of params (active < total only for MoE).
MODELS: dict[str, dict] = {
    "llama3.2-1b":   dict(total_b=1.24,  active_b=1.24,  layers=16, hidden=2048,  kv_heads=8,  head_dim=64),
    "llama3.2-3b":   dict(total_b=3.21,  active_b=3.21,  layers=28, hidden=3072,  kv_heads=8,  head_dim=128),
    "llama3.1-8b":   dict(total_b=8.03,  active_b=8.03,  layers=32, hidden=4096,  kv_heads=8,  head_dim=128),
    "llama3.1-70b":  dict(total_b=70.6,  active_b=70.6,  layers=80, hidden=8192,  kv_heads=8,  head_dim=128),
    "llama3.1-405b": dict(total_b=405.0, active_b=405.0, layers=126,hidden=16384, kv_heads=8,  head_dim=128),
    "mistral-7b":    dict(total_b=7.24,  active_b=7.24,  layers=32, hidden=4096,  kv_heads=8,  head_dim=128),
    "mixtral-8x7b":  dict(total_b=46.7,  active_b=12.9,  layers=32, hidden=4096,  kv_heads=8,  head_dim=128, moe=True),
    "qwen2.5-7b":    dict(total_b=7.6,   active_b=7.6,   layers=28, hidden=3584,  kv_heads=4,  head_dim=128),
    "qwen2.5-14b":   dict(total_b=14.7,  active_b=14.7,  layers=48, hidden=5120,  kv_heads=8,  head_dim=128),
    "qwen2.5-32b":   dict(total_b=32.5,  active_b=32.5,  layers=64, hidden=5120,  kv_heads=8,  head_dim=128),
    "qwen2.5-72b":   dict(total_b=72.7,  active_b=72.7,  layers=80, hidden=8192,  kv_heads=8,  head_dim=128),
    "gemma2-9b":     dict(total_b=9.24,  active_b=9.24,  layers=42, hidden=3584,  kv_heads=8,  head_dim=256),
    "gemma2-27b":    dict(total_b=27.2,  active_b=27.2,  layers=46, hidden=4608,  kv_heads=16, head_dim=128),
    "gpt-oss-20b":   dict(total_b=21.0,  active_b=3.6,   layers=24, hidden=2880,  kv_heads=8,  head_dim=64,  moe=True),
    "gpt-oss-120b":  dict(total_b=117.0, active_b=5.1,   layers=36, hidden=2880,  kv_heads=8,  head_dim=64,  moe=True),
    "deepseek-v2-lite": dict(total_b=15.7,  active_b=2.4,  layers=27, hidden=2048, kv_heads=16,  head_dim=128, moe=True, kv_style="mla", mla_dim=576),
    "deepseek-r1":      dict(total_b=671.0, active_b=37.0, layers=61, hidden=7168, kv_heads=128, head_dim=128, moe=True, kv_style="mla", mla_dim=576),
}

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text: str, code: str) -> str:
    """Wrap text in an ANSI color code, unless color output is disabled."""
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return c(t, "1")


def dim(t: str) -> str:
    return c(t, "2")


def red(t: str) -> str:
    return c(t, "31")


def green(t: str) -> str:
    return c(t, "32")


def yellow(t: str) -> str:
    return c(t, "33")


def bar(frac: float, width: int = 32) -> str:
    """Render a colored progress bar for a 0..1 fraction."""
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    color = green if frac < BAR_WARN_FRAC else (yellow if frac < BAR_CRIT_FRAC else red)
    return color("█" * filled) + dim("░" * (width - filled))


# ---------------------------------------------------------------------------
# Core memory & throughput math
# ---------------------------------------------------------------------------

def quant_bits(q: str) -> float:
    """Return bits-per-weight for a quantization name (case-insensitive)."""
    q = q.lower()
    if q not in QUANT_BITS:
        raise SystemExit(f"Unknown quant '{q}'. Known: {', '.join(QUANT_BITS)}")
    return QUANT_BITS[q]


def weight_bytes(params_b: float, quant: str) -> float:
    """Bytes occupied by `params_b` billion weights at the given quantization."""
    return params_b * 1e9 * quant_bits(quant) / 8.0


def kv_per_token_bytes(spec: dict, kv_dtype: str = "fp16") -> float:
    """Bytes of KV-cache consumed per generated token, per sequence.

    MLA models (DeepSeek V2/V3/R1) keep one compressed latent per layer instead
    of per-head K/V, so their cache is far smaller than a standard MHA/GQA model.
    """
    if spec.get("kv_style") == "mla":
        return spec["layers"] * spec["mla_dim"] * KV_BYTES[kv_dtype]
    return 2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"] * KV_BYTES[kv_dtype]


def kv_total_bytes(spec: dict, context: int, batch: int, kv_dtype: str = "fp16") -> float:
    """Total KV-cache bytes for `context` tokens across `batch` sequences."""
    return kv_per_token_bytes(spec, kv_dtype) * context * batch


def activation_bytes(spec: dict, context: int, batch: int) -> float:
    """Rough transient activation working set (small vs weights/KV)."""
    return batch * context * spec["hidden"] * 2 * 2  # two live buffers, fp16


def budget(spec: dict, quant: str, context: int, batch: int, concurrency: int = 1,
           kv_dtype: str = "fp16", os_reserve: float = DEFAULT_OS_RESERVE_GB,
           framework: float = DEFAULT_FRAMEWORK_GB,
           total_mem: float | None = None) -> dict:
    """Return the full unified-memory breakdown (all values in bytes)."""
    total = (total_mem if total_mem is not None else SPARK["total_mem_gb"]) * GB
    if total <= 0:
        raise SystemExit("--total-mem must be a positive number of GB.")
    streams = batch * concurrency
    w = weight_bytes(spec["total_b"], quant)
    kv = kv_total_bytes(spec, context, streams, kv_dtype)
    act = activation_bytes(spec, context, streams)
    overhead = (os_reserve + framework) * GB
    used = w + kv + act + overhead
    return {
        "total": total,
        "weights": w,
        "kv": kv,
        "activations": act,
        "os_reserve": os_reserve * GB,
        "framework": framework * GB,
        "overhead": overhead,
        "used": used,
        "free": total - used,
        "fits": used <= total,
        "util": used / total,
    }


def decode_tok_s(spec: dict, quant: str, context: int, batch: int = 1,
                 kv_dtype: str = "fp16", bandwidth: float | None = None,
                 efficiency: float = BW_EFFICIENCY) -> dict:
    """Memory-bandwidth roofline for decode (token generation) speed.

    Decode is memory-bound: each step streams the (active) weights once plus the
    whole KV-cache. This is exactly why Spark's 273 GB/s caps tok/s.
    """
    bw = (bandwidth if bandwidth is not None else SPARK["bandwidth_gbps"]) * 1e9 * efficiency
    w_active = weight_bytes(spec["active_b"], quant)
    kv_step = kv_per_token_bytes(spec, kv_dtype) * context * batch
    bytes_per_step = w_active + kv_step
    agg = batch * bw / bytes_per_step      # aggregate tokens/s across the batch
    per = agg / batch                      # tokens/s seen by one stream
    return {"aggregate": agg, "per_stream": per, "bytes_per_step": bytes_per_step}


def fmt_gb(b: float) -> str:
    """Format a byte count as a right-aligned 'NNN.N GB' string."""
    return f"{b / GB:6.1f} GB"


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

def speed_verdict(tok_s: float) -> str:
    """Colored one-word judgment of a decode speed."""
    if tok_s >= INTERACTIVE_TOK_S:
        return green("interactive")
    if tok_s >= USABLE_TOK_S:
        return yellow("usable")
    return red("slow - bandwidth bound")


def render_budget_table(bd: dict) -> None:
    """Print the unified-memory breakdown: per-component bars, used, capacity."""
    for label, val in (("Weights", bd["weights"]), ("KV-cache", bd["kv"]),
                       ("Activations", bd["activations"]), ("OS reserve", bd["os_reserve"]),
                       ("CUDA/framework", bd["framework"])):
        print(f"  {label:<16}{fmt_gb(val)}  {bar(val / bd['total'])}")
    print(dim("  " + "─" * 56))
    print(f"  {'USED':<16}{fmt_gb(bd['used'])}  {bar(bd['util'])} {bd['util'] * 100:4.1f}%")
    print(f"  {'of':<16}{fmt_gb(bd['total'])}")


def render_verdict(bd: dict, advise_hint: str | None = None) -> None:
    """Print the FITS / DOES NOT FIT line, optionally with a follow-up hint."""
    if bd["fits"]:
        print("  " + green("✓ FITS") + f"  | {fmt_gb(bd['free']).strip()} headroom")
    else:
        print("  " + red("✗ DOES NOT FIT") + f"  | over by {fmt_gb(-bd['free']).strip()}")
        if advise_hint:
            print(dim(advise_hint))


def render_speed(tp: dict, suffix: str = "", show_assumptions: bool = False,
                 bandwidth: float | None = None,
                 efficiency: float = BW_EFFICIENCY) -> None:
    """Print the decode-speed roofline block."""
    print(bold("  Decode speed (memory-bandwidth roofline)"))
    print(f"    {tp['per_stream']:6.1f} tok/s per stream{suffix}")
    print(f"    feel: {speed_verdict(tp['per_stream'])}")
    if show_assumptions:
        bw = bandwidth or SPARK["bandwidth_gbps"]
        print(dim(f"    (assumes ~{int(efficiency * 100)}% of {bw:.0f} GB/s; "
                  "prefill/compute not modeled)"))


# ---------------------------------------------------------------------------
# Model resolution (built-in DB or custom flags)
# ---------------------------------------------------------------------------

def resolve_model(args: argparse.Namespace) -> dict:
    """Resolve a model spec from --model (DB id) or custom --params/... flags."""
    if args.model and args.model in MODELS:
        spec = dict(MODELS[args.model])
        spec["name"] = args.model
    elif args.model:
        raise SystemExit(
            f"Unknown model '{args.model}'. Run `sparkfit models` or pass "
            f"--params/--layers/--hidden/--kv-heads/--head-dim for a custom one.")
    else:
        if not args.params:
            raise SystemExit("Provide a known --model or a custom --params (billions).")
        spec = {
            "name": "custom",
            "total_b": args.params,
            "active_b": args.active or args.params,
            "layers": args.layers, "hidden": args.hidden,
            "kv_heads": args.kv_heads, "head_dim": args.head_dim,
            "moe": args.active is not None and args.active < args.params,
        }
    return spec


def add_model_flags(p: argparse.ArgumentParser) -> None:
    """Attach the model-selection flags shared by plan/advise/fit."""
    p.add_argument("-m", "--model", help="model id from the built-in DB (see `sparkfit models`)")
    p.add_argument("--params", type=float, help="custom: total params (billions)")
    p.add_argument("--active", type=float, help="custom: active params for MoE (billions)")
    p.add_argument("--layers", type=int, default=32, help="custom: number of layers")
    p.add_argument("--hidden", type=int, default=4096, help="custom: hidden size")
    p.add_argument("--kv-heads", type=int, default=8, help="custom: KV heads (GQA)")
    p.add_argument("--head-dim", type=int, default=128, help="custom: head dim")


def _envf(name: str, default: "float | None") -> "float | None":
    """Read a float from an environment variable, falling back to default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def add_workload_flags(p: argparse.ArgumentParser) -> None:
    """Attach the workload/environment flags shared by plan/advise/fit."""
    p.add_argument("-q", "--quant", default="q4_k_m", help="quantization (default q4_k_m)")
    p.add_argument("-c", "--context", type=int, default=4096, help="context length")
    p.add_argument("-b", "--batch", type=int, default=1, help="batch size per stream")
    p.add_argument("-n", "--concurrency", type=int, default=1, help="concurrent streams/replicas")
    p.add_argument("--kv-dtype", default="fp16", choices=list(KV_BYTES), help="KV-cache dtype")
    p.add_argument("--os-reserve", type=float, default=_envf("SPARKFIT_OS_RESERVE", DEFAULT_OS_RESERVE_GB),
                   help="GB reserved for OS + Grace CPU side")
    p.add_argument("--framework", type=float, default=_envf("SPARKFIT_FRAMEWORK", DEFAULT_FRAMEWORK_GB),
                   help="GB for CUDA context + serving framework")
    p.add_argument("--total-mem", type=float, help="override total unified memory (GB)")
    p.add_argument("--bandwidth", type=float, default=_envf("SPARKFIT_BANDWIDTH", None),
                   help="override memory bandwidth (GB/s)")
    p.add_argument("--efficiency", type=float, default=_envf("SPARKFIT_EFFICIENCY", BW_EFFICIENCY),
                   help="fraction of peak bandwidth reached (calibrate on your device)")
    p.add_argument("--live", action="store_true",
                   help="plan against memory free NOW (read from the device)")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")


# ---------------------------------------------------------------------------
# Simplified entry: fuzzy name match + Hugging Face auto-fetch
# ---------------------------------------------------------------------------

def resolve_db_name(query: str) -> str | None:
    """Match a built-in model by exact id or partial tokens ('llama 70')."""
    if query in MODELS:
        return query
    toks = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]
    if not toks:
        return None
    matches = [n for n in MODELS if all(t in n.lower() for t in toks)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"Ambiguous '{query}'. Matches: {', '.join(matches)}")
    return None


def _mlp_params(hidden: int, inter: int) -> int:
    """Parameter count of one SwiGLU MLP block (gate + up + down)."""
    return 3 * hidden * inter


def parse_hf_config(cfg: dict) -> dict:
    """Build a best-effort sparkfit spec from a HF transformers config.json dict.

    Handles dense, Mixture-of-Experts, and Multi-head Latent Attention (MLA,
    DeepSeek V2/V3/R1) architectures.
    """
    g = cfg.get
    hidden = g("hidden_size") or g("n_embd")
    layers = g("num_hidden_layers") or g("n_layer")
    heads = g("num_attention_heads") or g("n_head")
    if not (hidden and layers and heads):
        raise SystemExit(
            "config.json lacks core fields "
            "(hidden_size/num_hidden_layers/num_attention_heads).")
    kv_heads = g("num_key_value_heads") or heads
    head_dim = g("head_dim") or (hidden // heads)
    vocab = g("vocab_size") or 32000
    inter = g("intermediate_size") or (4 * hidden)
    tie = bool(g("tie_word_embeddings"))

    # Attention parameters per layer, and the KV-cache style.
    kv_lora = g("kv_lora_rank")
    if kv_lora:
        # Multi-head Latent Attention: compressed latent KV cache (DeepSeek).
        qk_rope = g("qk_rope_head_dim") or 0
        qk_nope = g("qk_nope_head_dim") or head_dim
        v_head = g("v_head_dim") or head_dim
        q_lora = g("q_lora_rank")
        if q_lora:
            q = hidden * q_lora + q_lora * heads * (qk_nope + qk_rope)
        else:
            q = hidden * heads * (qk_nope + qk_rope)
        kv_down = hidden * (kv_lora + qk_rope)
        kv_up = kv_lora * heads * (qk_nope + v_head)
        o = heads * v_head * hidden
        attn = q + kv_down + kv_up + o
        kv_style = "mla"
        mla_dim = kv_lora + qk_rope
    else:
        q = hidden * heads * head_dim
        kv = 2 * hidden * kv_heads * head_dim
        o = heads * head_dim * hidden
        attn = q + kv + o
        kv_style = "mha"
        mla_dim = 0

    # MLP per layer: dense, or Mixture-of-Experts (incl. DeepSeek fine-grained).
    n_exp = g("num_local_experts") or g("num_experts") or g("n_routed_experts") or 0
    top = g("num_experts_per_tok") or 0
    if n_exp and top:
        moe_inter = g("moe_intermediate_size") or inter
        first_dense = g("first_k_dense_replace") or 0
        moe_layers = max(layers - first_dense, 0)
        shared = (g("n_shared_experts") or 0) * _mlp_params(hidden, moe_inter)
        shared_inter = g("shared_expert_intermediate_size")
        if shared_inter:  # Qwen2-MoE style single shared expert
            shared += _mlp_params(hidden, shared_inter)
        router = hidden * n_exp
        routed_total = n_exp * _mlp_params(hidden, moe_inter)
        routed_active = top * _mlp_params(hidden, moe_inter)
        dense_mlp = first_dense * _mlp_params(hidden, inter)
        mlp_total = dense_mlp + moe_layers * (routed_total + shared + router)
        mlp_active = dense_mlp + moe_layers * (routed_active + shared + router)
        moe = True
    else:
        mlp_total = mlp_active = layers * _mlp_params(hidden, inter)
        moe = False

    embed = vocab * hidden
    lm_head = 0 if tie else vocab * hidden
    attn_sum = layers * attn
    total = embed + lm_head + attn_sum + mlp_total
    active = embed + lm_head + attn_sum + mlp_active
    spec = {"total_b": total / 1e9, "active_b": active / 1e9, "layers": layers,
            "hidden": hidden, "kv_heads": kv_heads, "head_dim": head_dim, "moe": moe}
    if kv_style == "mla":
        spec["kv_style"] = "mla"
        spec["mla_dim"] = mla_dim
    return spec


def fetch_hf_config(repo_id: str, revision: str = "main") -> dict:
    """Download a model's config.json from Hugging Face and parse it into a spec."""
    repo_id = repo_id.strip().rstrip("/")
    m = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", repo_id)
    if m:
        repo_id = m.group(1)
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "sparkfit"})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            cfg = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SystemExit(f"'{repo_id}' is gated/private. Set HF_TOKEN and retry.")
        if e.code == 404:
            raise SystemExit(f"No config.json for '{repo_id}' (check the id).")
        raise SystemExit(f"HTTP {e.code} fetching '{repo_id}'.")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Network error for '{repo_id}': {e.reason} (auto-fetch needs internet).")
    spec = parse_hf_config(cfg)
    spec["name"] = repo_id
    spec["source"] = "huggingface"
    return spec


def resolve_target(target: str) -> dict:
    """Resolve a quick-mode target: a DB id (fuzzy) or a Hugging Face repo id/URL."""
    if "/" in target or target.startswith("http"):
        return fetch_hf_config(target)
    name = resolve_db_name(target)
    if name:
        spec = dict(MODELS[name])
        spec["name"] = name
        spec["source"] = "builtin"
        return spec
    raise SystemExit(
        f"Unknown model '{target}'. Try `sparkfit models`, a partial name "
        f"(e.g. 'llama 70'), or a Hugging Face id (e.g. 'Qwen/Qwen2.5-7B').")


# ---------------------------------------------------------------------------
# Live memory (when run ON the Spark)
# ---------------------------------------------------------------------------

def read_live_mem() -> dict:
    """Read current memory usage from nvidia-smi and /proc/meminfo, if available."""
    info: dict = {}
    # nvidia-smi (unified memory shows as GPU memory on GB10).
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"], text=True, timeout=5)
            t, u, f = (float(x) for x in out.strip().splitlines()[0].split(","))
            info["gpu"] = {"total_gb": t / 1024, "used_gb": u / 1024, "free_gb": f / 1024}
        except Exception:
            pass
    # /proc/meminfo (Linux host view).
    try:
        with open("/proc/meminfo") as fh:
            mi = {}
            for line in fh:
                k, _, v = line.partition(":")
                mi[k] = float(v.strip().split()[0]) / (1024 * 1024)  # kB -> GB
        info["host"] = {"total_gb": mi.get("MemTotal", 0),
                        "available_gb": mi.get("MemAvailable", 0)}
    except Exception:
        pass
    return info


def live_free_gb() -> float | None:
    """Currently-available memory in GB (unified shows as host MemAvailable on GB10)."""
    info = read_live_mem()
    g = info.get("gpu", {}).get("free_gb")
    if g:
        return g
    return info.get("host", {}).get("available_gb")


def apply_live(args: argparse.Namespace) -> float | None:
    """If --live, plan against memory free right now instead of the 128 GB total."""
    if not getattr(args, "live", False):
        return None
    free = live_free_gb()
    if not free:
        raise SystemExit("--live: could not read live memory. Run this ON the DGX Spark.")
    args.total_mem = free
    args.os_reserve = 0.0  # live-free already excludes OS + running processes
    if not getattr(args, "json", False):
        print()
        print(dim(f"  live: planning against {free:.1f} GB free right now "
                  "(OS + running processes already excluded)"))
    return free


# ---------------------------------------------------------------------------
# Subcommand: quick (default one-shot report)
# ---------------------------------------------------------------------------

def cmd_quick(args: argparse.Namespace) -> None:
    """One-shot smart report: auto-pick a quant, show budget, speed, alternatives."""
    apply_live(args)
    spec = resolve_target(args.target)
    ctx, conc = args.context, args.concurrency
    ladder = []
    auto = first_fit = None
    for qz in QUANT_LADDER:
        bd = budget(spec, qz, ctx, 1, conc, args.kv_dtype, args.os_reserve,
                    args.framework, args.total_mem)
        tp = decode_tok_s(spec, qz, ctx, 1, args.kv_dtype, args.bandwidth, args.efficiency)
        ladder.append((qz, bd, tp))
        if bd["fits"]:
            if first_fit is None:
                first_fit = qz
            if bd["free"] / bd["total"] >= SAFE_MARGIN and auto is None:
                auto = qz
    chosen = args.quant or auto or first_fit or QUANT_LADDER[-1]
    bd = budget(spec, chosen, ctx, 1, conc, args.kv_dtype, args.os_reserve,
                args.framework, args.total_mem)
    tp = decode_tok_s(spec, chosen, ctx, 1, args.kv_dtype, args.bandwidth, args.efficiency)

    if args.json:
        print(json.dumps({
            "model": spec, "chosen_quant": chosen, "auto": args.quant is None,
            "context": ctx, "concurrency": conc, "fits": bd["fits"],
            "used_gb": round(bd["used"] / GB, 2), "headroom_gb": round(bd["free"] / GB, 2),
            "decode_tok_s": round(tp["per_stream"], 1),
            "alternatives": [{"quant": q, "fits": b["fits"],
                              "used_gb": round(b["used"] / GB, 2),
                              "tok_s": round(t["per_stream"], 1)} for q, b, t in ladder],
        }, indent=2))
        return

    src = "Hugging Face" if spec.get("source") == "huggingface" else "built-in"
    moe = f" | MoE ({spec['active_b']:.1f}B active)" if spec.get("moe") else ""
    pick = "auto" if args.quant is None else "chosen"
    streams_suffix = f"  |  {conc} streams" if conc > 1 else ""
    print()
    print(bold(f"  {SPARK['name']}  |  {spec['name']}  |  {chosen} ({pick})  |  ctx {ctx}{streams_suffix}"))
    print(dim(f"  {spec['total_b']:.0f}B params | {spec['layers']} layers | "
              f"hidden {spec['hidden']} | kv_heads {spec['kv_heads']}{moe} | [{src}]"))
    print()
    render_budget_table(bd)
    print()
    render_verdict(bd)
    print()
    note = "" if bd["fits"] else dim("   (hypothetical, does not fit)")
    render_speed(tp, suffix=note)
    print()
    print(bold("  Other quantizations"))
    for qz, b2, t2 in ladder:
        mark = green("✓") if b2["fits"] else red("✗")
        star = dim("  (selected)") if qz == chosen else ""
        print(f"    {qz:<8}{b2['used'] / GB:6.0f}G  {mark} {t2['per_stream']:4.0f} tok/s{star}")
    print()
    ref = spec["name"] if spec.get("source") == "builtin" else "<model>"
    if bd["fits"]:
        print(dim("  tip: add -c 16384, -n 4 or -q q5_k_m to customize  |  "
                  f"full table: sparkfit advise -m {ref}"))
    else:
        fitting = [qz for qz, b2, _ in ladder if b2["fits"]]
        if fitting:
            print(dim(f"  tip: too big at {chosen} - try a lighter quant (-q {fitting[-1]}), "
                      "lower -c/-n, or free memory"))
        else:
            print(dim("  tip: doesn't fit even at q2_k - free memory, reduce -c/-n, "
                      "or pick a smaller model"))
    print()


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> None:
    """Full unified-memory budget plus decode-speed roofline for one config."""
    apply_live(args)
    spec = resolve_model(args)
    bd = budget(spec, args.quant, args.context, args.batch, args.concurrency,
                args.kv_dtype, args.os_reserve, args.framework, args.total_mem)
    tp = decode_tok_s(spec, args.quant, args.context, args.batch, args.kv_dtype,
                      args.bandwidth, args.efficiency)

    if args.json:
        print(json.dumps({"model": spec, "quant": args.quant, "context": args.context,
                          "batch": args.batch, "concurrency": args.concurrency,
                          "budget_gb": {k: round(v / GB, 3) for k, v in bd.items()
                                        if k not in ("fits", "util")},
                          "fits": bd["fits"], "util": round(bd["util"], 4),
                          "decode_tok_s": {k: round(v, 2) for k, v in tp.items()}}, indent=2))
        return

    streams = args.batch * args.concurrency
    print()
    print(bold(f"  {SPARK['name']}  |  {spec['name']}  |  {args.quant}"))
    print(dim(f"  context {args.context}  |  batch {args.batch}  |  concurrency {args.concurrency}"
              f"  |  {streams} stream(s)  |  KV {args.kv_dtype}"))
    print()
    render_budget_table(bd)
    print()
    hint = ("    try: sparkfit advise "
            + (f"-m {spec['name']} " if spec["name"] in MODELS else "")
            + f"-c {args.context} -n {args.concurrency}")
    render_verdict(bd, advise_hint=hint)
    print()
    suffix = f"   |   {tp['aggregate']:6.1f} tok/s aggregate" if streams > 1 else ""
    render_speed(tp, suffix=suffix, show_assumptions=True, bandwidth=args.bandwidth,
                 efficiency=args.efficiency)
    print()


# ---------------------------------------------------------------------------
# Subcommand: advise  (quantization + concurrency advisor)
# ---------------------------------------------------------------------------

def cmd_advise(args: argparse.Namespace) -> None:
    """Rank quantizations and recommend the highest quality that fits with margin."""
    apply_live(args)
    spec = resolve_model(args)
    results = []
    for q in QUANT_LADDER:
        bd = budget(spec, q, args.context, args.batch, args.concurrency,
                    args.kv_dtype, args.os_reserve, args.framework, args.total_mem)
        tp = decode_tok_s(spec, q, args.context, args.batch, args.kv_dtype, args.bandwidth, args.efficiency)
        margin = bd["free"] / bd["total"]
        results.append((q, bd, tp, margin))

    if args.json:
        print(json.dumps([
            {"quant": q, "fits": bd["fits"], "used_gb": round(bd["used"] / GB, 2),
             "headroom_gb": round(bd["free"] / GB, 2), "tok_s": round(tp["per_stream"], 1)}
            for q, bd, tp, _ in results], indent=2))
        return

    pct = int(SAFE_MARGIN * 100)
    print()
    print(bold(f"  Quantization advisor | {spec['name']} | ctx {args.context} | "
               f"{args.batch * args.concurrency} stream(s)"))
    print()
    print(dim(f"  {'quant':<9}{'used':>9}{'headroom':>11}{'tok/s':>8}   verdict"))
    print(dim("  " + "─" * 56))
    recommended = None
    for q, bd, tp, margin in results:
        if bd["fits"]:
            tag = green("✓ fits")
            if margin >= SAFE_MARGIN and recommended is None:
                recommended = q
                tag = green(f"✓ recommended (best quality, >={pct}% margin)")
        else:
            tag = red("✗ over")
        print(f"  {q:<9}{bd['used'] / GB:8.1f}G{bd['free'] / GB:10.1f}G"
              f"{tp['per_stream']:7.0f}   {tag}")
    print()
    if recommended:
        print("  " + green(f"> Use {bold(recommended)}")
              + f": highest-quality quant that still fits with a safe (>={pct}%) margin.")
    else:
        first_fit = next((q for q, bd, _, _ in results if bd["fits"]), None)
        if first_fit:
            print("  " + yellow(f"> {first_fit} fits but with thin margin.")
                  + " Reduce --context or --concurrency for safety.")
        else:
            print("  " + red("> Does not fit at any quant.")
                  + " Reduce context/concurrency or pick a smaller model.")
    print()


# ---------------------------------------------------------------------------
# Subcommand: fit  (largest model / max concurrency that fits)
# ---------------------------------------------------------------------------

def cmd_fit(args: argparse.Namespace) -> None:
    """Scan the DB for what fits, or (with --concurrency-scan) max parallel streams."""
    apply_live(args)
    if args.concurrency_scan:
        spec = resolve_model(args)
        rows = []
        n = 1
        while n <= 256:
            bd = budget(spec, args.quant, args.context, args.batch, n,
                        args.kv_dtype, args.os_reserve, args.framework, args.total_mem)
            if not bd["fits"]:
                break
            tp = decode_tok_s(spec, args.quant, args.context, args.batch * n,
                              args.kv_dtype, args.bandwidth, args.efficiency)
            rows.append((n, bd, tp))
            n += 1
        max_n = rows[-1][0] if rows else 0
        if args.json:
            print(json.dumps({"model": spec["name"], "quant": args.quant,
                              "max_concurrency": max_n}, indent=2))
            return
        print()
        print(bold(f"  Max concurrency | {spec['name']} | {args.quant} | ctx {args.context}"))
        print()
        if max_n == 0:
            print("  " + red("> does not fit even at 1 stream; reduce -c/-q "
                              "or pick a smaller model"))
        else:
            print("  " + green(f"> up to {max_n} concurrent stream(s) fit in "
                               f"{(args.total_mem or SPARK['total_mem_gb']):.0f} GB"))
        if rows:
            last = rows[-1]
            print(dim(f"    at {max_n} streams: {fmt_gb(last[1]['used']).strip()} used | "
                      f"{last[2]['aggregate']:.0f} tok/s aggregate | "
                      f"{last[2]['per_stream']:.0f} tok/s/stream"))
        print()
        return

    # Otherwise: scan the model DB for what fits at the given workload.
    fitting = []
    for name, base in MODELS.items():
        spec = dict(base)
        spec["name"] = name
        bd = budget(spec, args.quant, args.context, args.batch, args.concurrency,
                    args.kv_dtype, args.os_reserve, args.framework, args.total_mem)
        tp = decode_tok_s(spec, args.quant, args.context, args.batch, args.kv_dtype, args.bandwidth, args.efficiency)
        fitting.append((name, base["total_b"], bd, tp))
    fitting.sort(key=lambda r: r[1], reverse=True)

    if args.json:
        print(json.dumps([
            {"model": n, "params_b": p, "fits": bd["fits"],
             "used_gb": round(bd["used"] / GB, 2), "tok_s": round(tp["per_stream"], 1)}
            for n, p, bd, tp in fitting], indent=2))
        return
    print()
    print(bold(f"  What fits at {args.quant} | ctx {args.context} | "
               f"{args.batch * args.concurrency} stream(s)"))
    print()
    print(dim(f"  {'model':<16}{'params':>8}{'used':>9}{'tok/s':>8}   fit"))
    print(dim("  " + "─" * 52))
    for name, p, bd, tp in fitting:
        tag = green("✓") if bd["fits"] else red("✗")
        line = f"  {name:<16}{p:7.0f}B{bd['used'] / GB:8.1f}G{tp['per_stream']:7.0f}   {tag}"
        print(line if bd["fits"] else dim(line))
    print()
    biggest = next((n for n, p, bd, tp in fitting if bd["fits"]), None)
    if biggest:
        print("  " + green(f"> Largest model that fits: {bold(biggest)}"))
    print()


# ---------------------------------------------------------------------------
# Subcommand: scan  (live system snapshot)
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> None:
    """Show a live memory snapshot and how many weights would still fit."""
    info = read_live_mem()
    if args.json:
        print(json.dumps(info, indent=2))
        return
    print()
    print(bold("  Live memory scan"))
    print()
    if not info:
        print(yellow("  No live telemetry found (nvidia-smi / /proc/meminfo)."))
        print(dim("  This machine doesn't look like a DGX Spark, or you lack permissions."))
        print(dim("  Use `sparkfit plan` for offline planning instead."))
        print()
        return
    if "gpu" in info:
        g = info["gpu"]
        frac = g["used_gb"] / g["total_gb"] if g["total_gb"] else 0
        print(f"  GPU/unified  {g['used_gb']:.1f} / {g['total_gb']:.1f} GB  {bar(frac)} {frac * 100:.0f}%")
    if "host" in info:
        h = info["host"]
        used = h["total_gb"] - h["available_gb"]
        frac = used / h["total_gb"] if h["total_gb"] else 0
        print(f"  host RAM     {used:.1f} / {h['total_gb']:.1f} GB  {bar(frac)} {frac * 100:.0f}%")
    free = info.get("gpu", {}).get("free_gb") or info.get("host", {}).get("available_gb", 0)
    print()
    print(dim(f"  ~{free:.1f} GB free right now."))
    for q in ("q4_k_m", "fp16"):
        cap_b = (free * GB) * 8 / QUANT_BITS[q] / 1e9
        print(f"    at {q:<7}: room for ~{cap_b:.0f}B params of weights "
              + dim("(before KV-cache & overhead)"))
    print()


# ---------------------------------------------------------------------------
# Subcommand: models
# ---------------------------------------------------------------------------

def cmd_models(args: argparse.Namespace) -> None:
    """List the built-in model database."""
    if args.json:
        print(json.dumps(MODELS, indent=2))
        return
    print()
    print(bold("  Built-in models") + dim("  (override any field with --params/--layers/...)"))
    print()
    print(dim(f"  {'id':<16}{'total':>8}{'active':>8}{'layers':>8}{'hidden':>8}{'kv_h':>6}{'h_dim':>6}"))
    print(dim("  " + "─" * 60))
    for name, s in MODELS.items():
        moe = " (MoE)" if s.get("moe") else ""
        print(f"  {name:<16}{s['total_b']:7.0f}B{s['active_b']:7.1f}B"
              f"{s['layers']:8}{s['hidden']:8}{s['kv_heads']:6}{s['head_dim']:6}{moe}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all subcommands."""
    p = argparse.ArgumentParser(
        prog="sparkfit",
        description="DGX Spark LLM memory-capacity planner (unified-memory & "
                    "bandwidth aware).",
        epilog="examples:\n"
               "  sparkfit plan -m llama3.1-70b -q q4_k_m -c 8192\n"
               "  sparkfit advise -m qwen2.5-72b -c 16384\n"
               "  sparkfit fit -q q4_k_m -c 4096\n"
               "  sparkfit fit -m llama3.1-8b --concurrency-scan\n"
               "  sparkfit scan",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", "-V", action="version",
                   version="sparkfit " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="full unified-memory budget + decode speed for one config")
    add_model_flags(pl)
    add_workload_flags(pl)
    pl.set_defaults(func=cmd_plan)

    ad = sub.add_parser("advise", help="recommend the highest-quality quant that fits with margin")
    add_model_flags(ad)
    add_workload_flags(ad)
    ad.set_defaults(func=cmd_advise)

    ft = sub.add_parser("fit", help="scan the DB for what fits, or max concurrency for a model")
    add_model_flags(ft)
    add_workload_flags(ft)
    ft.add_argument("--concurrency-scan", action="store_true",
                    help="instead: find max concurrent streams for --model")
    ft.set_defaults(func=cmd_fit)

    sc = sub.add_parser("scan", help="read live memory when run ON the Spark")
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=cmd_scan)

    md = sub.add_parser("models", help="list the built-in model database")
    md.add_argument("--json", action="store_true")
    md.set_defaults(func=cmd_models)

    qk = sub.add_parser("quick", help="one-shot smart report (default when you pass just a model)")
    qk.add_argument("target", help="model id / partial name / Hugging Face repo id")
    qk.add_argument("-q", "--quant", default=None, help="force a quant (default: auto-pick)")
    qk.add_argument("-c", "--context", type=int, default=8192)
    qk.add_argument("-n", "--concurrency", type=int, default=1)
    qk.add_argument("--kv-dtype", default="fp16", choices=list(KV_BYTES))
    qk.add_argument("--os-reserve", type=float, default=_envf("SPARKFIT_OS_RESERVE", DEFAULT_OS_RESERVE_GB))
    qk.add_argument("--framework", type=float, default=_envf("SPARKFIT_FRAMEWORK", DEFAULT_FRAMEWORK_GB))
    qk.add_argument("--total-mem", type=float, default=None)
    qk.add_argument("--bandwidth", type=float, default=_envf("SPARKFIT_BANDWIDTH", None))
    qk.add_argument("--efficiency", type=float, default=_envf("SPARKFIT_EFFICIENCY", BW_EFFICIENCY))
    qk.add_argument("--live", action="store_true",
                    help="plan against memory free NOW (read from the device)")
    qk.add_argument("--json", action="store_true")
    qk.set_defaults(func=cmd_quick)

    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point. Routes a bare model argument to the quick subcommand."""
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"plan", "advise", "fit", "scan", "models", "quick",
             "-h", "--help", "--version", "-V"}
    if not argv:
        build_parser().print_help()
        return
    if argv[0] not in known:
        argv = ["quick"] + argv
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
