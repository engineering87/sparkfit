"""Tests for sparkfit core math and model resolution."""
import pytest

import sparkfit as sf


def test_quant_bits():
    assert sf.quant_bits("fp16") == 16.0
    assert sf.quant_bits("q4_k_m") == 4.85
    with pytest.raises(SystemExit):
        sf.quant_bits("does-not-exist")


def test_weight_bytes():
    # 8B params at fp16 (2 bytes) = 16 GB (decimal)
    assert sf.weight_bytes(8.0, "fp16") == pytest.approx(16e9)
    # half the bytes at fp8
    assert sf.weight_bytes(8.0, "fp8") == pytest.approx(8e9)


def test_kv_per_token():
    spec = sf.MODELS["llama3.1-8b"]
    # 2 * layers * kv_heads * head_dim * 2 bytes (fp16)
    expected = 2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"] * 2
    assert sf.kv_per_token_bytes(spec, "fp16") == expected


def test_budget_fits_and_overflows():
    small = dict(sf.MODELS["llama3.1-8b"])
    bd = sf.budget(small, "q4_k_m", context=4096, batch=1)
    assert bd["fits"] is True
    assert bd["free"] > 0

    huge = dict(sf.MODELS["llama3.1-405b"])
    bd = sf.budget(huge, "fp16", context=4096, batch=1)
    assert bd["fits"] is False
    assert bd["free"] < 0


def test_decode_tok_s_positive():
    spec = sf.MODELS["llama3.1-8b"]
    tp = sf.decode_tok_s(spec, "q4_k_m", context=4096, batch=1)
    assert tp["per_stream"] > 0
    # smaller model => faster than a 70B at the same quant
    big = sf.decode_tok_s(sf.MODELS["llama3.1-70b"], "q4_k_m", 4096, 1)
    assert tp["per_stream"] > big["per_stream"]


def test_parse_hf_dense_qwen():
    cfg = {
        "hidden_size": 3584, "intermediate_size": 18944, "num_hidden_layers": 28,
        "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064,
        "tie_word_embeddings": False,
    }
    spec = sf.parse_hf_config(cfg)
    assert spec["moe"] is False
    assert spec["total_b"] == pytest.approx(7.6, abs=0.15)  # real Qwen2.5-7B ~7.61B
    assert spec["kv_heads"] == 4
    assert spec["head_dim"] == 128


def test_parse_hf_moe_mixtral():
    cfg = {
        "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
        "num_local_experts": 8, "num_experts_per_tok": 2, "tie_word_embeddings": False,
    }
    spec = sf.parse_hf_config(cfg)
    assert spec["moe"] is True
    assert spec["total_b"] == pytest.approx(46.7, abs=0.5)
    assert spec["active_b"] == pytest.approx(12.9, abs=0.5)
    assert spec["active_b"] < spec["total_b"]


def test_resolve_db_name_fuzzy():
    assert sf.resolve_db_name("llama3.1-70b") == "llama3.1-70b"   # exact
    assert sf.resolve_db_name("llama 70") == "llama3.1-70b"        # tokens
    assert sf.resolve_db_name("72b") == "qwen2.5-72b"              # fragment
    assert sf.resolve_db_name("nope-not-real") is None


def test_resolve_db_name_ambiguous():
    with pytest.raises(SystemExit):
        sf.resolve_db_name("qwen2.5")  # matches several


def test_parse_hf_mla_deepseek_v2_lite():
    cfg = {
        "hidden_size": 2048, "intermediate_size": 10944, "num_hidden_layers": 27,
        "num_attention_heads": 16, "num_key_value_heads": 16, "vocab_size": 102400,
        "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128,
        "v_head_dim": 128, "q_lora_rank": None, "n_routed_experts": 64,
        "num_experts_per_tok": 6, "n_shared_experts": 2, "moe_intermediate_size": 1408,
        "first_k_dense_replace": 1, "tie_word_embeddings": False,
    }
    spec = sf.parse_hf_config(cfg)
    assert spec["kv_style"] == "mla"
    assert spec["mla_dim"] == 576
    assert spec["moe"] is True
    assert spec["total_b"] == pytest.approx(15.7, abs=0.6)  # real ~15.7B


def test_parse_hf_mla_deepseek_v3():
    cfg = {
        "hidden_size": 7168, "intermediate_size": 18432, "num_hidden_layers": 61,
        "num_attention_heads": 128, "num_key_value_heads": 128, "vocab_size": 129280,
        "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128,
        "v_head_dim": 128, "q_lora_rank": 1536, "n_routed_experts": 256,
        "num_experts_per_tok": 8, "n_shared_experts": 1, "moe_intermediate_size": 2048,
        "first_k_dense_replace": 3, "tie_word_embeddings": False,
    }
    spec = sf.parse_hf_config(cfg)
    assert spec["total_b"] == pytest.approx(671, abs=15)   # real 671B
    assert spec["active_b"] == pytest.approx(37, abs=4)    # real ~37B


def test_mla_kv_much_smaller_than_mha():
    spec = sf.parse_hf_config({
        "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128,
        "num_key_value_heads": 128, "vocab_size": 129280, "kv_lora_rank": 512,
        "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
        "q_lora_rank": 1536, "intermediate_size": 18432,
    })
    mla = sf.kv_per_token_bytes(spec, "fp16")
    naive = 2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"] * 2
    assert mla == 61 * 576 * 2
    assert mla < naive / 5


def test_deepseek_catalog_entry_uses_mla():
    spec = dict(sf.MODELS["deepseek-r1"])
    assert spec.get("kv_style") == "mla"
    # latent cache stays tiny even at long context
    kv = sf.kv_total_bytes(spec, context=8192, batch=1)
    assert kv < 2 * 1e9  # under 2 GB for 8k context


# --- CLI integration tests (exercise command handlers and rendering) ---

def test_cli_models(capsys):
    sf.main(["models"])
    assert "llama3.1-70b" in capsys.readouterr().out


def test_cli_quick_report(capsys):
    sf.main(["llama3.1-8b"])
    out = capsys.readouterr().out
    assert "FITS" in out and "tok/s per stream" in out


def test_cli_plan_json(capsys):
    sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "--json"])
    import json
    data = json.loads(capsys.readouterr().out)
    assert data["fits"] is True
    assert data["model"]["name"] == "llama3.1-8b"


def test_cli_advise(capsys):
    sf.main(["advise", "-m", "qwen2.5-72b", "-c", "16384"])
    assert "Use" in capsys.readouterr().out


def test_cli_fit_scan_db(capsys):
    sf.main(["fit", "-q", "q4_k_m", "-c", "4096"])
    assert "Largest model that fits" in capsys.readouterr().out


def test_cli_fit_concurrency_scan(capsys):
    sf.main(["fit", "-m", "llama3.1-8b", "-q", "q4_k_m", "--concurrency-scan"])
    assert "concurrent stream" in capsys.readouterr().out


def test_cli_quick_deepseek_mla(capsys):
    sf.main(["deepseek-v2-lite", "-c", "8192"])
    assert "MoE" in capsys.readouterr().out


def test_cli_scan_json(capsys):
    sf.main(["scan", "--json"])
    import json
    json.loads(capsys.readouterr().out)  # valid JSON, no exception


def test_cli_version():
    with pytest.raises(SystemExit):
        sf.main(["--version"])


def test_cli_unknown_model_errors():
    with pytest.raises(SystemExit):
        sf.main(["totally-unknown-model-xyz"])


def test_cli_no_args_prints_help(capsys):
    sf.main([])
    assert "usage" in capsys.readouterr().out.lower()


def test_total_mem_zero_errors():
    with pytest.raises(SystemExit):
        sf.budget(sf.MODELS["llama3.1-8b"], "q4_k_m", 4096, 1, total_mem=0)


def test_total_mem_negative_errors():
    with pytest.raises(SystemExit):
        sf.budget(sf.MODELS["llama3.1-8b"], "q4_k_m", 4096, 1, total_mem=-5)


def test_efficiency_flag_scales_tok_s(capsys):
    import json
    sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "--efficiency", "0.35", "--json"])
    low = json.loads(capsys.readouterr().out)["decode_tok_s"]["per_stream"]
    sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "--efficiency", "0.70", "--json"])
    high = json.loads(capsys.readouterr().out)["decode_tok_s"]["per_stream"]
    assert high == pytest.approx(2 * low, rel=0.02)


def test_env_efficiency_overrides_default(capsys, monkeypatch):
    import json
    sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "--json"])
    base = json.loads(capsys.readouterr().out)["decode_tok_s"]["per_stream"]
    monkeypatch.setenv("SPARKFIT_EFFICIENCY", "0.35")
    sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "--json"])
    low = json.loads(capsys.readouterr().out)["decode_tok_s"]["per_stream"]
    assert low < base


# --- local config.json support ---

def test_local_config_file(tmp_path, capsys):
    import json
    cfg = {"hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32,
           "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
           "tie_word_embeddings": False}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    sf.main([str(p), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["model"]["source"] == "local"
    assert out["model"]["total_b"] == pytest.approx(5.9, abs=0.5)  # these dims, GQA


def test_local_config_directory(tmp_path):
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 24,
        "num_attention_heads": 16, "num_key_value_heads": 16, "vocab_size": 32000}))
    spec = sf.load_local_config(str(tmp_path))
    assert spec["source"] == "local"
    assert spec["layers"] == 24


def test_local_config_text_config_nesting(tmp_path):
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 5120, "intermediate_size": 17408,
                        "num_hidden_layers": 64, "num_attention_heads": 24,
                        "num_key_value_heads": 4, "head_dim": 256,
                        "vocab_size": 248320, "tie_word_embeddings": False}}))
    spec = sf.load_local_config(str(tmp_path))
    assert spec["layers"] == 64 and spec["hidden"] == 5120


def test_local_config_dir_without_config_errors(tmp_path):
    with pytest.raises(SystemExit):
        sf.load_local_config(str(tmp_path))  # empty dir, no config.json


def test_local_config_label_shown(tmp_path, capsys):
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000}))
    sf.main([str(tmp_path)])
    assert "[local]" in capsys.readouterr().out


# --- hybrid attention (Qwen3.5-style): only full_attention layers keep a KV-cache ---

def test_hybrid_kv_layers_from_layer_types():
    cfg = {"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 8,
           "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
           "vocab_size": 248320,
           "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                           "full_attention", "linear_attention", "linear_attention",
                           "linear_attention", "full_attention"]}
    spec = sf.parse_hf_config(cfg)
    assert spec["kv_layers"] == 2  # 2 of 8 are full_attention
    assert sf.kv_per_token_bytes(spec, "fp16") == 2 * 2 * 4 * 256 * 2


def test_hybrid_kv_layers_from_interval():
    cfg = {"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
           "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
           "vocab_size": 248320, "full_attention_interval": 4}
    spec = sf.parse_hf_config(cfg)
    assert spec["kv_layers"] == 16  # 64 // 4


def test_standard_model_counts_all_kv_layers():
    spec = sf.parse_hf_config({"hidden_size": 4096, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
        "intermediate_size": 11008})
    assert spec["kv_layers"] == 32  # no hybrid markers -> every layer keeps KV


# --- weights override, measured weights, vision tower, sliding window ---

def test_weights_gb_override():
    spec = sf.MODELS["llama3.1-8b"]
    bd = sf.budget(spec, "q4_k_m", 4096, 1, weights_gb=35.9)
    assert bd["weights"] == pytest.approx(35.9 * sf.GB)
    assert bd["weights"] > sf.budget(spec, "q4_k_m", 4096, 1)["weights"]


def test_local_dir_weights_from_index(tmp_path):
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 32000, "intermediate_size": 11008}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 35_900_000_000}}))
    spec = sf.load_local_config(str(tmp_path))
    assert spec["weights_gb"] == pytest.approx(35.9, abs=0.1)


def test_local_dir_weights_from_files(tmp_path):
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16,
        "num_key_value_heads": 16, "vocab_size": 32000, "intermediate_size": 5632}))
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 1_000_000)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\0" * 2_000_000)
    spec = sf.load_local_config(str(tmp_path))
    assert spec["weights_gb"] == pytest.approx(0.003, abs=1e-4)


def test_vision_tower_adds_to_total_only():
    base = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
            "num_key_value_heads": 8, "vocab_size": 32000, "intermediate_size": 11008}
    no_v = sf.parse_hf_config(dict(base))
    with_v = sf.parse_hf_config({**base, "vision_config": {
        "hidden_size": 1152, "depth": 27, "intermediate_size": 4304}})
    assert with_v["total_b"] > no_v["total_b"]
    assert with_v["active_b"] == no_v["active_b"]  # vision not streamed at decode


def test_sliding_window_caps_kv():
    spec = sf.parse_hf_config({
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 32000, "intermediate_size": 14336,
        "sliding_window": 4096})
    assert spec["sliding_window"] == 4096
    assert sf.kv_total_bytes(spec, 32768, 1) == sf.kv_total_bytes(spec, 4096, 1)


def test_sliding_window_disabled_by_flag():
    spec = sf.parse_hf_config({
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 32000, "intermediate_size": 14336,
        "sliding_window": 4096, "use_sliding_window": False})
    assert spec["sliding_window"] == 0
    assert sf.kv_total_bytes(spec, 32768, 1) > sf.kv_total_bytes(spec, 4096, 1)


def test_weights_gb_nonpositive_falls_back():
    spec = sf.MODELS["llama3.1-8b"]
    base = sf.budget(spec, "fp16", 4096, 1)["weights"]
    assert sf.budget(spec, "fp16", 4096, 1, weights_gb=0)["weights"] == base
    assert sf.budget(spec, "fp16", 4096, 1, weights_gb=-5)["weights"] == base
    assert sf.decode_tok_s(spec, "fp16", 4096, 1, weights_gb=0)["per_stream"] > 0


# --- weights-override quick mode is single-config and consistent ---

def test_quick_weights_override_single_config(tmp_path, capsys):
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 32000, "intermediate_size": 11008}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 30_000_000_000}}))
    sf.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "Other quantizations" not in out          # no quant-shopping
    assert "measured on-disk size" in out
    assert "30.0 GB" in out                            # weights line uses measured size
    sf.main([str(tmp_path), "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["alternatives"] == []                     # consistent: no contradictory table
    assert d["weights_gb"] == pytest.approx(30.0, abs=0.1)


def test_quick_weights_gb_flag_single_config(capsys):
    import json
    sf.main(["llama3.1-8b", "--weights-gb", "20", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["weights_gb"] == pytest.approx(20.0)
    assert d["alternatives"] == []
    assert d["used_gb"] > 20  # 20 GB weights + reserves


def test_quick_normal_still_shows_alternatives(capsys):
    sf.main(["llama3.1-8b"])
    assert "Other quantizations" in capsys.readouterr().out


def test_nonpositive_efficiency_bandwidth_fall_back():
    spec = sf.MODELS["llama3.1-8b"]
    base = sf.decode_tok_s(spec, "q4_k_m", 4096, 1)["per_stream"]
    assert sf.decode_tok_s(spec, "q4_k_m", 4096, 1, efficiency=-1)["per_stream"] == pytest.approx(base)
    assert sf.decode_tok_s(spec, "q4_k_m", 4096, 1, efficiency=0)["per_stream"] == pytest.approx(base)
    assert sf.decode_tok_s(spec, "q4_k_m", 4096, 1, bandwidth=0)["per_stream"] == pytest.approx(base)


# --- v0.3.0 review regressions: concurrency in decode + input validation ---

def test_per_stream_decreases_with_concurrency(capsys):
    import json
    def per(n):
        sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "-c", "4096",
                 "-n", str(n), "--json"])
        return json.loads(capsys.readouterr().out)["decode_tok_s"]["per_stream"]
    p1, p4, p8 = per(1), per(4), per(8)
    # more concurrent streams contend for bandwidth -> each stream is slower
    assert p1 > p4 > p8


def test_aggregate_exceeds_per_stream_under_concurrency(capsys):
    import json
    sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "-c", "4096",
             "-n", "8", "--json"])
    t = json.loads(capsys.readouterr().out)["decode_tok_s"]
    assert t["aggregate"] > t["per_stream"] * 5  # ~8x minus weight amortization


def test_quick_tok_s_reflects_concurrency(capsys):
    import json
    sf.main(["llama3.1-8b", "-n", "1", "--json"])
    one = json.loads(capsys.readouterr().out)["decode_tok_s"]
    sf.main(["llama3.1-8b", "-n", "8", "--json"])
    eight = json.loads(capsys.readouterr().out)["decode_tok_s"]
    assert eight < one


def test_concurrency_one_matches_single_stream(capsys):
    import json
    # the default single-stream number must not change (validated on real HW)
    sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "-c", "4096", "-n", "1", "--json"])
    a = json.loads(capsys.readouterr().out)["decode_tok_s"]["per_stream"]
    sf.main(["plan", "-m", "llama3.1-8b", "-q", "q4_k_m", "-c", "4096", "--json"])
    b = json.loads(capsys.readouterr().out)["decode_tok_s"]["per_stream"]
    assert a == pytest.approx(b)


def test_decode_tok_s_zero_batch_no_crash():
    spec = sf.MODELS["llama3.1-8b"]
    t = sf.decode_tok_s(spec, "q4_k_m", 4096, 0)  # defensive guard, no ZeroDivision
    assert t["per_stream"] == 0.0


@pytest.mark.parametrize("flag,val", [("-c", "0"), ("-c", "-5"), ("-b", "0"), ("-n", "0")])
def test_nonpositive_workload_dims_rejected(flag, val):
    with pytest.raises(SystemExit):
        sf.main(["plan", "-m", "llama3.1-8b", flag, val])


def test_positive_int_type():
    import argparse
    assert sf._positive_int("3") == 3
    for bad in ("0", "-1", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            sf._positive_int(bad)
