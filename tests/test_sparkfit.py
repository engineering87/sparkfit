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
