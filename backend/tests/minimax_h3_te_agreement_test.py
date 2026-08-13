"""MiniMax-H3: per-installation measurement of text-encoder substitution agreement.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_agreement_test.py -v

`MEASURED_TE_SUBSTITUTIONS` used to present one machine's two numbers to every
user as fact. This file covers what replaced it: a tracked versioned prompt
suite, content identity for the three files a measurement belongs to, a
per-installation store that degrades to "none" rather than to a wrong number,
the metrics themselves against arithmetic with known answers, and the automatic
hook's obligation not to raise into a model load.

Synthetic tensors and a stub encoder only; no model, no GPU, nothing large on
disk. The one end-to-end run fabricates a small reference bank instead of
building one from a 32B encoder.
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.models.minimax_h3 import te_agreement as ta  # noqa: E402
from core.models.minimax_h3.te_projection import (  # noqa: E402
    PUBLISHED_TE_SUBSTITUTIONS,
    describe_te_substitution,
    load_te_projection,
    measured_te_substitution,
    published_te_substitution,
    read_te_projection_spec,
)

ENCODER_4B = "qwen3vl_4b_heretic_tap24_bf16.safetensors"
PROJECTION_4B = "mmh3-4b-ClipProj-celeb-mlp.safetensors"

D_IN, TEXT_DIM = 6, 9


# ---------------------------------------------------------------------------
# Fixtures: tiny stand-ins for every real file the measurement touches
# ---------------------------------------------------------------------------

class _Tokenizer:
    """Deterministic word-level tokenizer; `add_special_tokens` is honoured."""

    def __call__(self, text, add_special_tokens=True):
        ids = [(abs(hash(word)) % 900) + 10 for word in text.split()]
        if add_special_tokens:
            ids = [1] + ids + [2]
        return {"input_ids": ids}


class _Encoder:
    """A stand-in text encoder: one deterministic hidden row per token."""

    def __init__(self, width=D_IN, tokens_out=None):
        self.width = width
        self.tokens_out = tokens_out

    def encode(self, token_ids):
        rows = self.tokens_out if self.tokens_out is not None else len(token_ids)
        generator = torch.Generator().manual_seed(sum(token_ids) % 10007)
        return torch.randn(1, rows, self.width, generator=generator).to(torch.bfloat16)


@pytest.fixture
def stub_encode(monkeypatch):
    """Route `encode_presentation` at the stub encoder."""
    from core.models.minimax_h3 import h3_pipeline_ops as ops

    monkeypatch.setattr(
        ops, "encode_presentation",
        lambda encoder, token_ids, **kwargs: encoder.encode(list(token_ids)))


def _projection_file(directory, *, d_in=D_IN, d_out=TEXT_DIM, name=PROJECTION_4B, seed=7):
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        "W": torch.randn(d_in, d_out, generator=generator),
        "mean_in": torch.randn(d_in, generator=generator),
        "std_in": torch.rand(d_in, generator=generator) + 0.5,
        "mean_out": torch.randn(d_out, generator=generator),
        "std_out": torch.rand(d_out, generator=generator) + 0.5,
        "sink_out": torch.randn(d_out, generator=generator),
        "mlp.0.weight": torch.randn(4, d_in, generator=generator),
        "mlp.0.bias": torch.randn(4, generator=generator),
        "mlp.2.weight": torch.randn(d_out, 4, generator=generator),
        "mlp.2.bias": torch.randn(d_out, generator=generator),
    }
    path = str(directory / name)
    save_file(tensors, path, metadata={"d_in": str(d_in), "d_out": str(d_out), "tap": "24",
                                       "mlp_hidden": "4", "mlp_depth": "1"})
    return path


# Bound before any test can monkeypatch the module attribute, so building a
# second suite inside a test that has already installed a first one still reads
# the file it just wrote.
_LOAD_SUITE = ta.load_suite


def _small_suite(tmp_path, prompts=("alpha beta", "gamma delta epsilon", "zeta"),
                 version="test-suite-v1", target=6):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"version": version, "composite_target_tokens": target,
                                "prompts": list(prompts)}), encoding="utf-8")
    return _LOAD_SUITE(str(path))


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """A 3-prompt suite, installed as the one every code path resolves."""
    built = _small_suite(tmp_path / "suite")
    monkeypatch.setattr(ta, "load_suite", lambda path=None: built)
    return built


def _components(tmp_path, projection_path, *, encoder=ENCODER_4B, encoder_obj=None):
    encoder_file = tmp_path / encoder
    if not encoder_file.exists():
        save_file({"w": torch.zeros(4, 4)}, str(encoder_file))
    return {
        "text_encoder": encoder_obj if encoder_obj is not None else _Encoder(),
        "tokenizer": _Tokenizer(),
        "text_encoder_path": str(encoder_file),
        "te_projection": load_te_projection(read_te_projection_spec(projection_path)),
        "transformer_config": {"text_dim": TEXT_DIM},
        "dit_path": None,
        "official_dir": None,
    }


def _fabricate_bank(root, suite, components, *, reference_name="released_32b.safetensors",
                    scale=1.0, noise=0.0, tokens_override=None):
    """A reference bank without a 32B: the candidate's own output, perturbed.

    `scale`/`noise` set the answer the metrics must then reproduce.
    """
    from core.models.minimax_h3 import h3_pipeline_ops as ops

    reference_path = root / reference_name
    save_file({"w": torch.ones(8, 8)}, str(reference_path))
    corpus = ta.build_corpus(components["tokenizer"], suite)
    tensors, presentations = {}, []
    generator = torch.Generator().manual_seed(11)
    for name, token_ids in corpus:
        hidden = ops.encode_presentation(components["text_encoder"], token_ids)
        projected = ops.project_prompt_embeds(
            hidden, components["te_projection"], text_dim=TEXT_DIM, device="cpu")[0].float()
        reference = projected / scale
        if noise:
            reference = reference + noise * torch.randn(
                reference.shape, generator=generator)
        rows = tokens_override if tokens_override is not None else len(token_ids)
        tensors[name] = reference[:rows].contiguous().to(torch.bfloat16)
        presentations.append({"name": name, "tokens": rows})

    directory = (ta.store_dir(str(root / "store")) / "banks"
                 / ta.bank_key(suite["digest"], ta.file_identity(str(reference_path))))
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "bank.safetensors"))
    manifest = {
        "format": ta.RECORD_FORMAT,
        "suite_version": suite["version"],
        "suite_digest": suite["digest"],
        "reference": {"basename": reference_name,
                      "identity": ta.file_identity(str(reference_path))},
        "hidden_size": TEXT_DIM,
        "presentations": presentations,
        "token_total": sum(entry["tokens"] for entry in presentations),
        "built_at": "2026-08-13T00:00:00",
    }
    with open(directory / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return str(reference_path)


# ---------------------------------------------------------------------------
# 1. The prompt suite is a tracked, versioned asset
# ---------------------------------------------------------------------------

def test_the_shipped_suite_is_the_gate_corpus_and_carries_a_version():
    suite = ta.load_suite()
    assert suite["version"] == "h3-te-suite-v1"
    assert len(suite["prompts"]) == 102
    assert suite["composite_target_tokens"] == 220
    assert suite["digest"] and len(suite["digest"]) == 64
    assert os.path.isfile(suite["path"])


def test_the_digest_moves_when_the_prompts_do_even_at_the_same_version(tmp_path):
    first = _small_suite(tmp_path / "a", prompts=("one two", "three"))
    (tmp_path / "b").mkdir()
    second = _small_suite(tmp_path / "b", prompts=("one two", "four"))
    assert first["version"] == second["version"]
    assert first["digest"] != second["digest"]


def test_the_corpus_is_prompts_then_composites():
    suite = ta.load_suite()
    corpus = ta.build_corpus(_Tokenizer(), suite)
    names = [name for name, _ in corpus]
    assert names[:102] == [f"p{index:03d}" for index in range(102)]
    assert all(name.startswith("c") for name in names[102:])
    assert len(names) > 102
    # No special tokens: the tokenizer's own BOS/EOS must not appear.
    assert all(1 not in ids and 2 not in ids for _, ids in corpus)


def test_a_measurement_records_the_suite_that_produced_it(tmp_path, stub_encode, suite):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components)
    root = str(tmp_path / "store")

    record = ta.measure_substitution(components, root=root)
    assert record["suite_version"] == suite["version"]
    assert record["suite_digest"] == suite["digest"]


def test_a_bank_from_another_suite_is_not_compared_against(
        tmp_path, stub_encode, suite, monkeypatch):
    """Numbers from different suites are not comparable, so they are not made."""
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components)
    root = str(tmp_path / "store")

    (tmp_path / "b").mkdir()
    other = _small_suite(tmp_path / "b", prompts=("one two", "three four five", "six"),
                         version="test-suite-v2")
    monkeypatch.setattr(ta, "load_suite", lambda path=None: other)

    assert ta.measure_substitution(components, root=root) is None


# ---------------------------------------------------------------------------
# 2. Identity keying
# ---------------------------------------------------------------------------

def test_the_same_file_keys_the_same_way(tmp_path):
    path = str(tmp_path / "encoder.safetensors")
    save_file({"a": torch.arange(64, dtype=torch.float32)}, path)
    ta._IDENTITY_CACHE.clear()
    first = ta.file_identity(path)
    ta._IDENTITY_CACHE.clear()
    assert ta.file_identity(path) == first


def test_a_changed_header_changes_the_key(tmp_path):
    """A re-quantization changes dtypes and offsets; the key must follow."""
    tensors = {"a": torch.arange(64, dtype=torch.float32)}
    first = str(tmp_path / "a.safetensors")
    second = str(tmp_path / "b.safetensors")
    save_file(tensors, first)
    save_file({"renamed": tensors["a"]}, second)
    assert ta.file_identity(first) != ta.file_identity(second)


def test_a_changed_size_changes_the_key(tmp_path):
    first = str(tmp_path / "a.safetensors")
    second = str(tmp_path / "b.safetensors")
    save_file({"a": torch.zeros(64)}, first)
    save_file({"a": torch.zeros(128)}, second)
    assert ta.file_identity(first) != ta.file_identity(second)


def test_the_same_basename_with_other_content_does_not_inherit_the_key(tmp_path):
    """A re-quantized file dropped in under the old name must key differently."""
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    first = str(tmp_path / "old" / ENCODER_4B)
    second = str(tmp_path / "new" / ENCODER_4B)
    save_file({"a": torch.ones(4096)}, first)
    save_file({"a": torch.full((4096,), 2.0)}, second)
    assert os.path.getsize(first) == os.path.getsize(second)
    assert ta.file_identity(first) != ta.file_identity(second)


def test_identical_content_under_two_names_keys_identically(tmp_path):
    first = str(tmp_path / "one.safetensors")
    second = str(tmp_path / "two.safetensors")
    save_file({"a": torch.ones(4096)}, first)
    save_file({"a": torch.ones(4096)}, second)
    assert ta.file_identity(first) == ta.file_identity(second)


def _write_gguf(path, *, kv=(("general.name", "enc"),), tensor="blk.0.weight", payload=b"\x00" * 4096):
    """A minimal but real GGUF: magic, KV table, tensor table, then data."""
    def string(text):
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    body = b""
    for key, value in kv:
        body += string(key) + struct.pack("<I", 8) + string(value)
    body += string(tensor) + struct.pack("<I", 1) + struct.pack("<Q", 32)
    body += struct.pack("<I", 0) + struct.pack("<Q", 0)
    blob = struct.pack("<4sIQQ", b"GGUF", 3, 1, len(kv)) + body
    blob += b"\x00" * ((32 - len(blob) % 32) % 32) + payload
    with open(path, "wb") as fh:
        fh.write(blob)
    return str(path)


def test_a_gguf_keys_on_its_kv_metadata_and_tensor_table(tmp_path):
    first = _write_gguf(tmp_path / "a.gguf")
    second = _write_gguf(tmp_path / "b.gguf")
    assert ta.file_identity(first) == ta.file_identity(second)

    renamed_kv = _write_gguf(tmp_path / "c.gguf", kv=(("general.name", "other"),))
    renamed_tensor = _write_gguf(tmp_path / "d.gguf", tensor="blk.0.bias")
    other_weights = _write_gguf(tmp_path / "e.gguf", payload=b"\x01" * 4096)
    keys = {ta.file_identity(path)
            for path in (first, renamed_kv, renamed_tensor, other_weights)}
    assert len(keys) == 4


def test_the_measurement_key_depends_on_all_three_files_and_the_suite():
    base = ("suite", "enc", "proj", "ref", "dit")
    key = ta.measurement_key(*base)
    for index in range(len(base)):
        changed = list(base)
        changed[index] = "other"
        assert ta.measurement_key(*changed) != key


# ---------------------------------------------------------------------------
# 3. Storage: written, read back, and degrading to "none"
# ---------------------------------------------------------------------------

def test_a_measurement_is_stored_and_read_back(tmp_path, stub_encode, suite):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components)
    root = str(tmp_path / "store")

    record = ta.measure_substitution(components, root=root)
    assert record is not None

    read_back = ta.local_te_agreement(
        components["text_encoder_path"], projection, root=root)
    assert read_back is not None
    assert read_back["measured_at"] == record["measured_at"]
    assert read_back["stage_a"]["rows"] == record["stage_a"]["rows"]


def test_a_missing_store_is_no_measurement(tmp_path):
    projection = _projection_file(tmp_path)
    encoder = str(tmp_path / ENCODER_4B)
    save_file({"w": torch.zeros(4, 4)}, encoder)
    assert ta.local_te_agreement(encoder, projection,
                                 root=str(tmp_path / "nothing-here")) is None
    assert ta.list_reference_banks(root=str(tmp_path / "nothing-here")) == []


def test_a_corrupt_store_is_no_measurement(tmp_path, stub_encode, suite):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components)
    root = str(tmp_path / "store")
    ta.measure_substitution(components, root=root)

    for path in (ta.store_dir(root) / "measurements").glob("*.json"):
        path.write_text("{not json", encoding="utf-8")
    assert ta.local_te_agreement(components["text_encoder_path"], projection, root=root) is None

    for path in (ta.store_dir(root) / "banks").iterdir():
        (path / "manifest.json").write_text("[]", encoding="utf-8")
    assert ta.list_reference_banks(root=root) == []
    assert ta.measure_substitution(components, root=root) is None


def test_a_bank_whose_tensor_file_vanished_is_not_offered(tmp_path, stub_encode, suite):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components)
    root = str(tmp_path / "store")
    for path in (ta.store_dir(root) / "banks").iterdir():
        (path / "bank.safetensors").unlink()
    assert ta.list_reference_banks(root=root) == []
    assert ta.measure_substitution(components, root=root) is None


# ---------------------------------------------------------------------------
# 4. The metrics, on tensors with known answers
# ---------------------------------------------------------------------------

def _pairs(reference, candidate):
    return [(reference, candidate)]


def test_a_perfect_candidate_scores_one_everywhere():
    reference = torch.randn(9, 5)
    metrics = ta.agreement_metrics(_pairs(reference, reference.clone()),
                                   ta.bank_global_mean([reference]))
    assert metrics["cos_median"] == 1.0
    assert metrics["cos_mean_removed_median"] == 1.0
    assert metrics["norm_ratio"]["median"] == 1.0
    assert metrics["rel_rms"] == 0.0
    assert metrics["frac_norm_in_band"] == 1.0


def test_the_constant_predictor_scores_zero_mean_removed_and_high_raw():
    """The whole reason the mean-removed number is the load-bearing one.

    Rows share a large offset, so raw cosine is dominated by it while the
    mean-removed cosine of a constant predictor is exactly 0.
    """
    offset = torch.full((256,), 4.0)
    reference = offset + 0.3 * torch.randn(40, 256, generator=torch.Generator().manual_seed(3))
    global_mean = ta.bank_global_mean([reference])
    constant = global_mean.unsqueeze(0).expand(reference.shape[0], -1).contiguous()

    metrics = ta.agreement_metrics(_pairs(reference, constant), global_mean)
    assert metrics["cos_median"] > 0.9
    assert abs(metrics["cos_mean_removed_median"]) < 1e-6


def test_a_scaled_candidate_has_the_scale_as_its_norm_ratio_and_rel_rms():
    reference = torch.randn(7, 5, generator=torch.Generator().manual_seed(5))
    metrics = ta.agreement_metrics(_pairs(reference, reference * 1.5),
                                   ta.bank_global_mean([reference]))
    assert metrics["norm_ratio"]["median"] == pytest.approx(1.5, abs=1e-3)
    assert metrics["rel_rms"] == pytest.approx(0.5, abs=1e-3)
    # Direction is untouched by a positive scale.
    assert metrics["cos_median"] == pytest.approx(1.0, abs=1e-4)
    # 1.5 is outside [0.8, 1.25], so the registered band clause fails wholly.
    assert metrics["frac_norm_in_band"] == 0.0


def test_the_sink_row_is_excluded_from_the_aggregates_and_reported_apart():
    reference = torch.randn(6, 5, generator=torch.Generator().manual_seed(9))
    candidate = reference.clone()
    candidate[0] = -reference[0]  # only the sink row disagrees

    metrics = ta.agreement_metrics(_pairs(reference, candidate),
                                   ta.bank_global_mean([reference]))
    assert metrics["rows"] == 5
    assert metrics["cos_median"] == 1.0
    assert metrics["rel_rms"] == 0.0
    assert metrics["sink_row"]["cos_median"] == pytest.approx(-1.0, abs=1e-4)
    assert metrics["sink_row"]["norm_ratio_median"] == pytest.approx(1.0, abs=1e-4)


def test_the_global_mean_ignores_the_sink_row():
    tensor = torch.zeros(3, 4)
    tensor[0] = 100.0
    tensor[1] = 1.0
    tensor[2] = 3.0
    assert torch.allclose(ta.bank_global_mean([tensor]), torch.full((4,), 2.0))


def test_a_shape_mismatch_between_the_banks_is_refused():
    with pytest.raises(ValueError, match=r"not the same presentation"):
        ta.agreement_metrics(_pairs(torch.randn(4, 5), torch.randn(3, 5)),
                             torch.zeros(5))


# ---------------------------------------------------------------------------
# 5. Token counts must match position-for-position
# ---------------------------------------------------------------------------

def test_a_substitute_that_tokenizes_differently_voids_the_comparison(
        tmp_path, stub_encode, suite):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components, tokens_override=1)
    root = str(tmp_path / "store")

    with pytest.raises(ValueError, match=r"tokenizes the suite differently"):
        ta.measure_substitution(components, root=root)


def test_a_projection_that_changed_the_row_count_is_refused(tmp_path, stub_encode, suite):
    """`project_prompt_embeds` owns this refusal; the measurement inherits it."""
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components)
    root = str(tmp_path / "store")
    components["text_encoder"] = _Encoder(tokens_out=2)

    with pytest.raises(ValueError, match=r"per position and is void"):
        ta.measure_substitution(components, root=root)


# ---------------------------------------------------------------------------
# 6. Building a bank: the two refusals
# ---------------------------------------------------------------------------

def test_a_substitute_cannot_be_used_as_a_reference(tmp_path, stub_encode):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)

    with pytest.raises(ValueError, match=r"cannot be one"):
        ta.build_reference_bank(components, reference_basename=ENCODER_4B,
                                root=str(tmp_path / "store"))


def test_a_bank_cannot_be_mislabelled_as_another_encoders(tmp_path, stub_encode, suite):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    components["te_projection"] = None  # a released encoder carries none

    with pytest.raises(ValueError, match=r"Load the encoder you are naming"):
        ta.build_reference_bank(components, reference_basename="some_other_32b.safetensors",
                                root=str(tmp_path / "store"))


def test_a_bank_is_built_stored_and_found_again(tmp_path, stub_encode, suite):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    components["te_projection"] = None
    components["text_encoder"] = _Encoder(width=TEXT_DIM)
    root = str(tmp_path / "store")

    seen = []
    manifest = ta.build_reference_bank(
        components, reference_basename=ENCODER_4B, root=root,
        progress=lambda done, total, name: seen.append((done, total, name)))

    assert manifest["reference"]["basename"] == ENCODER_4B
    assert manifest["hidden_size"] == TEXT_DIM
    assert len(seen) == len(manifest["presentations"]) == len(ta.build_corpus(_Tokenizer(), suite))
    found = ta.find_reference_bank(components["text_encoder_path"], root=root, suite=suite)
    assert found is not None and found["suite_digest"] == suite["digest"]
    assert set(ta.load_reference_bank(found)) == {
        entry["name"] for entry in manifest["presentations"]}


def test_a_quantized_token_refiner_is_refused_rather_than_loaded(tmp_path):
    """Stage B needs the refiner unquantized; a quantized one has no view."""
    header = {
        "condition_proj.weight": {"dtype": "F8_E4M3", "shape": [8, 4], "data_offsets": [0, 0]},
        "token_refiner.blocks.0.attn.qkv_proj.weight": {
            "dtype": "BF16", "shape": [24, 8], "data_offsets": [0, 0]},
    }
    blob = json.dumps(header).encode("utf-8")
    path = tmp_path / "dit.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)

    with pytest.raises(ValueError, match=r"stores its token refiner quantized"):
        ta.build_stage_b(str(path))


def test_a_dit_without_a_refiner_is_refused(tmp_path):
    header = {"blocks.0.mlp.fc1.weight": {"dtype": "BF16", "shape": [4, 4],
                                          "data_offsets": [0, 0]}}
    blob = json.dumps(header).encode("utf-8")
    path = tmp_path / "dit.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)

    with pytest.raises(ValueError, match=r"no condition_proj/token_refiner tensors"):
        ta.build_stage_b(str(path))


# ---------------------------------------------------------------------------
# 7. The automatic hook
# ---------------------------------------------------------------------------

def test_the_hook_is_silent_when_there_is_no_bank(tmp_path, stub_encode):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    assert ta.maybe_measure_substitution(components, root=str(tmp_path / "store")) is None


def test_the_hook_measures_once_and_then_stops(tmp_path, stub_encode, suite):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components)
    root = str(tmp_path / "store")

    first = ta.maybe_measure_substitution(components, root=root)
    second = ta.maybe_measure_substitution(components, root=root)

    assert first is not None
    assert second is None


def test_the_hook_swallows_a_failure_rather_than_failing_the_load(tmp_path, monkeypatch, capsys):
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)

    def explode(*args, **kwargs):
        raise RuntimeError("bank store on fire")

    monkeypatch.setattr(ta, "list_reference_banks", explode)
    assert ta.maybe_measure_substitution(components, root=str(tmp_path / "store")) is None
    assert "bank store on fire" in capsys.readouterr().out


def test_the_load_path_calls_the_hook():
    """The hook must sit on the load, not be a function nobody invokes."""
    import inspect

    from core.models.minimax_h3 import loader

    source = inspect.getsource(loader.load_minimax_h3_from_path)
    assert "maybe_measure_substitution(components)" in source


# ---------------------------------------------------------------------------
# 8. The shipped table is a labelled fallback, and a local measurement wins
# ---------------------------------------------------------------------------

def test_the_shipped_table_is_labelled_as_measured_elsewhere():
    entry = published_te_substitution(f"a/{ENCODER_4B}", f"b/{PROJECTION_4B}")
    assert entry["source"] == "published"
    assert entry["cosine"] == 0.826
    assert not hasattr(PUBLISHED_TE_SUBSTITUTIONS, "MEASURED_TE_SUBSTITUTIONS")

    message = describe_te_substitution(f"a/{ENCODER_4B}", f"b/{PROJECTION_4B}")
    assert "not on this installation" in message
    assert "constant predictor" in message


def test_an_unknown_pairing_gets_nothing_from_the_table(tmp_path):
    assert published_te_substitution("a/unknown.safetensors", "b/unknown.safetensors") is None
    message = describe_te_substitution("a/unknown.safetensors", "b/unknown.safetensors")
    assert "No agreement with a released encoder is recorded" in message


def test_a_local_measurement_beats_the_shipped_number(
        tmp_path, stub_encode, suite, monkeypatch):
    projection = _projection_file(tmp_path, name=PROJECTION_4B)
    components = _components(tmp_path, projection)
    root = str(tmp_path / "store")
    _fabricate_bank(tmp_path, suite, components)
    record = ta.measure_substitution(components, root=root)

    monkeypatch.setattr(ta, "local_te_agreement",
                        lambda te, proj, root=None: record)
    resolved = measured_te_substitution(components["text_encoder_path"], projection)
    assert resolved["source"] == "local"
    assert resolved["cosine"] == record["stage_a"]["cos_mean_removed_median"]

    message = describe_te_substitution(components["text_encoder_path"], projection)
    assert "Measured on this installation" in message
    assert "0.826" not in message


def test_a_broken_store_falls_back_to_the_shipped_number(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(ta, "local_te_agreement", explode)
    resolved = measured_te_substitution(f"a/{ENCODER_4B}", f"b/{PROJECTION_4B}")
    assert resolved["source"] == "published"


# ---------------------------------------------------------------------------
# 9. End to end on a fabricated bank: the arithmetic, proven without a 32B
# ---------------------------------------------------------------------------

def test_the_cheap_half_end_to_end_against_a_fabricated_bank(
        tmp_path, stub_encode, monkeypatch, capsys):
    """A bank that is the candidate's own output divided by 1.25.

    Every number is then predictable: direction is exact (a positive scale moves
    no angle), the norm ratio is the scale, rel-RMS is |scale - 1|, and the
    constant predictor sits far below on the mean-removed cosine. The
    mean-removed cosine is NOT 1 here and must not be asserted so: subtracting
    the reference's own mean from a scaled copy does tilt it.
    """
    suite = _small_suite(tmp_path / "suite",
                         prompts=("alpha beta gamma delta", "epsilon zeta eta theta iota",
                                  "kappa lambda mu"), target=8)
    monkeypatch.setattr(ta, "load_suite", lambda path=None: suite)
    projection = _projection_file(tmp_path)
    components = _components(tmp_path, projection)
    _fabricate_bank(tmp_path, suite, components, scale=1.25)
    root = str(tmp_path / "store")

    record = ta.measure_substitution(components, root=root)

    stage_a = record["stage_a"]
    assert stage_a["cos_median"] == pytest.approx(1.0, abs=1e-3)
    assert stage_a["norm_ratio"]["median"] == pytest.approx(1.25, abs=1e-2)
    assert stage_a["rel_rms"] == pytest.approx(0.25, abs=1e-2)
    assert stage_a["cos_mean_removed_median"] > 0.9
    assert stage_a["baseline_cos_mean_removed_median"] < 0.5
    assert stage_a["baseline_rel_rms"] > stage_a["rel_rms"]
    # No DiT was loaded, so there is no post-refiner view and the record says so.
    assert record["stage_b"] is None
    assert "no DiT file" in record["stage_b_reason"]

    summary = ta.summarize_measurement(record)
    assert summary["source"] == "local"
    assert summary["stage"] == "raw"
    assert summary["reference"] == "released_32b.safetensors"
    print(f"[end-to-end] {json.dumps(summary)}")
    assert "end-to-end" in capsys.readouterr().out
