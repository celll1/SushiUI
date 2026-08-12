"""MiniMax-H3: pairing a converted small text encoder with its trained projection.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_projection_test.py -v

A converted encoder is 2560- (4B) or 4096-wide (8B) where the DiT takes 5120,
so it is only ever correct WITH the projection trained for that exact pair.
Every check here is about refusing the near-misses: the other size's
projection, a projection whose output does not fit the DiT, one fitted at
another tap, and -- the mistake this design actually made once -- a forward
that drops the linear skip ``W`` and runs the MLP alone.

Synthetic tensors only; no model, no GPU, nothing large on disk.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.models.minimax_h3 import te_projection  # noqa: E402
from core.models.minimax_h3.te_projection import (  # noqa: E402
    apply_te_projection,
    discover_te_projections,
    load_te_projection,
    read_te_projection_spec,
    resolve_te_projection,
)

# The two shipped pairs' real widths, so a message that names them is checkable.
D_IN_4B, D_IN_8B, TEXT_DIM = 2560, 4096, 5120


def _write_projection(path, *, d_in, d_out=8, tap=24, mlp_hidden=6, mlp_depth=1,
                      generator=None, corrupt=None):
    g = generator or torch.Generator().manual_seed(0)
    tensors = {
        "W": torch.randn(d_in, d_out, generator=g),
        "mean_in": torch.randn(d_in, generator=g),
        "std_in": torch.rand(d_in, generator=g) + 0.5,
        "mean_out": torch.randn(d_out, generator=g),
        "std_out": torch.rand(d_out, generator=g) + 0.5,
        "sink_out": torch.randn(d_out, generator=g),
        "mlp.0.weight": torch.randn(mlp_hidden, d_in, generator=g),
        "mlp.0.bias": torch.randn(mlp_hidden, generator=g),
        "mlp.2.weight": torch.randn(d_out, mlp_hidden, generator=g),
        "mlp.2.bias": torch.randn(d_out, generator=g),
    }
    if corrupt:
        tensors.update(corrupt)
    metadata = {"d_in": str(d_in), "d_out": str(d_out), "tap": str(tap),
                "mlp_hidden": str(mlp_hidden), "mlp_depth": str(mlp_depth)}
    save_file(tensors, str(path), metadata=metadata)
    return str(path)


def _tiny_projection(tmp_path, **kwargs):
    path = _write_projection(tmp_path / "proj.safetensors", d_in=5, d_out=4, mlp_hidden=6, **kwargs)
    return load_te_projection(read_te_projection_spec(path))


# ---------------------------------------------------------------------------
# The forward
# ---------------------------------------------------------------------------

def test_forward_is_the_linear_skip_plus_the_mlp(tmp_path):
    """``(z @ W + mlp(z)) * std_out + mean_out``, computed independently here."""
    projection = _tiny_projection(tmp_path)
    t = projection["tensors"]
    hidden = torch.randn(2, 3, 5)

    z = (hidden - t["mean_in"]) / t["std_in"]
    mlp = torch.nn.functional.gelu(z @ t["mlp.0.weight"].T + t["mlp.0.bias"]) @ t["mlp.2.weight"].T
    expected = (z @ t["W"] + mlp + t["mlp.2.bias"]) * t["std_out"] + t["mean_out"]
    expected[:, 0, :] = t["sink_out"]

    got = apply_te_projection(hidden, projection)
    assert torch.allclose(got, expected, atol=1e-5)


def test_dropping_the_W_skip_would_change_the_result(tmp_path):
    """The regression this file exists for: MLP-alone is a different projection.

    Measured on the real pair, the mean-removed cosine to the 32B reference is
    0.157 without the skip and 0.833 with it (gate G0c).
    """
    projection = _tiny_projection(tmp_path)
    t = projection["tensors"]
    hidden = torch.randn(1, 4, 5)

    z = (hidden - t["mean_in"]) / t["std_in"]
    mlp_only = (torch.nn.functional.gelu(z @ t["mlp.0.weight"].T + t["mlp.0.bias"])
                @ t["mlp.2.weight"].T + t["mlp.2.bias"]) * t["std_out"] + t["mean_out"]
    mlp_only[:, 0, :] = t["sink_out"]

    got = apply_te_projection(hidden, projection)
    # Rows 1.. differ; row 0 is sink_out in both, so compare only the rest.
    assert not torch.allclose(got[:, 1:, :], mlp_only[:, 1:, :], atol=1e-3)


def test_row_zero_is_replaced_by_sink_out_not_added(tmp_path):
    projection = _tiny_projection(tmp_path)
    hidden = torch.randn(2, 3, 5)
    got = apply_te_projection(hidden, projection)
    for batch in range(2):
        assert torch.allclose(got[batch, 0, :], projection["tensors"]["sink_out"], atol=1e-6)


def test_forward_refuses_a_hidden_state_of_the_wrong_width(tmp_path):
    projection = _tiny_projection(tmp_path)
    with pytest.raises(ValueError, match=r"d_in=5.*7-wide"):
        apply_te_projection(torch.randn(1, 3, 7), projection)


def test_gelu_is_exact_not_tanh(tmp_path):
    """A tanh-approximate GELU would move the output; the trained head is exact."""
    projection = _tiny_projection(tmp_path)
    t = projection["tensors"]
    hidden = torch.randn(1, 4, 5) * 3.0
    z = (hidden - t["mean_in"]) / t["std_in"]
    pre = z @ t["mlp.0.weight"].T + t["mlp.0.bias"]
    approx = (torch.nn.functional.gelu(pre, approximate="tanh") @ t["mlp.2.weight"].T
              + t["mlp.2.bias"] + z @ t["W"]) * t["std_out"] + t["mean_out"]
    approx[:, 0, :] = t["sink_out"]
    got = apply_te_projection(hidden, projection)
    assert not torch.allclose(got[:, 1:, :], approx[:, 1:, :], atol=1e-6)


# ---------------------------------------------------------------------------
# The header contract
# ---------------------------------------------------------------------------

def test_spec_reads_the_declared_geometry(tmp_path):
    path = _write_projection(tmp_path / "p.safetensors", d_in=D_IN_4B, d_out=TEXT_DIM, tap=24,
                             mlp_hidden=3)
    spec = read_te_projection_spec(path)
    assert (spec["d_in"], spec["d_out"], spec["tap"]) == (D_IN_4B, TEXT_DIM, 24)


def test_spec_refuses_metadata_that_contradicts_the_tensors(tmp_path):
    path = _write_projection(tmp_path / "p.safetensors", d_in=5, d_out=4,
                             corrupt={"W": torch.zeros(5, 6)})
    with pytest.raises(ValueError, match=r"contradict its declared d_in=5"):
        read_te_projection_spec(path)


def test_spec_refuses_an_unimplemented_mlp_depth(tmp_path):
    path = _write_projection(tmp_path / "p.safetensors", d_in=5, d_out=4, mlp_depth=2)
    with pytest.raises(ValueError, match=r"mlp_depth=2"):
        read_te_projection_spec(path)


def test_spec_refuses_a_missing_tensor(tmp_path):
    tensors = {"W": torch.zeros(5, 4)}
    save_file(tensors, str(tmp_path / "p.safetensors"),
              metadata={"d_in": "5", "d_out": "4", "tap": "24", "mlp_hidden": "6", "mlp_depth": "1"})
    with pytest.raises(ValueError, match=r"missing tensor"):
        read_te_projection_spec(str(tmp_path / "p.safetensors"))


# ---------------------------------------------------------------------------
# Discovery and the pairing gates
# ---------------------------------------------------------------------------

def _pair_dir(tmp_path):
    root = tmp_path / "minimax_h3"
    directory = root / te_projection.MINIMAX_H3_TE_PROJECTION_DIRNAME
    directory.mkdir(parents=True)
    _write_projection(directory / "mmh3-4b-ClipProj.safetensors", d_in=D_IN_4B, d_out=TEXT_DIM,
                      tap=24, mlp_hidden=2)
    _write_projection(directory / "mmh3-8b-ClipProj.safetensors", d_in=D_IN_8B, d_out=TEXT_DIM,
                      tap=24, mlp_hidden=2)
    return str(root)


def test_discovery_matches_d_in_to_the_encoder_width(tmp_path):
    root = _pair_dir(tmp_path)
    found = discover_te_projections(root, d_in=D_IN_4B)
    assert [os.path.basename(spec["path"]) for spec in found] == ["mmh3-4b-ClipProj.safetensors"]


def test_discovery_resolves_the_pair_for_a_4b_encoder(tmp_path):
    root = _pair_dir(tmp_path)
    spec = resolve_te_projection(root=root, te_path="qwen3vl_4b.safetensors",
                                 hidden_size=D_IN_4B, num_hidden_layers=24, text_dim=TEXT_DIM)
    assert os.path.basename(spec["path"]) == "mmh3-4b-ClipProj.safetensors"


def test_4b_encoder_with_the_8b_projection_is_refused_naming_both_widths(tmp_path):
    """The pairing that would silently mis-project: both numbers must be in the message."""
    root = _pair_dir(tmp_path)
    eight_b = os.path.join(root, te_projection.MINIMAX_H3_TE_PROJECTION_DIRNAME,
                           "mmh3-8b-ClipProj.safetensors")
    with pytest.raises(ValueError) as excinfo:
        resolve_te_projection(root=root, te_path="qwen3vl_4b.safetensors", hidden_size=D_IN_4B,
                              num_hidden_layers=24, text_dim=TEXT_DIM, override=eight_b)
    message = str(excinfo.value)
    assert f"d_in={D_IN_8B}" in message and f"hidden_size={D_IN_4B}" in message


def test_projection_that_does_not_fit_the_dit_is_refused_naming_both_widths(tmp_path):
    """Closes the hole where a wrong width only dies inside ``context_embedder``."""
    root = _pair_dir(tmp_path)
    directory = os.path.join(root, te_projection.MINIMAX_H3_TE_PROJECTION_DIRNAME)
    narrow = _write_projection(os.path.join(directory, "narrow.safetensors"),
                               d_in=64, d_out=1024, tap=24, mlp_hidden=2)
    with pytest.raises(ValueError) as excinfo:
        resolve_te_projection(root=root, te_path="te.safetensors", hidden_size=64,
                              num_hidden_layers=24, text_dim=TEXT_DIM, override=narrow)
    message = str(excinfo.value)
    assert "d_out=1024" in message and f"text_dim={TEXT_DIM}" in message


def test_projection_fitted_at_another_tap_is_refused_naming_both(tmp_path):
    root = _pair_dir(tmp_path)
    other = _write_projection(
        os.path.join(root, te_projection.MINIMAX_H3_TE_PROJECTION_DIRNAME, "tap16.safetensors"),
        d_in=64, d_out=TEXT_DIM, tap=16, mlp_hidden=2)
    with pytest.raises(ValueError) as excinfo:
        resolve_te_projection(root=root, te_path="te.safetensors", hidden_size=64,
                              num_hidden_layers=24, text_dim=TEXT_DIM, override=other)
    message = str(excinfo.value)
    assert "tap=16" in message and "num_hidden_layers=24" in message


def test_no_matching_projection_refuses_rather_than_running_unprojected(tmp_path):
    root = _pair_dir(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_te_projection(root=root, te_path="te.safetensors", hidden_size=1536,
                              num_hidden_layers=24, text_dim=TEXT_DIM)
    message = str(excinfo.value)
    assert "d_in=1536" in message and "Refusing to encode" in message


def test_two_candidates_of_the_same_width_refuse_rather_than_guess(tmp_path):
    root = _pair_dir(tmp_path)
    _write_projection(
        os.path.join(root, te_projection.MINIMAX_H3_TE_PROJECTION_DIRNAME, "another-4b.safetensors"),
        d_in=D_IN_4B, d_out=TEXT_DIM, tap=24, mlp_hidden=2)
    with pytest.raises(ValueError, match=r"Name one explicitly"):
        resolve_te_projection(root=root, te_path="te.safetensors", hidden_size=D_IN_4B,
                              num_hidden_layers=24, text_dim=TEXT_DIM)


def test_a_missing_override_path_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"nope\.safetensors"):
        resolve_te_projection(root=str(tmp_path), te_path="te.safetensors", hidden_size=64,
                              num_hidden_layers=24, text_dim=TEXT_DIM,
                              override=str(tmp_path / "nope.safetensors"))
