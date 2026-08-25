"""Phase U-3: understanding-branch training combined with reference conditioning.

SENSENOVA_TRAINING_DESIGN.md 13.4. The design's claim is that this needs zero
additional mechanism, because a reference item's ``<IMG_CONTEXT>`` rows traverse
the same understanding decoder layers in the same prefix pass. That is true of
the DECODER STACK and false of its entry: vendor ``_build_it2i_inputs`` returns
EMBEDS, and the shipped prefix loop only ever called ``embed_tokens``. One
keyword argument, asserted here rather than assumed.

The vendor ``_build_it2i_inputs`` / ``get_thw_indexes`` and the real target
enumerator are used unmodified; only the tokenizer, the ViT and the decoder
geometry are small.
"""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.sensenova.sensenova_lora import (  # noqa: E402
    iter_sensenova_lora_targets,
    und_gradient_unreachable_paths,
)
from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel  # noqa: E402
from core.training.ops import sensenova_ops  # noqa: E402
from core.training.sensenova_four_phase import (  # noqa: E402
    install_four_phase_backward,
)

_IMG_START, _IMG_END, _IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"
_SPECIAL = {_IMG_START: 100, _IMG_END: 101, _IMG_CONTEXT: 102}
_DIM = 8
_REF_GRID = (4, 4)  # ViT patches per reference; /2 merge -> 4 context tokens.
_LAYERS = 3


class _Tokenizer:
    """Special tokens keep their id; every other character is one id."""

    pattern = re.compile("|".join(re.escape(tok) for tok in _SPECIAL))

    def __call__(self, query, return_tensors=None):
        ids, cursor = [], 0
        for match in self.pattern.finditer(query):
            ids.extend([1] * (match.start() - cursor))
            ids.append(_SPECIAL[match.group(0)])
            cursor = match.end()
        ids.extend([1] * (len(query) - cursor))
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def convert_tokens_to_ids(self, token):
        return _SPECIAL[token]


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(_DIM, _DIM, bias=False))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(_DIM, _DIM, bias=False))


class _Layer(nn.Module):
    """The understanding half's shape: K/V leave, hidden goes on to the next layer."""

    attention_type = "full_attention"

    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _Mlp()

    def forward(self, hidden_states, **kwargs):
        assert kwargs.get("return_kv") is True
        attn = self.self_attn
        keys = attn.k_proj(hidden_states).unsqueeze(1)
        values = attn.v_proj(hidden_states).unsqueeze(1)
        attended = attn.o_proj(attn.q_proj(hidden_states))
        mlp = self.mlp
        hidden = attended + mlp.down_proj(mlp.gate_proj(hidden_states) * mlp.up_proj(hidden_states))
        return hidden_states + hidden, keys, values


class _UndModel(nn.Module):
    def __init__(self, layers=_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(layers)])
        self.embed_tokens = nn.Embedding(256, _DIM)
        self.config = SimpleNamespace(num_hidden_layers=layers, attention_dropout=0.0)


class _VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(_DIM, _DIM, bias=False)

    def forward(self, rows):
        return self.proj(rows)


class _Transformer(nn.Module):
    patch_size = 16
    downsample_ratio = 0.5
    img_start_token_id = _SPECIAL[_IMG_START]
    img_context_token_id = None

    # Vendor code under test, bound verbatim.
    get_thw_indexes = NEOChatModel.get_thw_indexes
    _build_it2i_inputs = NEOChatModel._build_it2i_inputs

    def __init__(self, layers=_LAYERS):
        super().__init__()
        self.und = _UndModel(layers)
        self.vision_model = _VisionTower()
        self.language_model = SimpleNamespace(
            model=self.und, get_input_embeddings=lambda: self.und.embed_tokens
        )
        self.extract_feature_calls = 0
        self.device = torch.device("cpu")
        # As load_components does before any adapter unfreezes its own targets.
        self.requires_grad_(False)

    def _build_t2i_query(self, text, system_message=None, append_text=None):
        return text + (append_text or "")

    def _build_t2i_text_inputs(self, tokenizer, query):
        input_ids = tokenizer(query)["input_ids"]
        t_idx = torch.arange(input_ids.shape[1], dtype=torch.long)
        zeros = torch.zeros_like(t_idx)
        return input_ids, torch.stack([t_idx, zeros, zeros], dim=0), {
            "full_attention": None
        }

    def _t2i_prefix_forward(self, input_ids, indexes, attention_mask):
        raise AssertionError("the trainable path must not use the vendor no-grad prefix")

    def extract_feature(self, pixel_values, grid_hw=None):
        self.extract_feature_calls += 1
        merged = int(1 / self.downsample_ratio) ** 2
        count = int((grid_hw[:, 0] * grid_hw[:, 1]).sum()) // merged
        return self.vision_model(torch.arange(count * _DIM, dtype=torch.float32).view(count, _DIM))


def _trainer(transformer, thaw_und=True, **overrides):
    if thaw_und:
        # What the adapter does after load_components froze the whole tree.
        for module in _und_targets(transformer).values():
            module.weight.requires_grad_(True)
    trainer = SimpleNamespace(
        transformer=transformer,
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=True,
        train_text_encoder=True,
    )
    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer


def _write_reference(tmp_path, name="ref.png"):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (64, 64), (200, 30, 30)).save(path)
    return str(path)


def _fake_load_image_native(image, patch_size, downsample_ratio, **kwargs):
    grid_h, grid_w = _REF_GRID
    return (
        torch.zeros(grid_h * grid_w, _DIM),
        torch.tensor([[grid_h, grid_w]], dtype=torch.long),
    )


def _encode(trainer, prompt, ref_paths=None, **kwargs):
    target = "core.models.sensenova.sensenova_pipeline_ops"
    with patch(f"{target}.load_image_native", side_effect=_fake_load_image_native), patch(
        f"{target}.raise_if_cancelled"
    ):
        return sensenova_ops.encode_prompt(
            trainer, prompt, reference_image_paths=ref_paths, **kwargs
        )


def _und_targets(transformer):
    return {
        path: module
        for path, _parent, _attr, module in iter_sensenova_lora_targets(
            transformer, branch="und"
        )
    }


def _kv_loss(cache):
    """Everything the generation pass can read, and nothing it cannot.

    The prefix keeps ``past_key_values`` and drops ``last_hidden_state``, which
    is why five understanding targets are structurally unreachable (U-0).
    """
    return sum(layer.keys.sum() + layer.values.sum() for layer in cache.layers)


# ---------------------------------------------------------------------------
# A. The refusal this phase lifts, and the seam the design's claim missed
# ---------------------------------------------------------------------------

# U-2-5, the commit that recorded this gap; its tree still carries both guards.
SHIPPED_COMMIT = "ce713b58"
_OPS_PATH = "backend/core/training/ops/sensenova_ops.py"
_REFUSAL = "text-only in this"


def _shipped_source(path: str) -> str:
    """The file as it shipped, rather than a transcription of it here.

    A hand-copied guard is byte-accurate on the day it is written and cannot
    say anything after that; `git show` reproduces the behaviour instead of
    restating it. Same mechanism as
    ``sensenova_four_phase_ui_exposure_test._git_show``.
    """
    try:
        result = subprocess.run(["git", "show", f"{SHIPPED_COMMIT}:{path}"],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if result.returncode != 0:  # pragma: no cover
        pytest.skip(f"{SHIPPED_COMMIT} not in this clone")
    return result.stdout.decode("utf-8")


def test_negative_control_the_shipped_tree_refused_on_both_paths():
    """Twice there, zero times now -- read off both trees, not transcribed.

    Each occurrence is tied to its path structurally rather than by position:
    the first precedes ``four_phase.cut``, which only the four-phase branch
    reaches, and the second precedes the eviction refusal, which only the
    single-backward branch carries.
    """
    shipped = _shipped_source(_OPS_PATH)
    sites = [
        index for index in range(len(shipped))
        if shipped.startswith(_REFUSAL, index)
    ]
    assert len(sites) == 2
    assert sites[0] < shipped.index("four_phase.cut(") < sites[1]
    assert sites[1] < shipped.index("cannot run with MoT phase ")

    now = Path(sensenova_ops.__file__).read_text(encoding="utf-8")
    assert _REFUSAL not in now
    # The eviction refusal on the same path is NOT what U-3 lifted.
    assert now.count("cannot run with MoT phase ") == 1


@pytest.mark.parametrize("four_phase", [False, True])
def test_both_paths_now_accept_a_reference_conditioned_item(tmp_path, four_phase):
    """The positive half, per path, asserted on what distinguishes the paths.

    Under the shipped tree each of these raised ``NotImplementedError`` after a
    25-32 GiB load (U-2-5); the refusal was never gated before the load.
    """
    transformer = _Transformer()
    trainer = _trainer(transformer)
    if four_phase:
        install_four_phase_backward(trainer)

    prefix = _encode(trainer, "a caption", [_write_reference(tmp_path)],
                     requires_grad=True)

    assert len(prefix.cache.layers) == _LAYERS
    keys = prefix.cache.layers[0].keys
    if four_phase:
        # A cut boundary: grad-requiring leaves, no graph behind them.
        assert keys.requires_grad and keys.is_leaf and keys.grad_fn is None
    else:
        # A live graph the generation backward runs on through.
        assert keys.grad_fn is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_ids": None, "inputs_embeds": None},
        {"input_ids": torch.ones(1, 3, dtype=torch.long), "inputs_embeds": torch.zeros(1, 3, _DIM)},
    ],
)
def test_prefix_loop_takes_exactly_one_entry(kwargs):
    model = _UndModel()
    idx = torch.arange(3)
    indexes = torch.stack([idx, torch.zeros_like(idx), torch.zeros_like(idx)], dim=0)
    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        sensenova_ops.forward_und_prefix_layers(
            model,
            kwargs["input_ids"],
            indexes,
            {"full_attention": None},
            inputs_embeds=kwargs["inputs_embeds"],
        )


# ---------------------------------------------------------------------------
# B. What flows
# ---------------------------------------------------------------------------


def test_reference_prefix_is_differentiable_and_carries_the_vit_rows(tmp_path):
    transformer = _Transformer()
    trainer = _trainer(transformer)

    prefix = _encode(trainer, "a caption", [_write_reference(tmp_path)], requires_grad=True)

    assert transformer.extract_feature_calls == 1
    assert transformer.img_context_token_id == _SPECIAL[_IMG_CONTEXT]
    sensenova_ops._assert_immutable_prefix_cache(prefix.cache, _LAYERS, trainable=True)
    # 13.4 7.5 differential 2: the t extent, not the token count.
    assert prefix.text_length < int(prefix.cache.get_seq_length())


def test_reference_conditioned_backward_reaches_the_same_targets_as_text_only(tmp_path):
    """The design says the same 289; measured here as the same SET, at L=3."""
    reached = {}
    for label, refs in (("text", None), ("reference", [_write_reference(tmp_path)])):
        transformer = _Transformer()
        targets = _und_targets(transformer)
        prefix = _encode(_trainer(transformer), "a caption", refs, requires_grad=True)
        _kv_loss(prefix.cache).backward()
        reached[label] = {
            path
            for path, module in targets.items()
            if module.weight.grad is not None and module.weight.grad.abs().sum() > 0
        }

    unreachable = und_gradient_unreachable_paths(_LAYERS)
    assert len(unreachable) == 5
    for label, paths in reached.items():
        assert len(paths) == 7 * _LAYERS - 5, label
        assert set(_und_targets(_Transformer())) - paths == unreachable, label
    # The claim under test: reference conditioning does not enlarge or shift it.
    assert reached["text"] == reached["reference"]


def test_a_reference_only_signal_still_reaches_the_understanding_layers(tmp_path):
    """Gradient from the spliced rows ALONE, with every text row zeroed out.

    Without this the previous test would pass on a prefix that ignored the
    reference entirely: the text rows would carry the whole gradient.
    """
    transformer = _Transformer()
    targets = _und_targets(transformer)
    trainer = _trainer(transformer)
    ids_seen = {}
    original = NEOChatModel.get_thw_indexes

    def spy(self, input_ids, grid_hw=None):
        ids_seen["ids"] = input_ids
        return original(self, input_ids, grid_hw)

    with patch.object(_Transformer, "get_thw_indexes", spy):
        prefix = _encode(trainer, "a caption", [_write_reference(tmp_path)], requires_grad=True)

    context_rows = (ids_seen["ids"] == _SPECIAL[_IMG_CONTEXT]).nonzero().flatten()
    assert context_rows.numel() > 0
    loss = sum(
        layer.keys[..., context_rows, :].sum() + layer.values[..., context_rows, :].sum()
        for layer in prefix.cache.layers
    )
    loss.backward()

    reached = {
        path
        for path, module in targets.items()
        if module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    }
    assert reached == set(targets) - und_gradient_unreachable_paths(_LAYERS)


# ---------------------------------------------------------------------------
# C. The vision tower stays frozen
# ---------------------------------------------------------------------------


def test_vision_tower_holds_no_gradient_after_a_reference_backward(tmp_path):
    transformer = _Transformer()
    for module in _und_targets(transformer).values():
        module.weight.requires_grad_(True)
    trainer = _trainer(transformer)

    prefix = _encode(trainer, "a caption", [_write_reference(tmp_path)], requires_grad=True)
    _kv_loss(prefix.cache).backward()

    tower = transformer.vision_model
    assert all(not p.requires_grad for p in tower.parameters())
    assert all(p.grad is None for p in tower.parameters())
    # And it is outside the enumeration every writer emits, not merely frozen.
    assert not any(
        path.startswith("vision_model") for path in _und_targets(transformer)
    )


def test_a_thawed_vision_tower_is_refused_rather_than_silently_trained(tmp_path):
    transformer = _Transformer()
    transformer.vision_model.proj.weight.requires_grad_(True)
    trainer = _trainer(transformer)

    with pytest.raises(RuntimeError, match="frozen understanding vision tower"):
        _encode(trainer, "a caption", [_write_reference(tmp_path)], requires_grad=True)

    # Text-only items never run the tower, so they are not refused for it.
    _encode(trainer, "a caption", None, requires_grad=True)


def test_the_tower_runs_under_no_grad_so_no_activation_is_retained(tmp_path):
    transformer = _Transformer()
    trainer = _trainer(transformer)
    seen = {}
    original = NEOChatModel._build_it2i_inputs

    def spy(self, tokenizer, query, pixel_values=None, grid_hw=None):
        embeds, indexes, mask = original(self, tokenizer, query, pixel_values, grid_hw)
        seen["grad_enabled"] = torch.is_grad_enabled()
        seen["embeds_grad_fn"] = embeds.grad_fn
        return embeds, indexes, mask

    with patch.object(_Transformer, "_build_it2i_inputs", spy):
        _encode(trainer, "a caption", [_write_reference(tmp_path)], requires_grad=True)

    assert seen["grad_enabled"] is False
    assert seen["embeds_grad_fn"] is None


# ---------------------------------------------------------------------------
# D. The four-phase split
# ---------------------------------------------------------------------------


def test_four_phase_reference_prefix_cuts_to_boundary_leaves(tmp_path):
    transformer = _Transformer()
    trainer = _trainer(transformer)
    four_phase = install_four_phase_backward(trainer)

    prefix = _encode(trainer, "a caption", [_write_reference(tmp_path)], requires_grad=True)

    sensenova_ops._assert_immutable_prefix_cache(
        prefix.cache, _LAYERS, boundary_leaf=True
    )
    assert four_phase._current is not None
    stored_inputs = four_phase._current[0]
    assert stored_inputs.embeds is True
    assert stored_inputs.tokens.dtype.is_floating_point


def test_four_phase_replays_the_same_reference_embeds_without_rerunning_the_vit(tmp_path):
    transformer = _Transformer()
    targets = _und_targets(transformer)
    trainer = _trainer(transformer)
    four_phase = install_four_phase_backward(trainer)

    prefix = _encode(trainer, "a caption", [_write_reference(tmp_path)], requires_grad=True)
    assert transformer.extract_feature_calls == 1

    # Phase 2 stands in for the generation backward: it terminates in the leaves.
    _kv_loss(prefix.cache).backward()
    four_phase.capture()
    four_phase.flush()

    # Phase 3 recomputed from the STORED embeds -- deterministic, no second ViT.
    assert transformer.extract_feature_calls == 1
    reached = {
        path
        for path, module in targets.items()
        if module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    }
    assert reached == set(targets) - und_gradient_unreachable_paths(_LAYERS)


def test_four_phase_negative_control_a_detached_boundary_trains_nothing(tmp_path):
    """Skipping phase 3 leaves the loss identical and the understanding half unmoved."""
    transformer = _Transformer()
    targets = _und_targets(transformer)
    trainer = _trainer(transformer)
    four_phase = install_four_phase_backward(trainer)

    prefix = _encode(trainer, "a caption", [_write_reference(tmp_path)], requires_grad=True)
    _kv_loss(prefix.cache).backward()
    four_phase.capture()

    assert all(module.weight.grad is None for module in targets.values())
