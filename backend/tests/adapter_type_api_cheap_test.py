"""What `GET /loras` reports about a file, and what `loras[].adapter_type` asserts.

The defect these gate: `LoRAManager._is_valid_lora_file` admitted a file on a
KEY-PREFIX test (`lora_unet_*` / `lora_te_*`) that says nothing about the
algebra, so a LoHa in the sd-scripts spelling was listed and selectable
"as an ordinary LoRA", while a Z-Image LyCORIS file (flattened
`lora_transformer_*` stems, no down/up keys) satisfied no arm at all and was
filtered OUT of the list on an architecture that loads and generates it.

Two claims are kept apart on purpose, because conflating them is how a UI ends
up calling a working file broken:
  * what the FILE is -- `adapter_type` / `adapter_state`, detected per file;
  * whether the loaded ARCHITECTURE can apply it -- `adapter_families`, read
    from `ENABLED_ADAPTER_PAIRS`.

No model loads, no CUDA. Synthetic safetensors files only. Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_type_api_cheap_test.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from safetensors.torch import save_file

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.adapter_types import (  # noqa: E402
    ADAPTER_TYPE_AUTO, ADAPTER_TYPE_CHOICES, ADAPTER_TYPE_MISMATCH_CODE,
    parse_lora_items,
)
from api.error_handlers import APIError, is_lora_refusal_code  # noqa: E402
from api.param_defaults import LORA_ITEM_DEFAULTS  # noqa: E402
from core.adapters.spec import FAMILY_NAMES, KNOWN_ARCHITECTURES  # noqa: E402
from core.extensions.lora_manager import LoRAManager  # noqa: E402

R, D = 4, 8


def _loha(stem: str):
    return {
        f"{stem}.hada_w1_a": torch.zeros(D, R),
        f"{stem}.hada_w1_b": torch.zeros(R, D),
        f"{stem}.hada_w2_a": torch.zeros(D, R),
        f"{stem}.hada_w2_b": torch.zeros(R, D),
        f"{stem}.alpha": torch.tensor(float(R)),
    }


def _lora(stem: str):
    return {
        f"{stem}.lora_down.weight": torch.zeros(R, D),
        f"{stem}.lora_up.weight": torch.zeros(D, R),
        f"{stem}.alpha": torch.tensor(float(R)),
    }


#: name -> (tensors, metadata). The four stem spellings are the ones measured
#: in the design doc's admission table.
FIXTURES = {
    # sd-scripts spelling: admitted by the old prefix test, and reported as an
    # ordinary LoRA.
    "ltx2_loha": (_loha("lora_unet_transformer_blocks_0_attn1_to_q"), None),
    "ltx2_lora": (_lora("lora_unet_transformer_blocks_0_attn1_to_q"), None),
    # Z-Image's flattened spelling: satisfied NO arm of the old test.
    "zimage_loha": (_loha("lora_transformer_layers_0_attn_to_q"), None),
    "zimage_lora": (_lora("lora_transformer_layers_0_attn_to_q"), None),
    # Unfactored LoKr: rank is legitimately absent, and upstream stores
    # lora_dim as the alpha.
    "lokr_full": ({
        "lora_unet_x.lokr_w1": torch.zeros(3, 4),
        "lora_unet_x.lokr_w2": torch.zeros(4, 3),
        "lora_unet_x.alpha": torch.tensor(1.0),
    }, None),
    # Tucker form: a real, file-intrinsic inconsistency for a Linear-only engine.
    "tucker_loha": (dict(_loha("lora_unet_y"),
                         **{"lora_unet_y.hada_t1": torch.zeros(R, R, 3, 3)}), None),
    # A valid `lora_bias=True` PEFT export: detection used to RAISE on its 1-D
    # bias, and "unknown" must stay a report rather than a refusal.
    "peft_bias": ({
        "base_model.model.x.lora_A.weight": torch.zeros(R, D),
        "base_model.model.x.lora_B.weight": torch.zeros(D, R),
        "base_model.model.x.lora_A.bias": torch.zeros(R),
    }, None),
    # Neither an adapter nor listable.
    "full_finetune": ({"model.diffusion_model.x.weight": torch.zeros(D, D)}, None),
    # DoRA: the decomposition axis, detected from dora_scale.
    "dora": (dict(_lora("lora_unet_z"),
                  **{"lora_unet_z.dora_scale": torch.zeros(D)}), None),
    # A LoHa whose stems classify_lora_keys cannot place, declaring an
    # architecture NAME in a foreign spelling. It generates on every enabled
    # architecture; validate()'s architecture axis called it broken.
    "foreign_model_type": (_loha("unplaceable.stem"),
                           {"model_type": "sdxl_base_v1-0"}),
    # The same foreign declaration on a file that IS inconsistent, so the
    # neutralised axis cannot be mistaken for a blanket "never invalid".
    "foreign_model_type_tucker": (
        dict(_loha("unplaceable.stem"),
             **{"unplaceable.stem.hada_t1": torch.zeros(R, R, 3, 3)}),
        {"model_type": "sdxl_base_v1-0"}),
}


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    """One LoRAManager over one directory of every fixture, scanned once."""
    directory = tmp_path_factory.mktemp("loras")
    for name, (tensors, metadata) in FIXTURES.items():
        save_file(tensors, str(directory / f"{name}.safetensors"), metadata=metadata)
    manager = LoRAManager(lora_dir=str(directory))
    manager.seeded_dirs = []
    entries = {e["name"].replace(".safetensors", ""): e
               for e in manager.get_available_loras(force_rescan=True)}
    return manager, directory, entries


# --- what the FILE is ------------------------------------------------------

def test_sd_scripts_loha_is_no_longer_reported_as_an_ordinary_lora(scanned):
    _, _, entries = scanned
    assert entries["ltx2_loha"]["adapter_type"] == "loha"
    assert entries["ltx2_loha"]["adapter_algorithm"] == "loha"
    assert entries["ltx2_lora"]["adapter_type"] == "lora"


def test_zimage_lycoris_spelling_is_listed_at_all(scanned):
    """It has no down/up keys and no lora_unet_/lora_te_ prefix, so the
    key-prefix admission test dropped it -- on one of the architectures where
    LoHa actually generates."""
    _, _, entries = scanned
    assert "zimage_loha" in entries
    assert entries["zimage_loha"]["adapter_type"] == "loha"
    assert entries["zimage_loha"]["arch"] == "zimage"


def test_full_finetune_is_still_excluded(scanned):
    _, _, entries = scanned
    assert "full_finetune" not in entries


def test_unknown_algebra_is_listed_and_reported_unknown_not_guessed(scanned):
    """`unknown` must not become a refusal on the LISTING path: a valid
    lora_bias=True PEFT export detects that way whenever the sniff misses."""
    _, _, entries = scanned
    entry = entries["peft_bias"]
    assert entry["adapter_type"] in ("lora", "unknown")
    if entry["adapter_type"] == "unknown":
        assert entry["adapter_state"] == "unknown"
    assert entry["adapter_state"] != "invalid"


def test_ordinary_lora_is_never_reported_invalid(scanned):
    """The engine does not validate ordinary LoRA either (its rank sniff covers
    three spellings only), so the listing must not judge it."""
    _, _, entries = scanned
    for name in ("ltx2_lora", "zimage_lora", "peft_bias"):
        assert entries[name]["adapter_state"] != "invalid", name


def test_unfactored_lokr_has_no_rank_and_is_still_ok(scanned):
    _, _, entries = scanned
    entry = entries["lokr_full"]
    assert entry["adapter_type"] == "lokr"
    assert entry["adapter_rank"] is None
    assert entry["adapter_state"] == "ok", entry["adapter_state_reason"]


def test_a_foreign_model_type_does_not_make_an_applicable_file_invalid(scanned):
    """`from_codec(architecture=None)` FALLS BACK to the file's `model_type`
    rather than ignoring it, so validating that axis without a loaded model to
    compare against refuses a file for naming an architecture this build does
    not spell -- one AdapterSession would apply, since it always passes the
    loaded arch and never reaches the fallback."""
    _, _, entries = scanned
    entry = entries["foreign_model_type"]
    assert entry["adapter_type"] == "loha"
    assert entry["adapter_state"] == "ok", entry["adapter_state_reason"]


def test_neutralising_that_axis_does_not_disarm_the_algebra_axes(scanned):
    _, _, entries = scanned
    entry = entries["foreign_model_type_tucker"]
    assert entry["adapter_state"] == "invalid"
    assert "Tucker" in (entry["adapter_state_reason"] or "")
    assert "architecture" not in (entry["adapter_state_reason"] or "")


def test_tucker_form_is_invalid_with_the_engines_own_reason(scanned):
    _, _, entries = scanned
    entry = entries["tucker_loha"]
    assert entry["adapter_state"] == "invalid"
    assert "Tucker" in (entry["adapter_state_reason"] or "")


def test_weight_decomposition_is_a_separate_axis(scanned):
    _, _, entries = scanned
    entry = entries["dora"]
    assert entry["adapter_algorithm"] == "lora"
    assert entry["weight_decompose"] is True
    assert entry["adapter_type"] == "dora"


def test_details_endpoint_reports_the_same_fields_as_the_list(scanned):
    manager, _, entries = scanned
    info = manager.get_lora_info("ltx2_loha.safetensors")
    for key in ("adapter_type", "adapter_algorithm", "weight_decompose",
                "adapter_format", "adapter_state", "adapter_rank"):
        assert info[key] == entries["ltx2_loha"][key], key
    # The details-only fields still work off the same single read.
    assert info["layers"] and info["exists"] is True


# --- the per-file cache ----------------------------------------------------

def test_rescan_does_not_reread_unchanged_files(scanned, monkeypatch):
    manager, _, _ = scanned
    reads = []
    original = manager._read_lora_header
    monkeypatch.setattr(manager, "_read_lora_header",
                        lambda p: (reads.append(p), original(p))[1])
    manager.get_available_loras(force_rescan=True)
    assert reads == []


def test_probe_cache_is_pruned_to_what_the_scan_saw(tmp_path):
    """The training output directory is a search path: a long run writing
    checkpoints must not grow the cache for the life of the process."""
    for name in ("a", "b", "c"):
        save_file(_lora("lora_unet_a"), str(tmp_path / f"{name}.safetensors"))
    manager = LoRAManager(lora_dir=str(tmp_path))
    manager.seeded_dirs = []
    manager.get_available_loras(force_rescan=True)
    assert len(manager._probe_cache) == 3

    (tmp_path / "b.safetensors").unlink()
    (tmp_path / "c.safetensors").unlink()
    manager.get_available_loras(force_rescan=True)
    assert len(manager._probe_cache) == 1


def test_an_edited_file_is_reread(tmp_path):
    path = tmp_path / "edited.safetensors"
    save_file(_lora("lora_unet_a"), str(path))
    manager = LoRAManager(lora_dir=str(tmp_path))
    manager.seeded_dirs = []
    assert manager.get_available_loras(force_rescan=True)[0]["adapter_type"] == "lora"

    save_file(_loha("lora_unet_a"), str(path))
    os.utime(path, (0, 0))  # mtime change is what the cache key notices
    assert manager.get_available_loras(force_rescan=True)[0]["adapter_type"] == "loha"


# --- the assertion ---------------------------------------------------------

def _items(manager, monkeypatch, payload):
    """Driven through `asyncio.run`, which is also what pins that the disk read
    stays off the caller's loop: `parse_lora_items` is a coroutine."""
    import asyncio

    import api.adapter_types as module
    import core.extensions.lora_manager as lm
    monkeypatch.setattr(lm, "lora_manager", manager)
    return asyncio.run(module.parse_lora_items(payload))


def test_omitted_adapter_type_defaults_to_auto(scanned, monkeypatch):
    manager, _, _ = scanned
    items = _items(manager, monkeypatch, '[{"path": "ltx2_loha.safetensors"}]')
    assert items[0]["adapter_type"] == ADAPTER_TYPE_AUTO == LORA_ITEM_DEFAULTS["adapter_type"]


def test_matching_assertion_passes(scanned, monkeypatch):
    manager, _, _ = scanned
    items = _items(manager, monkeypatch,
                   '[{"path": "ltx2_loha.safetensors", "adapter_type": "loha"}]')
    assert items[0]["adapter_type"] == "loha"


def test_mismatched_assertion_is_refused_not_overridden(scanned, monkeypatch):
    manager, _, _ = scanned
    with pytest.raises(APIError) as excinfo:
        _items(manager, monkeypatch,
               '[{"path": "ltx2_loha.safetensors", "adapter_type": "lora"}]')
    error = excinfo.value
    assert error.status_code == 400
    assert error.code == ADAPTER_TYPE_MISMATCH_CODE
    assert is_lora_refusal_code(error.code)
    # The message must name the real cause, not call the file broken.
    assert "is a loha checkpoint, not lora" in error.detail


def test_assertion_on_an_undetectable_file_is_refused_but_auto_still_works(tmp_path, monkeypatch):
    path = tmp_path / "opaque.safetensors"
    save_file({"lora_unet_marker": torch.zeros(2)}, str(path))
    manager = LoRAManager(lora_dir=str(tmp_path))
    manager.seeded_dirs = []
    assert manager.adapter_report("opaque.safetensors")["adapter_type"] == "unknown"

    with pytest.raises(APIError) as excinfo:
        _items(manager, monkeypatch,
               '[{"path": "opaque.safetensors", "adapter_type": "lora"}]')
    assert excinfo.value.code == ADAPTER_TYPE_MISMATCH_CODE
    assert "cannot be checked" in excinfo.value.detail
    # ... and the file itself is still applicable under auto.
    assert _items(manager, monkeypatch,
                  '[{"path": "opaque.safetensors"}]')[0]["adapter_type"] == "auto"


def test_a_missing_file_is_left_to_the_generation_paths_own_refusal(scanned, monkeypatch):
    """`lora_not_found` names the real cause; a type mismatch would not."""
    manager, _, _ = scanned
    items = _items(manager, monkeypatch,
                   '[{"path": "nope.safetensors", "adapter_type": "lokr"}]')
    assert items[0]["adapter_type"] == "lokr"


def test_auto_touches_no_disk_and_an_assertion_leaves_the_loop(scanned, monkeypatch):
    """The parse runs on 13 async routes: the file read has to go to a thread,
    and an all-`auto` request -- every request that does not assert -- must not
    pay for it."""
    import asyncio

    import api.adapter_types as module
    import core.extensions.lora_manager as lm

    manager, _, _ = scanned
    monkeypatch.setattr(lm, "lora_manager", manager)
    probes, hops = [], []
    monkeypatch.setattr(manager, "adapter_report",
                        lambda name: probes.append(name) or {"adapter_type": "lora"})
    real_to_thread = asyncio.to_thread
    async def counted(fn, *a, **kw):
        hops.append(fn.__name__)
        return await real_to_thread(fn, *a, **kw)
    monkeypatch.setattr(module.asyncio, "to_thread", counted)

    asyncio.run(module.parse_lora_items('[{"path": "ltx2_lora.safetensors"}]'))
    assert (probes, hops) == ([], [])

    asyncio.run(module.parse_lora_items(
        '[{"path": "ltx2_lora.safetensors", "adapter_type": "lora"}]'))
    assert probes == ["ltx2_lora.safetensors"] and hops == ["_assert_all_match_files"]


def test_unknown_assertion_value_is_refused(scanned, monkeypatch):
    manager, _, _ = scanned
    with pytest.raises(APIError) as excinfo:
        _items(manager, monkeypatch,
               '[{"path": "ltx2_lora.safetensors", "adapter_type": "locon"}]')
    assert excinfo.value.status_code == 400


def test_multipart_and_json_transports_read_the_same_item(scanned, monkeypatch):
    """One item model for both, so a field cannot mean different things."""
    manager, _, _ = scanned
    obj = {"path": "ltx2_lora.safetensors", "strength": 0.5,
           "apply_to_unet": False, "step_range": [10, 900],
           "unet_layer_weights": {"MID": 0.25}, "adapter_type": "lora"}
    import json
    from_json = _items(manager, monkeypatch, [dict(obj)])
    from_multipart = _items(manager, monkeypatch, json.dumps([obj]))
    assert from_json == from_multipart == [obj]


def test_every_other_item_field_survives_the_parse(scanned, monkeypatch):
    """The parse must not drop keys an architecture reads off the raw dict."""
    manager, _, _ = scanned
    item = _items(manager, monkeypatch,
                  '[{"path": "ltx2_lora.safetensors", "components": ["transformer"],'
                  ' "some_future_key": 3}]')[0]
    assert item["components"] == ["transformer"] and item["some_future_key"] == 3


# --- capability reporting --------------------------------------------------

def test_capability_payload_is_read_from_the_enablement_table():
    from api.arch_capabilities import adapter_families_payload
    from core.adapters.capability import ENABLED_ADAPTER_PAIRS

    payload = adapter_families_payload()
    assert set(payload) == set(ENABLED_ADAPTER_PAIRS)
    for arch, entry in payload.items():
        enabled = {FAMILY_NAMES[pair] for pair in ENABLED_ADAPTER_PAIRS[arch]}
        assert set(entry["supported"]) == enabled, arch
        # Every family is answered exactly once, either way.
        assert set(entry["supported"]) | set(entry["unsupported"]) == set(FAMILY_NAMES.values())
        assert not (set(entry["supported"]) & set(entry["unsupported"]))
        assert all(entry["unsupported"].values()), arch


def test_an_unenabled_architecture_is_not_advertised_as_accepting_loha():
    from api.arch_capabilities import adapter_families_payload
    payload = adapter_families_payload()
    assert "loha" not in payload["sd15"]["supported"]
    assert "loha" in payload["sd15"]["unsupported"]
    # ... while the four flipped rows are advertised, because they generate.
    for arch in ("zimage", "krea2", "minit2i", "ltx2"):
        assert {"lora", "loha", "lokr"} <= set(payload[arch]["supported"]), arch


def test_block_swap_effect_follows_the_declared_install_order():
    from api.arch_capabilities import adapter_families_payload
    from core.adapters.capability import (AFTER_SPLIT, BEFORE_SPLIT,
                                          BLOCK_SWAP_ADAPTER_ORDER,
                                          BLOCK_SWAP_REFUSAL_CODE,
                                          BLOCK_SWAP_WARNING_CODE)
    expected = {AFTER_SPLIT: ("refused", BLOCK_SWAP_REFUSAL_CODE),
                BEFORE_SPLIT: ("not_offloaded", BLOCK_SWAP_WARNING_CODE)}
    payload = adapter_families_payload()
    for arch, order in BLOCK_SWAP_ADAPTER_ORDER.items():
        entry = payload[arch]["block_swap"]
        assert entry["order"] == order
        assert (entry["effect"], entry["code"]) == expected.get(
            order, ("not_applicable", None)), arch
    for arch, entry in payload.items():
        assert ("block_swap" in entry) == (arch in BLOCK_SWAP_ADAPTER_ORDER)


def test_mismatch_reaches_the_client_as_400_with_its_code(scanned, monkeypatch):
    """Through the real error handlers and a real HTTP round trip: the taxonomy
    is only useful if a client can branch on it."""
    import asyncio

    import httpx
    from fastapi import FastAPI

    import api.adapter_types as module
    import core.extensions.lora_manager as lm
    from api.error_handlers import register_error_handlers

    manager, _, _ = scanned
    monkeypatch.setattr(lm, "lora_manager", manager)

    app = FastAPI()
    register_error_handlers(app)

    @app.post("/probe")
    async def probe(payload: dict):
        await module.parse_lora_items(payload["loras"])
        return {"ok": True}

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            return await client.post("/probe", json={"loras": [
                {"path": "ltx2_loha.safetensors", "adapter_type": "lora"}]})

    response = asyncio.run(call())
    assert response.status_code == 400
    assert response.json()["code"] == ADAPTER_TYPE_MISMATCH_CODE


# --- frontend transport ----------------------------------------------------
# The design doc's requirement is that a new `loras[]` FIELD survives queue and
# loop generation rather than each panel growing a control. Every carrier copies
# the ARRAY (`loras: p.loras`), so any per-item field rides along -- these gate
# that shape, since rebuilding an item field-by-field is exactly how this repo
# has dropped parameters before (ADD_A_PARAMETER.md, failure patterns 4 and 5).

_FRONTEND = os.path.join(_REPO, "frontend", "src")
_PANELS = ["Txt2ImgPanel", "Img2ImgPanel", "InpaintPanel"]


def _read_frontend(*parts):
    with open(os.path.join(_FRONTEND, *parts), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("panel", _PANELS)
def test_loop_generation_carries_the_whole_lora_array(panel):
    source = _read_frontend("components", "generation", f"{panel}.tsx")
    assert "stepParams.loras = step.useMainLoRAs ? (mainParams.loras || []) : [];" in source


def test_queue_dispatch_and_formdata_carry_the_whole_lora_array():
    assert "loras: p.loras," in _read_frontend(
        "components", "generation", "GenerationQueueProcessor.tsx")
    api = _read_frontend("utils", "api.ts")
    assert 'JSON.stringify(paramsWithImages.loras || [])' in api
    assert "adapter_type?: AdapterTypeAssertion;" in api


@pytest.mark.parametrize("panel", _PANELS)
def test_no_panel_rebuilds_a_lora_item_field_by_field(panel):
    """A rebuilt item silently drops every field the rebuilder does not list."""
    source = _read_frontend("components", "generation", f"{panel}.tsx")
    for line in source.splitlines():
        if "loras" in line and "path:" in line:
            pytest.fail(f"{panel} constructs a LoRA item inline: {line.strip()}")


# --- spec parity -----------------------------------------------------------

def _openapi_schema(name):
    import yaml
    with open(os.path.join(_REPO, "openapi.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["components"]["schemas"][name]


def test_openapi_assertion_enum_matches_the_backend():
    enum = _openapi_schema("LoRARequestItem")["properties"]["adapter_type"]
    assert set(enum["enum"]) == set(ADAPTER_TYPE_CHOICES)
    assert enum["default"] == LORA_ITEM_DEFAULTS["adapter_type"]


def test_openapi_detected_enums_match_the_backend():
    properties = _openapi_schema("LoRAInfo")["properties"]
    assert set(properties["adapter_type"]["enum"]) == set(FAMILY_NAMES.values()) | {"unknown"}
    assert set(properties["arch"]["enum"]) == KNOWN_ARCHITECTURES | {"unknown"}
    assert set(properties["adapter_state"]["enum"]) == {"ok", "unknown", "invalid"}
