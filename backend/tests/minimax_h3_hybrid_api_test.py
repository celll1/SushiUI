"""C5: the MiniMax-H3 hybrid over the API.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_hybrid_api_test.py -v

Design doc: `docs/guides/MINIMAX_H3_HYBRID_LOADER_DESIGN.md` (rev2) sections 6.1
and 6.2. Three things this layer can get wrong, each silently:

* sending something on a load that named no overlay, which would change every
  other architecture's load path for a MiniMax-H3 feature;
* handing the loader a request of a shape it does not accept, which surfaces as
  a refusal that reads like a bug rather than as the merge the user asked for;
* answering a refused pair with a 500, which says "the server broke" about a
  pair the server correctly declined to merge.

Everything here is header-only: the fixtures are the fake trees from
`minimax_h3_hybrid_preflight_test` and no real load ever runs, so no checkpoint
under `M:/model/minimax_h3` is opened.
"""

import asyncio
import inspect
import os
import sys

import pytest
import yaml

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, _BACKEND_DIR)

from minimax_h3_hybrid_preflight_test import _h3_header, _tree  # noqa: E402

from api.param_defaults import H3_HYBRID_LOAD_DEFAULTS  # noqa: E402
from core.models.minimax_h3.hybrid_spec import (  # noqa: E402
    DEFAULT_BLOCK_RANGE_END,
    DEFAULT_BLOCK_RANGE_START,
    HYBRID_REQUEST_KEYS,
    PRESET_BLOCK_RANGE_ADALN,
    MiniMaxH3HybridRefusal,
    describe_minimax_h3_hybrid_overlay_choices,
    normalize_hybrid_request,
    preflight_hybrid_request,
)

_OPENAPI = os.path.join(os.path.dirname(_BACKEND_DIR), "openapi.yaml")

# The fake trees hold 6 blocks, so the shipped 25..49 default names blocks that
# do not exist there; the load tests that reach a real preflight pass a range.
_RANGE = {"hybrid_block_range_start": 2, "hybrid_block_range_end": 3}


class _StubPipelineManager:
    """Records the load call. `current_model_info` is read by the response."""

    def __init__(self, raiser=None):
        self.current_model_info = {"type": "minimax_h3"}
        self.seen = None
        self._raiser = raiser

    def load_model(self, **kwargs):
        self.seen = kwargs
        if self._raiser is not None:
            self._raiser(**kwargs)


def _post(monkeypatch, manager, **fields):
    """POST a real multipart form at the real route.

    Not a direct `await routes.load_model(...)`: called that way, an unsent
    field is the `Form(...)` sentinel rather than its default, and `Form(None)`
    is TRUTHY -- a base-only load would look like a hybrid request. Resolving the
    defaults is the framework's job, so the test has to let it do it.
    """
    import httpx
    from fastapi import FastAPI

    import api.routes as routes
    from api.error_handlers import register_error_handlers

    monkeypatch.setattr(routes, "pipeline_manager", manager)
    app = FastAPI()
    register_error_handlers(app)
    app.post("/models/load")(routes.load_model)

    data = {"source_type": "safetensors", "source": "M:/model/x.safetensors"}
    data.update({k: v for k, v in fields.items() if v is not None})

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/models/load", data=data)
            return response.status_code, response.json()

    return asyncio.run(run())


def _load(monkeypatch, **fields):
    """A successful load; returns the kwargs the route forwarded."""
    manager = _StubPipelineManager()
    status, payload = _post(monkeypatch, manager, **fields)
    assert status == 200, payload
    return manager.seen


# ---------------------------------------------------------------------------
# defaults: one source, three places
# ---------------------------------------------------------------------------

def test_the_defaults_bundle_is_the_loaders_own_constants():
    """Re-typed 25/49 here would let the API advertise a range the loader does
    not apply."""
    assert H3_HYBRID_LOAD_DEFAULTS == {
        "overlay_file": None,
        "preset": PRESET_BLOCK_RANGE_ADALN,
        "block_range_start": DEFAULT_BLOCK_RANGE_START,
        "block_range_end": DEFAULT_BLOCK_RANGE_END,
        "final_adaln_from_overlay": False,
    }
    assert set(H3_HYBRID_LOAD_DEFAULTS) == set(HYBRID_REQUEST_KEYS)


@pytest.mark.parametrize("field,key", [
    ("overlay_file", "overlay_file"),
    ("hybrid_preset", "preset"),
    ("hybrid_block_range_start", "block_range_start"),
    ("hybrid_block_range_end", "block_range_end"),
    ("hybrid_final_adaln_from_overlay", "final_adaln_from_overlay"),
])
def test_every_form_default_comes_from_the_bundle(field, key):
    from api import routes

    default = inspect.signature(routes.load_model).parameters[field].default
    assert default.default == H3_HYBRID_LOAD_DEFAULTS[key]


# ---------------------------------------------------------------------------
# no overlay: the load path is the one it always was
# ---------------------------------------------------------------------------

def test_a_load_without_an_overlay_passes_no_hybrid_argument_at_all(monkeypatch):
    """Not `hybrid=None` -- ABSENT. The default in `load_model`'s own signature
    is what a base-only load has to keep hitting."""
    from core.pipeline import DiffusionPipelineManager

    assert "hybrid" not in _load(monkeypatch)
    assert inspect.signature(
        DiffusionPipelineManager.load_model).parameters["hybrid"].default is None


def test_an_empty_overlay_forwards_exactly_the_pre_c5_kwargs(monkeypatch):
    """An omitted multipart field arrives as "" from some clients, and a recipe
    left over in a client's state must not turn a base-only load into anything."""
    baseline = _load(monkeypatch)
    seen = _load(monkeypatch, overlay_file="",
                 hybrid_preset="whatever", hybrid_block_range_start=7,
                 hybrid_block_range_end=8, hybrid_final_adaln_from_overlay=True)
    assert seen == baseline
    assert set(seen) == {"source_type", "source", "pipeline_type", "force_reload",
                         "text_encoder_file", "clip_projection_file"}


# ---------------------------------------------------------------------------
# with an overlay: the request the loader accepts, and nothing else
# ---------------------------------------------------------------------------

def test_an_overlay_reaches_the_pipeline_as_a_hybrid_request(monkeypatch):
    seen = _load(monkeypatch, overlay_file="M:/model/minimax_h3/o.safetensors",
                 hybrid_block_range_start=3, hybrid_block_range_end=9,
                 hybrid_final_adaln_from_overlay=True)
    assert seen["hybrid"] == {
        "overlay_file": "M:/model/minimax_h3/o.safetensors",
        "preset": PRESET_BLOCK_RANGE_ADALN,
        "block_range_start": 3,
        "block_range_end": 9,
        "final_adaln_from_overlay": True,
    }


def test_the_assembled_request_is_one_the_loader_accepts_unchanged(monkeypatch):
    """The route builds the dict and validates nothing; if its SHAPE were wrong
    the whole feature would fail as an "unrecognised field" refusal."""
    seen = _load(monkeypatch, overlay_file="M:/o.safetensors")
    assert normalize_hybrid_request(seen["hybrid"]) == seen["hybrid"]


def test_omitting_the_recipe_sends_the_documented_defaults(monkeypatch):
    seen = _load(monkeypatch, overlay_file="M:/o.safetensors")
    assert seen["hybrid"]["block_range_start"] == DEFAULT_BLOCK_RANGE_START
    assert seen["hybrid"]["block_range_end"] == DEFAULT_BLOCK_RANGE_END
    assert seen["hybrid"]["final_adaln_from_overlay"] is False


# ---------------------------------------------------------------------------
# refusals are 400s, and only for a request that asked for a hybrid
# ---------------------------------------------------------------------------

def _refused(monkeypatch, raiser, **fields):
    """POST a load that fails; returns `(status, message)`."""
    status, payload = _post(monkeypatch, _StubPipelineManager(raiser), **fields)
    message = payload.get("detail") or payload.get("error") or ""
    return status, str(message)


@pytest.mark.parametrize("hybrid_preset,expected", [
    ("full_overlay", "preset_unsupported"),
    ("blocks", "preset_unknown"),
])
def test_a_refused_preset_is_a_400_naming_its_code(monkeypatch, hybrid_preset, expected):
    def raiser(**kwargs):
        normalize_hybrid_request(kwargs["hybrid"])

    status, message = _refused(monkeypatch, raiser, overlay_file="M:/o.safetensors",
                               hybrid_preset=hybrid_preset)
    assert status == 400
    assert message.startswith(f"[{expected}]")


def test_a_refused_pair_is_a_400_from_the_real_preflight(monkeypatch, tmp_path):
    """End to end through the loader's own checks: the base and overlay here are
    the same partition, which is the direction the recipe is not."""
    _base, overlay = _tree(tmp_path)

    def raiser(**kwargs):
        preflight_hybrid_request(kwargs["source"], kwargs["hybrid"])

    status, message = _refused(monkeypatch, raiser, source=overlay,
                               overlay_file=overlay, **_RANGE)
    assert status == 400
    assert message.startswith("[variant_direction]")


def test_an_out_of_range_block_range_is_a_400(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)

    def raiser(**kwargs):
        preflight_hybrid_request(kwargs["source"], kwargs["hybrid"])

    status, message = _refused(monkeypatch, raiser, source=base, overlay_file=overlay,
                               hybrid_block_range_start=2, hybrid_block_range_end=99)
    assert status == 400
    assert message.startswith("[block_range_out_of_range]")


def test_an_overlay_on_another_architecture_is_a_coded_400(monkeypatch):
    """Through the real guard, so the code and the message are the shipped ones.

    Unreachable from this route in practice (the preflight resolves the tree
    first and refuses `not_an_h3_tree`), which is exactly why it must not be the
    one refusal that arrives without a code: a client branching on the `[code]`
    prefix would drop it into its unknown-error branch.
    """
    from core.model_loader import ModelLoader

    def raiser(**kwargs):
        ModelLoader._refuse_hybrid_on_other_arch("sdxl", kwargs["hybrid"])

    status, message = _refused(monkeypatch, raiser, overlay_file="M:/o.safetensors")
    assert status == 400
    assert message.startswith("[hybrid_wrong_architecture]")
    assert "sdxl" in message


def test_the_same_guard_passes_a_minimax_h3_hybrid_through():
    from core.model_loader import ModelLoader

    assert ModelLoader._refuse_hybrid_on_other_arch(
        "minimax_h3", {"overlay_file": "o"}) is None
    assert ModelLoader._refuse_hybrid_on_other_arch("sdxl", None) is None


@pytest.mark.parametrize("fields", [{}, {"overlay_file": "M:/o.safetensors"}])
def test_a_fault_that_is_not_a_refusal_is_a_500_even_under_a_hybrid(monkeypatch, fields):
    """The merge itself raising `ValueError` -- a dtype complaint from the real
    read, a selector bug -- is a fault to debug, not a bad request. Answering it
    with a 400 would blame the caller and drop the traceback."""
    def raiser(**_kwargs):
        raise ValueError("some loader complaint from inside the merge")

    status, _message = _refused(monkeypatch, raiser, **fields)
    assert status == 500


def test_the_traceback_is_printed_for_a_refused_hybrid(monkeypatch, capfd):
    """The 400 branch sits AFTER the existing log line; moving it before would
    lose the stack for exactly the failures C7 has to diagnose."""
    def raiser(**kwargs):
        normalize_hybrid_request(dict(kwargs["hybrid"], nonsense=1))

    status, _message = _refused(monkeypatch, raiser, overlay_file="M:/o.safetensors")
    assert status == 400
    printed = capfd.readouterr().out
    assert "Error loading model:" in printed and "Traceback" in printed


def test_a_busy_lifecycle_gate_is_still_a_409_with_a_hybrid(monkeypatch):
    from core.model_state_coordinator import ModelStateBusyError

    def raiser(**_kwargs):
        raise ModelStateBusyError("a generation is running")

    status, _message = _refused(monkeypatch, raiser, overlay_file="M:/o.safetensors")
    assert status == 409


# ---------------------------------------------------------------------------
# GET /models/minimax-h3/hybrid-overlays
# ---------------------------------------------------------------------------

def test_the_listing_offers_the_other_partition_and_not_the_base(tmp_path):
    base, overlay = _tree(tmp_path)
    choices = describe_minimax_h3_hybrid_overlay_choices(base)

    assert choices["base"]["path"] == base
    assert choices["base"]["variant"] == "fl2va"
    assert choices["base"]["num_blocks"] == 6
    assert choices["checked_block_range"] == [0, 5]
    assert [entry["path"] for entry in choices["overlays"]] == [overlay]

    candidate = choices["overlays"][0]
    assert candidate["variant"] == "ref2va"
    assert candidate["compatible"] is True
    assert candidate["reason"] is None and candidate["refusal_code"] is None
    assert candidate["num_blocks"] == 6
    assert candidate["quantization_format"] == "unquantized"


def _listing(model_path):
    """The route's response: `defaults` is added there, not by the core call."""
    from api import routes

    return asyncio.run(routes.list_minimax_h3_hybrid_overlays(model_path=str(model_path)))


def test_the_listing_carries_the_defaults_a_load_would_apply(tmp_path, monkeypatch):
    base, _overlay = _tree(tmp_path)
    assert "defaults" not in describe_minimax_h3_hybrid_overlay_choices(base)

    defaults = _listing(base)["defaults"]
    assert defaults["preset"] == H3_HYBRID_LOAD_DEFAULTS["preset"]
    assert defaults["block_range_start"] == H3_HYBRID_LOAD_DEFAULTS["block_range_start"]
    assert defaults["block_range_end"] == H3_HYBRID_LOAD_DEFAULTS["block_range_end"]
    assert defaults["final_adaln_from_overlay"] == (
        H3_HYBRID_LOAD_DEFAULTS["final_adaln_from_overlay"])
    assert defaults["presets"] == [PRESET_BLOCK_RANGE_ADALN]


def test_an_override_in_param_defaults_reaches_the_listing(tmp_path, monkeypatch):
    """`param_defaults.py` is where an API default is changed; a copy built in
    core would leave that change invisible to the frontend."""
    base, _overlay = _tree(tmp_path)
    monkeypatch.setitem(H3_HYBRID_LOAD_DEFAULTS, "block_range_start", 7)
    assert _listing(base)["defaults"]["block_range_start"] == 7


def test_every_advertised_preset_is_one_the_loader_accepts(tmp_path):
    from core.models.minimax_h3.hybrid_spec import SUPPORTED_PRESETS, validate_preset

    base, _overlay = _tree(tmp_path)
    assert _listing(base)["defaults"]["presets"] == list(SUPPORTED_PRESETS)
    for preset in SUPPORTED_PRESETS:
        validate_preset(preset, "o.safetensors")


def test_an_incompatible_candidate_is_listed_with_its_code_not_dropped(tmp_path):
    """A client offering the set has to be able to say why an entry is not
    available -- the same rule the projection candidates follow."""
    base, overlay = _tree(tmp_path, overlay_header=_h3_header(num_blocks=4))
    candidate = describe_minimax_h3_hybrid_overlay_choices(base)["overlays"][0]

    assert candidate["path"] == overlay
    assert candidate["compatible"] is False
    assert candidate["refusal_code"] == "key_set_mismatch"
    # The bare message: the code is its own field here, unlike the 400 detail.
    assert candidate["reason"] and not candidate["reason"].startswith("[")
    assert candidate["quantization_format"] is None


def test_a_ref2va_base_offers_nothing_and_says_why(tmp_path):
    """The recipe is not symmetric; listing the reverse as available would send
    the user into a refusal at load time."""
    base, overlay = _tree(tmp_path)
    choices = describe_minimax_h3_hybrid_overlay_choices(overlay)

    assert choices["base"]["variant"] == "ref2va"
    assert [entry["path"] for entry in choices["overlays"]] == [base]
    assert choices["overlays"][0]["refusal_code"] == "variant_direction"


def test_the_listing_refuses_a_path_that_is_not_an_h3_tree(tmp_path):
    with pytest.raises(ValueError, match=r"does not resolve to a MiniMax-H3 model tree"):
        describe_minimax_h3_hybrid_overlay_choices(str(tmp_path))


def test_the_listing_route_answers_a_bad_path_with_400_not_500(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from api import routes

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(routes.list_minimax_h3_hybrid_overlays(model_path=str(tmp_path)))
    assert excinfo.value.status_code == 400


def test_the_listing_route_maps_a_refusal_to_400(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from api import routes

    def raiser(_path):
        raise MiniMaxH3HybridRefusal("header_unreadable", "no header")

    monkeypatch.setattr(
        "core.models.minimax_h3.hybrid_spec.describe_minimax_h3_hybrid_overlay_choices",
        raiser)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(routes.list_minimax_h3_hybrid_overlays(model_path="M:/whatever"))
    assert excinfo.value.status_code == 400
    assert "[header_unreadable]" in excinfo.value.detail


# ---------------------------------------------------------------------------
# the spec says what the implementation does
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def openapi():
    with open(_OPENAPI, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_openapi_declares_the_form_fields_with_the_implemented_defaults(openapi):
    schema = (openapi["paths"]["/models/load"]["post"]["requestBody"]["content"]
              ["multipart/form-data"]["schema"]["properties"])

    assert schema["overlay_file"]["type"] == "string"
    assert schema["overlay_file"]["nullable"] is True
    assert "default" not in schema["overlay_file"]  # null, i.e. no hybrid
    assert schema["hybrid_preset"]["default"] == H3_HYBRID_LOAD_DEFAULTS["preset"]
    assert schema["hybrid_preset"]["enum"] == [PRESET_BLOCK_RANGE_ADALN]
    assert (schema["hybrid_block_range_start"]["default"]
            == H3_HYBRID_LOAD_DEFAULTS["block_range_start"])
    assert (schema["hybrid_block_range_end"]["default"]
            == H3_HYBRID_LOAD_DEFAULTS["block_range_end"])
    assert (schema["hybrid_final_adaln_from_overlay"]["default"]
            == H3_HYBRID_LOAD_DEFAULTS["final_adaln_from_overlay"])


def test_openapi_documents_the_refusal_codes_the_loader_can_return(openapi):
    """A code the spec omits is one a client cannot branch on."""
    import re

    described = openapi["paths"]["/models/load"]["post"]["responses"]["400"]["description"]
    source = open(os.path.join(_BACKEND_DIR, "core", "models", "minimax_h3",
                               "hybrid_spec.py"), encoding="utf-8").read()
    # Every construction site, both spellings: the private helper in the spec
    # module and the class itself, which the load dispatcher raises directly.
    source += open(os.path.join(_BACKEND_DIR, "core", "model_loader.py"),
                   encoding="utf-8").read()
    codes = set(re.findall(
        r'(?:_refuse|MiniMaxH3HybridRefusal)\(\s*\n?\s*[fr]?"([a-z0-9_]+)"', source))
    codes |= {"base_missing", "overlay_missing"}   # built from an f-string
    assert "hybrid_wrong_architecture" in codes, "the scan missed the dispatcher's refusal"
    missing = sorted(code for code in codes if code not in described)
    assert not missing, f"undocumented refusal codes: {missing}"


def test_openapi_carries_the_overlay_listing_and_the_hybrid_provenance(openapi):
    path = openapi["paths"]["/models/minimax-h3/hybrid-overlays"]["get"]
    assert (path["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
            == "#/components/schemas/MiniMaxH3HybridOverlayListResponse")

    loaded = openapi["components"]["schemas"]["LoadedModelInfo"]["properties"]
    assert set(loaded) >= {"base_variant", "overlay_variant", "hybrid"}
    assert (loaded["hybrid"]["allOf"][0]["$ref"]
            == "#/components/schemas/MiniMaxH3HybridProvenance")


def test_the_provenance_schema_matches_what_a_load_actually_reports(openapi, tmp_path):
    """`/models/current` was already returning these ahead of the spec."""
    base, overlay = _tree(tmp_path)
    provenance = preflight_hybrid_request(
        base, {"overlay_file": overlay, "block_range_start": 2,
               "block_range_end": 3}).provenance()

    declared = openapi["components"]["schemas"]["MiniMaxH3HybridProvenance"]["properties"]
    assert set(provenance) == set(declared)
    recipe = openapi["components"]["schemas"]["MiniMaxH3HybridRecipe"]["properties"]
    assert set(provenance["hybrid_recipe"]) == set(recipe)


def test_the_overlay_candidate_schema_matches_the_listing(openapi, tmp_path):
    base, _overlay = _tree(tmp_path)
    choices = _listing(base)
    schemas = openapi["components"]["schemas"]

    listing = schemas["MiniMaxH3HybridOverlayListResponse"]["properties"]
    assert set(choices) == set(listing)
    assert set(choices["base"]) == set(listing["base"]["properties"])
    assert set(choices["defaults"]) == set(listing["defaults"]["properties"])
    assert set(choices["overlays"][0]) == set(
        schemas["MiniMaxH3HybridOverlayCandidate"]["properties"])
