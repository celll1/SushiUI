"""A LoRA refusal's machine-readable code must reach the client, not just the log.

Every architecture refuses a LoRA that would do nothing, and every refusal path
records a `lora_not_found` / `lora_load_failed` / `lora_incompatible` warning
before raising. Those codes used to be write-only: the routes read
``get_warnings()`` on the SUCCESS path only, so a refusal reached the client as
a generic 400/500 with the taxonomy discarded.

The mechanism is a ``code`` on the RAISED EXCEPTION -- ``with_error_code`` for
the plain builtins the backends raise, the ``code=`` argument for the
``APIError`` subclasses, and the ``AdapterRefusal`` class attribute for
architectures already on ``core.adapters.session`` -- which the route copies
onto the ``APIError`` it answers with. It is NOT inferred from the warning
bucket: a refusal's warned text and its exception text deliberately differ on
several architectures (ACE-Step and LTX-2.3 raise a shorter sentence than they
warn), so any match-the-message scheme silently reports no code for them, and a
last-warning-wins scheme would pin an unrelated failure on a bystander warning.
``test_unrelated_failure_does_not_inherit_a_bystander_warning_code`` is the gate
for that.

Five architectures raising four different types are driven end to end here
(`FileNotFoundError` from SD1.5/SDXL, `RuntimeError` from Anima and MiniT2I,
`ValidationError` from LTX-2.3, `AdapterFileMissing` from Z-Image's session),
through the real warning store, the real error handlers and a real HTTP
response. Every tagged LoRA refusal answers 400 regardless of that raised type.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/refusal_error_code_cheap_test.py -v
"""

from __future__ import annotations

import ast
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio

import httpx
import torch
import yaml
from torch import nn
from fastapi import FastAPI
from safetensors.torch import save_file

from api import generation_status as gs
from api.error_handlers import (
    GenerationError,
    NotFoundError,
    ValidationError,
    register_error_handlers,
    with_error_code,
)
from api.error_handlers import ValidationError as CustomValidationError

_ROUTES = os.path.join(os.path.dirname(__file__), "..", "api", "routes.py")
_OPENAPI = os.path.join(_REPO, "openapi.yaml")

D = 8


# ── the two except-clause shapes routes.py uses ──────────────────────────────
# Copied here so the chain can run without a model; the AST gate below proves
# routes.py still has these shapes at all 13 generation endpoints.
def _run_like_a_generation_route(work, gen_id):
    try:
        return work()
    except (GenerationError, CustomValidationError, NotFoundError) as e:
        gs.fail_generation(str(e), generation_id=gen_id)
        gs.attach_error_context(e, gen_id)
        raise
    except Exception as e:
        gs.fail_generation(str(e), generation_id=gen_id)
        raise GenerationError(
            "Text-to-image generation failed",
            detail=f"{str(e)}",
            **gs.error_context(e, gen_id)
        )


def _request(app, method, path):
    """One real ASGI round trip. ``raise_app_exceptions=False`` because
    Starlette's ServerErrorMiddleware re-raises after sending the response, and
    the body is what this file is about."""
    async def _go():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.request(method, path)
    return asyncio.run(_go())


def _post(work):
    """Drive ``work`` through a real endpoint, real handlers, real response."""
    app = FastAPI()
    register_error_handlers(app)

    @app.post("/generate/txt2img")
    def _endpoint():
        gen_id = gs.start_generation("txt2img")
        _run_like_a_generation_route(work, gen_id)
        gs.complete_generation(generation_id=gen_id)
        return {"success": True, "warnings": gs.get_warnings(gen_id)}

    return _request(app, "POST", "/generate/txt2img")


def _assert_well_formed(response, status):
    body = response.json()
    assert response.status_code == status
    for field in ("error", "code", "status_code", "timestamp", "path"):
        assert field in body, f"{field} missing from {sorted(body)}"
    assert body["status_code"] == status
    assert body["path"] == "/generate/txt2img"
    assert isinstance(body["error"], str) and body["error"]
    return body


def _codes(body):
    return [w["code"] for w in body.get("warnings", [])]


# ── architecture stubs ───────────────────────────────────────────────────────
class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(D, D, bias=False)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])


def _anima_backend():
    from core.pipeline_backends.anima import AnimaMixin

    class _B(AnimaMixin):
        def __init__(self):
            self.anima_components = {"transformer": _Transformer()}

    return _B()


def _ltx2_backend():
    from core.pipeline_backends.ltx2 import LTX2Mixin

    class _B(LTX2Mixin):
        def __init__(self):
            self.ltx2_components = {"transformer": _Transformer()}

    return _B()


def _zimage_backend():
    from core.pipeline_backends.zimage import ZImageMixin

    class _B(ZImageMixin):
        def __init__(self):
            self.zimage_components = {"transformer": _Transformer()}

    return _B()


def _minit2i_backend():
    from core.pipeline_backends.minit2i import MiniT2IMixin

    class _B(MiniT2IMixin):
        def __init__(self):
            self.minit2i_components = {"transformer": _Transformer()}

    return _B()


MISSING = "no_such_lora_for_the_refusal_code_test.safetensors"


# ── the chain, per architecture and per raised type ──────────────────────────
def test_sd15_file_not_found_carries_lora_not_found():
    """SD1.5/SDXL go through diffusers, and refuse with a FileNotFoundError."""
    from core.extensions.lora_manager import LoRAManager

    response = _post(lambda: LoRAManager().load_loras(None, [{"path": MISSING}]))
    body = _assert_well_formed(response, 400)
    assert body["code"] == "lora_not_found"
    assert _codes(body) == ["lora_not_found"]


def test_anima_runtime_error_carries_lora_not_found():
    """Anima accumulates per-file failures and raises ONE joined RuntimeError."""
    response = _post(lambda: _anima_backend()._load_lora_anima([{"path": MISSING}]))
    body = _assert_well_formed(response, 400)
    assert body["code"] == "lora_not_found"


def test_ltx2_validation_error_keeps_its_400_and_carries_the_code():
    """LTX-2.3 refuses with an APIError, so the status must stay 400."""
    response = _post(lambda: _ltx2_backend()._load_lora_ltx2([{"path": MISSING}]))
    body = _assert_well_formed(response, 400)
    assert body["code"] == "lora_not_found"
    # A re-raised APIError gets its warnings from attach_error_context.
    assert _codes(body) == ["lora_not_found"]


def test_zimage_adapter_session_refusal_carries_its_class_code():
    """Z-Image is on ``core.adapters.session``, whose ``AdapterRefusal`` carries
    the code as a class attribute instead of being tagged at the raise. The
    route reads ``.code`` either way, so both mechanisms answer the same."""
    response = _post(lambda: _zimage_backend()._load_lora_zimage([{"path": MISSING}]))
    body = _assert_well_formed(response, 400)
    assert body["code"] == "lora_not_found"


def test_minit2i_unreadable_file_carries_lora_load_failed(tmp_path):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    response = _post(
        lambda: _minit2i_backend()._minit2i_prepare_loras([{"path": str(broken)}]))
    body = _assert_well_formed(response, 400)
    assert body["code"] == "lora_load_failed"


def test_anima_reports_the_first_failure_and_warnings_carry_every_code(tmp_path):
    """Two bad files, two different reasons: `code` names the first, and
    `warnings[]` still carries both."""
    ghost = tmp_path / "ghost.safetensors"
    save_file({"lora_unet_transformer_blocks_9_attn_q_proj.lora_down.weight": torch.zeros(2, D),
               "lora_unet_transformer_blocks_9_attn_q_proj.lora_up.weight": torch.zeros(D, 2)},
              str(ghost), metadata={"model_type": "anima"})

    response = _post(lambda: _anima_backend()._load_lora_anima(
        [{"path": str(ghost)}, {"path": MISSING}]))
    body = _assert_well_formed(response, 400)
    assert body["code"] == "lora_incompatible"
    assert set(_codes(body)) >= {"lora_incompatible", "lora_not_found"}


def test_every_lora_refusal_code_answers_400():
    """The open warning taxonomy is not an enum, but the codes which currently
    identify refusals form a deliberate HTTP-status contract."""
    codes = (
        "lora_not_found",
        "lora_load_failed",
        "lora_incompatible",
        "lora_partial",
        "lora_uncond_unavailable",
        "ltx2_lora_arch_mismatch",
        "minimax_h3_lora_variant_mismatch",
    )
    for code in codes:
        def _refuse(code=code):
            gs.add_warning("refused", code=code)
            raise with_error_code(RuntimeError("refused"), code)

        body = _assert_well_formed(_post(_refuse), 400)
        assert body["code"] == code
        assert _codes(body) == [code]


def test_tagged_api_error_is_normalized_to_400():
    """The first route clause re-raises APIError subclasses rather than wrapping
    them, so attach_error_context must normalize those too."""
    response = _post(lambda: (_ for _ in ()).throw(
        GenerationError("refused", code="lora_incompatible")))
    body = _assert_well_formed(response, 400)
    assert body["code"] == "lora_incompatible"


# ── the negative gates ───────────────────────────────────────────────────────
def test_unrelated_failure_does_not_inherit_a_bystander_warning_code():
    """The reason the code rides on the exception rather than being read back
    out of the warning bucket."""
    def _work():
        gs.add_warning("LoRA 'x': applied 3 of 4 pairs.", code="lora_partial")
        raise RuntimeError("CUDA out of memory")

    response = _post(_work)
    body = _assert_well_formed(response, 500)
    assert body["code"] is None
    assert _codes(body) == ["lora_partial"]


def test_untagged_exception_still_produces_a_well_formed_error():
    response = _post(lambda: (_ for _ in ()).throw(RuntimeError("something else")))
    body = _assert_well_formed(response, 500)
    assert body["code"] is None
    assert "warnings" not in body


def test_generic_handler_without_a_generation_is_well_formed():
    """An exception that never passed a generation route still answers with the
    documented shape rather than crashing on a missing code."""
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/boom")
    def _boom():
        raise RuntimeError("no generation here")

    body = _request(app, "GET", "/boom").json()
    assert body["status_code"] == 500
    assert body["code"] is None
    assert body["path"] == "/boom"


def test_success_path_warnings_are_unchanged():
    response = _post(lambda: gs.add_warning("degraded", code="quantization_fallback"))
    assert response.status_code == 200
    assert response.json() == {"success": True,
                               "warnings": [{"code": "quantization_fallback",
                                             "message": "degraded"}]}


# ── the production-code gate ─────────────────────────────────────────────────
def test_every_generation_route_carries_the_error_context():
    """The clause shape replicated above must be the one routes.py really has.

    Any generation endpoint that reports a failure without the context would
    answer a refusal with no code, which is the defect this file exists for.
    """
    tree = ast.parse(open(_ROUTES, encoding="utf-8").read())
    checked, missing = 0, []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "start_generation" not in {getattr(n.func, "id", None)
                                      for n in ast.walk(fn) if isinstance(n, ast.Call)}:
            continue
        for handler in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
            names = {getattr(n.func, "id", None)
                     for n in ast.walk(handler) if isinstance(n, ast.Call)}
            if "fail_generation" not in names:
                continue
            checked += 1
            if not names & {"error_context", "attach_error_context"}:
                missing.append((fn.name, handler.lineno))
    assert not missing, f"failure clauses with no error context: {missing}"
    assert checked >= 26, f"only {checked} failure clauses found; did a route move?"


def test_openapi_documents_partial_lora_as_a_400_refusal():
    with open(_OPENAPI, encoding="utf-8") as handle:
        schemas = yaml.safe_load(handle)["components"]["schemas"]

    for schema_name in ("GenerationParams", "Txt2VidRequest", "Txt2AudRequest"):
        description = schemas[schema_name]["properties"]["loras"]["description"]
        assert "lora_partial" in description
        assert "answers 400" in description
        assert "warns `lora_partial`" not in description

    error_code = schemas["ErrorResponse"]["properties"]["code"]["description"]
    assert "only partially matches" in error_code
    assert "answers 400" in error_code
