"""Adapter selection resolves through ARCH_REGISTRY, identically to the if-chain.

``LoRATrainer._create_adapter`` used to be a thirteen-branch if-chain over the
trainer's ``is_<arch>`` flags. It now asks the architecture's ArchHandler
(``lora_adapter_class`` / ``lora_adapter_kwargs``). This file is the
differential gate on that move: ``_legacy_plan`` below is a VERBATIM
transcription of the removed chain -- same parse function, same default scope
CSV, same three-tier attribute/config/default resolution, same log suffix --
and every case asserts the registry answer equals it.

Driven from ``ARCH_REGISTRY`` itself, so an architecture added to the registry
without a legacy transcription here fails rather than going untested.

No model is loaded: the adapters' ``__init__`` only stores its arguments, so
both sides are constructed against a stub trainer on CPU.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/arch_adapter_registry_test.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import torch  # noqa: E402

from core.training.arch import ARCH_REGISTRY, resolve_arch_name  # noqa: E402

RANK, ALPHA = 16, 8
DTYPE = torch.float32


def _trainer(arch: str, *, attrs=None, config=None):
    """A stub trainer carrying only what adapter selection reads."""
    ns = SimpleNamespace(config=dict(config or {}))
    # No trainer carries an ``is_sd15``: SD1.5 is the fallthrough of both the
    # chain and resolve_arch_name, so it is spelled as "no flag set".
    for name in ARCH_REGISTRY:
        if name != "sd15":
            setattr(ns, f"is_{name}", name == arch)
    for key, value in (attrs or {}).items():
        setattr(ns, key, value)
    return ns


# ---------------------------------------------------------------------------
# The removed if-chain, transcribed. Returns (adapter class, kwargs, log suffix)
# instead of assigning self.adapter, so it can be compared without a trainer.
# ---------------------------------------------------------------------------

def _legacy_plan(trainer):
    from core.training.adapters import (
        AceStepLoRAAdapter, AnimaLoRAAdapter, FLUX2LoRAAdapter,
        Ideogram4LoRAAdapter, Krea2LoRAAdapter, LensLoRAAdapter,
        Ltx2LoRAAdapter, MiniMaxH3LoRAAdapter, MiniT2ILoRAAdapter,
        SD15LoRAAdapter, SDXLLoRAAdapter, SenseNovaLoRAAdapter,
        ZImageLoRAAdapter,
    )

    if getattr(trainer, "is_sensenova", False):
        return SenseNovaLoRAAdapter, {}, ""
    elif trainer.is_zimage:
        return ZImageLoRAAdapter, {}, ""
    elif trainer.is_flux2:
        return FLUX2LoRAAdapter, {}, ""
    elif trainer.is_lens:
        from core.models.lens.lens_lora import parse_scope_csv
        scope_csv = (getattr(trainer, "lens_lora_scope", "")
                     or trainer.config.get("lens_lora_scope", "")
                     or "img_attn,txt_attn,img_mlp,txt_mlp")
        scope = parse_scope_csv(scope_csv)
        return LensLoRAAdapter, {"scope": scope}, f" (scope={scope})"
    elif trainer.is_ideogram4:
        from core.models.ideogram4.ideogram4_lora import parse_scope_csv
        scope_csv = (getattr(trainer, "ideogram4_lora_scope", "")
                     or trainer.config.get("ideogram4_lora_scope", "")
                     or "attn,mlp")
        scope = parse_scope_csv(scope_csv)
        return Ideogram4LoRAAdapter, {"scope": scope}, f" (scope={scope})"
    elif trainer.is_minit2i:
        from core.models.minit2i.minit2i_lora import parse_scope_csv, parse_te_scope_csv
        scope_csv = (getattr(trainer, "minit2i_lora_scope", "")
                     or trainer.config.get("minit2i_lora_scope", "")
                     or "attn,mlp,txt_embed")
        scope = parse_scope_csv(scope_csv)
        te_scope_csv = (getattr(trainer, "minit2i_te_lora_scope", "")
                        or trainer.config.get("minit2i_te_lora_scope", "")
                        or "attn,ff")
        te_scope = parse_te_scope_csv(te_scope_csv)
        return (MiniT2ILoRAAdapter, {"scope": scope, "te_scope": te_scope},
                f" (scope={scope}, te_scope={te_scope})")
    elif trainer.is_krea2:
        from core.models.krea2.krea2_lora import parse_scope_csv
        scope_csv = (getattr(trainer, "krea2_lora_scope", "")
                     or trainer.config.get("krea2_lora_scope", "")
                     or "attn,mlp")
        scope = parse_scope_csv(scope_csv)
        return Krea2LoRAAdapter, {"scope": scope}, f" (scope={scope})"
    elif trainer.is_anima:
        scope_csv = (getattr(trainer, "anima_lora_scope", "")
                     or trainer.config.get("anima_lora_scope", "")
                     or "attention,mlp,llm_adapter")
        wanted = {tok.strip(): True for tok in scope_csv.split(",") if tok.strip()}
        if hasattr(trainer, "train_llm_adapter") or "train_llm_adapter" in trainer.config:
            wanted["llm_adapter"] = bool(
                getattr(trainer, "train_llm_adapter",
                        trainer.config.get("train_llm_adapter", True)))
        scope = {
            "attention": wanted.get("attention", True),
            "mlp": wanted.get("mlp", True),
            "mod": wanted.get("mod", False),
            "llm_adapter": wanted.get("llm_adapter", True),
        }
        return AnimaLoRAAdapter, {"scope": scope}, f" (scope={scope})"
    elif trainer.is_ltx2:
        scope_csv = (getattr(trainer, "ltx2_lora_scope", "")
                     or trainer.config.get("ltx2_lora_scope", "")
                     or "attention")
        wanted = {tok.strip(): True for tok in scope_csv.split(",") if tok.strip()}
        scope = {
            "attention": wanted.get("attention", True),
            "ff": wanted.get("ff", False),
            "audio": wanted.get("audio", False),
            "av_cross": wanted.get("av_cross", False),
        }
        return Ltx2LoRAAdapter, {"scope": scope}, f" (scope={scope})"
    elif trainer.is_minimax_h3:
        from core.training.adapters.minimax_h3_adapter import parse_scope_csv
        scope_csv = (getattr(trainer, "minimax_h3_lora_scope", "")
                     or trainer.config.get("minimax_h3_lora_scope", "")
                     or "attention,ff")
        scope = parse_scope_csv(scope_csv)
        return MiniMaxH3LoRAAdapter, {"scope": scope}, f" (scope={scope})"
    elif trainer.is_acestep:
        scope_csv = (getattr(trainer, "acestep_lora_scope", "")
                     or trainer.config.get("acestep_lora_scope", "")
                     or "attention")
        wanted = {tok.strip(): True for tok in scope_csv.split(",") if tok.strip()}
        scope = {
            "attention": wanted.get("attention", True),
            "mlp": wanted.get("mlp", False),
        }
        return AceStepLoRAAdapter, {"scope": scope}, f" (scope={scope})"
    elif trainer.is_sdxl:
        return SDXLLoRAAdapter, {}, ""
    else:
        return SD15LoRAAdapter, {}, ""


# Per-architecture probes. Each is (attributes, config) fed to the stub trainer;
# every architecture gets the default probe, and every architecture that reads a
# scope also gets a NARROWING one through the config and through the attribute,
# so a handler that dropped a tier fails.
PROBES = {
    "sd15": [({}, {})],
    "sdxl": [({}, {})],
    "zimage": [({}, {})],
    "flux2": [({}, {})],
    "sensenova": [({}, {})],
    "lens": [
        ({}, {}),
        ({}, {"lens_lora_scope": "img_attn"}),
        ({"lens_lora_scope": "txt_mlp"}, {"lens_lora_scope": "img_attn"}),
    ],
    "ideogram4": [
        ({}, {}),
        ({}, {"ideogram4_lora_scope": "attn"}),
        ({"ideogram4_lora_scope": "mlp"}, {"ideogram4_lora_scope": "attn"}),
    ],
    "krea2": [
        ({}, {}),
        ({}, {"krea2_lora_scope": "mlp"}),
        ({"krea2_lora_scope": "attn"}, {"krea2_lora_scope": "mlp"}),
    ],
    "minit2i": [
        ({}, {}),
        ({}, {"minit2i_lora_scope": "attn", "minit2i_te_lora_scope": "ff"}),
        ({"minit2i_lora_scope": "txt_embed"},
         {"minit2i_lora_scope": "attn", "minit2i_te_lora_scope": "ff"}),
    ],
    "anima": [
        ({}, {}),
        ({}, {"anima_lora_scope": "attention,mod"}),
        ({}, {"train_llm_adapter": False}),
        ({"train_llm_adapter": False}, {"anima_lora_scope": "mlp"}),
    ],
    "ltx2": [
        ({}, {}),
        ({}, {"ltx2_lora_scope": "attention,ff,audio,av_cross"}),
        ({"ltx2_lora_scope": "ff"}, {"ltx2_lora_scope": "attention"}),
    ],
    "minimax_h3": [
        ({}, {}),
        ({}, {"minimax_h3_lora_scope": "ff"}),
        ({"minimax_h3_lora_scope": "attention"}, {"minimax_h3_lora_scope": "ff"}),
    ],
    "acestep": [
        ({}, {}),
        ({}, {"acestep_lora_scope": "attention,mlp"}),
        ({"acestep_lora_scope": "mlp"}, {"acestep_lora_scope": "attention"}),
    ],
}


class AdapterRegistryParityTest(unittest.TestCase):

    def test_every_registry_arch_is_probed(self):
        """Driven from the registry: a new architecture must be transcribed
        here, or it ships with adapter selection untested."""
        self.assertEqual(set(PROBES), set(ARCH_REGISTRY))

    def test_registry_plan_matches_the_if_chain(self):
        for arch, probes in PROBES.items():
            handler = ARCH_REGISTRY[arch](None)
            for attrs, config in probes:
                with self.subTest(arch=arch, attrs=attrs, config=config):
                    trainer = _trainer(arch, attrs=attrs, config=config)
                    expected_cls, expected_kwargs, expected_log = _legacy_plan(trainer)
                    plan = handler.lora_adapter_plan(trainer)
                    self.assertIs(plan.adapter_cls, expected_cls)
                    self.assertEqual(plan.kwargs, expected_kwargs)
                    # The log line the trainer prints is derived from the kwargs;
                    # this pins it to the wording each branch used.
                    self.assertEqual(plan.log_detail, expected_log)

    def test_constructed_adapter_is_indistinguishable(self):
        """Not just the same class -- the same instance state, which is what
        'same constructor arguments' means for these adapters."""
        for arch, probes in PROBES.items():
            handler = ARCH_REGISTRY[arch](None)
            for attrs, config in probes:
                with self.subTest(arch=arch, attrs=attrs, config=config):
                    trainer = _trainer(arch, attrs=attrs, config=config)
                    expected_cls, expected_kwargs, _ = _legacy_plan(trainer)
                    expected = expected_cls(trainer, RANK, ALPHA, DTYPE,
                                            **expected_kwargs)
                    actual = handler.lora_adapter_plan(trainer).build(
                        trainer, RANK, ALPHA, DTYPE)
                    self.assertIs(type(actual), type(expected))
                    self.assertIs(actual.trainer, trainer)
                    self.assertEqual(actual.__dict__, expected.__dict__)

    def test_flags_resolve_to_the_registry_key_that_owns_the_adapter(self):
        """The handler bound at load time (``trainer.arch``) is what selection
        now reads, so its flag-priority order must agree with the chain's for
        every architecture -- not just for mutually exclusive flags."""
        for arch in ARCH_REGISTRY:
            with self.subTest(arch=arch):
                trainer = _trainer(arch)
                self.assertEqual(resolve_arch_name(trainer), arch)

    def test_a_handler_without_a_declared_adapter_refuses(self):
        """The base declares no class, so a future handler that forgets one
        raises here instead of silently falling through to SD1.5."""
        from core.training.arch.base_arch import ArchHandler

        class Nameless(ArchHandler):
            name = "nameless"

            def load_components(self, trainer): ...
            def setup_block_swap(self, trainer): ...
            def setup_attention_backend(self, trainer): ...
            def encode_prompt(self, trainer, prompt, *, requires_grad=False): ...
            def vae_encode(self, trainer, image_tensor, **kwargs): ...
            def vae_decode(self, trainer, latents, *, latent_h, latent_w): ...
            def train_step(self, trainer, ctx): ...
            def sample(self, trainer, sample_ctx): ...

        with self.assertRaises(NotImplementedError):
            Nameless().lora_adapter_plan(_trainer("sd15"))


if __name__ == "__main__":
    unittest.main()
