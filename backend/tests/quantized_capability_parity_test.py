"""Cross-architecture parity for the weight-only-quantized capability.

Three defect classes have now recurred on EVERY architecture that gained
weight-only quantized Linear layers, and each was found and fixed one arch at a
time:

1. **The `isinstance(x, nn.Linear)` trap.** `Int8Linear` / `Fp8Linear` are
   `nn.Module`s, NOT `nn.Linear` subclasses, so a LoRA target predicate written
   that way skips every quantized layer SILENTLY. Measured: 75% of intended
   targets dropped on Anima, 8 sites on FLUX.2 (plus an adapter whose dtype was
   taken from an int8 base weight), 3 sites on LTX-2.3.
2. **Advertised-but-unwired capabilities.** `QUANTIZED_LINEAR_ARCHS` gives an
   arch the `quantized_gemm` capability, the frontend renders the control from
   that capability, and the routes that serve the arch never accepted the
   parameter. LTX-2.3 hit this the day it joined the tuple; the same tuple had
   already left `extract_fp8_gemm_info` naming three archs out of four, so every
   FLUX.2 generation recorded no `fp8_gemm` at all. A THIRD hand-written map of
   the same shape, `extract_vae_info`'s, was found stale in the same file
   (missing `ltx2`/`acestep`, so LTX-2.3 videos recorded no VAE identity).
3. **Hook placement/ordering.** A guard that fires before the request is
   consulted. LTX-2.3's INT8 offloader guard refused every second block-swap
   generation of a session even with no quantization requested, because its
   offloader is persistent wrapper state; FLUX.2's identical-looking guard was
   safe only because a `finally` in three other functions cleared its flag.

Individually fixing each occurrence has not stopped the next one appearing, so
these tests are driven off `RUNTIME_INT8_ARCHS` / `QUANTIZED_LINEAR_ARCHS`
themselves. A newly added architecture is covered the moment it joins a tuple.

WHAT THESE TESTS DO NOT COVER. Nothing here loads a checkpoint, builds a real
architecture, or runs a forward: an LTX-2.3 DiT alone is 18.98 G parameters and
the FLUX.2/Ideogram 4 builds are of the same order, so the suite would not be
runnable. Where a property can only be shown on a real model, the strongest
available substitute is used and named as such:

* the LoRA predicates are exercised FUNCTIONALLY, on real `Int8Linear` /
  `Fp8Linear` / `nn.Linear` instances, but on those modules alone -- not on an
  arch's actual module tree. That the enumerator walks the right tree is the
  arch's own concern; that its predicate cannot see a quantized Linear is the
  bug class here, and that is what is tested.
* the LoRA adapter's dtype independence IS tested end to end, because
  `LoRALinearLayer` can be built directly over a quantized base.
* route/parameter coherence is read off the live FastAPI router, so it reflects
  the real signatures rather than a transcription of them.
* guard ordering is tested functionally where a fake pipeline manager suffices,
  and statically (AST) where it does not.

Registries below (`_GUARD_CASES`, `_QUANTIZED_GEMM_ROUTES` +
`_UNQUANTIZED_GENERATE_ROUTES`) are checked for COMPLETENESS against the tuples
and against the live router, so a new arch or a new route that is not covered
fails loudly rather than passing silently.

FOUR WAYS THIS SUITE WAS ITSELF WRONG, all found by adversarial audit and fixed
here, because they are the failure mode of a suite like this one:

* A test that pins a PROXY for the property instead of the property. Defect
  class 2's `extract_fp8_gemm_info` test grepped `pipeline.py` for the string
  `<arch>_components`, which is a property of the pipeline manager, not of the
  reporter; reverting the reporter to the exact broken map it was written for
  left all 20 tests green, because nothing in `backend/tests/` ever CALLED it.
  Every arch-map test here is now functional (it calls the function).
* A test written as a RELATION rather than a set. "A route that accepts
  `unet_quantization` must accept `quantized_gemm_mode`" is satisfied by a route
  that accepts neither, so deleting both from `/generate/txt2vid` was invisible.
  The route sets are now pinned absolutely and cross-checked against openapi.
* A test CIRCULAR against the table it was checking. The functional reporter
  test filed its fake module under a component name read from
  `layout_module_specs`, which is where `extract_fp8_gemm_info` reads its names
  too, so both sides agreed on any name -- right or wrong. Setting
  `EXPORT_LAYOUTS["acestep"]["modules"] = (("transformer", ""),)`, precisely the
  map drift this suite exists to catch, left every test green while production
  would look up `acestep_components["transformer"]`, find nothing, and record no
  `fp8_gemm` on every ACE-Step generation. `LoaderComponentNameAnchorTest` adds
  the third source: the loader's own return dict, which shares no source with
  either side.
* NO POSITIVE CONTROL for the runtime-int8 hooks. Every test that touched them
  passed only values that must convert NOTHING, so replacing a hook with
  `lambda self, params, progress_callback=None: None` was invisible (measured on
  acestep; krea2 and anima were in the same state).
  `RuntimeInt8ConversionPositiveControlTest` runs each arch's hook with an int8
  request over a fixture whose shape no arch policy filters, and requires the
  module tree to come back holding `Int8Linear`/`Fp8Linear`.

WHAT IS STILL NOT COVERED, after that pass:

* The LoRA predicate is still located BY NAME, and a name imported from
  `base_adapter` satisfies the lookup. EVERY conventional predicate an arch
  exposes is now exercised (it used to be the first in candidate order alone,
  which is how `pipeline_backends/acestep._is_lora_target` went functionally
  untested while `acestep_adapter._is_target` stood in for it) -- but a site
  that tests the type inline, rather than through a named predicate, is still
  reached only by the AST scan, and only if it is spelled as an
  `isinstance`/`type(x) is` test in a function whose source mentions "lora".
  Deliberately not hardened further: requiring each arch to DEFINE its own
  predicate would push archs to copy the shared one, which is the condition that
  produced this defect class in the first place.
* Nothing here runs a real architecture, so "the enumerator walks the right
  module tree" and "the conversion produces numerically correct weights" are out
  of scope by construction (see below). The positive control above converts ONE
  synthetic Linear per component: it separates a working hook from an empty one
  and from one that reads the wrong component key, and it says nothing about
  which of a real DiT's layers are selected.
* The static guard-ordering check is a complement, not a proof: it only requires
  that a top-level `raise` follow the first look at the request. On LTX-2.3 the
  hook now consults the request in its first statements, so that check is weak
  THERE and the functional `_GUARD_CASES` are what actually hold the line.
"""

import ast
import importlib
import inspect
import sys
import textwrap
import unittest
from unittest import mock
from pathlib import Path

import torch
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common.int8_runtime_quantize import (  # noqa: E402
    QUANTIZED_LINEAR_ARCHS, RUNTIME_INT8_ARCHS,
)

_REPO = Path(_BACKEND).parent


def _quantized_linears(in_features=8, out_features=8):
    """One instance of each weight-only quantized Linear class, plus nn.Linear."""
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    return {
        "nn.Linear": nn.Linear(in_features, out_features, bias=True,
                               dtype=torch.bfloat16),
        "Int8Linear": Int8Linear(in_features, out_features, bias=True,
                                 compute_dtype=torch.bfloat16),
        "Fp8Linear": Fp8Linear(in_features, out_features, bias=True,
                               compute_dtype=torch.bfloat16),
    }


# Modules that may own an architecture's LoRA target selection, by convention.
# Derived from the arch id so a new arch is searched without being listed.
def _lora_modules(arch):
    candidates = [
        f"core.models.{arch}.{arch}_lora",
        f"core.training.adapters.{arch}_adapter",
        f"core.pipeline_backends.{arch}",
        f"core.models.{arch}.loader",
    ]
    found = []
    for name in candidates:
        try:
            found.append(importlib.import_module(name))
        except ModuleNotFoundError as exc:
            # The candidate not existing is NORMAL -- this list is a convention,
            # not a manifest, and no arch has all four. A module that exists and
            # fails to import because one of ITS imports is missing is not: it
            # would drop out of the scan below silently, taking every isinstance
            # site in it with it. Only the first case is swallowed.
            absent = (exc.name or "")
            if absent != name and not name.startswith(absent + "."):
                _LORA_MODULE_IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            _LORA_MODULE_IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
    return found


# Populated by _lora_modules: candidate modules that EXIST but could not be
# imported. Asserted empty, because such a module is invisible to the AST scan.
_LORA_MODULE_IMPORT_ERRORS = {}


_PREDICATE_NAMES = ("_is_lora_target", "_is_target", "is_lora_wrappable_linear")


def _witness_component(arch):
    """The component-dict KEY under which `arch` keeps its quantized module.

    NOT always "transformer": ACE-Step's DiT is `acestep_components["dit"]`, and
    a fake that always said "transformer" would have let
    `extract_fp8_gemm_info`'s hardcoded name tuple pass for an arch it could not
    actually read -- the same shape as the map drift the test below exists for,
    one level down. Read from `EXPORT_LAYOUTS`, which is where the on-disk layout
    (and therefore the component that holds the quantized Linears) is declared,
    and which the reporter itself now derives its names from. Falls back to
    "transformer" for a quantized arch with no export layout.

    THIS IS THE SAME SOURCE THE REPORTER READS, so on its own it is circular:
    `extract_fp8_gemm_info` looks names up in `layout_module_specs` too, and a
    WRONG name in `EXPORT_LAYOUTS` makes both sides agree on it -- the fake would
    file the module under the wrong key, the reporter would find it there, and
    production (which looks the name up in the loader's real component dict)
    would find nothing. Measured: setting
    ``EXPORT_LAYOUTS["acestep"]["modules"] = (("transformer", ""),)`` left this
    file green. `LoaderComponentNameAnchorTest` below is the anchor that closes
    it: it checks the same name against the arch's LOADER, which shares no source
    with either side.
    """
    try:
        from core.models.common.quantized_export import layout_module_specs

        return layout_module_specs(arch)[0][0]
    except Exception:
        return "transformer"


# arch -> the loader entry point(s) that BUILD its component dict, as
# (module, dotted attribute). Resolved by import, so a renamed or moved loader
# fails here loudly instead of dropping the arch out of the anchor below.
# Checked for completeness against QUANTIZED_LINEAR_ARCHS.
#
# Why a written-down registry rather than a convention scan: a scan wide enough
# to find these six (they live in four different module shapes, one of them a
# @staticmethod on `ModelLoader`) also sweeps up helpers like
# `detect_anima_split_layout`, whose own `return {"dit": ...}` is a PATH dict,
# not a component dict. Folding those in would only ADD accepted key names,
# i.e. weaken exactly the assertion this table exists to make.
_LOADER_ENTRY_POINTS = {
    "acestep": (("core.models.acestep.loader", "load_acestep_from_path"),),
    "anima": (("core.models.anima.anima_loader", "load_anima_components"),),
    "flux2": (("core.model_loader", "ModelLoader.load_flux2_from_safetensors"),),
    "ideogram4": (
        ("core.models.ideogram4.ideogram4_loader", "load_ideogram4_components"),
        ("core.models.ideogram4.ideogram4_loader", "load_ideogram4_single_file"),
    ),
    "krea2": (("core.models.krea2.krea2_loader", "load_krea2_components"),),
    "ltx2": (("core.models.ltx2.loader", "load_ltx2_from_diffusers"),),
}


def _resolve_attr(module_name, dotted):
    obj = importlib.import_module(module_name)
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _return_dict_key_sets(fn):
    """The string keys of every ``return {...}`` literal in `fn`'s OWN body.

    Nested functions are excluded (a closure's dict is not what the loader
    hands back), and a `return` of anything other than a dict DISPLAY yields no
    entry -- so a loader that builds its dict incrementally produces an empty
    list here and is reported as un-anchorable rather than silently passing.
    """
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    nested = {id(n)
              for f in ast.walk(func)
              if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f is not func
              for n in ast.walk(f)}
    key_sets = []
    for node in ast.walk(func):
        if id(node) in nested or not isinstance(node, ast.Return):
            continue
        if isinstance(node.value, ast.Dict):
            key_sets.append({k.value for k in node.value.keys
                             if isinstance(k, ast.Constant) and isinstance(k.value, str)})
    return key_sets


class LoaderComponentNameAnchorTest(unittest.TestCase):
    """The component NAME, anchored against the loader that produces it.

    `extract_fp8_gemm_info` resolves an arch's quantized module as
    ``<arch>_components[name]`` with `name` from
    ``EXPORT_LAYOUTS[arch]["modules"]``, and the functional reporter test above
    files its fake module under a name read from THE SAME table. That pair is
    self-consistent for any name, right or wrong: with
    ``EXPORT_LAYOUTS["acestep"]["modules"] = (("transformer", ""),)`` -- the
    map-drift class this suite exists for -- the whole file stayed green while
    production would look up ``acestep_components["transformer"]``, find nothing,
    and record no `fp8_gemm` on every ACE-Step generation, silently and
    permanently.

    The third source is the LOADER: the dict literal it returns IS the component
    dict `pipeline.py` stores as ``<arch>_components``, and it shares no source
    with the layout table or with the reporter. Every arch here returns one that
    can be read statically; if a future loader builds its dict incrementally, the
    first test below fails with that arch named rather than skipping it.
    """

    def _key_sets(self, arch):
        out = []
        for module_name, dotted in _LOADER_ENTRY_POINTS[arch]:
            fn = _resolve_attr(module_name, dotted)
            for keys in _return_dict_key_sets(fn):
                out.append((f"{module_name}.{dotted}", keys))
        return out

    def test_every_quantized_arch_has_a_readable_loader_entry_point(self):
        missing = sorted(set(QUANTIZED_LINEAR_ARCHS) - set(_LOADER_ENTRY_POINTS))
        self.assertEqual(
            missing, [],
            "these archs own weight-only quantized Linear layers but have no loader "
            "entry point registered in _LOADER_ENTRY_POINTS, so nothing checks that "
            "the component name their export layout declares is a key their loader "
            "really returns")
        unreadable = []
        for arch in QUANTIZED_LINEAR_ARCHS:
            key_sets = self._key_sets(arch)
            if not key_sets or any(not keys for _where, keys in key_sets):
                unreadable.append(arch)
        self.assertEqual(
            unreadable, [],
            "these archs' loader entry points return no statically readable dict "
            "literal, so the anchor below would pass vacuously for them. Point the "
            "registry at the function that DOES return the component dict, or "
            "anchor the arch some other way -- do not skip it.")

    def test_the_layout_component_names_are_keys_the_loader_really_returns(self):
        from core.models.common.quantized_export import layout_module_specs

        for arch in QUANTIZED_LINEAR_ARCHS:
            for component, _prefix in layout_module_specs(arch):
                for where, keys in self._key_sets(arch):
                    with self.subTest(arch=arch, component=component, loader=where):
                        self.assertIn(
                            component, keys,
                            f"EXPORT_LAYOUTS[{arch!r}] declares the component "
                            f"{component!r}, but {where} returns no such key "
                            f"(it returns {sorted(keys)}). Everything that has to FIND "
                            f"this arch's quantized module -- the export job and "
                            f"generation_utils.extract_fp8_gemm_info -- looks the name "
                            f"up in the live <arch>_components dict, which is this "
                            f"loader's return value, so a name only the layout table "
                            f"believes in resolves to None: no fp8_gemm is recorded on "
                            f"any generation and the export refuses the model.")

    def test_the_anchor_discriminates(self):
        """The premise: a name from ANOTHER arch's layout is really rejected here.

        Without this, "the loader returns every name anyone might ask for" would
        satisfy the test above while proving nothing.
        """
        from core.models.common.quantized_export import layout_module_specs

        every = {name for a in QUANTIZED_LINEAR_ARCHS
                 for name, _p in layout_module_specs(a)}
        for arch in QUANTIZED_LINEAR_ARCHS:
            own = {name for name, _p in layout_module_specs(arch)}
            foreign = every - own
            if not foreign:
                continue
            for where, keys in self._key_sets(arch):
                with self.subTest(arch=arch, loader=where):
                    self.assertTrue(
                        foreign - keys,
                        f"{where} returns every component name any arch's layout "
                        f"declares ({sorted(foreign)}), so the check above cannot "
                        f"fail for a wrong name")


class LoraQuantizedTargetParityTest(unittest.TestCase):
    """Defect class 1: the `isinstance(x, nn.Linear)` trap.

    For every arch that can own quantized Linear layers, the predicate that
    decides "is this a LoRA target" must answer the SAME for an `Int8Linear` /
    `Fp8Linear` as for the `nn.Linear` it replaced. The predicate is located by
    convention (a module-level `_is_lora_target` / `_is_target` /
    `is_lora_wrappable_linear` in one of the arch's LoRA-owning modules); an
    arch that exposes none fails, which is the point -- an inline lambda cannot
    be checked, and Ideogram 4's used to be one.
    """

    def _predicates_for(self, arch):
        """EVERY conventional predicate the arch exposes, not just the first.

        This used to return the first match in candidate order, and an arch with
        two of them had only that one exercised: on ACE-Step
        ``acestep_adapter._is_target`` won, so ``pipeline_backends.acestep``'s
        own ``_is_lora_target`` -- the one the GENERATION path calls, and the
        site the quantized-Linear trap would actually be spelled at -- was
        covered by the AST scan alone. Both are checked now, and an arch that
        grows a third gets it for free.
        """
        found = []
        for module in _lora_modules(arch):
            for name in _PREDICATE_NAMES:
                fn = getattr(module, name, None)
                if callable(fn):
                    found.append((f"{module.__name__}.{name}", fn))
        return found

    def test_every_quantized_arch_exposes_a_findable_target_predicate(self):
        missing = [a for a in QUANTIZED_LINEAR_ARCHS if not self._predicates_for(a)]
        self.assertEqual(
            missing, [],
            f"these architectures own weight-only quantized Linear layers but expose "
            f"no module-level LoRA target predicate named one of {_PREDICATE_NAMES} "
            f"in core.models.<arch>.<arch>_lora / core.training.adapters.<arch>_adapter "
            f"/ core.pipeline_backends.<arch>. Without one the isinstance(x, nn.Linear) "
            f"trap cannot be tested for that arch (an inline lambda is invisible here).")

    def test_the_predicate_selects_quantized_linears_exactly_like_plain_ones(self):
        modules = _quantized_linears()
        for arch in QUANTIZED_LINEAR_ARCHS:
            for where, predicate in self._predicates_for(arch):
                with self.subTest(arch=arch, predicate=where):
                    plain = bool(predicate(modules["nn.Linear"]))
                    self.assertTrue(plain, f"{where} does not accept a plain nn.Linear")
                    for name in ("Int8Linear", "Fp8Linear"):
                        self.assertEqual(
                            bool(predicate(modules[name])), plain,
                            f"{arch}: {where} answers differently for {name} than for "
                            f"nn.Linear. {name} is an nn.Module but NOT an nn.Linear "
                            f"subclass, so an isinstance(x, nn.Linear) predicate drops every "
                            f"quantized layer silently -- the run 'succeeds' with a smaller "
                            f"target count that looks like a narrower scope.")

    def test_the_trap_is_real(self):
        """Pins the premise: the naive predicate really does reject both classes.

        If a future torch/vendor change made `Int8Linear` an `nn.Linear`
        subclass, the test above would pass for a reason that has nothing to do
        with what it checks, so the premise is asserted rather than assumed.
        """
        modules = _quantized_linears()
        for name in ("Int8Linear", "Fp8Linear"):
            self.assertFalse(isinstance(modules[name], nn.Linear), name)


class LoraAdapterDtypeTest(unittest.TestCase):
    """Defect class 1b: the adapter dtype must never come from the base weight.

    A LoRA branch built at `base.weight.dtype` over an int8 base is quantized to
    8 uniform levels; over an e4m3 base it loses most of its precision. Tested
    end to end because `LoRALinearLayer` can be constructed directly over a
    quantized base -- no architecture needed.
    """

    def test_lora_branch_dtype_ignores_a_quantized_base(self):
        from core.training.adapters.base_adapter import lora_branch_dtype

        modules = _quantized_linears()
        self.assertIs(lora_branch_dtype(modules["nn.Linear"]), torch.bfloat16)
        for name in ("Int8Linear", "Fp8Linear"):
            dtype = lora_branch_dtype(modules[name])
            self.assertTrue(dtype.is_floating_point, name)
            self.assertNotIn(dtype, (torch.int8, torch.float8_e4m3fn), name)

    def test_an_adapter_over_a_quantized_base_holds_no_quantized_parameter(self):
        from core.training.adapters.sd15_adapter import LoRALinearLayer

        modules = _quantized_linears(in_features=16, out_features=16)
        for name, base in modules.items():
            layer = LoRALinearLayer(base, rank=4, alpha=4, lora_name=f"t_{name}")
            for pname, param in layer.named_parameters():
                if "original_module" in pname:
                    continue  # the frozen base itself, which IS quantized
                self.assertTrue(
                    param.dtype.is_floating_point
                    and param.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2),
                    f"LoRA branch {pname} over a {name} base has dtype "
                    f"{param.dtype}: the adapter took its dtype from the base "
                    f"weight, so it is stored at the base's quantized precision.")


class LoraIsinstanceTrapScanTest(unittest.TestCase):
    """Static half of defect class 1: no bare `isinstance(_, nn.Linear)`.

    Functional predicate coverage above only reaches the ONE predicate an arch
    exposes; a wrapping loop can still test the type inline (which is how FLUX.2
    accumulated 8 sites). This scans every LoRA-owning module of every
    quantized-capable arch for an `isinstance` whose ONLY class is `nn.Linear`,
    inside a function that deals with LoRA at all.

    `isinstance(m, (nn.Linear, Int8Linear, ...))` is fine and is not flagged --
    the trap is nn.Linear ALONE.
    """

    @staticmethod
    def _is_nn_linear(node):
        if isinstance(node, ast.Attribute) and node.attr == "Linear":
            value = node.value
            if isinstance(value, ast.Name) and value.id in ("nn", "torch"):
                return True
            if isinstance(value, ast.Attribute) and value.attr == "nn":
                return True
        # `from torch.nn import Linear` spells the same class as a bare Name.
        # Nothing else in this codebase is called `Linear` unqualified (the
        # quantized ones are Int8Linear/Fp8Linear/LoRALinearLayer), so this
        # cannot collide with a class that WOULD be a correct thing to test for.
        if isinstance(node, ast.Name) and node.id == "Linear":
            return True
        return False

    def _type_comparisons(self, func):
        """`type(x) is nn.Linear` -- the same trap without the isinstance."""
        out = []
        for cmp_node in [n for n in ast.walk(func) if isinstance(n, ast.Compare)]:
            left = cmp_node.left
            if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                    and left.func.id == "type"):
                continue
            for op, comparator in zip(cmp_node.ops, cmp_node.comparators):
                if isinstance(op, (ast.Is, ast.Eq)) and self._is_nn_linear(comparator):
                    out.append(cmp_node.lineno)
        return out

    def _offenders(self, module):
        path = Path(inspect.getsourcefile(module))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        out = []
        for func in [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            source = ast.get_source_segment(path.read_text(encoding="utf-8"), func) or ""
            if "lora" not in source.lower():
                continue
            for lineno in self._type_comparisons(func):
                out.append(f"{path.name}:{lineno} in {func.name}() (type(x) is nn.Linear)")
            for call in [n for n in ast.walk(func) if isinstance(n, ast.Call)]:
                if not (isinstance(call.func, ast.Name) and call.func.id == "isinstance"):
                    continue
                if len(call.args) != 2:
                    continue
                classes = call.args[1]
                if isinstance(classes, (ast.Tuple, ast.List)):
                    continue  # a tuple can name the quantized classes; not the trap
                if self._is_nn_linear(classes):
                    out.append(f"{path.name}:{call.lineno} in {func.name}()")
        return out

    def test_every_lora_owning_module_was_actually_scanned(self):
        """A module that fails to IMPORT drops out of the scan silently.

        `_lora_modules` has to tolerate a candidate that does not exist (the
        candidate list is a naming convention, and no arch has all four), and
        that tolerance used to swallow every exception -- so a module that DOES
        exist but raises on import (a missing optional dependency is the
        realistic one) would take all of its isinstance sites out of the scan
        while the suite stayed green.
        """
        for arch in QUANTIZED_LINEAR_ARCHS:
            _lora_modules(arch)
        self.assertEqual(
            _LORA_MODULE_IMPORT_ERRORS, {},
            "these LoRA-owning modules exist but could not be imported, so the "
            "isinstance scan below never saw them")

    def test_no_lora_site_of_a_quantized_arch_tests_nn_linear_alone(self):
        offenders = {}
        for arch in QUANTIZED_LINEAR_ARCHS:
            for module in _lora_modules(arch):
                found = self._offenders(module)
                if found:
                    offenders.setdefault(arch, []).extend(found)
        self.assertEqual(
            offenders, {},
            "isinstance(x, nn.Linear) alone, in LoRA code owned by an architecture "
            "whose checkpoints can hold Int8Linear/Fp8Linear layers. Those classes "
            "are nn.Modules but not nn.Linear subclasses, so the test skips them "
            "silently. Use is_lora_wrappable_linear (or name the classes in the "
            "isinstance tuple).")


# Every /generate route, classified by whether it can serve an architecture that
# owns weight-only quantized Linear layers. PINNED ABSOLUTELY, and checked for
# completeness against the live router.
#
# Why absolutely and not as a relation. The first form of this test was "if a
# route accepts unet_quantization it must accept quantized_gemm_mode", which a
# route accepting NEITHER satisfies: deleting both parameters from
# /generate/txt2vid left the whole suite green while openapi.yaml went on
# advertising them -- the same defect class (advertised but unwired) the test
# exists for, one step further along.
#
# Why not derived. Nothing in the backend maps a route to the architectures it
# can serve: the routes dispatch on whatever model is LOADED, and
# `arch_capabilities` is keyed by arch, not by path. The one authoritative
# cross-check that does exist is openapi.yaml, and it is used as one (see
# `test_the_spec_and_the_routes_agree_on_the_quantized_parameters`) -- but a
# derivation from the spec alone would go vacuous again the moment a change
# dropped the parameter from BOTH sides, which is precisely the case that has to
# fail. So the set is written down here. The cost is real and accepted: adding a
# /generate route means adding its path to one of the two sets below, and the
# completeness test says so by name rather than passing silently.
_QUANTIZED_GEMM_ROUTES = frozenset({
    "/generate/txt2img",
    "/generate/img2img",
    "/generate/inpaint",
    "/generate/outpaint",
    "/generate/txt2img/training-preview",
    "/generate/img2img/training-preview",
    "/generate/inpaint/training-preview",
    "/generate/txt2vid",
    "/generate/img2vid",
    "/generate/outpaint/video",
    # ACE-Step audio. These three were in the unquantized set below while
    # `acestep` was in neither tuple. It is now in BOTH: its loader swaps in
    # Int8Linear/Fp8Linear for a weight-only quantized DiT and
    # `_acestep_runtime_int8` produces the same classes from a bf16 one, so both
    # parameters govern modules these routes really run -- the same test LTX-2.3
    # passed. Moving them here (rather than leaving them classified as
    # unquantized while the capability table advertised `quantized_gemm` for
    # acestep) is defect class 2 avoided rather than repeated.
    "/generate/txt2aud",
    "/generate/aud2aud",
    "/generate/outpaint/audio",
})

_UNQUANTIZED_GENERATE_ROUTES = frozenset({
    # Upscale. PIL/spandrel backends run no diffusion model at all, and the
    # diffusion backend runs the loaded model in whatever state it is already
    # in; the route exposes no quantization control of its own.
    "/generate/upscale",
})


class QuantizedCapabilityCoherenceTest(unittest.TestCase):
    """Defect class 2: a capability must have something behind it.

    `QUANTIZED_LINEAR_ARCHS` grants `quantized_gemm`, the frontend renders
    `QuantizedGemmSelect` from that capability, and the request has to survive
    the trip. This checks the three places the trip has actually broken.
    """

    @staticmethod
    def _generate_routes():
        from api.routes import router

        return [r for r in router.routes
                if getattr(r, "path", "").startswith("/generate/")]

    @staticmethod
    def _accepted_params(route):
        """Every request field a route accepts: signature params + body model fields."""
        from pydantic import BaseModel

        names = set()
        for name, param in inspect.signature(route.endpoint).parameters.items():
            names.add(name)
            annotation = param.annotation
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                names.update(annotation.model_fields.keys())
        return names

    def test_every_route_that_takes_unet_quantization_also_takes_quantized_gemm_mode(self):
        """The two are one axis; a route that offers one alone is the F2 shape.

        `unet_quantization` quantizes an unquantized model's weights and
        `quantized_gemm_mode` selects how already-quantized weights are
        multiplied -- an architecture reachable by a route always has both or
        neither. LTX-2.3's three video routes accepted the first and not the
        second while the UI rendered the control for it and wrote the value.
        """
        missing = []
        for route in self._generate_routes():
            params = self._accepted_params(route)
            if "unet_quantization" in params and "quantized_gemm_mode" not in params:
                missing.append(route.path)
        self.assertEqual(
            missing, [],
            "these /generate routes accept unet_quantization but not "
            "quantized_gemm_mode. The frontend renders the quantized-GEMM control "
            "from the arch capability, not from the route, so the control appears "
            "and its value is silently dropped.")

    def test_every_generate_route_is_classified(self):
        """A new /generate route must be put in one of the two sets, deliberately.

        This is what a written-down set buys and a relation does not: an
        unclassified route fails here instead of quietly satisfying an
        implication.
        """
        live = {r.path for r in self._generate_routes()}
        known = _QUANTIZED_GEMM_ROUTES | _UNQUANTIZED_GENERATE_ROUTES
        self.assertEqual(
            sorted(live - known), [],
            "these /generate routes are new since this test was written. Add each "
            "to _QUANTIZED_GEMM_ROUTES (it can serve an arch with weight-only "
            "quantized Linear layers, so it must accept unet_quantization AND "
            "quantized_gemm_mode) or to _UNQUANTIZED_GENERATE_ROUTES.")
        self.assertEqual(
            sorted(known - live), [],
            "these paths are pinned here but no longer exist on the router")

    def test_the_pinned_routes_accept_both_quantized_parameters(self):
        """Absolute, in both directions: exactly these routes, exactly both params.

        The relation above is satisfied by a route that dropped BOTH parameters
        (verified: removing both from /generate/txt2vid left all tests green).
        Naming the set makes that a failure.
        """
        for param in ("unet_quantization", "quantized_gemm_mode"):
            actual = {r.path for r in self._generate_routes()
                      if param in self._accepted_params(r)}
            self.assertEqual(
                actual, set(_QUANTIZED_GEMM_ROUTES),
                f"the set of /generate routes accepting {param} is not the pinned "
                f"set. Missing: {sorted(_QUANTIZED_GEMM_ROUTES - actual)}; "
                f"unexpected: {sorted(actual - _QUANTIZED_GEMM_ROUTES)}. A route "
                f"that serves a quantized-capable arch and does not accept this "
                f"parameter drops the value the UI sent; one that accepts it "
                f"without serving such an arch is advertising something it cannot "
                f"honor.")

    def test_the_gemm_reporter_finds_every_quantized_arch(self):
        """FUNCTIONAL coverage of `extract_fp8_gemm_info`, per arch.

        The static test below pins that `<arch>_components` EXISTS on the
        pipeline manager -- it does not pin that the reporter derives that name,
        and the reporter used to hand-write the map. Reverting it to its
        shipped-broken three-entry form left all 20 tests of this suite green,
        because nothing anywhere in backend/tests called the function (11
        non-test call sites, 0 test call sites). This calls it, for every arch in
        the tuple, over one real `Int8Linear`: a map that omits an arch reports
        "" for it, which is how a checkpoint full of quantized layers came to
        record no `fp8_gemm` at all.
        """
        from api.generation_utils import extract_fp8_gemm_info

        for arch in QUANTIZED_LINEAR_ARCHS:
            with self.subTest(arch=arch):
                manager = _FakeManager(current_model_info={"type": arch})
                setattr(manager, f"{arch}_components", {
                    _witness_component(arch):
                        nn.Sequential(_quantized_linears()["Int8Linear"]),
                })
                self.assertNotEqual(
                    extract_fp8_gemm_info(manager), "",
                    f"extract_fp8_gemm_info reported nothing for a {arch} model whose "
                    f"transformer holds an Int8Linear. Its arch -> component-dict map "
                    f"does not cover {arch}, so every {arch} generation records no "
                    f"fp8_gemm and every quantized_gemm_mode request there is reported "
                    f"as 'the checkpoint carries no quantized Linear layers'.")

    def test_the_gemm_reporter_is_silent_for_an_unquantized_checkpoint(self):
        """The premise of the test above: a bf16 model must still report nothing."""
        from api.generation_utils import extract_fp8_gemm_info

        for arch in QUANTIZED_LINEAR_ARCHS:
            manager = _FakeManager(current_model_info={"type": arch})
            setattr(manager, f"{arch}_components",
                    {_witness_component(arch): _tiny_transformer()})
            self.assertEqual(extract_fp8_gemm_info(manager), "", arch)

    def test_every_quantized_arch_has_a_component_dict_the_gemm_reporter_can_find(self):
        """`extract_fp8_gemm_info` derives `<arch>_components`; pin the convention.

        It used to hand-write the map, and the map drifted: FLUX.2 joined
        `QUANTIZED_LINEAR_ARCHS` while the map kept three entries, so FLUX.2
        generations recorded no `fp8_gemm` and every `w8a8` request there was
        reported as "the checkpoint carries no quantized Linear layers".
        """
        from core.pipeline import DiffusionPipelineManager

        source = Path(inspect.getsourcefile(DiffusionPipelineManager)).read_text(encoding="utf-8")
        missing = [a for a in QUANTIZED_LINEAR_ARCHS if f"{a}_components" not in source]
        self.assertEqual(
            missing, [],
            "extract_fp8_gemm_info resolves an arch's components as "
            "'<arch>_components' on the pipeline manager; these archs have no such "
            "attribute, so their generations would record no fp8_gemm at all.")

    def test_every_runtime_int8_arch_has_a_runtime_hook(self):
        """`unet_quantization: "int8"` is exempted per arch; the hook must exist."""
        for arch in RUNTIME_INT8_ARCHS:
            module = importlib.import_module(f"core.pipeline_backends.{arch}")
            source = Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")
            self.assertIn(
                f"def _{arch}_runtime_int8", source,
                f"{arch} is in RUNTIME_INT8_ARCHS (so the API accepts and the UI "
                f"offers unet_quantization='int8' for it) but "
                f"core/pipeline_backends/{arch}.py defines no _{arch}_runtime_int8 "
                f"hook, so the value would be accepted and ignored.")

    def test_the_capability_table_grants_quantized_gemm_to_exactly_these_archs(self):
        from api.arch_capabilities import ARCH_UNSUPPORTED

        for arch in QUANTIZED_LINEAR_ARCHS:
            self.assertNotIn(
                "quantized_gemm", ARCH_UNSUPPORTED.get(arch, {}),
                f"{arch} owns quantized Linear layers but is listed as not "
                f"supporting quantized_gemm")


def _component_archs():
    """Every arch PipelineManager keeps a `<arch>_components` dict for.

    Read out of `pipeline.py`'s `__init__` declarations rather than listed, for
    the same reason the functions under test derive their maps: a list here
    would drift exactly as the lists there did. SD1.5/SDXL declare none (their
    identity lives on the loaded pipeline), which is the branch this separates.
    """
    from core.pipeline import DiffusionPipelineManager

    source = Path(inspect.getsourcefile(DiffusionPipelineManager)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    init = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    archs = []
    for node in ast.walk(init):
        target = (node.target if isinstance(node, ast.AnnAssign)
                  else node.targets[0] if isinstance(node, ast.Assign) and node.targets
                  else None)
        if (isinstance(target, ast.Attribute) and target.attr.endswith("_components")
                and isinstance(target.value, ast.Name) and target.value.id == "self"):
            archs.append(target.attr[: -len("_components")])
    return sorted(set(archs))


class HandWrittenArchMapTest(unittest.TestCase):
    """Defect class 2, third occurrence: a hand-written arch -> attribute map.

    `extract_fp8_gemm_info` had one and it drifted (FLUX.2 recorded no
    `fp8_gemm`). `extract_vae_info`, 90 lines above it in the same file, had the
    SAME map with the same shape and was ALREADY stale: `ltx2` and `acestep` were
    missing, so every LTX-2.3 video fell through to the SD1.5/SDXL branch, found
    no `_sushi_vae_source`, and recorded no `vae_name`/`vae_hash` at all --
    although `ltx2_components["vae"]` exists and would have produced an identity.

    Both are now derived from the `<arch>_components` convention, and both are
    covered FUNCTIONALLY here (`extract_fp8_gemm_info` in
    `QuantizedCapabilityCoherenceTest`), over every arch the convention covers,
    so a fourth occurrence of the map cannot pass.
    """

    def test_the_component_arch_scan_finds_something(self):
        archs = _component_archs()
        self.assertGreaterEqual(len(archs), 5, archs)
        for arch in QUANTIZED_LINEAR_ARCHS:
            self.assertIn(arch, archs, arch)

    def test_the_vae_reporter_reads_every_arch_component_dict(self):
        from api.generation_utils import extract_vae_info

        for arch in _component_archs():
            with self.subTest(arch=arch):
                manager = _FakeManager(
                    current_model_info={"type": arch},
                    # A stale pipeline from an earlier SD1.5 load, which is what
                    # the fall-through branch would read if the arch were not
                    # recognised. It must NOT win.
                    txt2img_pipeline=_FakeManager(_sushi_vae_source="stale-sd15-vae"),
                )
                setattr(manager, f"{arch}_components", {"vae_source": f"vae-of-{arch}"})
                name, _hash = extract_vae_info(manager)
                self.assertEqual(
                    name, f"vae-of-{arch}",
                    f"extract_vae_info did not read {arch}_components. An arch missing "
                    f"from its map falls into the SD1.5/SDXL branch, which reads a "
                    f"pipeline this arch does not use -- so the generation records no "
                    f"VAE identity (or, worse, a stale one).")

    def test_the_pipeline_branch_still_serves_sd15_and_sdxl(self):
        """The premise: archs with no components dict must keep the old path."""
        from api.generation_utils import extract_vae_info

        for arch in ("sd15", "sdxl"):
            manager = _FakeManager(
                current_model_info={"type": arch},
                txt2img_pipeline=_FakeManager(_sushi_vae_source="sd-vae"))
            self.assertEqual(extract_vae_info(manager)[0], "sd-vae", arch)


class OpenapiArchListParityTest(unittest.TestCase):
    """Defect class 3 of the spec kind: a hand-written arch list in openapi.yaml.

    Four separate occurrences so far (`quantized_gemm_mode`'s description, the
    `runtime_int8_archs` example, the `quantized_linear_archs` example, the
    `sdxl.quantized_gemm` reason prose). Where the backend SERVES the value, the
    spec's example must equal what it serves; where the spec only describes a
    set, it must name exactly that set's ids.
    """

    @classmethod
    def setUpClass(cls):
        import yaml

        with open(_REPO / "openapi.yaml", encoding="utf-8") as fh:
            cls.spec = yaml.safe_load(fh)

    @staticmethod
    def _served_capabilities():
        import asyncio

        from api.routes import get_arch_capabilities

        return asyncio.run(get_arch_capabilities())

    def _capabilities_example(self):
        return (self.spec["paths"]["/schema/arch-capabilities"]["get"]["responses"]
                ["200"]["content"]["application/json"]["example"])

    def test_the_example_arch_lists_equal_what_the_endpoint_serves(self):
        served = self._served_capabilities()
        example = self._capabilities_example()
        for key in ("runtime_int8_archs", "quantized_linear_archs"):
            self.assertEqual(
                list(example[key]), list(served[key]),
                f"openapi's {key} example does not match GET "
                f"/schema/arch-capabilities")

    def test_the_example_unsupported_reasons_are_the_served_strings(self):
        from api.arch_capabilities import ARCH_UNSUPPORTED

        example = self._capabilities_example()
        for arch, features in (example.get("unsupported") or {}).items():
            for feature, reason in features.items():
                self.assertEqual(
                    reason, ARCH_UNSUPPORTED.get(arch, {}).get(feature),
                    f"openapi's example reason for {arch}.{feature} is not the "
                    f"string the backend serves (these are generated from the arch "
                    f"tuples, so a hand-edited copy drifts every time an arch joins)")

    def test_the_schema_field_examples_equal_the_served_lists(self):
        served = self._served_capabilities()
        props = self.spec["components"]["schemas"]["ArchCapabilities"]["properties"]
        for key in ("runtime_int8_archs", "quantized_linear_archs"):
            self.assertEqual(
                list(props[key]["example"]), list(served[key]),
                f"ArchCapabilities.{key}'s example does not match what the endpoint "
                f"serves")

    def test_the_unet_quantization_description_names_exactly_the_int8_archs(self):
        from core.models.common.int8_runtime_quantize import ARCH_DISPLAY_NAMES

        description = (self.spec["components"]["schemas"]["GenerationParams"]
                       ["properties"]["unet_quantization"]["description"])
        # Only the paragraph that enumerates the int8 archs; the FP8 paragraph
        # above it legitimately names other architectures.
        int8_paragraph = description.split("`int8`", 1)[1].split("\n\n", 1)[0]
        for arch in RUNTIME_INT8_ARCHS:
            self.assertIn(f"`{arch}`", int8_paragraph, arch)
        for arch in ARCH_DISPLAY_NAMES:
            if arch not in RUNTIME_INT8_ARCHS:
                self.assertNotIn(f"`{arch}`", int8_paragraph, arch)

    def test_the_export_job_arch_description_names_exactly_the_export_layouts(self):
        from core.models.common.quantized_export import EXPORT_LAYOUTS

        description = (self.spec["components"]["schemas"]["QuantizedExportJob"]
                       ["properties"]["arch"]["description"])
        for arch in EXPORT_LAYOUTS:
            self.assertIn(f"`{arch}`", description, arch)

    def test_the_video_routes_declare_quantized_gemm_mode(self):
        """The spec half of the F2 fix: the parameter the routes now accept."""
        for schema in ("Txt2VidRequest", "OutpaintVideoRequest"):
            props = self.spec["components"]["schemas"][schema]["properties"]
            self.assertIn("quantized_gemm_mode", props, schema)

    def _request_properties(self, path_item):
        """Every request property a /generate path declares, refs resolved."""
        names = set()
        post = path_item.get("post") or {}
        for body in ((post.get("requestBody") or {}).get("content") or {}).values():
            stack = [body.get("schema") or {}]
            seen = set()
            while stack:
                schema = stack.pop()
                ref = schema.get("$ref")
                if ref:
                    if ref in seen:
                        continue
                    seen.add(ref)
                    schema = self.spec["components"]["schemas"][ref.split("/")[-1]]
                for key in ("allOf", "anyOf", "oneOf"):
                    stack.extend(schema.get(key) or [])
                names.update((schema.get("properties") or {}).keys())
        return names

    def test_the_spec_and_the_routes_agree_on_the_quantized_parameters(self):
        """Every /generate path: what the spec declares == what the route accepts.

        The two halves used to be checked separately -- the routes against each
        other (a relation, which a route accepting neither satisfies) and the
        spec against two hand-named schemas -- so a route could stop accepting a
        parameter the spec kept advertising, which is defect class 2 verbatim.
        Checked in BOTH directions and over every path, so neither side can drift
        alone.
        """
        from api.routes import router

        by_path = {r.path: r for r in router.routes
                   if getattr(r, "path", "").startswith("/generate/")}
        for path, path_item in self.spec["paths"].items():
            if not path.startswith("/generate/"):
                continue
            route = by_path.get(path)
            self.assertIsNotNone(route, f"openapi declares {path}; the router does not")
            declared = self._request_properties(path_item)
            accepted = QuantizedCapabilityCoherenceTest._accepted_params(route)
            for param in ("unet_quantization", "quantized_gemm_mode"):
                self.assertEqual(
                    param in declared, param in accepted,
                    f"{path}: openapi declares {param}={param in declared} but the "
                    f"route accepts {param}={param in accepted}. The frontend is "
                    f"written against the spec, so the disagreeing side is the one "
                    f"that silently drops the value.")


# ---------------------------------------------------------------------------
# Defect class 3: runtime-hook guard ordering.
# ---------------------------------------------------------------------------
class _FakeManager:
    """Only the attributes a runtime-int8 hook reads."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


def _tiny_transformer():
    """A bf16 module with one Linear -- enough for any hook's walk."""
    return nn.Sequential(nn.Linear(8, 8, dtype=torch.bfloat16))


class _FakeOffloader:
    """A block offloader stub: only `cleanup()`, which the unwrap path calls."""

    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


def _ltx2_manager():
    """An LTX-2.3 fake carrying a STALE, LIVE block offloader.

    On LTX-2.3 the offloader is persistent state on the wrapper: it survives the
    generation that created it and is torn down only by the NEXT generation's
    `_ensure_ltx2_block_swap_wrapper` call, which runs strictly after the
    runtime-int8 hook. That is what made the pre-fix guard fire on the second
    block-swap generation of a session.
    """
    from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
    from core.pipeline_backends.ltx2 import LTX2Mixin
    from diffusers import LTX2VideoTransformer3DModel

    inner = LTX2VideoTransformer3DModel.from_config(_LTX2_CONFIG).to(torch.bfloat16)
    offloader = _FakeOffloader()
    wrapper = Ltx2BlockLoopWrapper(inner, block_offloader=offloader)
    # The pipeline reference the unwrap path needs, plus the shared-object rule
    # it maintains (pipeline.transformer IS components["transformer"]).
    pipeline = _FakeManager(transformer=wrapper)
    manager = _FakeManager(
        ltx2_components={"transformer": wrapper, "pipeline": pipeline},
        _ltx2_block_swap_count=22)
    manager.__class__ = type("FakeLtx2", (_FakeManager, LTX2Mixin), {})
    return manager, offloader


def _ltx2_case(quantization, blocks_to_swap=22):
    manager, _offloader = _ltx2_manager()
    manager._ltx2_runtime_int8({"unet_quantization": quantization,
                                "blocks_to_swap": blocks_to_swap})


def _flux2_case(quantization):
    from core.pipeline_backends.flux2 import Flux2Mixin

    transformer = _tiny_transformer()
    manager = _FakeManager(flux2_components={"transformer": transformer},
                           _flux2_active_block_offloader=object())
    manager.__class__ = type("FakeFlux2", (_FakeManager, Flux2Mixin), {})
    manager._flux2_runtime_int8({"unet_quantization": quantization}, transformer)


def _ideogram4_case(quantization):
    from core.pipeline_backends.ideogram4 import Ideogram4Mixin

    transformer = _tiny_transformer()
    transformer._block_offloader = object()
    manager = _FakeManager(ideogram4_components={"transformer": transformer})
    manager.__class__ = type("FakeIdeogram4", (_FakeManager, Ideogram4Mixin), {})
    manager._ideogram4_runtime_int8({"unet_quantization": quantization})


def _krea2_case(quantization):
    from core.pipeline_backends.krea2 import Krea2Mixin

    transformer = _tiny_transformer()
    transformer._block_offloader = object()
    manager = _FakeManager(krea2_components={"transformer": transformer})
    manager.__class__ = type("FakeKrea2", (_FakeManager, Krea2Mixin), {})
    manager._krea2_runtime_int8({"unet_quantization": quantization})


def _anima_case(quantization):
    from core.pipeline_backends.anima import AnimaMixin

    transformer = _tiny_transformer()
    transformer._block_offloader = object()
    manager = _FakeManager(anima_components={"transformer": transformer})
    manager.__class__ = type("FakeAnima", (_FakeManager, AnimaMixin), {})
    manager._anima_runtime_int8(quantization)


# arch -> a callable that runs the arch's runtime-int8 hook with a STALE block
# offloader present. Checked for completeness against RUNTIME_INT8_ARCHS below,
# so a new arch cannot join the tuple and skip this class silently.
def _acestep_case(quantization):
    """ACE-Step's stale state is a LoRA WRAPPER, not a block offloader.

    There is no block-swap path in this backend at all, so the invariant the hook
    has to survive is the other kind of leftover: `_acestep_lora_original_modules`
    and a wrapped Linear left behind by an earlier generation. A request that
    converts nothing must not care -- and `_acestep_runtime_int8` is called AFTER
    `_apply_or_clear_lora_acestep`, so by the time it runs the wrappers present
    are only the ones this request asked for.

    SCOPE, because this shape used to read as if it established more than it
    does: this case is only ever run with a quantization value that converts
    NOTHING (see `_GUARD_CASES`' caller). It says the guard is quiet, not that
    the hook does anything -- replacing `_acestep_runtime_int8` with a no-op
    satisfied it, and satisfied every other test in this file. The positive
    control that fails for a no-op hook is
    `RuntimeInt8ConversionPositiveControlTest` below, for every arch in
    RUNTIME_INT8_ARCHS.
    """
    from core.pipeline_backends.acestep import AceStepMixin
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    inner = nn.Linear(8, 8, dtype=torch.bfloat16)
    wrapped = LoRALinearLayer(inner, rank=4, alpha=4, lora_name="stale")
    dit = nn.Sequential(wrapped, nn.Linear(8, 8, dtype=torch.bfloat16))
    manager = _FakeManager(acestep_components={"dit": dit},
                           _acestep_lora_original_modules={"stale": inner},
                           _acestep_lora_wrapped_modules={"stale"})
    manager.__class__ = type("FakeAceStep", (_FakeManager, AceStepMixin), {})
    manager._acestep_runtime_int8({"unet_quantization": quantization})


_GUARD_CASES = {
    "acestep": _acestep_case,
    "ltx2": _ltx2_case,
    "flux2": _flux2_case,
    "ideogram4": _ideogram4_case,
    "krea2": _krea2_case,
    "anima": _anima_case,
}


# ---------------------------------------------------------------------------
# Defect class 3b: the hook that is a no-op.
#
# Every test above is satisfied by a hook that does nothing at all: the guard
# cases only ever pass values that convert NOTHING (None/"none"/""/fp8_e4m3fn),
# `test_the_genuine_violation_is_still_refused` names three archs by hand, and
# the existence test greps for `def _<arch>_runtime_int8`, which an empty body
# satisfies. Measured: replacing `AceStepMixin._acestep_runtime_int8` with
# `lambda self, params, progress_callback=None: None` left the whole file green,
# and krea2/anima were in the same state. So each arch's hook is also run with an
# int8 request that MUST convert.
# ---------------------------------------------------------------------------

def _quantizable_component():
    """One bf16 Linear that no arch's selection policy may filter out.

    2048x2048 is 8-aligned (`min_align`, which costs ACE-Step its 2048x6 FSQ
    projections) and sits at or above the runtime min-work gate (k>=2048,
    n>=1024) that acestep/anima/ideogram4/ltx2 apply, so "the policy skipped it"
    is not available as an explanation for a module that comes back unconverted.
    """
    return nn.Sequential(nn.Linear(2048, 2048, bias=True, dtype=torch.bfloat16))


def _mixin_for(arch):
    """The one `*Mixin` class `core.pipeline_backends.<arch>` defines."""
    module = importlib.import_module(f"core.pipeline_backends.{arch}")
    mixins = [obj for name, obj in vars(module).items()
              if isinstance(obj, type) and name.endswith("Mixin")
              and obj.__module__ == module.__name__]
    if len(mixins) != 1:
        raise AssertionError(
            f"core/pipeline_backends/{arch}.py defines {len(mixins)} *Mixin classes; "
            f"this helper assumes exactly one")
    return mixins[0]


# arch -> how its hook is CALLED. Signatures genuinely differ (Anima takes the
# quantization string itself, FLUX.2 takes the transformer as a second argument,
# LTX-2.3 also reads blocks_to_swap), so the invocation is per arch; the
# components, their NAMES and the assertion are not. Completeness is checked
# against RUNTIME_INT8_ARCHS below.
_INT8_INVOCATIONS = {
    "acestep": lambda m, mods, q: m._acestep_runtime_int8({"unet_quantization": q}),
    "anima": lambda m, mods, q: m._anima_runtime_int8(q),
    "krea2": lambda m, mods, q: m._krea2_runtime_int8({"unet_quantization": q}),
    "flux2": lambda m, mods, q: m._flux2_runtime_int8({"unet_quantization": q},
                                                      mods["transformer"]),
    "ideogram4": lambda m, mods, q: m._ideogram4_runtime_int8({"unet_quantization": q}),
    "ltx2": lambda m, mods, q: m._ltx2_runtime_int8({"unet_quantization": q,
                                                     "blocks_to_swap": 0}),
}


def _run_runtime_int8(arch, quantization):
    """Run `arch`'s hook over fake components; return the modules it was given.

    The component NAMES come from `layout_module_specs`, i.e. the same names the
    F1 anchor validates against the arch's loader -- a hook reading a key nothing
    puts a module under would find None and return quietly, which is the failure
    this control has to be able to see.

    CPU only: `torch.cuda.is_available` is patched False so the converter picks
    no work device, which keeps the run identical on a machine with a GPU and on
    one without (and allocates nothing on the device).
    """
    from core.models.common.quantized_export import layout_module_specs

    modules = {name: _quantizable_component()
               for name, _prefix in layout_module_specs(arch)}
    manager = _FakeManager(**{f"{arch}_components": dict(modules)})
    manager.__class__ = type(f"Fake{arch}", (_FakeManager, _mixin_for(arch)), {})
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        _INT8_INVOCATIONS[arch](manager, modules, quantization)
    return modules


class RuntimeInt8ConversionPositiveControlTest(unittest.TestCase):
    """`unet_quantization="int8"` must really replace the Linear modules.

    Nothing else in this file distinguishes a working hook from an empty one.
    This does not check numerics (that is `int8_linear`'s own concern) -- only
    that the request reaches the converter and the module tree comes back holding
    the quantized classes instead of `nn.Linear`.
    """

    def test_every_runtime_int8_arch_is_covered_here(self):
        self.assertEqual(
            sorted(set(RUNTIME_INT8_ARCHS) - set(_INT8_INVOCATIONS)), [],
            "these architectures honor unet_quantization='int8' but have no "
            "invocation in _INT8_INVOCATIONS, so nothing checks that their hook does "
            "anything at all. Add one -- or, if the hook needs more setup than a fake "
            "manager can provide, say so in the entry rather than leaving the arch "
            "out.")

    def test_the_hook_actually_converts(self):
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
        from core.models.ideogram4.vendor.int8_linear import Int8Linear

        for arch in RUNTIME_INT8_ARCHS:
            invoke = _INT8_INVOCATIONS.get(arch)
            if invoke is None:
                continue  # reported above
            with self.subTest(arch=arch):
                modules = _run_runtime_int8(arch, "int8")
                self.assertTrue(modules, arch)
                for name, module in modules.items():
                    layer = module[0]
                    self.assertIsInstance(
                        layer, (Int8Linear, Fp8Linear),
                        f"{arch}: after _{arch}_runtime_int8 with "
                        f"unet_quantization='int8', {name}'s 2048x2048 Linear is still "
                        f"a {type(layer).__name__}. The hook converted nothing -- an "
                        f"empty body passes every other test in this file, because "
                        f"they only ever call it with values that must NOT convert.")

    def test_the_control_is_a_control(self):
        """The same fixture with no int8 request must leave the Linears alone.

        Otherwise the assertion above could be satisfied by the fixture rather
        than by the hook.
        """
        for arch in RUNTIME_INT8_ARCHS:
            if arch not in _INT8_INVOCATIONS:
                continue
            for quantization in (None, "none", "fp8_e4m3fn"):
                with self.subTest(arch=arch, quantization=quantization):
                    modules = _run_runtime_int8(arch, quantization)
                    for name, module in modules.items():
                        self.assertIs(
                            type(module[0]), nn.Linear,
                            f"{arch}: a request for {quantization!r} converted {name} "
                            f"anyway")

# ~105 k parameters: 1 layer, 2 heads, 62 Linears. Small enough to build in a
# test, and needed because Ltx2BlockLoopWrapper pins the diffusers module tree.
_LTX2_CONFIG = {
    "activation_fn": "gelu-approximate", "attention_bias": True,
    "attention_head_dim": 16, "attention_out_bias": True,
    "audio_attention_head_dim": 8, "audio_cross_attention_dim": 32,
    "audio_cross_attn_mod": True, "audio_gated_attn": True,
    "audio_hop_length": 160, "audio_in_channels": 8,
    "audio_num_attention_heads": 2, "audio_out_channels": 8,
    "audio_patch_size": 1, "audio_patch_size_t": 1,
    "audio_pos_embed_max_pos": 20, "audio_sampling_rate": 16000,
    "audio_scale_factor": 4, "base_height": 64, "base_width": 64,
    "caption_channels": 48, "causal_offset": 1, "cross_attention_dim": 64,
    "cross_attn_mod": True, "cross_attn_timestep_scale_multiplier": 1000,
    "gated_attn": True, "in_channels": 8, "norm_elementwise_affine": False,
    "norm_eps": 1e-06, "num_attention_heads": 2, "num_layers": 1,
    "out_channels": 8, "patch_size": 1, "patch_size_t": 1,
    "perturbed_attn": True, "pos_embed_max_pos": 20,
    "qk_norm": "rms_norm_across_heads", "rope_double_precision": True,
    "rope_theta": 10000.0, "rope_type": "split",
    "timestep_scale_multiplier": 1000, "use_prompt_embeddings": False,
    "vae_scale_factors": [8, 32, 32],
}


class RuntimeInt8GuardOrderingTest(unittest.TestCase):
    """Defect class 3: a no-quantization request must be a NO-OP.

    Every runtime-int8 hook guards an invariant ("no block offloader may be live
    when the conversion runs") whose violation is silent. The guard is correct;
    firing it for a request that would convert NOTHING is not. On LTX-2.3 the
    offloader is persistent state on the wrapper, so the pre-fix guard turned
    every second block-swap generation of a session into a RuntimeError even when
    `unet_quantization` was unset -- and block swap is the standard mode for a
    37 GB model.
    """

    def test_every_runtime_int8_arch_is_covered_here(self):
        self.assertEqual(
            sorted(set(RUNTIME_INT8_ARCHS) - set(_GUARD_CASES)), [],
            "these architectures honor unet_quantization='int8' but have no case in "
            "_GUARD_CASES, so nothing checks that their guard leaves an "
            "unquantized request alone. Add one.")

    def test_a_request_with_no_quantization_is_a_no_op_with_a_stale_offloader(self):
        for arch in RUNTIME_INT8_ARCHS:
            case = _GUARD_CASES.get(arch)
            if case is None:
                continue  # reported above
            for quantization in (None, "none", "", "fp8_e4m3fn"):
                with self.subTest(arch=arch, quantization=quantization):
                    try:
                        case(quantization)
                    except RuntimeError as exc:
                        self.fail(
                            f"{arch}: a generation requesting "
                            f"{quantization!r} raised with a stale block offloader "
                            f"present: {exc}")

    def test_the_genuine_violation_is_still_refused(self):
        """The guard must survive the fix -- an int8 request with a live offloader.

        For LTX-2.3 the genuine violation is an int8 request that ALSO enables
        block swap in the same generation (`_ltx2_case`'s default): there the two
        orderings really do conflict. A stale offloader with `blocks_to_swap=0`
        is not a violation any more -- see the test below.
        """
        for arch in ("ltx2", "flux2", "ideogram4"):
            with self.subTest(arch=arch):
                with self.assertRaises(RuntimeError):
                    _GUARD_CASES[arch]("int8")

    def test_ltx2_int8_without_block_swap_tears_the_stale_offloader_down(self):
        """The refusal's ADVICE must work, not just read well.

        Scenario, reachable in one session: generate a video with
        `blocks_to_swap=22` (the offloader attaches and persists), then request
        `int8` WITH `blocks_to_swap=0` -- which is what the error message tells
        the user to do. The guard saw the stale offloader and raised, i.e. the
        advice was false in exactly the session where a user reads it, and the
        wrapper it refused over would have been torn down microseconds later by
        `_ensure_ltx2_swap_and_offload(0)`. The hook now performs that unwrap
        itself, first, so the request proceeds.
        """
        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper

        manager, offloader = _ltx2_manager()
        manager._ltx2_runtime_int8({"unet_quantization": "int8",
                                    "blocks_to_swap": 0})
        self.assertTrue(
            offloader.cleaned,
            "the stale block offloader was not cleaned up before the conversion")
        transformer = manager.ltx2_components["transformer"]
        self.assertNotIsInstance(
            transformer, Ltx2BlockLoopWrapper,
            "the transformer is still wrapped, so the conversion ran against a "
            "wrapper whose offloader holds references to the replaced modules")
        self.assertIs(manager.ltx2_components["pipeline"].transformer, transformer,
                      "the pipeline still points at the discarded wrapper")

    def test_ltx2_keeps_the_wrapper_when_a_feature_still_needs_the_block_loop(self):
        """`force_wrap` (FBCache/Spectrum/style at blocks_to_swap=0) is not an unwrap.

        Those features need the wrapper's custom block loop, so the hook must not
        tear it down behind the generate path's back; with a LIVE offloader that
        is still the genuine violation.
        """
        manager, offloader = _ltx2_manager()
        with self.assertRaises(RuntimeError):
            manager._ltx2_runtime_int8({"unet_quantization": "int8",
                                        "blocks_to_swap": 0}, force_wrap=True)
        self.assertFalse(offloader.cleaned)

    def test_no_hook_raises_before_it_has_looked_at_the_request(self):
        """Static complement: covers archs whose fake cannot reach every branch.

        A `raise` in the hook's own body (not in a nested function, which is the
        `precheck` form -- called only when a conversion is really about to
        start) must be preceded by a look at the request value.
        """
        for arch in RUNTIME_INT8_ARCHS:
            module = importlib.import_module(f"core.pipeline_backends.{arch}")
            path = Path(inspect.getsourcefile(module))
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            hook = next((n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef)
                         and n.name == f"_{arch}_runtime_int8"), None)
            self.assertIsNotNone(hook, arch)

            nested = {id(n) for f in ast.walk(hook)
                      if isinstance(f, ast.FunctionDef) and f is not hook
                      for n in ast.walk(f)}
            consulted = min(
                [n.lineno for n in ast.walk(hook)
                 if id(n) not in nested
                 and isinstance(n, ast.Constant)
                 and n.value == "unet_quantization"]
                + [n.lineno for n in ast.walk(hook)
                   if id(n) not in nested and isinstance(n, ast.Name)
                   and n.id in ("runtime_int8_requested", "transformer_quantization")],
                default=None)
            for node in ast.walk(hook):
                if id(node) in nested or not isinstance(node, ast.Raise):
                    continue
                self.assertIsNotNone(
                    consulted,
                    f"{arch}: _{arch}_runtime_int8 raises at line {node.lineno} "
                    f"without ever looking at the requested quantization")
                self.assertGreater(
                    node.lineno, consulted,
                    f"{arch}: _{arch}_runtime_int8 raises at line {node.lineno}, "
                    f"before the requested quantization is consulted (line "
                    f"{consulted}). A guard that fires for a request that converts "
                    f"nothing turns a leftover attribute into a failed generation; "
                    f"pass it to apply_runtime_int8_quantization as `precheck` "
                    f"instead, which runs only when a conversion really starts.")


class RuntimeInt8PartialConversionTest(unittest.TestCase):
    """`converted is False` does NOT mean "the model is unchanged".

    `apply_runtime_int8_quantization` returns False for a PARTIAL conversion too
    -- the CUDA-OOM-at-layer-N path it explicitly designs for, which sets
    `manager._runtime_int8_partial` and leaves the layers converted so far as
    `Int8Linear`. Any hook cleanup that a completed conversion needs is therefore
    needed after a partial one as well. ACE-Step's was gated on `converted`
    alone: after an OOM at layer 200 of 383, the pre-conversion bf16 modules
    stayed in `_acestep_lora_original_modules`, so the next LoRA load/unload
    cycle restored them over the Int8Linear modules (silently un-quantizing those
    layers) and held 2.4 GB of bf16 resident meanwhile. Anima's hook already
    consulted the latch, which is why this is a parity test and not an ACE-Step
    one.
    """

    def test_a_hook_that_branches_on_converted_also_consults_the_partial_latch(self):
        for arch in RUNTIME_INT8_ARCHS:
            module = importlib.import_module(f"core.pipeline_backends.{arch}")
            source = Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")
            hook = next((n for n in ast.walk(ast.parse(source))
                         if isinstance(n, ast.FunctionDef)
                         and n.name == f"_{arch}_runtime_int8"), None)
            self.assertIsNotNone(hook, arch)

            # Names bound to the converter's SECOND return value.
            converted_names = set()
            for node in ast.walk(hook):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    continue
                func = node.value.func
                name = func.attr if isinstance(func, ast.Attribute) else \
                    getattr(func, "id", "")
                if not name.startswith("apply_runtime_int8_quantization"):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Tuple) and len(target.elts) == 2 \
                            and isinstance(target.elts[1], ast.Name):
                        converted_names.add(target.elts[1].id)
            branched = [n.lineno for n in ast.walk(hook) if isinstance(n, ast.If)
                        and any(isinstance(x, ast.Name) and x.id in converted_names
                                for x in ast.walk(n.test))]
            if not branched:
                continue  # the hook ignores the flag; nothing to get wrong
            hook_source = ast.get_source_segment(source, hook) or ""
            self.assertIn(
                "_runtime_int8_partial", hook_source,
                f"{arch}: _{arch}_runtime_int8 branches on the converter's "
                f"'converted' return (line {branched[0]}) without ever consulting "
                f"manager._runtime_int8_partial. 'converted' is False for a PARTIAL "
                f"conversion as well as for a no-op, so whatever that branch does "
                f"after a full conversion is skipped in exactly the half-converted "
                f"state that needs it most.")

    def test_acestep_drops_the_stale_lora_base_cache_after_a_partial_conversion(self):
        """The functional half, on the arch where the branch has a body.

        The latch is the observable an OOM leaves behind, so it is set directly
        rather than by exhausting a GPU: with `_runtime_int8_partial` set, the DiT
        already holds Int8Linear modules whose pre-conversion bf16 originals are
        in the LoRA cache, and the cache must not survive this hook.
        """
        from core.pipeline_backends.acestep import AceStepMixin

        pre_conversion = nn.Linear(8, 8, dtype=torch.bfloat16)
        dit = nn.Sequential(nn.Linear(8, 8, dtype=torch.bfloat16))
        manager = _FakeManager(
            acestep_components={"dit": dit},
            _acestep_lora_original_modules={"decoder.layers.0.q_proj": pre_conversion},
            _acestep_lora_wrapped_modules=set(),
            _runtime_int8_partial=True,
            _runtime_int8_partial_done=200)
        manager.__class__ = type("FakeAceStep", (_FakeManager, AceStepMixin), {})
        manager._acestep_runtime_int8({"unet_quantization": "none"})
        self.assertEqual(
            manager._acestep_lora_original_modules, {},
            "the pre-conversion bf16 base modules are still cached after a partial "
            "INT8 conversion. A later LoRA load/unload cycle restores them over the "
            "converted Int8Linear modules (silently un-quantizing those layers), and "
            "the cache holds their weights resident until the model is reloaded.")


if __name__ == "__main__":
    unittest.main()
