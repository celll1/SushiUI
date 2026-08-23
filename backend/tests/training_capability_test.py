"""The training capability tables must cover every trainable architecture, and
the training form must read them instead of re-deriving them from arch names.

Two defects motivated this file, both of the same shape -- a per-architecture
fact spelled out in the UI and then never revisited when an architecture was
added:

1. `TrainingConfig.tsx`'s base-model filter had one `useState` per architecture
   for five of the thirteen in `ARCH_REGISTRY`. A model of any other
   architecture was shown only because no checkbox governed it; the filter was
   simply blind to lens/ideogram4/minit2i/krea2/ltx2/acestep/minimax_h3/
   sensenova.
2. The Block Swap section was rendered for every base model, including
   SenseNova, whose arch handler raises on `setup_block_swap` and whose runs are
   refused before they start.

Both are now backend declarations (`TRAINING_FEATURE_UNSUPPORTED`,
`ARCH_DISPLAY_NAMES`) that the form reads. What this file fixes is the
COVERAGE: that the declarations know about every architecture the registry
does, and that the mechanisms which refuse at runtime are declared.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_capability_test.py -v
"""

import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from api.arch_capabilities import (  # noqa: E402
    ARCH_DISPLAY_NAMES,
    TRAINING_DECLARED_ARCHS,
    TRAINING_FEATURE_LABELS,
    TRAINING_FEATURE_PARAMS,
    TRAINING_FEATURE_UNSUPPORTED,
    TRAINING_METHODS,
    TRAINING_UNSUPPORTED,
    training_feature_unsupported_reason,
)

ARCH_DIR = BACKEND / "core" / "training" / "arch"
TRAINING_CONFIG_TSX = REPO / "frontend" / "src" / "components" / "training" / "TrainingConfig.tsx"


def _arch_source(arch: str) -> str:
    return (ARCH_DIR / f"{arch}.py").read_text(encoding="utf-8")


def _method_body(source: str, name: str) -> str:
    """The body of `def <name>(` up to the next top-level-in-class `def`."""
    match = re.search(rf"\n    def {name}\(.*?(?=\n    def |\Z)", source, re.S)
    assert match, f"{name} not found"
    return match.group(0)


def test_declared_archs_match_the_arch_registry():
    """The coverage invariant. `TRAINING_DECLARED_ARCHS` restates ARCH_REGISTRY's
    keys because arch_capabilities cannot import the trainer package; this is
    what keeps the restatement honest."""
    from core.training.arch import ARCH_REGISTRY

    assert set(ARCH_REGISTRY) == set(TRAINING_DECLARED_ARCHS)


def test_every_declared_arch_has_a_display_name():
    """The architecture filter labels itself from this map."""
    missing = sorted(TRAINING_DECLARED_ARCHS - set(ARCH_DISPLAY_NAMES))
    assert missing == [], f"no display name for {missing}"


def test_tables_only_name_known_archs_features_and_methods():
    assert set(TRAINING_UNSUPPORTED) <= set(TRAINING_DECLARED_ARCHS)
    assert set(TRAINING_FEATURE_UNSUPPORTED) <= set(TRAINING_DECLARED_ARCHS)
    assert set(TRAINING_FEATURE_LABELS) == set(TRAINING_FEATURE_PARAMS)
    for arch, features in TRAINING_FEATURE_UNSUPPORTED.items():
        for feature, entry in features.items():
            assert feature in TRAINING_FEATURE_PARAMS, (arch, feature)
            assert entry["reason"].strip()
            assert set(entry.get("methods", TRAINING_METHODS)) <= set(TRAINING_METHODS)


def test_block_swap_refusals_are_declared():
    """An arch handler whose setup_block_swap raises must say so in the table --
    otherwise the form offers a control whose run is refused."""
    for arch in sorted(TRAINING_DECLARED_ARCHS):
        body = _method_body(_arch_source(arch), "setup_block_swap")
        if "raise NotImplementedError" not in body:
            continue
        assert training_feature_unsupported_reason(arch, "block_swap"), (
            f"arch/{arch}.py's setup_block_swap raises but block_swap is not "
            f"declared unsupported for it")


def test_sampling_refusals_are_declared():
    for arch in sorted(TRAINING_DECLARED_ARCHS):
        body = _method_body(_arch_source(arch), "sample")
        refuses = ("raise NotImplementedError" in body
                   or re.search(r"not yet\s+.*supported", body, re.S) is not None)
        if not refuses:
            continue
        assert training_feature_unsupported_reason(arch, "training_samples"), (
            f"arch/{arch}.py cannot sample but training_samples is not declared "
            f"unsupported for it")


def test_method_scope_narrows_rather_than_hides():
    """A scoped entry answers only for the methods it names."""
    assert training_feature_unsupported_reason("zimage", "text_encoder_training", "lora")
    assert training_feature_unsupported_reason("zimage", "text_encoder_training",
                                               "full_finetune") is None
    # Unknown arch / unknown feature: supported, never a silent hide.
    assert training_feature_unsupported_reason("brand_new_arch", "block_swap") is None
    assert training_feature_unsupported_reason(None, "block_swap") is None
    assert training_feature_unsupported_reason("sensenova", "no_such_feature") is None


def test_route_serves_the_new_capability_keys():
    source = (BACKEND / "api" / "routes.py").read_text(encoding="utf-8")
    for key in ("training_feature_unsupported", "training_feature_params",
                "training_feature_labels", "arch_display_names"):
        assert f'"{key}"' in source, f"GET /schema/arch-capabilities does not serve {key}"


def test_openapi_declares_the_new_capability_keys():
    spec = (REPO / "openapi.yaml").read_text(encoding="utf-8")
    for key in ("training_feature_unsupported", "training_feature_params",
                "training_feature_labels", "arch_display_names"):
        assert re.search(rf"^        {key}:$", spec, re.M), f"ArchCapabilities lacks {key}"


def test_training_form_is_data_driven():
    """The form must not carry per-architecture booleans for the filter, and its
    capability gates must read the backend table."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    for stale in ("showSD15", "showSDXL", "showZImage", "showFlux2", "showAnima"):
        assert stale not in tsx, f"{stale} is a per-arch filter flag; derive the list instead"
    assert "trainingFeatureUnsupportedReason(" in tsx
    assert "archDisplayName(" in tsx
    for gate in ("blockSwapUnsupported", "fusedGroupsUnsupported",
                 "referenceImagesUnsupported", "textEncoderTrainingUnsupported",
                 "trainingSamplesUnsupported", "vaeUnsupported"):
        assert gate in tsx, f"{gate} gate missing from the training form"
