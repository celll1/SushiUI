"""Strategy section 7's weightless half of the SenseNova null parity gate.

Steps 1 and 2 of the gate -- token-level parity and image-index parity between
the training null and inference's own uncond arm -- run without a checkpoint,
so they live here as ordinary tests as well as in the probe. They drive the very
functions the probe's weight arms drive
(`core.training.probes.sensenova_cfg_null_parity`), so the two cannot diverge.

The real checkpoint tokenizer is used when the configured model tree is staged
(tokenizer files only, no tensors); the structural claims that do not depend on
a vocabulary run everywhere with a deterministic stub.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/cfg_null_sensenova_parity_test.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.probes import sensenova_cfg_null_parity as probe  # noqa: E402
from tests.model_root import model_path  # noqa: E402  (tests dir is on sys.path)

_CHECKPOINT_DIR = model_path("sensenova")
_HAS_TOKENIZER = os.path.isfile(os.path.join(_CHECKPOINT_DIR, "tokenizer_config.json"))
_needs_tokenizer = pytest.mark.skipif(
    not _HAS_TOKENIZER,
    reason=f"SenseNova tokenizer files are not staged at {_CHECKPOINT_DIR}",
)


class _StubTokenizer:
    """One token id per character: deterministic, and monotone in the query."""

    def __call__(self, query, return_tensors=None):
        ids = torch.arange(1, len(query) + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids}


def _builders():
    template = (
        probe.checkpoint_template(_CHECKPOINT_DIR)
        if _HAS_TOKENIZER
        else probe.DEFAULT_TEMPLATE
    )
    return probe.QueryBuilders(template=template)


# ---------------------------------------------------------------------------
# Step 1: tokens
# ---------------------------------------------------------------------------


def test_the_null_query_equals_inferences_uncond_query_with_a_stub_tokenizer():
    result = probe.token_parity(_builders(), _StubTokenizer())
    assert result["query_equal"], result
    assert result["ids_equal"]
    assert result["text_length_equal"]
    assert result["null_differs_from_conditional"]


@_needs_tokenizer
def test_the_null_token_ids_equal_inferences_under_the_real_tokenizer():
    """The vocabulary is the checkpoint's own, so this is token parity and not
    only string parity: the ids the prefix forward would consume are compared."""
    transformer = _builders()
    result = probe.token_parity(transformer, probe.load_tokenizer(_CHECKPOINT_DIR))
    assert result["query_equal"]
    assert result["ids_equal"]
    assert result["text_length_equal"]
    assert result["null_token_count"] < result["conditional_token_count"]


# ---------------------------------------------------------------------------
# Step 2: image indexes
# ---------------------------------------------------------------------------


def test_the_image_indexes_equal_inferences_at_several_resolutions():
    result = probe.index_parity(_builders(), _StubTokenizer())
    assert len(result["resolutions"]) >= 2
    assert result["all_equal"], result


@_needs_tokenizer
def test_the_image_indexes_equal_inferences_under_the_real_tokenizer():
    result = probe.index_parity(
        _builders(), probe.load_tokenizer(_CHECKPOINT_DIR)
    )
    assert result["all_equal"], result
    # Every image token carries the null prefix length in its t coordinate.
    assert len({entry["t_value"] for entry in result["resolutions"]}) == 1
