from __future__ import annotations

import ast
from pathlib import Path


BASE_TRAINER = Path(__file__).resolve().parents[1] / "core" / "training" / "base_trainer.py"


def _calls(node):
    return [
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
    ]


def test_coincident_refills_skip_the_intermediate_main_model_round_trip():
    tree = ast.parse(BASE_TRAINER.read_text(encoding="utf-8"))
    guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "_coalesced_refill"
    ]

    assert len(guards) == 2
    guarded_calls = [_calls(node) for node in guards]
    assert any("move_main_model_to_gpu" in calls and "empty_cache" in calls
               for calls in guarded_calls)
    assert any(calls == ["move_main_model_to_cpu"] for calls in guarded_calls)


def test_due_flags_require_each_swap_buffer():
    source = BASE_TRAINER.read_text(encoding="utf-8")

    assert "_text_refill_due = (\n                        swap_buffer is not None" in source
    assert "_latent_refill_due = (\n                        latent_swap_buffer is not None" in source
    assert "_coalesced_refill = _text_refill_due and _latent_refill_due" in source
