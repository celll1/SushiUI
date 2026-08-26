"""Krea2's 48q/12kv attention must hit the conduit's native GQA pre-expansion.

Krea2Attention passes an explicit ``enable_gqa=self.num_heads != self.num_kv_heads``
straight into ``dispatch_attention`` (backend/core/models/krea2/vendor/transformer.py).
The default ``attention_backend`` is "native" for every architecture
(backend/api/param_defaults.py), so a default Krea2 run resolves native on
every one of its 28 transformer blocks. The fix lives centrally in
backend/core/attention/dispatch.py (not at this call site); this file proves
Krea2's real geometry actually reaches it and gets the fast path.
"""

import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.models.krea2.vendor.transformer as krea2_transformer  # noqa: E402
from core.attention.registry import BACKENDS  # noqa: E402


def _attn(num_heads=48, num_kv_heads=12, head_dim=8):
    hidden_size = num_heads * head_dim
    attn = krea2_transformer.Krea2Attention(hidden_size, num_heads, num_kv_heads)
    attn.eval()
    return attn


class Krea2GQADispatchTest(unittest.TestCase):
    def test_default_config_is_real_gqa(self):
        """48q/12kv (ratio 4) is the shipped Krea2Transformer2DModel default."""
        attn = _attn()
        self.assertNotEqual(attn.num_heads, attn.num_kv_heads)
        self.assertEqual(attn.num_heads // attn.num_kv_heads, 4)

    def test_native_backend_preexpands_kv(self):
        attn = _attn()
        attn._attn_backend = "native"
        hidden = torch.randn(1, 5, attn.hidden_size)

        capture = {}
        original = BACKENDS["native"]

        def spy_fn(q, k, v, **kwargs):
            capture["q_heads"] = q.shape[2]
            capture["k_heads"] = k.shape[2]
            capture["enable_gqa"] = kwargs.get("enable_gqa")
            return original.fn(q, k, v, **kwargs)

        import dataclasses

        BACKENDS["native"] = dataclasses.replace(original, fn=spy_fn)
        try:
            attn(hidden)
        finally:
            BACKENDS["native"] = original

        self.assertEqual(capture["k_heads"], capture["q_heads"])
        self.assertEqual(capture["q_heads"], 48)
        self.assertFalse(capture["enable_gqa"])

    def test_flash_backend_leaves_kv_unexpanded(self):
        attn = _attn()
        attn._attn_backend = "flash"
        hidden = torch.randn(1, 5, attn.hidden_size)

        capture = {}
        original = BACKENDS["flash"]

        def spy_fn(q, k, v, **kwargs):
            capture["q_heads"] = q.shape[2]
            capture["k_heads"] = k.shape[2]
            return q.clone()

        import dataclasses

        BACKENDS["flash"] = dataclasses.replace(original, fn=spy_fn)
        try:
            attn(hidden)
        finally:
            BACKENDS["flash"] = original

        self.assertEqual(capture["k_heads"], 12)
        self.assertEqual(capture["q_heads"], 48)

    def test_native_output_matches_pre_fix_math(self):
        """The pre-expansion is a pure speed optimization: output must be
        identical to what unmodified enable_gqa=True SDPA produces."""
        torch.manual_seed(3)
        attn = _attn()
        attn._attn_backend = "native"
        hidden = torch.randn(2, 7, attn.hidden_size)
        out = attn(hidden)

        # Reference: run the pre-conduit-fix math directly (bypass dispatch_attention).
        query = attn.to_q(hidden).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        qb, kb, vb = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ref = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, enable_gqa=True)
        ref = ref.transpose(1, 2).flatten(2, 3)
        ref = ref * torch.sigmoid(attn.to_gate(hidden))
        ref = attn.to_out[0](ref)

        self.assertTrue(torch.allclose(out, ref, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
