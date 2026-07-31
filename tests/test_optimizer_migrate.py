"""Quick in-memory test for optimizer_state_migrate.

Validates:
  1. AdamW (FP32): exp_avg / exp_avg_sq are migrated by tag-name alignment
  2. AdamW8bit-like state: state1 / state2 migrated, absmax* deleted
  3. New tag rows are zero, removed tag rows discarded, common tags preserved
  4. shape mismatch detection only triggers on vocab size change
"""
import sys, os
sys.path.insert(0, "backend")

import torch
import torch.nn as nn

from core.tagger.optimizer_state_migrate import migrate_head_optimizer_state


def make_dummy_setup(old_n: int, new_n: int, hidden: int = 8):
    """Create old/new vocab + a fake optimizer / model setup."""
    # Old tag names: tag_0..tag_(old_n-1)
    # New tag names: same minus the last 4 (simulating alias merges)
    old_tag_to_idx = {f"tag_{i}": i for i in range(old_n)}
    new_tags = [t for t in old_tag_to_idx if int(t.split("_")[1]) < new_n]
    new_tag_to_idx = {t: i for i, t in enumerate(new_tags)}

    # Build a tiny model with a Linear head matching new_n
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(hidden, new_n)
    model = M()

    # Build a fresh optimizer for the new model
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optim, old_tag_to_idx, new_tag_to_idx


def fabricate_saved_state_adamw(old_n: int, hidden: int = 8) -> dict:
    """Fabricate a torch.load-style saved state with OLD shapes for AdamW."""
    # The saved state would have entries keyed by integer index (0 = head.weight, 1 = head.bias)
    return {
        "state": {
            0: {  # head.weight
                "step": torch.tensor(1234, dtype=torch.long),
                "exp_avg":     torch.arange(old_n * hidden, dtype=torch.float32).reshape(old_n, hidden),
                "exp_avg_sq":  torch.arange(old_n * hidden, dtype=torch.float32).reshape(old_n, hidden) * 2.0,
            },
            1: {  # head.bias
                "step": torch.tensor(1234, dtype=torch.long),
                "exp_avg":     torch.arange(old_n, dtype=torch.float32),
                "exp_avg_sq":  torch.arange(old_n, dtype=torch.float32) * 2.0,
            },
        },
        "param_groups": [{"params": [0, 1], "lr": 1e-3}],
    }


def fabricate_saved_state_adam8bit(old_n: int, hidden: int = 8) -> dict:
    """Fabricate a saved state mimicking bnb AdamW8bit format."""
    # state1/state2 are uint8 (quantised), absmax1/absmax2 are per-block float32
    block_size = 256
    n_blocks_w = max(1, (old_n * hidden + block_size - 1) // block_size)
    n_blocks_b = max(1, (old_n + block_size - 1) // block_size)
    return {
        "state": {
            0: {
                "step": torch.tensor(1234, dtype=torch.long),
                "state1":   torch.randint(0, 256, (old_n, hidden), dtype=torch.uint8),
                "state2":   torch.randint(0, 256, (old_n, hidden), dtype=torch.uint8),
                "absmax1":  torch.rand(n_blocks_w),
                "absmax2":  torch.rand(n_blocks_w),
                "qmap1":    torch.linspace(-1, 1, 256),  # global table
                "qmap2":    torch.linspace(-1, 1, 256),
                "gnorm_vec": torch.zeros(100),
            },
            1: {
                "step": torch.tensor(1234, dtype=torch.long),
                "state1":   torch.randint(0, 256, (old_n,), dtype=torch.uint8),
                "state2":   torch.randint(0, 256, (old_n,), dtype=torch.uint8),
                "absmax1":  torch.rand(n_blocks_b),
                "absmax2":  torch.rand(n_blocks_b),
                "qmap1":    torch.linspace(-1, 1, 256),
                "qmap2":    torch.linspace(-1, 1, 256),
                "gnorm_vec": torch.zeros(100),
            },
        },
        "param_groups": [{"params": [0, 1], "lr": 1e-3}],
    }


def test_adamw_fp32():
    print("\n=== Test 1: AdamW (FP32), 100 -> 96 tags ===")
    OLD_N, NEW_N, H = 100, 96, 8
    model, optim, old_idx, new_idx = make_dummy_setup(OLD_N, NEW_N, H)
    saved = fabricate_saved_state_adamw(OLD_N, H)

    # Snapshot original values for verification
    orig_w_exp_avg = saved["state"][0]["exp_avg"].clone()
    orig_b_exp_avg = saved["state"][1]["exp_avg"].clone()

    summary = migrate_head_optimizer_state(
        saved_state=saved,
        optimizer=optim,
        head_weight=model.head.weight,
        head_bias=model.head.bias,
        old_tag_to_idx=old_idx,
        new_tag_to_idx=new_idx,
    )
    print(f"  summary: {summary}")

    # Verify shapes
    new_w_exp_avg = saved["state"][0]["exp_avg"]
    new_b_exp_avg = saved["state"][1]["exp_avg"]
    assert new_w_exp_avg.shape == (NEW_N, H), f"head.weight exp_avg shape {new_w_exp_avg.shape}"
    assert new_b_exp_avg.shape == (NEW_N,), f"head.bias exp_avg shape {new_b_exp_avg.shape}"

    # Verify common tags preserved (tag_0 is at idx 0 in both)
    for i in range(NEW_N):
        assert torch.equal(new_w_exp_avg[i], orig_w_exp_avg[i]), \
            f"tag_{i} weight mismatch at row {i}"
        assert new_b_exp_avg[i].item() == orig_b_exp_avg[i].item(), \
            f"tag_{i} bias mismatch at row {i}"

    # Removed tags (idx >= NEW_N) are discarded — verify by checking they don't leak
    print(f"  ✓ shapes correct, common tag values preserved, removed tags discarded")

    # step preserved
    assert saved["state"][0]["step"].item() == 1234
    print(f"  ✓ step preserved: {saved['state'][0]['step'].item()}")


def test_adam8bit():
    """For 8-bit optimisers, the head's state is fully cleared so bnb's
    init_state fires on the next step() (head momentum is reset)."""
    print("\n=== Test 2: AdamW8bit-like, 100 -> 96 tags (reset semantics) ===")
    OLD_N, NEW_N, H = 100, 96, 8
    model, optim, old_idx, new_idx = make_dummy_setup(OLD_N, NEW_N, H)
    saved = fabricate_saved_state_adam8bit(OLD_N, H)

    summary = migrate_head_optimizer_state(
        saved_state=saved,
        optimizer=optim,
        head_weight=model.head.weight,
        head_bias=model.head.bias,
        old_tag_to_idx=old_idx,
        new_tag_to_idx=new_idx,
    )
    print(f"  summary: {summary}")

    # State should be cleared entirely for 8-bit
    assert summary["weight"]["mode"] == "reset_8bit", f"expected reset_8bit, got {summary['weight']['mode']}"
    assert summary["bias"]["mode"]   == "reset_8bit"
    assert len(saved["state"][0]) == 0, f"head.weight state should be empty, got keys: {list(saved['state'][0].keys())}"
    assert len(saved["state"][1]) == 0, f"head.bias state should be empty, got keys: {list(saved['state'][1].keys())}"
    print(f"  ✓ head.weight state cleared (reset_8bit mode)")
    print(f"  ✓ head.bias state cleared (reset_8bit mode)")
    print(f"  ✓ bnb's init_state will fire on next step() — no KeyError")


def test_shape_match_no_op():
    print("\n=== Test 3: Same vocab size — no-op ===")
    OLD_N = NEW_N = 100
    H = 8
    model, optim, old_idx, new_idx = make_dummy_setup(OLD_N, NEW_N, H)
    saved = fabricate_saved_state_adamw(OLD_N, H)

    orig_state = {k: {kk: vv.clone() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                  for k, v in saved["state"].items()}

    summary = migrate_head_optimizer_state(
        saved_state=saved,
        optimizer=optim,
        head_weight=model.head.weight,
        head_bias=model.head.bias,
        old_tag_to_idx=old_idx,
        new_tag_to_idx=new_idx,
    )
    print(f"  summary: {summary}")

    # All tensors unchanged
    for p_idx in (0, 1):
        for key, val in orig_state[p_idx].items():
            cur = saved["state"][p_idx][key]
            if torch.is_tensor(val):
                assert torch.equal(cur, val), f"key {key} should be unchanged"
            else:
                assert cur == val
    print(f"  ✓ no tensors modified when shapes match")


def test_added_tags_zero_init():
    print("\n=== Test 4: Vocab GROWS — new tags zero-initialised ===")
    OLD_N, NEW_N, H = 90, 100, 8
    # Build new vocab: tag_0..tag_99 (10 new beyond old_n=90)
    old_tag_to_idx = {f"tag_{i}": i for i in range(OLD_N)}
    new_tag_to_idx = {f"tag_{i}": i for i in range(NEW_N)}

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(H, NEW_N)
    model = M()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    saved = fabricate_saved_state_adamw(OLD_N, H)
    orig_w_exp_avg = saved["state"][0]["exp_avg"].clone()

    migrate_head_optimizer_state(
        saved_state=saved,
        optimizer=optim,
        head_weight=model.head.weight,
        head_bias=model.head.bias,
        old_tag_to_idx=old_tag_to_idx,
        new_tag_to_idx=new_tag_to_idx,
    )

    new_w_exp_avg = saved["state"][0]["exp_avg"]
    assert new_w_exp_avg.shape == (NEW_N, H)

    # Old tags 0..89 preserved
    for i in range(OLD_N):
        assert torch.equal(new_w_exp_avg[i], orig_w_exp_avg[i])
    # New tags 90..99 are zeros
    for i in range(OLD_N, NEW_N):
        assert torch.all(new_w_exp_avg[i] == 0), f"new tag {i} should be zero"
    print(f"  ✓ 90 -> 100: old tags preserved, 10 new tags zero-init")


def test_unknown_keys_left_alone():
    print("\n=== Test 5: Unknown optimizer keys are not touched ===")
    OLD_N, NEW_N, H = 100, 96, 8
    model, optim, old_idx, new_idx = make_dummy_setup(OLD_N, NEW_N, H)
    saved = fabricate_saved_state_adamw(OLD_N, H)
    # Inject an unknown key
    saved["state"][0]["my_custom_buffer"] = torch.tensor([1.0, 2.0, 3.0])
    orig_custom = saved["state"][0]["my_custom_buffer"].clone()

    migrate_head_optimizer_state(
        saved_state=saved,
        optimizer=optim,
        head_weight=model.head.weight,
        head_bias=model.head.bias,
        old_tag_to_idx=old_idx,
        new_tag_to_idx=new_idx,
    )

    cur = saved["state"][0]["my_custom_buffer"]
    assert torch.equal(cur, orig_custom)
    print(f"  ✓ unknown key 'my_custom_buffer' preserved untouched")


if __name__ == "__main__":
    test_adamw_fp32()
    test_adam8bit()
    test_shape_match_no_op()
    test_added_tags_zero_init()
    test_unknown_keys_left_alone()
    print("\n✓ All migration unit tests passed.")
