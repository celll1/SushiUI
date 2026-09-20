import torch

from core.training.optimizers.optimizer_state_convert import (
    incompatible_optimizer_algorithm,
)


class AdamW8bit_RingBuffer:
    pass


class Lion8bit_RingBuffer:
    pass


def _lion_state():
    return {
        "_sushi_opt_class": "Lion8bit_RingBuffer",
        "state": {
            0: {
                "exp_avg": torch.zeros(256, dtype=torch.uint8),
                "absmax": torch.ones(1),
            }
        },
        "param_groups": [{"params": [0]}],
    }


def test_lion_to_adamw_requires_fresh_optimizer_state():
    assert incompatible_optimizer_algorithm(
        _lion_state(), AdamW8bit_RingBuffer()
    ) == "rb_lion8bit -> rb_adamw8bit"


def test_same_algorithm_does_not_trigger_reset_guard():
    assert incompatible_optimizer_algorithm(
        _lion_state(), Lion8bit_RingBuffer()
    ) is None
