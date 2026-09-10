import torch

from backend.core.memory_management.layer_offload_strategy import LayerOffloadStrategy


def test_strategy_summary_is_encodable_by_windows_console(capsys):
    strategy = LayerOffloadStrategy(
        num_layers=4,
        blocks_to_swap=2,
        device=torch.device("cuda"),
    )

    strategy.print_strategy()
    output = capsys.readouterr().out

    assert "CPU <-> GPU" in output
    output.encode("cp932")
