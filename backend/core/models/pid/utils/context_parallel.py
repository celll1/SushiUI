# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Context-parallel helpers, vendored subset (from `pid/_src/utils/context_parallel.py`).

SushiUI never initializes a context-parallel process group for PiD inference
(single-GPU decode only), so `cp_group` is always `None` at every call site in
the vendored networks (`pid_net.py`, `pixeldit_official.py`) and these functions
are never actually invoked — they exist purely so the (verbatim) network code
that guards on `if cp_group is not None:` continues to import cleanly.

Only `split_inputs_cp` and `cat_outputs_cp_with_grad` are vendored (the only two
consumed by the inference-only network code); `cat_outputs_cp` / `broadcast` /
`robust_broadcast` / `broadcast_split_tensor` from the original module are not
needed and were dropped to avoid depending on `pid._ext.imaginaire.utils.distributed`.
"""

from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, get_process_group_ranks
import torch


def split_inputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """Split input tensor along the sequence dimension for context parallelism."""
    cp_ranks = get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)

    assert x.shape[seq_dim] % cp_size == 0, f"{x.shape[seq_dim]} cannot divide cp_size {cp_size}"
    x = x.view(*x.shape[:seq_dim], cp_size, x.shape[seq_dim] // cp_size, *x.shape[(seq_dim + 1) :])
    seq_idx = torch.tensor([cp_group.rank()], device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x


def cat_outputs_cp_with_grad(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """Gather + concatenate tensors across a context-parallel group, keeping the
    local rank's slice attached to its own compute graph."""
    cp_size = cp_group.size()
    assert cp_size > 0, "cp_size should be greater than 0"

    gathered_tensors = [torch.zeros_like(x) for _ in range(cp_size)]
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    rank = cp_group.rank()
    gathered_tensors[rank] = x
    return torch.cat(gathered_tensors, dim=seq_dim)
