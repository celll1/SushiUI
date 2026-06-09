"""
Live vocabulary expansion coordinator for Danbooru-augmented tagger training.

VocabExpander is a thread-safe bridge between the Danbooru fetch thread
(which proposes new tags) and the training thread (which performs the
actual expansion).

expand_vocab_and_head() is the single function that atomically:
  1. Extends TagVocabulary with new tags
  2. Expands model.head (nn.Linear) with zero-initialized new rows
  3. Updates optimizer param_groups to reference the new head parameters
  4. Migrates (or resets) the optimizer's per-parameter state for the head
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .tag_vocabulary import TagVocabulary


# ---------------------------------------------------------------------------
# VocabExpander — thread-safe tag proposal queue
# ---------------------------------------------------------------------------

class VocabExpander:
    """Collects new-tag proposals from the buffer thread for the training thread.

    The buffer thread calls ``propose()`` when it encounters Danbooru tags that
    are not in the current vocabulary AND have been approved by the surveyor.
    The training thread calls ``has_pending()`` / ``consume_pending()`` once
    per batch to drain and act on those proposals.

    ``_already_proposed`` prevents re-proposing tags that were already consumed
    but not yet added (e.g. the expansion failed silently) or that were
    added and then somehow ended up in another image's tag list.
    """

    def __init__(self) -> None:
        self._pending: Set[str] = set()
        self._already_proposed: Set[str] = set()
        self._lock = threading.Lock()

    def propose(self, tags: Set[str]) -> None:
        """Called from the buffer (fetch) thread."""
        with self._lock:
            new = tags - self._already_proposed
            if new:
                self._pending |= new
                self._already_proposed |= new

    def has_pending(self) -> bool:
        """Called from the training thread."""
        with self._lock:
            return bool(self._pending)

    def consume_pending(self) -> List[str]:
        """Return and clear all pending tags. Called from the training thread."""
        with self._lock:
            tags = sorted(self._pending)
            self._pending.clear()
            return tags


# ---------------------------------------------------------------------------
# Optimizer state helpers
# ---------------------------------------------------------------------------

def _is_8bit_state(state: Dict[str, Any]) -> bool:
    return any(k in state for k in ("state1", "state2", "absmax1", "absmax2"))


def _expand_param_state(
    state: Dict[str, Any],
    n_new: int,
    is_bias: bool = False,
) -> None:
    """Expand per-parameter optimizer state in-place for n_new new rows/elements.

    For FP32 optimizers (AdamW, Lion):
      - ``exp_avg``, ``exp_avg_sq`` : concatenate zeros along dim 0 (weight)
        or extend length (bias)
    For 8-bit optimizers (AdamW8bit, Lion8bit):
      - Clear the entire state so bitsandbytes re-initialises on the next step.
        (Same policy as optimizer_state_migrate._migrate_one_param_state for
        8-bit params — re-quantising expanded block-wise tensors correctly
        requires non-trivial dequant/pad/requant logic; a one-step warm-up cost
        is far cheaper and safer.)
    """
    if not state:
        return

    if _is_8bit_state(state):
        state.clear()
        return

    # FP32: append zeros to momentum tensors along the tag axis.
    for key in list(state.keys()):
        val = state[key]
        if not torch.is_tensor(val):
            continue
        if key in ("exp_avg", "exp_avg_sq"):
            if is_bias:
                # bias state shape: [num_tags]
                zeros = torch.zeros(n_new, dtype=val.dtype, device=val.device)
                state[key] = torch.cat([val, zeros], dim=0)
            else:
                # weight state shape: [num_tags, in_features]
                zeros = torch.zeros(n_new, val.shape[1], dtype=val.dtype, device=val.device)
                state[key] = torch.cat([val, zeros], dim=0)
        # Other keys (step, gnorm_vec, …): leave untouched


def _find_param_in_optimizer(
    optimizer: torch.optim.Optimizer,
    param: nn.Parameter,
) -> Optional[int]:
    """Return the flat parameter index used as key in optimizer.state_dict()['state']."""
    idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is param:
                return idx
            idx += 1
    return None


# ---------------------------------------------------------------------------
# Main expansion function
# ---------------------------------------------------------------------------

def expand_vocab_and_head(
    new_tags: List[str],
    vocabulary: "TagVocabulary",
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Expand vocabulary, model head, and optimizer state atomically.

    Parameters
    ----------
    new_tags   : raw tag strings to add (will be normalized internally by add_tags)
    vocabulary : TagVocabulary — mutated in place
    model      : SigLIP2TaggerModel (or LoRA variant); must have ``model.head``
                 and ``model.expand_head()``
    optimizer  : the training optimizer; param_groups[1] must be the head group

    Returns
    -------
    Number of tags actually added (0 if all were already present).
    """
    from .tag_vocabulary import normalize_tag

    # Step 1: snapshot old mapping before mutation
    old_tag_to_idx: Dict[str, int] = dict(vocabulary.tag_to_idx)
    old_num_tags = vocabulary.num_tags

    # Step 2: extend vocabulary
    added: List[Tuple[str, int]] = vocabulary.add_tags(new_tags)
    if not added:
        return 0

    n_new = vocabulary.num_tags - old_num_tags

    # Step 3: save references to old head parameters BEFORE expand_head()
    #         replaces model.head (the old Parameter objects become orphans
    #         but are still referenced here for optimizer.state key lookup).
    old_head_weight: nn.Parameter = model.head.weight
    old_head_bias: Optional[nn.Parameter] = model.head.bias if model.head.bias is not None else None

    # Step 4: expand the head linear layer (zeros for new rows)
    new_w, new_b = model.expand_head(vocabulary.num_tags)

    # Step 5: update optimizer param_groups[1] to reference new parameters
    head_params = list(model.head.parameters())
    # Guard: if the optimizer has fewer than 2 groups, append rather than crash.
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["params"] = head_params
    else:
        optimizer.param_groups[0]["params"] = head_params

    # Step 6: migrate optimizer state
    #   The optimizer.state dict uses the live Parameter object as key.
    #   We need to:
    #   a) Look up old_head_weight's existing state (by object identity)
    #   b) Expand the state tensors for n_new new tags
    #   c) Move the state entry to the new parameter key
    #   d) Repeat for bias

    for old_param, new_param, is_bias in [
        (old_head_weight, new_w, False),
        (old_head_bias,   new_b, True),
    ]:
        if old_param is None or new_param is None:
            continue
        if old_param in optimizer.state:
            state = optimizer.state.pop(old_param)
            _expand_param_state(state, n_new, is_bias=is_bias)
            optimizer.state[new_param] = state
        # If no state yet (first step hasn't run), nothing to migrate.

    added_names = [tag for tag, _ in added]
    print(
        f"[VocabExpander] +{n_new} tag(s): "
        f"{added_names[:5]}{'...' if len(added_names) > 5 else ''}  "
        f"(total vocabulary: {vocabulary.num_tags})"
    )
    return n_new
