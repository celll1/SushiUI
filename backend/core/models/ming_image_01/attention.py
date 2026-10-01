# Copyright (c) Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native Q/K norms and RoPE adapted to the architecture-neutral conduit."""
import torch
from core.attention import (AttentionMode, dispatch_attention, dispatch_attention_varlen,
                            normalize_backend, validate_backend)


def joint_attention(query, key, value, mask, *, backend):
    mode = AttentionMode.TRAINING if torch.is_grad_enabled() else AttentionMode.INFERENCE
    if mask is not None and mask.dtype == torch.bool:
        valid = mask if mask.ndim == 2 else mask[:, 0, 0] if mask.ndim == 4 and mask.shape[1:3] == (1, 1) else None
        if valid is not None and valid.shape[-1] == key.shape[1] and valid.shape[0] in (1, key.shape[0]):
            valid = valid.expand(key.shape[:2])
            lengths = valid.sum(-1, dtype=torch.int32)
            if bool((lengths > 0).all()) and not bool(valid.all()):
                batch, queries, heads, width = query.shape
                cu_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32) * queries
                cu_k = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
                # Retain padded query rows: native masks exclude keys, not queries.
                output = dispatch_attention_varlen(query.reshape(-1, heads, width), key[valid], value[valid],
                    cu_q, cu_k, queries, int(lengths.max().item()), backend=backend, mode=mode)
                return output.reshape_as(query)
    if mask is not None and mask.ndim == 2:
        mask = mask[:, None, None, :]
    return dispatch_attention(query, key, value, attn_mask=mask, backend=backend, mode=mode)


class MingAttentionProcessor:
    def __init__(self, backend):
        self._attention_backend = backend

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
        query, key, value = (projection(hidden_states).unflatten(-1, (attn.heads, -1))
                             for projection in (attn.to_q, attn.to_k, attn.to_v))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if freqs_cis is not None:
            def rotary(value):
                with torch.amp.autocast(value.device.type, enabled=False):
                    complex_value = torch.view_as_complex(value.float().reshape(*value.shape[:-1], -1, 2))
                    return torch.view_as_real(complex_value * freqs_cis.unsqueeze(2)).flatten(3).type_as(value)
            query, key = rotary(query), rotary(key)
        output = joint_attention(query, key, value, attention_mask, backend=self._attention_backend)
        output = attn.to_out[0](output.flatten(2, 3).to(query.dtype))
        return attn.to_out[1](output) if len(attn.to_out) > 1 else output


def install_attention(model, backend):
    from .vendor.diffusion.transformer import SingleStreamAttentionProcessor
    from diffusers.models.attention_processor import Attention
    backend = normalize_backend(validate_backend(backend))
    if backend == "sla":
        raise ValueError("Ming does not implement SLA attention")
    count = 0
    for module in model.modules():
        if isinstance(module, Attention) and isinstance(module.processor, (SingleStreamAttentionProcessor, MingAttentionProcessor)):
            if getattr(module.processor, "_parallel_config", None) is not None:
                raise ValueError("Ming conduit does not implement distributed parallel attention")
            module.set_processor(MingAttentionProcessor(backend))
            count += 1
    if not count:
        raise ValueError("Ming native DiT attention processors were not found")
    model._ming_attention_backend = backend
    return count
