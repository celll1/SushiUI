"""Spatial resampling and a caption decoder with request-local KV caches."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .concepts import ConceptBatch
from .features import CaptionFeatures


@dataclass(frozen=True)
class CaptionConfig:
    spatial_dim: int
    global_dim: int
    num_tags: int
    vocab_size: int
    hidden_dim: int
    heads: int
    layers: int
    ffn_dim: int
    visual_queries: int
    max_length: int
    dropout: float
    num_categories: int
    pad_id: int
    bos_id: int
    eos_id: int

    def __post_init__(self):
        dimensions = (self.spatial_dim, self.global_dim, self.num_tags, self.vocab_size,
                      self.hidden_dim, self.heads, self.layers, self.ffn_dim,
                      self.visual_queries, self.max_length, self.num_categories)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("Caption dimensions must be positive integers")
        if self.hidden_dim % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Invalid attention dimensions or dropout")
        special = (self.pad_id, self.bos_id, self.eos_id)
        if len(set(special)) != 3 or any(type(i) is not int or not 0 <= i < self.vocab_size for i in special):
            raise ValueError("PAD, BOS, EOS must be distinct vocabulary indices")


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.hidden_dim // config.heads
        self.q = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = config.dropout

    def split(self, value):
        return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def project_kv(self, value):
        return self.split(self.k(value)), self.split(self.v(value))

    def attend(self, value, kv, mask=None, causal=False):
        result = F.scaled_dot_product_attention(
            self.split(self.q(value)), *kv, attn_mask=mask, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(result.transpose(1, 2).contiguous().flatten(2))


class _DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = _Attention(config)
        self.cross_attn = _Attention(config)
        self.norms = nn.ModuleList(nn.LayerNorm(config.hidden_dim) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(config.hidden_dim, config.ffn_dim), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)

    def full(self, value, memory, memory_mask, token_mask):
        normalized = self.norms[0](value)
        value = value + self.dropout(self.self_attn.attend(
            normalized, self.self_attn.project_kv(normalized), token_mask,
        ))
        value = value + self.dropout(self.cross_attn.attend(
            self.norms[1](value), self.cross_attn.project_kv(memory), memory_mask,
        ))
        return value + self.dropout(self.ffn(self.norms[2](value)))

    def step(self, value, previous, memory_kv, memory_mask):
        normalized = self.norms[0](value)
        current = self.self_attn.project_kv(normalized)
        kv = current if previous is None else tuple(torch.cat((old, new), dim=2) for old, new in zip(previous, current))
        # A one-token query may see all cached keys; causal=True would mask the past incorrectly.
        value = value + self.self_attn.attend(normalized, kv)
        value = value + self.cross_attn.attend(self.norms[1](value), memory_kv, memory_mask)
        return value + self.ffn(self.norms[2](value)), kv


@dataclass(frozen=True)
class GenerationResult:
    token_ids: torch.Tensor
    lengths: torch.Tensor
    reached_eos: torch.Tensor


class CompactCaptioner(nn.Module):
    def __init__(self, config: CaptionConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.spatial_projection = nn.Linear(config.spatial_dim, dim)
        self.xy_projection = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.queries = nn.Parameter(torch.randn(config.visual_queries, dim) * 0.02)
        self.resampler = nn.MultiheadAttention(dim, config.heads, dropout=config.dropout, batch_first=True)
        self.visual_norm = nn.LayerNorm(dim)
        self.visual_ffn = nn.Sequential(nn.Linear(dim, config.ffn_dim), nn.GELU(),
                                        nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, dim))
        self.global_projection = nn.Linear(config.global_dim, dim)
        self.tag_embedding = nn.Embedding(config.num_tags, dim)
        self.category_embedding = nn.Embedding(config.num_categories, dim)
        self.source_embedding = nn.Embedding(2, dim)
        self.score_projection = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.type_embedding = nn.Embedding(3, dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.word_embedding = nn.Embedding(config.vocab_size, dim)
        self.position_embedding = nn.Embedding(config.max_length, dim)
        self.layers = nn.ModuleList(_DecoderLayer(config) for _ in range(config.layers))
        self.output_norm = nn.LayerNorm(dim)

    def encode_memory(self, features: CaptionFeatures, concepts: ConceptBatch):
        spatial = features.spatial_tokens
        if spatial.ndim != 3 or spatial.shape[0] < 1:
            raise ValueError("Expected a nonempty batch of spatial tokens")
        batch, count = spatial.shape[:2]
        if (spatial.shape[-1] != self.config.spatial_dim
                or features.spatial_mask.shape != (batch, count)
                or features.spatial_xy.shape != (batch, count, 2)
                or features.global_features.shape != (batch, self.config.global_dim)):
            raise ValueError("Caption feature dimensions do not match model config")
        if not features.spatial_mask.any(dim=1).all():
            raise ValueError("Each image must have a valid spatial token")
        if concepts.ids.ndim != 2 or concepts.ids.shape[0] != batch:
            raise ValueError("Concept batch size mismatch")
        if any(field.shape != concepts.ids.shape for field in
               (concepts.scores, concepts.categories, concepts.sources, concepts.mask)):
            raise ValueError("Concept tensor dimensions must agree")
        dtype = self.spatial_projection.weight.dtype
        # Clear padding before arithmetic; even NaN padding must not pollute attention.
        spatial = spatial.masked_fill(~features.spatial_mask.unsqueeze(-1), 0).to(dtype)
        xy = features.spatial_xy.masked_fill(~features.spatial_mask.unsqueeze(-1), 0).to(dtype)
        projected = self.spatial_projection(spatial) + self.xy_projection(xy)
        query = self.queries.unsqueeze(0).expand(batch, -1, -1)
        visual, _ = self.resampler(query, projected, projected,
                                   key_padding_mask=~features.spatial_mask, need_weights=False)
        visual = self.visual_norm(visual + query)
        visual = visual + self.visual_ffn(visual) + self.type_embedding.weight[0]
        global_token = self.global_projection(features.global_features.to(dtype)).unsqueeze(1)
        global_token = global_token + self.type_embedding.weight[1]
        mask = concepts.mask
        tag_ids = concepts.ids.masked_fill(~mask, 0)
        category_ids = concepts.categories.masked_fill(~mask, 0)
        source_ids = concepts.sources.masked_fill(~mask, 0)
        scores = concepts.scores.masked_fill(~mask, 0).to(dtype)
        concept = (self.tag_embedding(tag_ids) + self.category_embedding(category_ids)
                   + self.source_embedding(source_ids) + self.score_projection(scores.unsqueeze(-1))
                   + self.type_embedding.weight[2])
        memory = self.memory_norm(torch.cat((visual, global_token, concept), dim=1))
        visible = torch.ones(batch, visual.shape[1] + 1, dtype=torch.bool, device=memory.device)
        return memory, torch.cat((visible, mask), dim=1)

    def _embed(self, tokens, positions):
        return self.word_embedding(tokens) + self.position_embedding(positions)

    def decode(self, tokens, memory, memory_mask):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.max_length:
            raise ValueError("Decoder input length exceeds model config")
        if tokens.shape[0] != memory.shape[0] or (tokens[:, 0] != self.config.bos_id).any():
            raise ValueError("Every decoder sequence must start with BOS")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool).tril()
        valid = tokens != self.config.pad_id
        token_mask = causal[None, None] & valid[:, None, None, :]
        value = self._embed(tokens, positions)
        for layer in self.layers:
            value = layer.full(value, memory, memory_mask[:, None, None, :], token_mask)
        return F.linear(self.output_norm(value), self.word_embedding.weight)

    def forward(self, features, concepts, input_ids):
        memory, mask = self.encode_memory(features, concepts)
        return self.decode(input_ids, memory, mask)

    def teacher_loss(self, features, concepts, sequences, *, label_smoothing: float):
        """Sequences contain BOS, at least one target token (EOS), and right padding."""
        if sequences.ndim != 2 or not 2 <= sequences.shape[1] <= self.config.max_length + 1:
            raise ValueError("Expected BOS plus target sequence within the model limit")
        if not 0 <= label_smoothing < 1:
            raise ValueError("Invalid label smoothing")
        valid = sequences != self.config.pad_id
        lengths = valid.sum(dim=1)
        prefix = torch.arange(sequences.shape[1], device=sequences.device)[None] < lengths[:, None]
        if not torch.equal(valid, prefix) or (lengths < 2).any():
            raise ValueError("Caption sequences must be nonempty and right padded")
        if (sequences.gather(1, (lengths - 1)[:, None]).squeeze(1) != self.config.eos_id).any():
            raise ValueError("Caption targets must end with EOS")
        if (sequences.eq(self.config.eos_id).sum(dim=1) != 1).any():
            raise ValueError("Caption targets must contain exactly one EOS")
        if (sequences[:, 1:] == self.config.bos_id).any():
            raise ValueError("BOS is only valid at the beginning")
        logits = self(features, concepts, sequences[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1).float(), sequences[:, 1:].flatten(),
                               ignore_index=self.config.pad_id, label_smoothing=label_smoothing)

    @torch.no_grad()
    def generate(self, features, concepts, *, max_new_tokens: int, cancelled=None):
        if self.training:
            raise ValueError("Generation requires model.eval()")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.config.max_length:
            raise ValueError("Generation limit exceeds model config")
        memory, mask = self.encode_memory(features, concepts)
        memory_mask = mask[:, None, None, :]
        cross_kv = [layer.cross_attn.project_kv(memory) for layer in self.layers]
        caches = [None] * len(self.layers)
        batch = memory.shape[0]
        token = torch.full((batch, 1), self.config.bos_id, device=memory.device, dtype=torch.long)
        done = torch.zeros(batch, device=memory.device, dtype=torch.bool)
        lengths = torch.zeros(batch, device=memory.device, dtype=torch.long)
        output = []
        for position in range(max_new_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError("Caption generation cancelled")
            value = self._embed(token, torch.tensor([position], device=memory.device))
            for index, layer in enumerate(self.layers):
                value, caches[index] = layer.step(value, caches[index], cross_kv[index], memory_mask)
            logits = F.linear(self.output_norm(value[:, 0]), self.word_embedding.weight)
            logits[:, self.config.pad_id] = -torch.inf
            logits[:, self.config.bos_id] = -torch.inf
            next_token = logits.argmax(dim=-1).masked_fill(done, self.config.pad_id)
            lengths = lengths + (~done).long()
            done = done | next_token.eq(self.config.eos_id)
            output.append(next_token)
            token = next_token[:, None]
            if done.all():
                break
        return GenerationResult(torch.stack(output, dim=1), lengths, done)
