"""
Prompt parser for A1111-style emphasis syntax.

Syntax:
- (text) - increases attention by 1.1x
- ((text)) - increases attention by 1.1^2 = 1.21x
- (text:1.5) - increases attention by 1.5x
- [text] - decreases attention by 1/1.1 = 0.909x
- \(text\) - literal parentheses (escaped)
"""

import re
from typing import List, Tuple

# Regex pattern for parsing attention syntax
re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\\]|
\\\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
\]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str) -> List[Tuple[str, float]]:
    """
    Parse prompt with A1111-style attention/emphasis syntax.

    Args:
        text: Prompt text with emphasis syntax like (word:1.2) or ((word))

    Returns:
        List of tuples (text_fragment, weight)

    Examples:
        >>> parse_prompt_attention("a (cat:1.2) and dog")
        [('a ', 1.0), ('cat', 1.2), (' and dog', 1.0)]

        >>> parse_prompt_attention("a ((cat)) and [dog]")
        [('a ', 1.0), ('cat', 1.21), (' and ', 1.0), ('dog', 0.909)]
    """
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        """Multiply weights of all fragments from start_position onwards"""
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text_match = m.group(0)
        weight = m.group(1)

        if text_match.startswith('\\'):
            # Escaped character - add literal
            res.append([text_match[1:], 1.0])
        elif text_match == '(':
            # Start of round bracket emphasis
            round_brackets.append(len(res))
        elif text_match == '[':
            # Start of square bracket de-emphasis
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            # Explicit weight like (text:1.5)
            multiply_range(round_brackets.pop(), float(weight))
        elif text_match == ')' and round_brackets:
            # End of round bracket - apply 1.1x multiplier
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text_match == ']' and square_brackets:
            # End of square bracket - apply 1/1.1x multiplier
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            # Regular text or BREAK keyword
            parts = re.split(re_break, text_match)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                if part:
                    res.append([part, 1.0])

    # Close any unclosed brackets with their respective multipliers
    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    # Convert to list of tuples and filter out empty strings
    return [(text, weight) for text, weight in res if text and text != "BREAK"]


def prompt_has_negative_weight(prompt: str, threshold: float = 0.0) -> bool:
    """True if any fragment of the prompt carries a weight below ``threshold``.

    A negative weight (e.g. ``(worst quality:-1)``) is the trigger for NegPip:
    the token's attention value is negated so the concept is subtracted rather
    than added. Plain emphasis scaling cannot represent that.
    """
    if not prompt:
        return False
    try:
        parsed = parse_prompt_attention(prompt)
    except Exception:
        return False
    return any(weight < threshold for _, weight in parsed)


def build_signed_weight_vector(prompt: str, embed_seq_len: int, tokenizer, device, dtype):
    """Build the per-token (embedding-position-aligned) signed weight vector.

    Returns a 1-D tensor of length ``embed_seq_len`` with 1.0 on BOS/EOS/padding
    and the parsed (possibly negative) emphasis weight on each content token,
    using the same chunk-aware position mapping as :func:`apply_emphasis_to_embeds`.
    Used by NegPip to scale V per token instead of scaling the embedding.
    """
    import torch

    token_weights = torch.ones(embed_seq_len, dtype=dtype, device=device)
    parsed = parse_prompt_attention(prompt) if prompt else []
    if len(parsed) == 0:
        return token_weights

    current_text = ""
    previous_token_count = 0
    for text, weight in parsed:
        if not text:
            continue
        current_text += text
        current_tokens = tokenizer(
            current_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        current_token_count = current_tokens.input_ids.shape[1]
        for token_idx in range(previous_token_count, current_token_count):
            chunk_idx = token_idx // 75
            token_in_chunk = token_idx % 75
            embed_pos = chunk_idx * 77 + 1 + token_in_chunk
            if embed_pos < embed_seq_len:
                token_weights[embed_pos] = weight
        previous_token_count = current_token_count

    return token_weights


def _flat_content_weights(prompt, tokenizer):
    """Per-content-token signed weights in tokenization order (no BOS/EOS/padding)."""
    parsed = parse_prompt_attention(prompt) if prompt else []
    weights = []
    current = ""
    prev = 0
    for text, w in parsed:
        if not text:
            continue
        current += text
        cnt = tokenizer(current, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
        weights.extend([w] * (cnt - prev))
        prev = cnt
    return weights


def build_signed_weight_vector_chunked(prompt, embed_seq_len, tokenizer, device, dtype,
                                       mode="a1111", max_chunks=0):
    """Signed per-token weight vector aligned to the encoder's chunk concatenation.

    Mirrors how _encode_prompt_chunked / _encode_prompt_nobos_single_chunk lay out the
    embedding for each chunking ``mode`` (a1111 keeps [BOS,75,EOS] per chunk; sd_scripts
    strips the inter-chunk BOS/EOS; nobos strips every BOS/EOS), so a negative-weighted
    token's V scale lands on exactly its embedding position regardless of mode. BOS/EOS/
    padding stay 1.0. The result is padded (1.0) / truncated to ``embed_seq_len``.
    """
    import torch

    out = torch.ones(embed_seq_len, dtype=dtype, device=device)
    cw = _flat_content_weights(prompt, tokenizer)
    n = len(cw)
    if n == 0:
        return out

    # Single chunk (<=75 content tokens)
    if n <= 75:
        if mode == "nobos":
            # BOS + last token stripped: content starts at position 0
            for j, w in enumerate(cw):
                if j < embed_seq_len:
                    out[j] = w
        else:
            # [BOS, content, EOS, pad]: content at position 1+j
            for j, w in enumerate(cw):
                if 1 + j < embed_seq_len:
                    out[1 + j] = w
        return out

    # Multi-chunk: build a full [77] weight vector per chunk, then slice/concat per mode
    chunks = [cw[i:i + 75] for i in range(0, n, 75)]
    if max_chunks > 0 and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
    chunk_vecs = []
    for c in chunks:
        v = torch.ones(77, dtype=dtype, device=device)
        for j, w in enumerate(c):
            v[1 + j] = w  # content at 1..len; BOS(0)/EOS/pad stay 1.0
        chunk_vecs.append(v)

    if mode == "sd_scripts":
        parts = []
        last = len(chunk_vecs) - 1
        for idx, v in enumerate(chunk_vecs):
            if len(chunk_vecs) == 1:
                parts.append(v)
            elif idx == 0:
                parts.append(v[:-1])      # drop EOS
            elif idx == last:
                parts.append(v[1:])       # drop BOS
            else:
                parts.append(v[1:-1])     # drop BOS and EOS
        concat = torch.cat(parts)
    elif mode == "nobos":
        concat = torch.cat([v[1:-1] for v in chunk_vecs])
    else:  # a1111
        concat = torch.cat(chunk_vecs)

    if concat.shape[0] < embed_seq_len:
        pad = torch.ones(embed_seq_len - concat.shape[0], dtype=dtype, device=device)
        concat = torch.cat([concat, pad])
    elif concat.shape[0] > embed_seq_len:
        concat = concat[:embed_seq_len]
    return concat


def apply_emphasis_to_embeds(prompt: str, prompt_embeds, tokenizer, device, dtype):
    """
    Apply A1111-style emphasis weights to already-encoded prompt embeddings.
    Supports both single-chunk (77 tokens) and multi-chunk embeddings (77*N tokens).

    Args:
        prompt: Original prompt with emphasis syntax
        prompt_embeds: Already encoded embeddings (shape: [1, seq_len, dim])
        tokenizer: Tokenizer used for the prompt
        device: torch device
        dtype: torch dtype

    Returns:
        Weighted embeddings with same shape as input
    """
    import torch

    # Parse prompt into weighted fragments
    parsed = parse_prompt_attention(prompt)

    if len(parsed) == 0:
        return prompt_embeds

    # Get actual embeddings length (may be 77, 154, 231, etc. for chunked prompts)
    embed_seq_len = prompt_embeds.size(1)

    # Build token weight multipliers for the actual embeddings length
    # (shared chunk-aware mapping with NegPip's signed-weight vector)
    token_weights = build_signed_weight_vector(prompt, embed_seq_len, tokenizer, device, dtype)

    # Apply weights to embeddings
    weighted_embeds = prompt_embeds * token_weights.unsqueeze(0).unsqueeze(-1)

    return weighted_embeds.to(dtype=dtype)
