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


def get_weighted_prompt_embeds(prompt: str, tokenizer, text_encoder, device, dtype, return_pooled=False):
    """
    Generate weighted prompt embeddings using A1111-style emphasis.

    Following A1111's implementation:
    1. Parse prompt to get text fragments and their weights
    2. Encode the full prompt normally with CLIP
    3. Apply weights to the encoded embeddings (AFTER encoding, not before)

    Args:
        return_pooled: If True, also return pooled embeddings (needed for SDXL)

    Returns:
        If return_pooled=False: prompt_embeds
        If return_pooled=True: (prompt_embeds, pooled_prompt_embeds)
    """
    import torch

    # Parse prompt into weighted fragments
    parsed = parse_prompt_attention(prompt)

    print(f"[Prompt Weighting] Parsed prompt: {parsed}")

    if len(parsed) == 0:
        parsed = [("", 1.0)]

    # Reconstruct the full text without emphasis syntax for tokenization
    full_text = "".join([text for text, _ in parsed])

    print(f"[Prompt Weighting] Full text for tokenization: {full_text}")

    # Tokenize the entire prompt
    text_inputs = tokenizer(
        full_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    # First, encode the prompt normally through CLIP
    # IMPORTANT: SDXL uses the penultimate (second-to-last) hidden state, not the last one!
    with torch.no_grad():
        encoder_output = text_encoder(text_input_ids, output_hidden_states=True)

        print(f"[DEBUG] encoder_output type: {type(encoder_output)}")
        print(f"[DEBUG] encoder_output has hidden_states: {hasattr(encoder_output, 'hidden_states')}")

        # Use penultimate hidden state for SDXL (second-to-last layer)
        if hasattr(encoder_output, 'hidden_states') and encoder_output.hidden_states:
            # hidden_states[-1] is last layer, hidden_states[-2] is penultimate
            prompt_embeds = encoder_output.hidden_states[-2]  # Shape: (1, 77, 768) or (1, 77, 1280)
            print(f"[DEBUG] Using hidden_states[-2] (penultimate layer)")
        elif hasattr(encoder_output, 'last_hidden_state'):
            prompt_embeds = encoder_output.last_hidden_state
            print(f"[DEBUG] Using last_hidden_state (fallback)")
        else:
            prompt_embeds = encoder_output[0]
            print(f"[DEBUG] Using encoder_output[0] (fallback)")

        if return_pooled:
            # For SDXL, get pooled output from the penultimate layer
            # Pooled output is the hidden state at the EOS token position
            # Since we're using hidden_states[-2], we need to extract pooled from there

            # Find EOS token position (last non-padding token)
            # EOS token is typically where the sequence ends
            eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.vocab.get('</w>', None)

            if eos_token_id is not None:
                # Find where EOS token is in the input_ids
                eos_positions = (text_input_ids == eos_token_id).nonzero(as_tuple=True)
                if len(eos_positions[1]) > 0:
                    # Use the hidden state at EOS position from penultimate layer
                    eos_idx = eos_positions[1][0]
                    pooled_prompt_embeds = prompt_embeds[:, eos_idx, :]
                    print(f"[DEBUG] Extracted pooled from hidden_states[-2] at EOS position {eos_idx}")
                else:
                    # Fallback: use last token position
                    pooled_prompt_embeds = prompt_embeds[:, -1, :]
                    print(f"[DEBUG] Extracted pooled from hidden_states[-2] at last position")
            else:
                # Fallback: try to use pooler_output or text_embeds
                if hasattr(encoder_output, 'pooler_output') and encoder_output.pooler_output is not None:
                    pooled_prompt_embeds = encoder_output.pooler_output
                    print(f"[DEBUG] Using pooler_output (fallback)")
                elif hasattr(encoder_output, 'text_embeds') and encoder_output.text_embeds is not None:
                    pooled_prompt_embeds = encoder_output.text_embeds
                    print(f"[DEBUG] Using text_embeds (fallback)")
                else:
                    print(f"[WARNING] No pooler_output or text_embeds found")
                    raise ValueError("Cannot find pooled embeddings")

    print(f"[Prompt Weighting] Encoded prompt shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")

    # Build token weight multipliers using A1111's approach:
    # 1. For each fragment, tokenize it in context with what comes before
    # 2. This ensures token boundaries match the actual full tokenization
    token_weights = torch.ones(tokenizer.model_max_length, dtype=dtype, device=device)

    # Build the multiplier array by tokenizing progressively
    current_text = ""
    previous_token_count = 0

    for text, weight in parsed:
        if not text:
            continue

        # Add this fragment to accumulated text
        current_text += text

        # Tokenize the accumulated text so far
        current_tokens = tokenizer(
            current_text,
            add_special_tokens=False,  # Don't add BOS/EOS yet
            return_tensors="pt",
        )
        current_token_count = current_tokens.input_ids.shape[1]

        # Apply weight to the NEW tokens (from previous_token_count to current_token_count)
        # Add 1 to skip BOS token position
        start_idx = previous_token_count + 1
        end_idx = min(current_token_count + 1, tokenizer.model_max_length)
        token_weights[start_idx:end_idx] = weight

        print(f"[Prompt Weighting] Fragment '{text[:20]}...' -> tokens {start_idx}:{end_idx}, weight {weight}")

        previous_token_count = current_token_count

    print(f"[Prompt Weighting] Token weights (first 20): {token_weights[:20]}")

    # Apply weights to the ENCODED embeddings (A1111's EmphasisOriginalNoNorm approach)
    # For SDXL, normalization is NOT performed (unlike SD1.5)
    # This follows A1111's EmphasisOriginalNoNorm class which is recommended for SDXL

    # Multiply each token's embedding vector by its weight
    # Shape: (1, 77, 768) * (1, 77, 1) = (1, 77, 768)
    weighted_prompt_embeds = prompt_embeds * token_weights.unsqueeze(0).unsqueeze(-1)

    print(f"[Prompt Weighting] Applied weights to encoded embeddings (no normalization for SDXL)")
    print(f"[DEBUG] Original embeds mean: {prompt_embeds.mean():.6f}, std: {prompt_embeds.std():.6f}")
    print(f"[DEBUG] Weighted embeds mean: {weighted_prompt_embeds.mean():.6f}, std: {weighted_prompt_embeds.std():.6f}")
    print(f"[DEBUG] Are they equal? {torch.allclose(prompt_embeds, weighted_prompt_embeds)}")

    if return_pooled:
        return weighted_prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)

    return weighted_prompt_embeds.to(dtype=dtype)


def apply_emphasis_to_embeds(prompt: str, prompt_embeds, tokenizer, device, dtype):
    """
    Apply A1111-style emphasis weights to already-encoded prompt embeddings.

    This function takes embeddings that were already encoded by the pipeline
    and applies token-level weights to them.

    Args:
        prompt: Original prompt with emphasis syntax
        prompt_embeds: Already encoded embeddings (shape: [1, 77, dim])
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

    # Reconstruct the full text without emphasis syntax
    full_text = "".join([text for text, _ in parsed])

    # Build token weight multipliers
    token_weights = torch.ones(tokenizer.model_max_length, dtype=dtype, device=device)

    # Build the multiplier array by tokenizing progressively
    current_text = ""
    previous_token_count = 0

    for text, weight in parsed:
        if not text:
            continue

        # Add this fragment to accumulated text
        current_text += text

        # Tokenize the accumulated text so far
        current_tokens = tokenizer(
            current_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        current_token_count = current_tokens.input_ids.shape[1]

        # Apply weight to the NEW tokens
        start_idx = previous_token_count + 1  # +1 to skip BOS token
        end_idx = min(current_token_count + 1, tokenizer.model_max_length)
        token_weights[start_idx:end_idx] = weight

        previous_token_count = current_token_count

    # Apply weights to embeddings
    # Shape: (1, 77, dim) * (1, 77, 1) = (1, 77, dim)
    weighted_embeds = prompt_embeds * token_weights.unsqueeze(0).unsqueeze(-1)

    return weighted_embeds.to(dtype=dtype)
