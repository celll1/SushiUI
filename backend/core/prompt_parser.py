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


def apply_emphasis_to_embeds(prompt: str, prompt_embeds, tokenizer, device, dtype):
    """
    Apply A1111-style emphasis weights to already-encoded prompt embeddings.

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

        # Apply weight to the NEW tokens
        start_idx = previous_token_count + 1  # +1 to skip BOS token
        end_idx = min(current_token_count + 1, tokenizer.model_max_length)
        token_weights[start_idx:end_idx] = weight

        previous_token_count = current_token_count

    # Apply weights to embeddings
    # Shape: (1, 77, dim) * (1, 77, 1) = (1, 77, dim)
    weighted_embeds = prompt_embeds * token_weights.unsqueeze(0).unsqueeze(-1)

    return weighted_embeds.to(dtype=dtype)
