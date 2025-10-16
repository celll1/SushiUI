"""
Prompt chunking for handling prompts longer than 75 tokens.

Supports different chunking modes:
- A1111 (Automatic1111): <BOS>75tokens<EOS><BOS>75tokens<EOS>...
- sd-scripts: <BOS>75tokens 75tokens 75tokens...<EOS>
- NoBOS: 75tokens 75tokens 75tokens (no BOS/EOS tokens)

Also supports A1111-style emphasis syntax: (word:1.2), ((word)), [word]
"""

import torch
from typing import Tuple, Literal, Optional, List

ChunkingMode = Literal["a1111", "sd_scripts", "nobos"]


def encode_prompt_chunked(
    tokenizer,
    text_encoder,
    prompt: str,
    device: str,
    dtype: torch.dtype,
    chunking_mode: ChunkingMode = "a1111",
    max_length: int = 75,
    emphasis_weights: Optional[torch.Tensor] = None,
    use_penultimate_hidden_state: bool = False,
    max_chunks: int = 0,
) -> torch.Tensor:
    """
    Encode a prompt with support for lengths beyond the model's max token limit.
    Also supports emphasis weights from parse_prompt_attention().

    Args:
        tokenizer: The tokenizer to use
        text_encoder: The text encoder model
        prompt: The prompt text to encode (should be clean text if using emphasis_weights)
        device: Device to place tensors on
        dtype: Data type for the output
        chunking_mode: How to handle chunking
            - "a1111": Each chunk gets its own BOS/EOS tokens (AUTOMATIC1111 style)
            - "sd_scripts": Single BOS/EOS for entire sequence (sd-scripts style)
            - "nobos": No BOS/EOS tokens at all
        max_length: Maximum tokens per chunk (default 75, matching SD's limit)
        emphasis_weights: Optional tensor of per-token weights for emphasis

    Returns:
        Encoded prompt embeddings with shape [1, total_tokens, hidden_dim]
    """
    # Get special token IDs
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id or eos_token_id

    # Tokenize the entire prompt without special tokens first
    full_tokens = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    ).input_ids[0]

    # If prompt fits in single chunk, use standard encoding
    if len(full_tokens) <= max_length:
        # Standard encoding with BOS/EOS
        encoded = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length + 2,  # +2 for BOS/EOS
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)

        with torch.no_grad():
            if use_penultimate_hidden_state:
                # For SDXL text_encoder_2, use penultimate hidden state
                encoder_output = text_encoder(input_ids, output_hidden_states=True)
                embeddings = encoder_output.hidden_states[-2]
            else:
                # Standard: use last hidden state
                embeddings = text_encoder(input_ids)[0]

        # Apply emphasis weights if provided
        if emphasis_weights is not None:
            # Build weight tensor for all positions
            # Position 0: BOS token (weight 1.0)
            # Positions 1 to len(emphasis_weights): actual tokens (use emphasis_weights)
            # Remaining positions: EOS and padding (weight 1.0)
            full_weights = torch.ones(embeddings.size(1), device=device, dtype=dtype)

            # Apply emphasis weights to token positions (skip BOS at position 0)
            num_tokens = min(len(emphasis_weights), embeddings.size(1) - 1)
            if num_tokens > 0:
                full_weights[1:1+num_tokens] = emphasis_weights[:num_tokens].to(dtype=dtype)

            embeddings = embeddings * full_weights.unsqueeze(0).unsqueeze(-1)

        return embeddings.to(dtype=dtype)

    # Split into chunks
    chunk_size = max_length
    chunks = []

    for i in range(0, len(full_tokens), chunk_size):
        chunk = full_tokens[i:i + chunk_size]
        chunks.append(chunk)

    # Limit number of chunks if max_chunks is specified
    if max_chunks > 0 and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    # Encode chunks based on mode
    chunk_embeddings = []

    if chunking_mode == "a1111":
        # AUTOMATIC1111 mode: Each chunk gets BOS/EOS
        # <BOS>chunk1<EOS><BOS>chunk2<EOS>...
        current_token_idx = 0
        for chunk in chunks:
            # Add BOS and EOS to each chunk
            chunk_with_tokens = torch.cat([
                torch.tensor([bos_token_id], device=device),
                chunk.to(device),
                torch.tensor([eos_token_id], device=device),
            ])

            # Pad to max_length + 2
            chunk_len = len(chunk_with_tokens)
            if chunk_len < max_length + 2:
                padding = torch.full(
                    (max_length + 2 - chunk_len,),
                    pad_token_id,
                    device=device
                )
                chunk_with_tokens = torch.cat([chunk_with_tokens, padding])

            # Encode this chunk
            with torch.no_grad():
                if use_penultimate_hidden_state:
                    encoder_output = text_encoder(chunk_with_tokens.unsqueeze(0), output_hidden_states=True)
                    chunk_emb = encoder_output.hidden_states[-2]
                else:
                    chunk_emb = text_encoder(chunk_with_tokens.unsqueeze(0))[0]

            # Apply emphasis weights to this chunk if provided
            if emphasis_weights is not None:
                # Extract weights for this chunk (skip BOS, include tokens, skip EOS/padding)
                chunk_weights = torch.ones(chunk_emb.size(1), device=device, dtype=dtype)
                # BOS token (index 0): weight 1.0
                # Actual tokens (index 1 to len(chunk)+1): use emphasis_weights
                end_idx = min(current_token_idx + len(chunk), len(emphasis_weights))
                token_count = end_idx - current_token_idx
                if token_count > 0:
                    chunk_weights[1:1+token_count] = emphasis_weights[current_token_idx:end_idx].to(dtype=dtype)
                # EOS and padding: weight 1.0

                chunk_emb = chunk_emb * chunk_weights.unsqueeze(0).unsqueeze(-1)

            chunk_embeddings.append(chunk_emb)
            current_token_idx += len(chunk)

        # Concatenate all chunk embeddings
        final_embeddings = torch.cat(chunk_embeddings, dim=1)

    elif chunking_mode == "sd_scripts":
        # sd-scripts mode: Single BOS at start, single EOS at end
        # <BOS>chunk1 chunk2 chunk3...<EOS>

        # Add BOS at the very beginning
        all_tokens = [torch.tensor([bos_token_id], device=device)]

        # Add all chunks
        for chunk in chunks:
            all_tokens.append(chunk.to(device))

        # Add EOS at the very end
        all_tokens.append(torch.tensor([eos_token_id], device=device))

        # Concatenate all tokens
        all_tokens = torch.cat(all_tokens)

        # Now split into chunks for encoding (each chunk max_length+2)
        chunk_embeddings = []
        current_token_idx = 0
        for i in range(0, len(all_tokens), max_length + 2):
            chunk = all_tokens[i:i + max_length + 2]

            # Pad if needed
            chunk_len = len(chunk)
            if chunk_len < max_length + 2:
                padding = torch.full(
                    (max_length + 2 - chunk_len,),
                    pad_token_id,
                    device=device
                )
                chunk = torch.cat([chunk, padding])

            # Encode
            with torch.no_grad():
                if use_penultimate_hidden_state:
                    encoder_output = text_encoder(chunk.unsqueeze(0), output_hidden_states=True)
                    chunk_emb = encoder_output.hidden_states[-2]
                else:
                    chunk_emb = text_encoder(chunk.unsqueeze(0))[0]

            # Apply emphasis weights if provided
            if emphasis_weights is not None:
                chunk_weights = torch.ones(chunk_emb.size(1), device=device, dtype=dtype)
                # For sd-scripts mode, we need to map global token positions
                # First chunk: BOS + tokens
                # Later chunks: just tokens
                token_start = 0 if i == 0 else 1  # Skip BOS in first chunk
                for j in range(token_start, min(chunk_len, chunk_emb.size(1))):
                    global_token_idx = current_token_idx + j - token_start
                    if global_token_idx < len(emphasis_weights):
                        chunk_weights[j] = emphasis_weights[global_token_idx].to(dtype=dtype)

                chunk_emb = chunk_emb * chunk_weights.unsqueeze(0).unsqueeze(-1)
                current_token_idx += chunk_len - token_start

            chunk_embeddings.append(chunk_emb)

        # Concatenate
        final_embeddings = torch.cat(chunk_embeddings, dim=1)

    elif chunking_mode == "nobos":
        # No BOS/EOS mode: Just the tokens
        # chunk1 chunk2 chunk3...

        # Concatenate all chunks
        all_tokens = torch.cat([chunk.to(device) for chunk in chunks])

        # Split into encoding chunks
        chunk_embeddings = []
        current_token_idx = 0
        for i in range(0, len(all_tokens), max_length + 2):
            chunk = all_tokens[i:i + max_length + 2]

            # Pad if needed
            chunk_len = len(chunk)
            if chunk_len < max_length + 2:
                padding = torch.full(
                    (max_length + 2 - chunk_len,),
                    pad_token_id,
                    device=device
                )
                chunk = torch.cat([chunk, padding])

            # Encode
            with torch.no_grad():
                if use_penultimate_hidden_state:
                    encoder_output = text_encoder(chunk.unsqueeze(0), output_hidden_states=True)
                    chunk_emb = encoder_output.hidden_states[-2]
                else:
                    chunk_emb = text_encoder(chunk.unsqueeze(0))[0]

            # Apply emphasis weights if provided
            if emphasis_weights is not None:
                chunk_weights = torch.ones(chunk_emb.size(1), device=device, dtype=dtype)
                for j in range(min(chunk_len, chunk_emb.size(1))):
                    global_token_idx = current_token_idx + j
                    if global_token_idx < len(emphasis_weights):
                        chunk_weights[j] = emphasis_weights[global_token_idx].to(dtype=dtype)

                chunk_emb = chunk_emb * chunk_weights.unsqueeze(0).unsqueeze(-1)

            chunk_embeddings.append(chunk_emb)
            current_token_idx += chunk_len

        # Concatenate
        final_embeddings = torch.cat(chunk_embeddings, dim=1)

    else:
        raise ValueError(f"Unknown chunking mode: {chunking_mode}")

    return final_embeddings.to(dtype=dtype)


def encode_prompt_chunked_sdxl(
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    prompt: str,
    device: str,
    dtype: torch.dtype,
    chunking_mode: ChunkingMode = "a1111",
    max_length: int = 75,
    emphasis_weights: Optional[torch.Tensor] = None,
    emphasis_weights_2: Optional[torch.Tensor] = None,
    max_chunks: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a prompt for SDXL with dual text encoders.

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds)
    """
    # Encode with first text encoder (CLIP ViT-L)
    prompt_embeds_1 = encode_prompt_chunked(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        device=device,
        dtype=dtype,
        chunking_mode=chunking_mode,
        max_length=max_length,
        emphasis_weights=emphasis_weights,
        max_chunks=max_chunks,
    )

    # Encode with second text encoder (OpenCLIP ViT-G)
    # For SDXL, we need both embeddings and pooled output
    full_tokens_2 = tokenizer_2(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    ).input_ids[0]

    if len(full_tokens_2) <= max_length:
        # Single chunk - use standard encoding
        encoded_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_length + 2,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_2 = encoded_2.input_ids.to(device)

        with torch.no_grad():
            text_encoder_output = text_encoder_2(input_ids_2, output_hidden_states=True)
            prompt_embeds_2 = text_encoder_output.hidden_states[-2]
            pooled_prompt_embeds = text_encoder_output[0]

        # Apply emphasis weights if provided
        if emphasis_weights_2 is not None:
            # Build weight tensor for all positions
            # Position 0: BOS token (weight 1.0)
            # Positions 1 to len(emphasis_weights_2): actual tokens (use emphasis_weights_2)
            # Remaining positions: EOS and padding (weight 1.0)
            full_weights = torch.ones(prompt_embeds_2.size(1), device=device, dtype=dtype)

            # Apply emphasis weights to token positions (skip BOS at position 0)
            num_tokens = min(len(emphasis_weights_2), prompt_embeds_2.size(1) - 1)
            if num_tokens > 0:
                full_weights[1:1+num_tokens] = emphasis_weights_2[:num_tokens].to(dtype=dtype)

            prompt_embeds_2 = prompt_embeds_2 * full_weights.unsqueeze(0).unsqueeze(-1)
    else:
        # Multi-chunk encoding for second encoder
        # For SDXL's text_encoder_2, we use hidden_states[-2] for prompt embeddings
        prompt_embeds_2 = encode_prompt_chunked(
            tokenizer=tokenizer_2,
            text_encoder=text_encoder_2,
            prompt=prompt,
            device=device,
            dtype=dtype,
            chunking_mode=chunking_mode,
            max_length=max_length,
            emphasis_weights=emphasis_weights_2,
            use_penultimate_hidden_state=True,  # Use hidden_states[-2] for TE2
            max_chunks=max_chunks,
        )

        # For pooled embeddings, use the FIRST chunk's pooler output [0]
        # SDXL only uses the first 75 tokens for pooling
        first_chunk_tokens = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_length + 2,
            truncation=True,  # This truncates to first 75 tokens
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            pooled_prompt_embeds = text_encoder_2(first_chunk_tokens)[0]

    # Concatenate embeddings from both encoders
    # SDXL expects concatenated embeddings: [TE1_embeds, TE2_embeds]
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    return prompt_embeds, pooled_prompt_embeds
