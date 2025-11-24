# Copyright (c) UnieAI. All rights reserved.
"""Bad words ban kernel optimized with Triton.

This module implements blocking of unwanted token sequences during generation:
- Single-token banning
- Multi-token sequence banning
- Per-batch or shared bad word lists
- Beam search support

Based on TurboMind's implementation but using Triton for cross-platform support.
"""
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _ban_bad_words_kernel(
    logits_ptr,
    output_ids_ptr,
    parent_ids_ptr,
    bad_words_ptr,
    bad_words_offsets_ptr,
    batch_size,
    beam_width,
    vocab_size_padded,
    bad_words_len,
    step,
    share_words: tl.constexpr,
    has_beam_search: tl.constexpr,
):
    """Ban bad words by setting their logits to -inf.

    Bad words are stored as a flat list with offsets indicating boundaries.
    Format:
        bad_words: [token1_1, token1_2, ..., token2_1, ...]
        bad_words_offsets: [end_idx_of_word1, end_idx_of_word2, ...]

    Single-token bad words are always banned.
    Multi-token bad words are checked against generation history.

    Args:
        logits_ptr: Logits [batch_size * beam_width, vocab_size_padded]
        output_ids_ptr: Generated tokens [step, batch_size * beam_width]
        parent_ids_ptr: Parent beam indices [step, batch_size * beam_width] (for beam search)
        bad_words_ptr: Bad word tokens [num_bad_words * max_word_len] or [batch_size, ...]
        bad_words_offsets_ptr: End indices for each bad word [num_bad_words] or [batch_size, ...]
        batch_size: Batch size
        beam_width: Beam width
        vocab_size_padded: Padded vocabulary size
        bad_words_len: Number of bad words
        step: Current generation step
        share_words: If True, all batches share the same bad words list
        has_beam_search: If True, beam_width > 1
    """
    bad_word_idx = tl.program_id(0)
    batch_beam_idx = tl.program_id(1)

    if bad_word_idx >= bad_words_len:
        return

    batch_idx = batch_beam_idx // beam_width
    beam_idx = batch_beam_idx % beam_width

    # Get bad words for this batch
    if share_words:
        base_bad_words_offset = 0
        base_offsets_offset = 0
    else:
        base_bad_words_offset = batch_idx * 2 * bad_words_len
        base_offsets_offset = batch_idx * 2 * bad_words_len + bad_words_len

    # Load offset for this bad word
    offset_idx = base_offsets_offset + bad_word_idx
    item_end = tl.load(bad_words_offsets_ptr + offset_idx)

    # Skip if invalid
    if item_end < 0:
        return

    # Calculate item boundaries
    if bad_word_idx > 0:
        item_start = tl.load(bad_words_offsets_ptr + offset_idx - 1)
    else:
        item_start = 0

    item_size = item_end - item_start

    # Single-token case: always ban
    should_ban = item_size == 1

    # Multi-token case: check history
    if item_size > 1 and step >= item_size - 1:
        should_ban = tl.constexpr(True)
        parent_id = beam_idx

        # Check each previous token in the sequence
        for token_idx in range(item_size - 2, -1, -1):
            # Calculate step index
            history_step = step - (item_size - 1) + token_idx
            history_idx = history_step * batch_size * beam_width + batch_idx * beam_width + parent_id

            # Load previous token
            previous_token = tl.load(output_ids_ptr + history_idx)

            # Load expected bad word token
            expected_token = tl.load(bad_words_ptr + base_bad_words_offset + item_start + token_idx)

            # Check if it matches
            if previous_token != expected_token:
                should_ban = tl.constexpr(False)
                break

            # Update parent for beam search
            if has_beam_search:
                parent_idx = history_step * batch_size * beam_width + batch_idx * beam_width + parent_id
                parent_id = tl.load(parent_ids_ptr + parent_idx)

                if parent_id < 0 or parent_id >= beam_width:
                    should_ban = tl.constexpr(False)
                    break

    # Ban the token if conditions are met
    if should_ban:
        banned_token = tl.load(bad_words_ptr + base_bad_words_offset + item_end - 1)

        if 0 < banned_token and banned_token < vocab_size_padded:
            logit_idx = batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token
            tl.store(logits_ptr + logit_idx, float('-inf'))


def ban_bad_words(
    logits: torch.Tensor,
    output_ids: torch.Tensor,
    bad_words: torch.Tensor,
    bad_words_offsets: torch.Tensor,
    step: int,
    parent_ids: Optional[torch.Tensor] = None,
    beam_width: int = 1,
    share_words: bool = True,
) -> None:
    """Ban bad words by setting their logits to -infinity (in-place).

    Args:
        logits: Logits tensor [batch_size * beam_width, vocab_size_padded]
        output_ids: Generated token IDs [step, batch_size * beam_width]
        bad_words: Bad word tokens. If share_words=True: [bad_words_len, max_word_len],
                   else: [batch_size, bad_words_len, max_word_len]
        bad_words_offsets: End indices for each bad word. If share_words=True: [bad_words_len],
                          else: [batch_size, bad_words_len]
        step: Current generation step
        parent_ids: Parent beam indices [step, batch_size * beam_width] (for beam search)
        beam_width: Beam width (default: 1)
        share_words: If True, all batches share the same bad words list (default: True)
    """
    batch_beam_size, vocab_size_padded = logits.shape
    batch_size = batch_beam_size // beam_width

    if share_words:
        bad_words_len = bad_words_offsets.numel()
    else:
        bad_words_len = bad_words_offsets.shape[1]

    has_beam_search = beam_width > 1

    # Flatten bad_words for easier indexing
    bad_words_flat = bad_words.reshape(-1)
    bad_words_offsets_flat = bad_words_offsets.reshape(-1)

    # Launch kernel
    grid = (bad_words_len, batch_size * beam_width)

    _ban_bad_words_kernel[grid](
        logits,
        output_ids,
        parent_ids if parent_ids is not None else output_ids,  # Dummy pointer
        bad_words_flat,
        bad_words_offsets_flat,
        batch_size,
        beam_width,
        vocab_size_padded,
        bad_words_len,
        step,
        share_words,
        has_beam_search,
    )


@triton.jit
def _ban_tokens_kernel(
    logits_ptr,
    banned_tokens_ptr,
    num_banned_tokens,
    batch_size,
    vocab_size_padded,
):
    """Simple kernel to ban specific tokens (set logits to -inf).

    Args:
        logits_ptr: Logits [batch_size, vocab_size_padded]
        banned_tokens_ptr: List of token IDs to ban [num_banned_tokens]
        num_banned_tokens: Number of banned tokens
        batch_size: Batch size
        vocab_size_padded: Padded vocabulary size
    """
    batch_idx = tl.program_id(0)
    token_idx = tl.program_id(1)

    if batch_idx >= batch_size or token_idx >= num_banned_tokens:
        return

    # Load banned token ID
    banned_token = tl.load(banned_tokens_ptr + token_idx)

    if 0 <= banned_token and banned_token < vocab_size_padded:
        logit_idx = batch_idx * vocab_size_padded + banned_token
        tl.store(logits_ptr + logit_idx, float('-inf'))


def ban_tokens(
    logits: torch.Tensor,
    banned_tokens: torch.Tensor,
) -> None:
    """Ban specific tokens by setting their logits to -infinity (in-place).

    This is a simplified version for banning single tokens without history checking.

    Args:
        logits: Logits tensor [batch_size, vocab_size_padded]
        banned_tokens: Token IDs to ban [num_banned_tokens]
    """
    batch_size, vocab_size_padded = logits.shape
    num_banned_tokens = banned_tokens.numel()

    # Launch kernel
    grid = (batch_size, num_banned_tokens)

    _ban_tokens_kernel[grid](
        logits,
        banned_tokens,
        num_banned_tokens,
        batch_size,
        vocab_size_padded,
    )
