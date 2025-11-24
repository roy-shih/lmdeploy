# Copyright (c) UnieAI. All rights reserved.
"""Stop criteria kernels optimized with Triton.

This module implements stopping criteria for text generation:
- Stop words/sequences detection
- Maximum length criterion
- Combined stopping conditions

Based on TurboMind's implementation but using Triton for cross-platform support.
"""
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _stop_words_criterion_kernel(
    output_ids_ptr,
    parent_ids_ptr,
    stop_words_ptr,
    stop_words_offsets_ptr,
    finished_ptr,
    batch_size,
    beam_width,
    stop_words_len,
    step,
    has_beam_search: tl.constexpr,
):
    """Check if generated sequence matches any stop word/sequence.

    Stop words are stored as a flat list with offsets indicating boundaries.
    Format:
        stop_words: [word1_token1, word1_token2, ..., word2_token1, ...]
        stop_words_offsets: [end_idx_of_word1, end_idx_of_word2, ...]

    Args:
        output_ids_ptr: Generated tokens [step, batch_size * beam_width]
        parent_ids_ptr: Parent beam indices [step, batch_size * beam_width] (for beam search)
        stop_words_ptr: Stop word tokens [batch_size, stop_words_len, max_word_len]
        stop_words_offsets_ptr: End indices for each stop word [batch_size, stop_words_len]
        finished_ptr: Finished flags [batch_size * beam_width]
        batch_size: Batch size
        beam_width: Beam width
        stop_words_len: Number of stop words
        step: Current generation step
        has_beam_search: If True, beam_width > 1
    """
    stop_word_idx = tl.program_id(0)
    batch_beam_idx = tl.program_id(1)

    if stop_word_idx >= stop_words_len:
        return

    batch_idx = batch_beam_idx // beam_width
    beam_idx = batch_beam_idx % beam_width

    # Get stop words for this batch
    base_stop_words_offset = batch_idx * 2 * stop_words_len
    base_offsets_offset = batch_idx * 2 * stop_words_len + stop_words_len

    # Load offset for this stop word
    offset_idx = base_offsets_offset + stop_word_idx
    item_end = tl.load(stop_words_offsets_ptr + offset_idx)

    # Skip if invalid
    if item_end < 0:
        return

    # Calculate item boundaries
    if stop_word_idx > 0:
        item_start = tl.load(stop_words_offsets_ptr + offset_idx - 1)
    else:
        item_start = 0

    item_size = item_end - item_start

    # Check if we have enough generated tokens
    if step + 1 < item_size:
        return

    # Check if sequence matches stop word
    should_stop = tl.constexpr(True)
    parent_id = beam_idx

    # Check each token in the stop word
    for token_idx in range(item_size - 1, -1, -1):
        # Calculate step index
        history_step = step - (item_size - 1) + token_idx
        history_idx = history_step * batch_size * beam_width + batch_idx * beam_width + parent_id

        # Load previous token
        previous_token = tl.load(output_ids_ptr + history_idx)

        # Load expected stop word token
        expected_token = tl.load(stop_words_ptr + base_stop_words_offset + item_start + token_idx)

        # Check if it matches
        if previous_token != expected_token:
            should_stop = tl.constexpr(False)
            break

        # Update parent for beam search
        if has_beam_search:
            parent_idx = history_step * batch_size * beam_width + batch_idx * beam_width + parent_id
            parent_id = tl.load(parent_ids_ptr + parent_idx)

            if parent_id < 0 or parent_id >= beam_width:
                should_stop = tl.constexpr(False)
                break

    # Mark as finished if stop word matched
    if should_stop:
        finished_idx = batch_idx * beam_width + beam_idx
        tl.store(finished_ptr + finished_idx, tl.constexpr(True))


def check_stop_words(
    output_ids: torch.Tensor,
    stop_words: torch.Tensor,
    stop_words_offsets: torch.Tensor,
    finished: torch.Tensor,
    step: int,
    parent_ids: Optional[torch.Tensor] = None,
    beam_width: int = 1,
) -> None:
    """Check if generated sequences match any stop words (in-place update of finished flags).

    Args:
        output_ids: Generated token IDs [step, batch_size * beam_width]
        stop_words: Stop word tokens [batch_size, stop_words_len, max_word_len]
        stop_words_offsets: End indices for each stop word [batch_size, stop_words_len]
        finished: Finished flags [batch_size * beam_width]
        step: Current generation step
        parent_ids: Parent beam indices [step, batch_size * beam_width] (for beam search)
        beam_width: Beam width (default: 1)
    """
    batch_beam_size = finished.numel()
    batch_size = batch_beam_size // beam_width
    stop_words_len = stop_words_offsets.shape[1]

    has_beam_search = beam_width > 1

    # Flatten tensors for easier indexing
    stop_words_flat = stop_words.reshape(-1)
    stop_words_offsets_flat = stop_words_offsets.reshape(-1)

    # Launch kernel
    grid = (stop_words_len, batch_size * beam_width)

    _stop_words_criterion_kernel[grid](
        output_ids,
        parent_ids if parent_ids is not None else output_ids,  # Dummy pointer
        stop_words_flat,
        stop_words_offsets_flat,
        finished,
        batch_size,
        beam_width,
        stop_words_len,
        step,
        has_beam_search,
    )


@triton.jit
def _length_criterion_kernel(
    finished_ptr,
    sequence_limit_length_ptr,
    batch_size,
    beam_width,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    """Check if sequences have reached maximum length.

    Args:
        finished_ptr: Finished flags [batch_size * beam_width]
        sequence_limit_length_ptr: Maximum lengths [batch_size]
        batch_size: Batch size
        beam_width: Beam width
        step: Current generation step
        BLOCK_SIZE: Block size
    """
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < batch_size * beam_width

    # Get batch index for each sequence
    batch_idx = idx // beam_width

    # Load maximum length for this batch
    max_length = tl.load(sequence_limit_length_ptr + batch_idx, mask=mask, other=0)

    # Check if step has reached or exceeded max length
    reached_limit = step >= max_length

    # Load current finished flag
    current_finished = tl.load(finished_ptr + idx, mask=mask, other=False)

    # Update finished flag (OR with length criterion)
    new_finished = current_finished | reached_limit

    # Store updated flag
    tl.store(finished_ptr + idx, new_finished, mask=mask)


def check_length_criterion(
    finished: torch.Tensor,
    sequence_limit_length: torch.Tensor,
    step: int,
    beam_width: int = 1,
) -> None:
    """Check if sequences have reached maximum length (in-place update of finished flags).

    Args:
        finished: Finished flags [batch_size * beam_width]
        sequence_limit_length: Maximum lengths [batch_size]
        step: Current generation step
        beam_width: Beam width (default: 1)
    """
    batch_beam_size = finished.numel()
    batch_size = batch_beam_size // beam_width

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(batch_beam_size, BLOCK_SIZE),)

    _length_criterion_kernel[grid](
        finished,
        sequence_limit_length,
        batch_size,
        beam_width,
        step,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _check_all_finished_kernel(
    finished_ptr,
    all_finished_ptr,
    batch_size,
    beam_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Check if all sequences in the batch are finished.

    Args:
        finished_ptr: Finished flags [batch_size * beam_width]
        all_finished_ptr: Output flag [1]
        batch_size: Batch size
        beam_width: Beam width
        BLOCK_SIZE: Block size
    """
    # Accumulate finished flags
    all_finished = tl.constexpr(True)

    for idx_start in range(0, batch_size * beam_width, BLOCK_SIZE):
        idx = idx_start + tl.arange(0, BLOCK_SIZE)
        mask = idx < batch_size * beam_width

        # Load finished flags
        finished = tl.load(finished_ptr + idx, mask=mask, other=True)

        # Check if any sequence is not finished
        any_not_finished = tl.sum(tl.where(~finished, 1, 0))
        if any_not_finished > 0:
            all_finished = tl.constexpr(False)

    # Store result (only thread 0)
    if tl.program_id(0) == 0:
        tl.store(all_finished_ptr, all_finished)


def check_all_finished(finished: torch.Tensor) -> bool:
    """Check if all sequences in the batch are finished.

    Args:
        finished: Finished flags [batch_size * beam_width]

    Returns:
        True if all sequences are finished, False otherwise
    """
    # Create output buffer
    all_finished = torch.zeros(1, dtype=torch.bool, device=finished.device)

    batch_beam_size = finished.numel()
    BLOCK_SIZE = 256

    # Launch kernel
    grid = (1,)

    _check_all_finished_kernel[grid](
        finished,
        all_finished,
        batch_beam_size,
        1,  # beam_width doesn't matter for this check
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return all_finished.item()


def apply_stop_criteria(
    output_ids: torch.Tensor,
    finished: torch.Tensor,
    step: int,
    sequence_limit_length: torch.Tensor,
    stop_words: Optional[torch.Tensor] = None,
    stop_words_offsets: Optional[torch.Tensor] = None,
    parent_ids: Optional[torch.Tensor] = None,
    beam_width: int = 1,
) -> bool:
    """Apply all stopping criteria and check if generation should stop.

    Args:
        output_ids: Generated token IDs [step, batch_size * beam_width]
        finished: Finished flags [batch_size * beam_width]
        step: Current generation step
        sequence_limit_length: Maximum lengths [batch_size]
        stop_words: Optional stop word tokens [batch_size, stop_words_len, max_word_len]
        stop_words_offsets: Optional end indices for stop words [batch_size, stop_words_len]
        parent_ids: Optional parent beam indices [step, batch_size * beam_width]
        beam_width: Beam width (default: 1)

    Returns:
        True if all sequences are finished, False otherwise
    """
    # Check length criterion
    check_length_criterion(finished, sequence_limit_length, step, beam_width)

    # Check stop words criterion if provided
    if stop_words is not None and stop_words_offsets is not None:
        check_stop_words(
            output_ids,
            stop_words,
            stop_words_offsets,
            finished,
            step,
            parent_ids,
            beam_width,
        )

    # Check if all sequences are finished
    return check_all_finished(finished)
