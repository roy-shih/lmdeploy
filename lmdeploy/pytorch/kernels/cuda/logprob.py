# Copyright (c) UnieAI. All rights reserved.
"""Log probability computation kernels optimized with Triton.

This module implements efficient log probability calculations for language models:
- Log softmax computation
- Token-level log probabilities
- Cumulative log probability (sequence-level)

Based on TurboMind's implementation but using Triton for cross-platform support.
"""
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_V': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_V': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE_V': 2048}, num_warps=8),
    ],
    key=['vocab_size'],
)
@triton.jit
def _log_prob_kernel(
    log_probs_ptr,
    logits_ptr,
    token_ids_ptr,
    lengths_ptr,
    batch_size,
    max_length,
    vocab_size: tl.constexpr,
    vocab_size_padded: tl.constexpr,
    batch_first: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    """Compute log probabilities from logits.

    Formula: log_probs[t] = log(softmax(logits[t]))[token_ids[t+1]]
              = logits[t][token_ids[t+1]] - max(logits[t]) - log(sum(exp(logits[t] - max)))

    Args:
        log_probs_ptr: Output log probabilities [batch_size, max_length-1] or [max_length-1, batch_size]
        logits_ptr: Input logits [batch_size, max_length, vocab_size_padded] or [max_length, batch_size, vocab_size_padded]
        token_ids_ptr: Token IDs [batch_size, max_length] or [max_length, batch_size]
        lengths_ptr: Sequence lengths [batch_size]
        batch_size: Batch size
        max_length: Maximum sequence length
        vocab_size: Actual vocabulary size
        vocab_size_padded: Padded vocabulary size
        batch_first: If True, batch dimension is first
        BLOCK_SIZE_V: Block size for vocabulary dimension
    """
    # Get batch index and step index
    if batch_first:
        batch_idx = tl.program_id(0)
        step = tl.program_id(1)
    else:
        step = tl.program_id(0)
        batch_idx = tl.program_id(1)

    if batch_idx >= batch_size:
        return

    # Load sequence length
    length = tl.load(lengths_ptr + batch_idx)

    # Only process valid steps (up to length - 1)
    if step >= length - 1:
        return

    # Calculate logits offset
    if batch_first:
        logits_offset = batch_idx * max_length * vocab_size_padded + step * vocab_size_padded
        token_idx_offset = batch_idx * max_length + step + 1  # Next token
    else:
        logits_offset = step * batch_size * vocab_size_padded + batch_idx * vocab_size_padded
        token_idx_offset = (step + 1) * batch_size + batch_idx

    # Load next token ID (the token we want probability for)
    next_token_id = tl.load(token_ids_ptr + token_idx_offset)

    # Find max logit (for numerical stability)
    max_logit = float('-inf')
    for v_start in range(0, vocab_size, BLOCK_SIZE_V):
        v_offs = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = v_offs < vocab_size
        logits = tl.load(logits_ptr + logits_offset + v_offs, mask=v_mask, other=float('-inf'))
        block_max = tl.max(logits, axis=0)
        max_logit = tl.maximum(max_logit, block_max)

    # Calculate sum of exp(logits - max_logit)
    sum_exp = 0.0
    for v_start in range(0, vocab_size, BLOCK_SIZE_V):
        v_offs = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = v_offs < vocab_size
        logits = tl.load(logits_ptr + logits_offset + v_offs, mask=v_mask, other=float('-inf'))
        exp_logits = tl.exp(logits - max_logit)
        sum_exp += tl.sum(tl.where(v_mask, exp_logits, 0.0), axis=0)

    # Load logit for the next token
    next_token_logit = tl.load(logits_ptr + logits_offset + next_token_id)

    # Calculate log probability: logit - max - log(sum_exp)
    log_prob = next_token_logit - max_logit - tl.log(sum_exp + 1e-9)

    # Store result
    if batch_first:
        output_offset = batch_idx * (max_length - 1) + step
    else:
        output_offset = step * batch_size + batch_idx

    tl.store(log_probs_ptr + output_offset, log_prob)


def compute_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    batch_first: bool = True,
) -> torch.Tensor:
    """Compute log probabilities from logits.

    Args:
        logits: Logits tensor of shape [batch_size, max_length, vocab_size] or [max_length, batch_size, vocab_size]
        token_ids: Token IDs of shape [batch_size, max_length] or [max_length, batch_size]
        lengths: Sequence lengths of shape [batch_size]
        batch_first: If True, batch dimension is first (default: True)

    Returns:
        Log probabilities of shape [batch_size, max_length-1] or [max_length-1, batch_size]
    """
    if batch_first:
        batch_size, max_length, vocab_size_padded = logits.shape
        output_shape = (batch_size, max_length - 1)
    else:
        max_length, batch_size, vocab_size_padded = logits.shape
        output_shape = (max_length - 1, batch_size)

    vocab_size = vocab_size_padded  # Assume no padding unless specified

    # Output buffer
    log_probs = torch.empty(output_shape, dtype=torch.float32, device=logits.device)

    # Launch kernel
    if batch_first:
        grid = (batch_size, max_length - 1)
    else:
        grid = (max_length - 1, batch_size)

    BLOCK_SIZE_V = min(triton.next_power_of_2(vocab_size), 2048)

    _log_prob_kernel[grid](
        log_probs,
        logits,
        token_ids,
        lengths,
        batch_size,
        max_length,
        vocab_size,
        vocab_size_padded,
        batch_first,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    return log_probs


@triton.jit
def _accumulate_log_probs_kernel(
    cum_log_probs_ptr,
    log_probs_ptr,
    lengths_ptr,
    batch_size,
    max_length_minus_1,
    batch_first: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    """Accumulate log probabilities across sequence dimension.

    Formula: cum_log_probs[b] = sum_{t=0}^{length-2} log_probs[b, t]

    Args:
        cum_log_probs_ptr: Output cumulative log probabilities [batch_size]
        log_probs_ptr: Input log probabilities [batch_size, max_length-1] or [max_length-1, batch_size]
        lengths_ptr: Sequence lengths [batch_size]
        batch_size: Batch size
        max_length_minus_1: max_length - 1
        batch_first: If True, batch dimension is first
        BLOCK_SIZE_S: Block size for sequence dimension
    """
    batch_idx = tl.program_id(0)

    if batch_idx >= batch_size:
        return

    # Load sequence length
    length = tl.load(lengths_ptr + batch_idx)
    num_steps = length - 1  # Number of log probs to sum

    # Calculate log_probs offset for this batch
    if batch_first:
        log_probs_offset = batch_idx * max_length_minus_1
        stride = 1
    else:
        log_probs_offset = batch_idx
        stride = batch_size

    # Accumulate log probabilities
    local_sum = 0.0
    for step_start in range(0, num_steps, BLOCK_SIZE_S):
        step_offs = step_start + tl.arange(0, BLOCK_SIZE_S)
        step_mask = step_offs < num_steps

        # Load log probabilities
        offsets = log_probs_offset + step_offs * stride
        log_prob_vals = tl.load(log_probs_ptr + offsets, mask=step_mask, other=0.0)

        # Accumulate
        local_sum += tl.sum(log_prob_vals, axis=0)

    # Store cumulative log probability
    tl.store(cum_log_probs_ptr + batch_idx, local_sum)


def accumulate_log_probs(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    batch_first: bool = True,
) -> torch.Tensor:
    """Accumulate log probabilities across sequence dimension.

    Args:
        log_probs: Log probabilities of shape [batch_size, max_length-1] or [max_length-1, batch_size]
        lengths: Sequence lengths of shape [batch_size]
        batch_first: If True, batch dimension is first (default: True)

    Returns:
        Cumulative log probabilities of shape [batch_size]
    """
    if batch_first:
        batch_size, max_length_minus_1 = log_probs.shape
    else:
        max_length_minus_1, batch_size = log_probs.shape

    # Output buffer
    cum_log_probs = torch.empty(batch_size, dtype=torch.float32, device=log_probs.device)

    # Launch kernel
    grid = (batch_size,)
    BLOCK_SIZE_S = min(triton.next_power_of_2(max_length_minus_1), 1024)

    _accumulate_log_probs_kernel[grid](
        cum_log_probs,
        log_probs,
        lengths,
        batch_size,
        max_length_minus_1,
        batch_first,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
    )

    return cum_log_probs


def compute_cumulative_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    batch_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cumulative log probabilities from logits.

    This is a combined function that computes both token-level and cumulative log probabilities.

    Args:
        logits: Logits tensor of shape [batch_size, max_length, vocab_size] or [max_length, batch_size, vocab_size]
        token_ids: Token IDs of shape [batch_size, max_length] or [max_length, batch_size]
        lengths: Sequence lengths of shape [batch_size]
        batch_first: If True, batch dimension is first (default: True)

    Returns:
        Tuple of (log_probs, cumulative_log_probs)
        - log_probs: Token-level log probabilities
        - cumulative_log_probs: Sequence-level cumulative log probabilities [batch_size]
    """
    # Compute token-level log probabilities
    log_probs = compute_log_probs(logits, token_ids, lengths, batch_first)

    # Accumulate to get cumulative log probabilities
    cum_log_probs = accumulate_log_probs(log_probs, lengths, batch_first)

    return log_probs, cum_log_probs


@triton.jit
def _log_softmax_kernel(
    output_ptr,
    input_ptr,
    batch_size,
    vocab_size: tl.constexpr,
    vocab_size_padded: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute log softmax.

    Formula: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

    Args:
        output_ptr: Output tensor [batch_size, vocab_size_padded]
        input_ptr: Input tensor [batch_size, vocab_size_padded]
        batch_size: Batch size
        vocab_size: Actual vocabulary size
        vocab_size_padded: Padded vocabulary size
        BLOCK_SIZE: Block size for vocabulary dimension
    """
    batch_idx = tl.program_id(0)

    if batch_idx >= batch_size:
        return

    # Calculate offset for this batch
    offset = batch_idx * vocab_size_padded

    # Find max value
    max_val = float('-inf')
    for v_start in range(0, vocab_size, BLOCK_SIZE):
        v_offs = v_start + tl.arange(0, BLOCK_SIZE)
        v_mask = v_offs < vocab_size
        vals = tl.load(input_ptr + offset + v_offs, mask=v_mask, other=float('-inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Calculate sum of exp(x - max)
    sum_exp = 0.0
    for v_start in range(0, vocab_size, BLOCK_SIZE):
        v_offs = v_start + tl.arange(0, BLOCK_SIZE)
        v_mask = v_offs < vocab_size
        vals = tl.load(input_ptr + offset + v_offs, mask=v_mask, other=float('-inf'))
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(v_mask, exp_vals, 0.0), axis=0)

    # Compute log_softmax and store
    log_sum_exp = tl.log(sum_exp + 1e-9)
    for v_start in range(0, vocab_size_padded, BLOCK_SIZE):
        v_offs = v_start + tl.arange(0, BLOCK_SIZE)
        v_mask = v_offs < vocab_size_padded

        # Load input
        vals = tl.load(input_ptr + offset + v_offs, mask=v_mask, other=float('-inf'))

        # Compute log_softmax
        is_valid = v_offs < vocab_size
        log_softmax_vals = tl.where(is_valid, vals - max_val - log_sum_exp, float('-inf'))

        # Store
        tl.store(output_ptr + offset + v_offs, log_softmax_vals, mask=v_mask)


def log_softmax(
    input: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute log softmax.

    Args:
        input: Input tensor of shape [batch_size, vocab_size]
        dim: Dimension to apply log softmax (default: -1)

    Returns:
        Log softmax output
    """
    assert dim == -1 or dim == 1, "Only last dimension is supported"

    batch_size, vocab_size_padded = input.shape
    vocab_size = vocab_size_padded

    # Output buffer
    output = torch.empty_like(input)

    # Launch kernel
    grid = (batch_size,)
    BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 2048)

    _log_softmax_kernel[grid](
        output,
        input,
        batch_size,
        vocab_size,
        vocab_size_padded,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
