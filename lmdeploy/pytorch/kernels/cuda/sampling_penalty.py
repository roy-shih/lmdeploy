# Copyright (c) UnieAI. All rights reserved.
"""Sampling penalty kernels optimized with Triton.

This module implements various penalties for text generation:
- Temperature penalty: Controls randomness (higher = more random)
- Repetition penalty: Discourages repeated tokens (additive/multiplicative)
- Frequency penalty: Penalizes based on token frequency
- Presence penalty: Penalizes based on token presence
- Min length penalty: Prevents EOS before minimum length

Based on TurboMind's implementation but using Triton for cross-platform support.
"""
import torch
import triton
import triton.language as tl
from typing import Optional
from enum import IntEnum


class RepetitionPenaltyType(IntEnum):
    """Repetition penalty types."""
    NONE = 0
    ADDITIVE = 1
    MULTIPLICATIVE = 2


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['vocab_size'],
)
@triton.jit
def _apply_temperature_kernel(
    logits_ptr,
    bias_ptr,
    batch_size,
    vocab_size: tl.constexpr,
    vocab_size_padded: tl.constexpr,
    temperature_inverse: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply temperature penalty to logits.

    Formula: logits = (logits + bias) / temperature
    Padded values are set to -inf.

    Args:
        logits_ptr: Logits tensor [batch_size, vocab_size_padded]
        bias_ptr: Optional bias [vocab_size_padded]
        batch_size: Number of sequences
        vocab_size: Actual vocabulary size
        vocab_size_padded: Padded vocabulary size
        temperature_inverse: 1/temperature
        has_bias: Whether bias is provided
        BLOCK_SIZE: Block size for vocabulary dimension
    """
    batch_idx = tl.program_id(0)
    vocab_block_idx = tl.program_id(1)

    if batch_idx >= batch_size:
        return

    # Calculate vocabulary offsets
    vocab_start = vocab_block_idx * BLOCK_SIZE
    vocab_offs = vocab_start + tl.arange(0, BLOCK_SIZE)
    vocab_mask = vocab_offs < vocab_size_padded

    # Load logits
    logits_offset = batch_idx * vocab_size_padded + vocab_offs
    logits = tl.load(logits_ptr + logits_offset, mask=vocab_mask, other=0.0)

    # Add bias if provided
    if has_bias:
        bias = tl.load(bias_ptr + vocab_offs, mask=vocab_mask, other=0.0)
        logits = logits + bias

    # Apply temperature (divide by temperature = multiply by inverse)
    logits = logits * temperature_inverse

    # Mask out padded values
    is_valid = vocab_offs < vocab_size
    logits = tl.where(is_valid, logits, float('-inf'))

    # Store result
    tl.store(logits_ptr + logits_offset, logits, mask=vocab_mask)


def apply_temperature_penalty(
    logits: torch.Tensor,
    temperature: float,
    bias: Optional[torch.Tensor] = None,
) -> None:
    """Apply temperature penalty to logits (in-place).

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size_padded]
        temperature: Temperature value (> 0). Higher = more random.
        bias: Optional bias tensor of shape [vocab_size_padded]
    """
    batch_size, vocab_size_padded = logits.shape
    vocab_size = vocab_size_padded  # Assume no padding unless specified

    # Compute temperature inverse
    temperature_inverse = 1.0 / max(temperature, 1e-6)

    # Launch kernel
    BLOCK_SIZE = min(triton.next_power_of_2(vocab_size_padded), 1024)
    grid = (batch_size, triton.cdiv(vocab_size_padded, BLOCK_SIZE))

    _apply_temperature_kernel[grid](
        logits,
        bias if bias is not None else logits,  # Dummy pointer if no bias
        batch_size,
        vocab_size,
        vocab_size_padded,
        temperature_inverse,
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['vocab_size'],
)
@triton.jit
def _batch_apply_temperature_kernel(
    logits_ptr,
    temperatures_ptr,
    batch_size,
    vocab_size: tl.constexpr,
    vocab_size_padded: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply per-batch temperature penalty.

    Args:
        logits_ptr: Logits tensor [batch_size, vocab_size_padded]
        temperatures_ptr: Temperature values [batch_size]
        batch_size: Number of sequences
        vocab_size: Actual vocabulary size
        vocab_size_padded: Padded vocabulary size
        BLOCK_SIZE: Block size for vocabulary dimension
    """
    batch_idx = tl.program_id(0)
    vocab_block_idx = tl.program_id(1)

    if batch_idx >= batch_size:
        return

    # Load temperature for this batch
    temperature = tl.load(temperatures_ptr + batch_idx)
    temperature_inverse = 1.0 / tl.maximum(temperature, 1e-6)

    # Calculate vocabulary offsets
    vocab_start = vocab_block_idx * BLOCK_SIZE
    vocab_offs = vocab_start + tl.arange(0, BLOCK_SIZE)
    vocab_mask = vocab_offs < vocab_size_padded

    # Load logits
    logits_offset = batch_idx * vocab_size_padded + vocab_offs
    logits = tl.load(logits_ptr + logits_offset, mask=vocab_mask, other=0.0)

    # Apply temperature
    logits = logits * temperature_inverse

    # Mask out padded values
    is_valid = vocab_offs < vocab_size
    logits = tl.where(is_valid, logits, float('-inf'))

    # Store result
    tl.store(logits_ptr + logits_offset, logits, mask=vocab_mask)


def batch_apply_temperature_penalty(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> None:
    """Apply per-batch temperature penalty (in-place).

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size_padded]
        temperatures: Temperature values of shape [batch_size]
    """
    batch_size, vocab_size_padded = logits.shape
    vocab_size = vocab_size_padded

    # Launch kernel
    BLOCK_SIZE = min(triton.next_power_of_2(vocab_size_padded), 1024)
    grid = (batch_size, triton.cdiv(vocab_size_padded, BLOCK_SIZE))

    _batch_apply_temperature_kernel[grid](
        logits,
        temperatures,
        batch_size,
        vocab_size,
        vocab_size_padded,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _apply_repetition_penalty_kernel(
    logits_ptr,
    output_ids_ptr,
    penalty: tl.constexpr,
    penalty_type: tl.constexpr,
    batch_idx,
    batch_size,
    vocab_size,
    vocab_size_padded,
    input_length,
    max_input_length,
    step,
):
    """Apply repetition penalty for a single batch.

    Penalties:
    - Additive: logit = logit - penalty
    - Multiplicative: logit = logit * penalty (if logit < 0) else logit / penalty

    Args:
        logits_ptr: Logits tensor [batch_size, vocab_size_padded]
        output_ids_ptr: Output token IDs [step, batch_size]
        penalty: Penalty value
        penalty_type: 1=Additive, 2=Multiplicative
        batch_idx: Current batch index
        batch_size: Total batch size
        vocab_size: Actual vocabulary size
        vocab_size_padded: Padded vocabulary size
        input_length: Input length for this batch
        max_input_length: Maximum input length
        step: Current generation step
    """
    # Process each token in the sequence
    for seq_idx in range(step):
        # Skip padding tokens in input
        if seq_idx >= input_length and seq_idx < max_input_length:
            continue

        # Get the token ID at this position
        # output_ids shape: [step, batch_size]
        token_id = tl.load(output_ids_ptr + seq_idx * batch_size + batch_idx)

        # Skip if token_id is out of range
        if token_id >= vocab_size:
            continue

        # Load current logit value
        logit_offset = batch_idx * vocab_size_padded + token_id
        logit = tl.load(logits_ptr + logit_offset)

        # Apply penalty based on type
        if penalty_type == 1:  # Additive
            penalized_logit = logit - penalty
        elif penalty_type == 2:  # Multiplicative
            if logit < 0.0:
                penalized_logit = logit * penalty
            else:
                penalized_logit = logit / penalty
        else:  # None
            penalized_logit = logit

        # Store penalized logit
        tl.store(logits_ptr + logit_offset, penalized_logit)


@triton.jit
def _batch_apply_repetition_penalty_kernel(
    logits_ptr,
    penalties_ptr,
    output_ids_ptr,
    input_lengths_ptr,
    penalty_type: tl.constexpr,
    batch_size,
    vocab_size,
    vocab_size_padded,
    max_input_length,
    step,
    has_input_lengths: tl.constexpr,
):
    """Apply per-batch repetition penalty.

    Args:
        logits_ptr: Logits tensor [batch_size, vocab_size_padded]
        penalties_ptr: Penalty values [batch_size]
        output_ids_ptr: Output token IDs [step, batch_size]
        input_lengths_ptr: Input lengths [batch_size] (optional)
        penalty_type: 1=Additive, 2=Multiplicative
        batch_size: Number of sequences
        vocab_size: Actual vocabulary size
        vocab_size_padded: Padded vocabulary size
        max_input_length: Maximum input length
        step: Current generation step
        has_input_lengths: Whether input_lengths is provided
    """
    batch_idx = tl.program_id(0)

    if batch_idx >= batch_size:
        return

    # Load penalty for this batch
    penalty = tl.load(penalties_ptr + batch_idx)

    # Load input length for this batch
    if has_input_lengths:
        input_length = tl.load(input_lengths_ptr + batch_idx)
    else:
        input_length = max_input_length

    # Apply repetition penalty
    _apply_repetition_penalty_kernel(
        logits_ptr,
        output_ids_ptr,
        penalty,
        penalty_type,
        batch_idx,
        batch_size,
        vocab_size,
        vocab_size_padded,
        input_length,
        max_input_length,
        step,
    )


def apply_repetition_penalty(
    logits: torch.Tensor,
    penalties: torch.Tensor,
    output_ids: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    max_input_length: int = 0,
    penalty_type: RepetitionPenaltyType = RepetitionPenaltyType.MULTIPLICATIVE,
) -> None:
    """Apply repetition penalty to logits (in-place).

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size_padded]
        penalties: Penalty values of shape [batch_size]
        output_ids: Output token IDs of shape [step, batch_size]
        input_lengths: Optional input lengths [batch_size]
        max_input_length: Maximum input length
        penalty_type: Type of penalty (ADDITIVE or MULTIPLICATIVE)
    """
    batch_size, vocab_size_padded = logits.shape
    step = output_ids.shape[0]
    vocab_size = vocab_size_padded

    if penalty_type == RepetitionPenaltyType.NONE:
        return  # No penalty to apply

    # Launch kernel
    grid = (batch_size,)

    _batch_apply_repetition_penalty_kernel[grid](
        logits,
        penalties,
        output_ids,
        input_lengths if input_lengths is not None else logits,  # Dummy pointer
        int(penalty_type),
        batch_size,
        vocab_size,
        vocab_size_padded,
        max_input_length,
        step,
        input_lengths is not None,
    )


@triton.jit
def _apply_frequency_presence_penalty_kernel(
    logits_ptr,
    token_counts_ptr,
    frequency_penalty: tl.constexpr,
    presence_penalty: tl.constexpr,
    batch_idx,
    vocab_size,
    vocab_size_padded,
):
    """Apply frequency and presence penalties.

    Frequency penalty: logit = logit - frequency_penalty * count
    Presence penalty: logit = logit - presence_penalty * (1 if count > 0 else 0)

    Args:
        logits_ptr: Logits tensor [batch_size, vocab_size_padded]
        token_counts_ptr: Token occurrence counts [batch_size, vocab_size]
        frequency_penalty: Frequency penalty coefficient
        presence_penalty: Presence penalty coefficient
        batch_idx: Current batch index
        vocab_size: Actual vocabulary size
        vocab_size_padded: Padded vocabulary size
    """
    token_id = tl.program_id(0)

    if token_id >= vocab_size:
        return

    # Load token count
    count_offset = batch_idx * vocab_size + token_id
    count = tl.load(token_counts_ptr + count_offset)

    # Load logit
    logit_offset = batch_idx * vocab_size_padded + token_id
    logit = tl.load(logits_ptr + logit_offset)

    # Apply frequency penalty
    if frequency_penalty != 0.0:
        logit = logit - frequency_penalty * count.to(tl.float32)

    # Apply presence penalty
    if presence_penalty != 0.0:
        has_appeared = count > 0
        logit = logit - tl.where(has_appeared, presence_penalty, 0.0)

    # Store result
    tl.store(logits_ptr + logit_offset, logit)


def apply_frequency_presence_penalty(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> None:
    """Apply frequency and presence penalties (in-place).

    Frequency penalty discourages tokens based on how many times they've appeared.
    Presence penalty applies a flat penalty if a token has appeared at all.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size_padded]
        token_counts: Token occurrence counts [batch_size, vocab_size]
        frequency_penalty: Frequency penalty coefficient (default: 0.0)
        presence_penalty: Presence penalty coefficient (default: 0.0)
    """
    batch_size, vocab_size_padded = logits.shape
    vocab_size = token_counts.shape[1]

    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return  # No penalty to apply

    # Launch kernel for each batch and token
    for batch_idx in range(batch_size):
        grid = (vocab_size,)
        _apply_frequency_presence_penalty_kernel[grid](
            logits,
            token_counts,
            frequency_penalty,
            presence_penalty,
            batch_idx,
            vocab_size,
            vocab_size_padded,
        )


@triton.jit
def _apply_min_length_penalty_kernel(
    logits_ptr,
    min_lengths_ptr,
    sequence_lengths_ptr,
    end_ids_ptr,
    batch_size,
    vocab_size_padded,
    end_ids_size,
):
    """Apply minimum length penalty.

    Sets logits for EOS tokens to -inf if sequence length < min_length.

    Args:
        logits_ptr: Logits tensor [batch_size, vocab_size_padded]
        min_lengths_ptr: Minimum lengths [batch_size]
        sequence_lengths_ptr: Current sequence lengths [batch_size]
        end_ids_ptr: End-of-sequence token IDs [batch_size, end_ids_size]
        batch_size: Number of sequences
        vocab_size_padded: Padded vocabulary size
        end_ids_size: Number of EOS tokens per batch
    """
    tid = tl.program_id(0)
    batch_idx = tid // end_ids_size
    eos_idx = tid % end_ids_size

    if batch_idx >= batch_size:
        return

    # Load end token ID
    end_id = tl.load(end_ids_ptr + batch_idx * end_ids_size + eos_idx)

    if end_id <= 0:
        return

    # Load sequence length and min length
    seq_len = tl.load(sequence_lengths_ptr + batch_idx)
    min_len = tl.load(min_lengths_ptr + batch_idx)

    # If sequence is shorter than min_length, mask out EOS token
    if seq_len + 1 < min_len:
        logit_offset = batch_idx * vocab_size_padded + end_id
        tl.store(logits_ptr + logit_offset, float('-inf'))


def apply_min_length_penalty(
    logits: torch.Tensor,
    min_lengths: torch.Tensor,
    sequence_lengths: torch.Tensor,
    end_ids: torch.Tensor,
) -> None:
    """Apply minimum length penalty (in-place).

    Prevents generation from ending before reaching minimum length.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size_padded]
        min_lengths: Minimum lengths of shape [batch_size]
        sequence_lengths: Current sequence lengths of shape [batch_size]
        end_ids: End-of-sequence token IDs of shape [batch_size, end_ids_size]
    """
    batch_size, vocab_size_padded = logits.shape
    end_ids_size = end_ids.shape[1] if end_ids.ndim > 1 else 1

    # Launch kernel
    grid = (batch_size * end_ids_size,)

    _apply_min_length_penalty_kernel[grid](
        logits,
        min_lengths,
        sequence_lengths,
        end_ids,
        batch_size,
        vocab_size_padded,
        end_ids_size,
    )


def update_token_counts(
    token_counts: torch.Tensor,
    new_token_ids: torch.Tensor,
) -> None:
    """Update token occurrence counts (in-place).

    Helper function for frequency/presence penalties.

    Args:
        token_counts: Token counts of shape [batch_size, vocab_size]
        new_token_ids: New token IDs of shape [batch_size]
    """
    batch_size = new_token_ids.shape[0]
    for i in range(batch_size):
        token_id = new_token_ids[i].item()
        if 0 <= token_id < token_counts.shape[1]:
            token_counts[i, token_id] += 1
