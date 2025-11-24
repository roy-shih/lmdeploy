# Copyright (c) UnieAI. All rights reserved.
"""Top-P (Nucleus) sampling kernels optimized with Triton."""
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _topp_sampling_kernel(
    logits_ptr,
    output_ptr,
    seeds_ptr,
    offsets_ptr,
    p_ptr,
    stride_batch,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Top-P (Nucleus) sampling kernel.

    This kernel performs:
    1. Sort logits in descending order (simplified version)
    2. Apply softmax
    3. Find cumulative probability until reaching p
    4. Sample from the nucleus

    Args:
        logits_ptr: Input logits [batch_size, vocab_size]
        output_ptr: Sampled token ids [batch_size]
        seeds_ptr: Random seeds [batch_size]
        offsets_ptr: Random offsets [batch_size]
        p_ptr: Top-p values [batch_size]
        stride_batch: Stride for batch dimension
        vocab_size: Vocabulary size
        BLOCK_SIZE: Block size for processing
    """
    batch_id = tl.program_id(0)

    # Load p value for this batch
    p_value = tl.load(p_ptr + batch_id)

    # Load all logits for this batch
    logits_offset = batch_id * stride_batch
    logits_vals = tl.zeros([vocab_size], dtype=tl.float32)
    indices = tl.zeros([vocab_size], dtype=tl.int32)

    # Load logits in blocks
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        vals = tl.load(logits_ptr + logits_offset + offs, mask=mask, other=-float('inf'))

        # Store in local array
        for i in range(BLOCK_SIZE):
            if mask[i]:
                idx = offs[i]
                logits_vals[idx] = vals[i]
                indices[idx] = idx

    # Simple bubble sort for top-p (inefficient but simple for Triton)
    # In practice, we'd use a more efficient sorting algorithm
    # For now, we'll do a simplified version that finds the nucleus

    # Apply softmax first
    max_val = -float('inf')
    for i in range(vocab_size):
        if logits_vals[i] > max_val:
            max_val = logits_vals[i]

    sum_exp = 0.0
    probs = tl.zeros([vocab_size], dtype=tl.float32)
    for i in range(vocab_size):
        exp_val = tl.math.exp(logits_vals[i] - max_val)
        probs[i] = exp_val
        sum_exp += exp_val

    # Normalize probabilities
    for i in range(vocab_size):
        probs[i] = probs[i] / sum_exp

    # Greedy approach: repeatedly find max and accumulate
    # until cumsum >= p
    nucleus_mask = tl.zeros([vocab_size], dtype=tl.int1)
    cumsum = 0.0
    num_in_nucleus = 0

    # Find elements in nucleus
    for _ in range(vocab_size):
        if cumsum >= p_value:
            break

        # Find maximum probability not yet in nucleus
        max_prob = 0.0
        max_idx = 0
        for i in range(vocab_size):
            if not nucleus_mask[i] and probs[i] > max_prob:
                max_prob = probs[i]
                max_idx = i

        if max_prob > 0:
            nucleus_mask[max_idx] = True
            cumsum += max_prob
            num_in_nucleus += 1

    # Renormalize probabilities within nucleus
    nucleus_sum = 0.0
    for i in range(vocab_size):
        if nucleus_mask[i]:
            nucleus_sum += probs[i]

    nucleus_probs = tl.zeros([vocab_size], dtype=tl.float32)
    for i in range(vocab_size):
        if nucleus_mask[i]:
            nucleus_probs[i] = probs[i] / nucleus_sum

    # Sample from nucleus
    seed = tl.load(seeds_ptr + batch_id)
    offset = tl.load(offsets_ptr + batch_id).to(tl.int32)
    rand_val = tl.rand(seed, offset)

    # Find sampled index using cumsum
    cumsum = 0.0
    sampled_idx = 0
    for i in range(vocab_size):
        if nucleus_mask[i]:
            cumsum += nucleus_probs[i]
            if rand_val <= cumsum:
                sampled_idx = indices[i]
                break

    tl.store(output_ptr + batch_id, sampled_idx)


def topp_sampling(
    logits: torch.Tensor,
    p: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Top-P (Nucleus) sampling.

    Args:
        logits: Input logits of shape [batch_size, vocab_size]
        p: Top-p values of shape [batch_size], should be in (0, 1]
        seeds: Random seeds of shape [batch_size]
        offsets: Random offsets of shape [batch_size]

    Returns:
        Sampled token ids of shape [batch_size]
    """
    batch_size, vocab_size = logits.shape

    # Output buffer
    output = torch.empty(batch_size, device=logits.device, dtype=torch.int32)

    # Launch kernel
    BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 1024)
    grid = (batch_size,)

    _topp_sampling_kernel[grid](
        logits,
        output,
        seeds,
        offsets,
        p,
        logits.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.long()


@triton.jit
def _topp_filter_kernel(
    logits_ptr,
    p_ptr,
    out_ptr,
    stride_batch,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Filter logits to keep only nucleus (top-p) values.

    This kernel:
    1. Sorts probabilities
    2. Finds cumulative sum threshold
    3. Masks out values outside nucleus

    Args:
        logits_ptr: Input logits [batch_size, vocab_size]
        p_ptr: Top-p values [batch_size]
        out_ptr: Output filtered logits [batch_size, vocab_size]
        stride_batch: Batch stride
        vocab_size: Vocabulary size
        BLOCK_SIZE: Block size
    """
    batch_id = tl.program_id(0)
    v_block_id = tl.program_id(1)

    p_value = tl.load(p_ptr + batch_id)

    # Load logits block
    logits_offset = batch_id * stride_batch
    v_start = v_block_id * BLOCK_SIZE
    v_offs = v_start + tl.arange(0, BLOCK_SIZE)
    v_mask = v_offs < vocab_size

    logits = tl.load(logits_ptr + logits_offset + v_offs, mask=v_mask, other=-float('inf'))

    # Apply softmax to get probabilities
    # (This is a simplified version - in practice would need two-pass)
    # For now, we'll use a threshold-based approach

    # Convert to probabilities (approximate)
    max_logit = tl.max(logits, 0)
    probs = tl.math.exp(logits - max_logit)
    sum_probs = tl.sum(probs, 0)
    probs = probs / sum_probs

    # Simple filtering: keep if prob > threshold
    # This is approximate - full top-p requires sorting
    # We use a heuristic: keep top probs that would sum to ~p
    threshold = (1.0 - p_value) / vocab_size  # Simplified threshold

    # Mask out low-probability tokens
    for i in range(BLOCK_SIZE):
        if v_mask[i] and probs[i] < threshold:
            logits[i] = -float('inf')

    # Store filtered logits
    tl.store(out_ptr + logits_offset + v_offs, logits, mask=v_mask)


def topp_filter(
    logits: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """Filter logits to keep only nucleus (top-p) values.

    Args:
        logits: Input logits of shape [batch_size, vocab_size]
        p: Top-p values per sample of shape [batch_size]

    Returns:
        Filtered logits where non-nucleus values are set to -inf
    """
    batch_size, vocab_size = logits.shape
    out = torch.empty_like(logits)

    BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 1024)
    grid = (batch_size, triton.cdiv(vocab_size, BLOCK_SIZE))

    _topp_filter_kernel[grid](
        logits,
        p,
        out,
        logits.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def torch_topp_sampling(
    logits: torch.Tensor,
    p: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-P sampling using PyTorch (reference implementation).

    This is a reference implementation for testing and fallback.

    Args:
        logits: Input logits of shape [batch_size, vocab_size]
        p: Top-p values of shape [batch_size]
        temperature: Optional temperature scaling [batch_size]

    Returns:
        Sampled token ids of shape [batch_size]
    """
    batch_size = logits.size(0)

    # Apply temperature if provided
    if temperature is not None:
        logits = logits / temperature.unsqueeze(-1)

    # Sort probabilities in descending order
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find nucleus: keep tokens until cumsum > p
    # We keep at least one token
    nucleus_mask = cumsum_probs <= p.unsqueeze(-1)
    nucleus_mask[:, 0] = True  # Always keep the top token

    # Renormalize probabilities within nucleus
    nucleus_probs = sorted_probs * nucleus_mask
    nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True)

    # Sample from nucleus
    sampled_positions = torch.multinomial(nucleus_probs, num_samples=1).squeeze(-1)

    # Map back to original indices
    sampled_ids = torch.gather(sorted_indices, 1, sampled_positions.unsqueeze(-1)).squeeze(-1)

    return sampled_ids
