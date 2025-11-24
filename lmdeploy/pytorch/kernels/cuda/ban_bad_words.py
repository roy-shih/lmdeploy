# Copyright (c) UnieAI.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def ban_bad_words_kernel(
    logits_ptr,
    bad_words_ptr,
    bad_words_mask_ptr,
    batch_size,
    vocab_size,
    max_bad_words: tl.constexpr,
    stride_lb,
    stride_lv,
    stride_bwb,
    stride_bww,
    stride_mb,
    stride_mw,
    filter_value,
):
    """
    ROCm-optimized version (no branching, pure predication).
    
    Reference: src/turbomind/kernels/ban_bad_words.cu
    
    Args:
        logits_ptr: [batch_size, vocab_size] logits to modify in-place
        bad_words_ptr: [batch_size, max_bad_words] bad word token IDs
        bad_words_mask_ptr: [batch_size, max_bad_words] which entries are valid
        filter_value: value to set for banned tokens (typically -inf)
    
    Logic:
        For each batch, for each valid bad word:
            logits[batch, bad_word_id] = filter_value
    """
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    # base pointers
    logits_row_ptr = logits_ptr + batch_idx * stride_lb
    bw_row_ptr = bad_words_ptr + batch_idx * stride_bwb
    mask_row_ptr = bad_words_mask_ptr + batch_idx * stride_mb

    # iterate via static_range (compile-time unrolled)
    for bw_idx in tl.static_range(max_bad_words):
        token_id = tl.load(bw_row_ptr + bw_idx * stride_bww)
        valid = tl.load(mask_row_ptr + bw_idx * stride_mw)

        in_vocab = (token_id >= 0) & (token_id < vocab_size)
        mask = valid & in_vocab

        # predicated store: ROCm prefers mask-style writes, no branching
        tl.store(
            logits_row_ptr + token_id * stride_lv,
            filter_value,
            mask=mask,
        )


def ban_bad_words(
    logits: Tensor,
    bad_words: Tensor,
    bad_words_mask: Tensor,
    filter_value: float = -float('inf'),
) -> Tensor:
    """
    Ban (mask) bad words by setting their logits to -inf.
    
    ROCm-optimized version with predicated stores and no branching.
    
    Args:
        logits: [batch_size, vocab_size] logits tensor (modified in-place)
        bad_words: [batch_size, max_bad_words] bad word token IDs
        bad_words_mask: [batch_size, max_bad_words] bool mask for valid entries
        filter_value: value to set for banned tokens (default: -inf)
    
    Returns:
        Modified logits tensor (same as input)
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [batch, vocab], got {logits.shape}")

    batch_size, vocab_size = logits.shape

    if bad_words.dim() != 2:
        raise ValueError(f"bad_words must be [batch, max_bad_words], got {bad_words.shape}")
    if bad_words.shape[0] != batch_size:
        raise ValueError(
            f"bad_words.shape[0] ({bad_words.shape[0]}) != batch_size ({batch_size})"
        )

    if bad_words_mask.dim() != 2:
        raise ValueError(f"bad_words_mask must be 2D, got {bad_words_mask.shape}")
    if bad_words_mask.shape != bad_words.shape:
        raise ValueError(
            f"bad_words_mask.shape ({bad_words_mask.shape}) != bad_words.shape ({bad_words.shape})"
        )

    max_bad_words = bad_words.shape[1]

    # Ensure contiguous and on same device
    logits = logits.contiguous()
    bad_words = bad_words.to(device=logits.device).contiguous()
    bad_words_mask = bad_words_mask.to(device=logits.device).contiguous()

    # Launch one program per batch item
    grid = (batch_size,)

    ban_bad_words_kernel[grid](
        logits,
        bad_words,
        bad_words_mask,
        batch_size=batch_size,
        vocab_size=vocab_size,
        max_bad_words=max_bad_words,
        stride_lb=logits.stride(0),
        stride_lv=logits.stride(1),
        stride_bwb=bad_words.stride(0),
        stride_bww=bad_words.stride(1),
        stride_mb=bad_words_mask.stride(0),
        stride_mw=bad_words_mask.stride(1),
        filter_value=filter_value,
    )

    return logits
