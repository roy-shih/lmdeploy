# Copyright (c) UnieAI.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def build_repetition_mask_kernel(
    input_ids_ptr,   # int32 / int64
    visited_ptr,     # uint8, shape [batch_size, vocab_size]
    batch_size,
    seq_len,
    vocab_size,
    stride_is,       # input_ids.stride(0)  (seq dimension)
    stride_ib,       # input_ids.stride(1)  (batch dimension)
    stride_vb,       # visited.stride(0)    (batch)
    stride_vv,       # visited.stride(1)    (vocab)
    BLOCK_SEQ: tl.constexpr,
):
    """
    Phase 1: Build visited mask for each batch, marking which vocab ids appear in history.

    Reference: src/turbomind/kernels/sampling_penalty_kernels.cu (batchApplyRepetitionPenalty)
    
    input_ids: [seq_len, batch_size]
    visited:   [batch_size, vocab_size]  (0/1)
    """
    batch_id = tl.program_id(0)
    seq_block = tl.program_id(1)

    offs_s = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mask_s = offs_s < seq_len

    # Load tokens for this batch in this seq_block
    inp_ptr = input_ids_ptr + offs_s * stride_is + batch_id * stride_ib
    token_ids = tl.load(inp_ptr, mask=mask_s, other=0)

    # Convert to int32 for comparison and indexing
    token_ids = token_ids.to(tl.int32)

    # Valid token: 0 <= tid < vocab_size
    valid = mask_s & (token_ids >= 0) & (token_ids < vocab_size)

    # Each batch has one row in visited
    visited_row_ptr = visited_ptr + batch_id * stride_vb

    # Set visited[batch_id, token_ids] = 1
    tl.store(visited_row_ptr + token_ids, 1, mask=valid)


@triton.jit
def apply_repetition_penalty_kernel(
    logits_ptr,      # [batch_size, vocab_size]
    visited_ptr,     # [batch_size, vocab_size] uint8
    penalties_ptr,   # [batch_size]
    batch_size,
    vocab_size,
    stride_lb,       # logits.stride(0)
    stride_lv,       # logits.stride(1)
    stride_vb,       # visited.stride(0)
    stride_vv,       # visited.stride(1)
    penalty_type: tl.constexpr,  # 0=None, 1=Additive, 2=Multiplicative
    BLOCK_V: tl.constexpr,
):
    """
    Phase 2: Apply repetition penalty to logits where visited==1.

    logits:   [batch_size, vocab_size]
    visited:  [batch_size, vocab_size] (0/1)
    penalties: [batch_size]
    """
    batch_id = tl.program_id(0)
    vblock = tl.program_id(1)

    offs_v = vblock * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < vocab_size

    # Row pointers
    logits_row_ptr = logits_ptr + batch_id * stride_lb + offs_v * stride_lv
    visited_row_ptr = visited_ptr + batch_id * stride_vb + offs_v * stride_vv

    logits = tl.load(logits_row_ptr, mask=mask_v, other=0.0)
    visited = tl.load(visited_row_ptr, mask=mask_v, other=0)

    visited = visited.to(tl.int32)
    has_seen = visited > 0

    penalty = tl.load(penalties_ptr + batch_id)

    if penalty_type == 1:
        # Additive: logit - penalty
        penalized = logits - penalty
        new_logits = tl.where(has_seen, penalized, logits)
    elif penalty_type == 2:
        # Multiplicative: <0 -> * penalty; >=0 -> / penalty
        penalized = tl.where(logits < 0.0, logits * penalty, logits / penalty)
        new_logits = tl.where(has_seen, penalized, logits)
    else:
        # No penalty
        new_logits = logits

    tl.store(logits_row_ptr, new_logits, mask=mask_v)


def apply_repetition_penalty(
    logits: Tensor,
    input_ids: Tensor,
    penalties: Tensor,
    penalty_type: str = 'multiplicative',
) -> Tensor:
    """
    High-performance Repetition Penalty (TurboMind-style Triton implementation).

    Args:
        logits:    [batch_size, vocab_size] logits tensor (modified in-place)
        input_ids: [seq_len, batch_size] or [batch_size, seq_len] token history
        penalties: [batch_size] repetition penalty values (typically > 1.0)
        penalty_type: 'additive', 'multiplicative', or 'none'

    Returns:
        Modified logits (same tensor as input)
    """
    assert logits.dim() == 2, f"logits must be 2D [batch, vocab], got {logits.shape}"
    batch_size, vocab_size = logits.shape

    # Normalize input_ids layout to [seq_len, batch_size]
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

    if input_ids.shape[0] == batch_size:
        # [batch, seq_len] -> [seq_len, batch]
        input_ids = input_ids.t().contiguous()
    elif input_ids.shape[1] == batch_size:
        # [seq_len, batch], ensure contiguous
        input_ids = input_ids.contiguous()
    else:
        raise ValueError(
            f"input_ids shape {input_ids.shape} not compatible with batch_size={batch_size}"
        )

    seq_len = input_ids.shape[0]

    # Penalties: [batch]
    if penalties.dim() != 1 or penalties.shape[0] != batch_size:
        raise ValueError(
            f"penalties must be 1D with shape [batch_size], got {penalties.shape}"
        )
    penalties = penalties.to(dtype=logits.dtype, device=logits.device).contiguous()

    # Penalty type mapping
    penalty_type_map = {
        'none': 0,
        'additive': 1,
        'multiplicative': 2,
    }
    penalty_type_const = penalty_type_map.get(penalty_type.lower(), 2)

    # Build visited mask (uint8 to save memory)
    visited = torch.zeros(
        (batch_size, vocab_size),
        dtype=torch.uint8,
        device=logits.device,
    ).contiguous()

    # ---- Phase 1: Build visited mask ----
    BLOCK_SEQ = 128
    grid1 = (
        batch_size,
        triton.cdiv(seq_len, BLOCK_SEQ),
    )
    build_repetition_mask_kernel[grid1](
        input_ids,
        visited,
        batch_size,
        seq_len,
        vocab_size,
        stride_is=input_ids.stride(0),
        stride_ib=input_ids.stride(1),
        stride_vb=visited.stride(0),
        stride_vv=visited.stride(1),
        BLOCK_SEQ=BLOCK_SEQ,
    )

    # ---- Phase 2: Apply penalty on logits ----
    if not logits.is_contiguous():
        logits = logits.contiguous()

    BLOCK_V = 128
    grid2 = (
        batch_size,
        triton.cdiv(vocab_size, BLOCK_V),
    )
    apply_repetition_penalty_kernel[grid2](
        logits,
        visited,
        penalties,
        batch_size,
        vocab_size,
        stride_lb=logits.stride(0),
        stride_lv=logits.stride(1),
        stride_vb=visited.stride(0),
        stride_vv=visited.stride(1),
        penalty_type=penalty_type_const,
        BLOCK_V=BLOCK_V,
    )

    return logits
