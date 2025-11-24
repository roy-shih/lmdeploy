# Copyright (c) UnieAI. All rights reserved.
"""Embedding lookup and position encoding kernels optimized with Triton."""
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_D': 512}, num_warps=8),
    ],
    key=['hidden_dim'],
)
@triton.jit
def _embedding_lookup_kernel(
    output_ptr,
    embedding_table_ptr,
    token_ids_ptr,
    num_tokens,
    hidden_dim: tl.constexpr,
    vocab_size,
    stride_out_n: tl.constexpr,
    stride_out_d: tl.constexpr,
    stride_emb_v: tl.constexpr,
    stride_emb_d: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Embedding lookup kernel.

    Performs: output[i] = embedding_table[token_ids[i]]

    Args:
        output_ptr: Output embeddings [num_tokens, hidden_dim]
        embedding_table_ptr: Embedding table [vocab_size, hidden_dim]
        token_ids_ptr: Token IDs [num_tokens]
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        vocab_size: Vocabulary size
        stride_out_n: Output stride for token dimension
        stride_out_d: Output stride for hidden dimension
        stride_emb_v: Embedding table stride for vocab dimension
        stride_emb_d: Embedding table stride for hidden dimension
        BLOCK_SIZE_D: Block size for hidden dimension
    """
    token_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    if token_id >= num_tokens:
        return

    # Load token ID
    token_idx = tl.load(token_ids_ptr + token_id)

    # Compute offsets
    d_start = d_block_id * BLOCK_SIZE_D
    d_offs = d_start + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offs < hidden_dim

    # Load embedding
    emb_ptr = embedding_table_ptr + token_idx * stride_emb_v + d_offs * stride_emb_d
    embedding = tl.load(emb_ptr, mask=d_mask, other=0.0)

    # Store output
    out_ptr = output_ptr + token_id * stride_out_n + d_offs * stride_out_d
    tl.store(out_ptr, embedding, mask=d_mask)


def embedding_lookup(
    embedding_table: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Embedding lookup.

    Args:
        embedding_table: Embedding table of shape [vocab_size, hidden_dim]
        token_ids: Token IDs of shape [num_tokens]

    Returns:
        Output embeddings of shape [num_tokens, hidden_dim]
    """
    vocab_size, hidden_dim = embedding_table.shape
    num_tokens = token_ids.numel()

    # Output buffer
    output = torch.empty(
        num_tokens, hidden_dim,
        device=embedding_table.device,
        dtype=embedding_table.dtype
    )

    # Launch kernel
    BLOCK_SIZE_D = min(triton.next_power_of_2(hidden_dim), 512)
    grid = (num_tokens, triton.cdiv(hidden_dim, BLOCK_SIZE_D))

    _embedding_lookup_kernel[grid](
        output,
        embedding_table,
        token_ids,
        num_tokens,
        hidden_dim,
        vocab_size,
        output.stride(0),
        output.stride(1),
        embedding_table.stride(0),
        embedding_table.stride(1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_D': 512}, num_warps=8),
    ],
    key=['hidden_dim'],
)
@triton.jit
def _embedding_lookup_pos_encoding_kernel(
    output_ptr,
    embedding_table_ptr,
    position_encoding_ptr,
    token_ids_ptr,
    position_ids_ptr,
    num_tokens,
    hidden_dim: tl.constexpr,
    scale: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_d: tl.constexpr,
    stride_emb_v: tl.constexpr,
    stride_emb_d: tl.constexpr,
    stride_pos_p: tl.constexpr,
    stride_pos_d: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Fused embedding lookup and position encoding kernel.

    Performs: output[i] = embedding_table[token_ids[i]] * scale + position_encoding[position_ids[i]]

    Args:
        output_ptr: Output embeddings [num_tokens, hidden_dim]
        embedding_table_ptr: Embedding table [vocab_size, hidden_dim]
        position_encoding_ptr: Position encoding [max_seq_len, hidden_dim]
        token_ids_ptr: Token IDs [num_tokens]
        position_ids_ptr: Position IDs [num_tokens]
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        scale: Scaling factor for embeddings
        stride_out_n: Output stride for token dimension
        stride_out_d: Output stride for hidden dimension
        stride_emb_v: Embedding stride for vocab dimension
        stride_emb_d: Embedding stride for hidden dimension
        stride_pos_p: Position encoding stride for position dimension
        stride_pos_d: Position encoding stride for hidden dimension
        BLOCK_SIZE_D: Block size for hidden dimension
    """
    token_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    if token_id >= num_tokens:
        return

    # Load token ID and position ID
    token_idx = tl.load(token_ids_ptr + token_id)
    position_idx = tl.load(position_ids_ptr + token_id)

    # Compute offsets for hidden dimension
    d_start = d_block_id * BLOCK_SIZE_D
    d_offs = d_start + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offs < hidden_dim

    # Load embedding
    emb_ptr = embedding_table_ptr + token_idx * stride_emb_v + d_offs * stride_emb_d
    embedding = tl.load(emb_ptr, mask=d_mask, other=0.0)

    # Load position encoding
    pos_ptr = position_encoding_ptr + position_idx * stride_pos_p + d_offs * stride_pos_d
    pos_encoding = tl.load(pos_ptr, mask=d_mask, other=0.0)

    # Fused: embedding * scale + position_encoding
    output = embedding * scale + pos_encoding

    # Store output
    out_ptr = output_ptr + token_id * stride_out_n + d_offs * stride_out_d
    tl.store(out_ptr, output, mask=d_mask)


def embedding_lookup_pos_encoding(
    embedding_table: torch.Tensor,
    position_encoding: torch.Tensor,
    token_ids: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused embedding lookup and position encoding.

    Args:
        embedding_table: Embedding table of shape [vocab_size, hidden_dim]
        position_encoding: Position encoding of shape [max_seq_len, hidden_dim]
        token_ids: Token IDs of shape [num_tokens]
        position_ids: Optional position IDs of shape [num_tokens].
                     If None, uses range(num_tokens).
        scale: Scaling factor for embeddings (default: 1.0)

    Returns:
        Output embeddings of shape [num_tokens, hidden_dim]
    """
    vocab_size, hidden_dim = embedding_table.shape
    num_tokens = token_ids.numel()

    # Create position IDs if not provided
    if position_ids is None:
        position_ids = torch.arange(num_tokens, device=token_ids.device, dtype=token_ids.dtype)

    # Output buffer
    output = torch.empty(
        num_tokens, hidden_dim,
        device=embedding_table.device,
        dtype=embedding_table.dtype
    )

    # Launch kernel
    BLOCK_SIZE_D = min(triton.next_power_of_2(hidden_dim), 512)
    grid = (num_tokens, triton.cdiv(hidden_dim, BLOCK_SIZE_D))

    _embedding_lookup_pos_encoding_kernel[grid](
        output,
        embedding_table,
        position_encoding,
        token_ids,
        position_ids,
        num_tokens,
        hidden_dim,
        scale,
        output.stride(0),
        output.stride(1),
        embedding_table.stride(0),
        embedding_table.stride(1),
        position_encoding.stride(0),
        position_encoding.stride(1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return output


@triton.jit
def _add_position_encoding_kernel(
    output_ptr,
    input_ptr,
    position_encoding_ptr,
    position_ids_ptr,
    num_tokens,
    hidden_dim: tl.constexpr,
    stride_n: tl.constexpr,
    stride_d: tl.constexpr,
    stride_pos_p: tl.constexpr,
    stride_pos_d: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Add position encoding to input embeddings.

    Performs: output[i] = input[i] + position_encoding[position_ids[i]]
    """
    token_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    if token_id >= num_tokens:
        return

    # Load position ID
    position_idx = tl.load(position_ids_ptr + token_id)

    # Compute offsets
    d_start = d_block_id * BLOCK_SIZE_D
    d_offs = d_start + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offs < hidden_dim

    # Load input embedding
    inp_ptr = input_ptr + token_id * stride_n + d_offs * stride_d
    inp_emb = tl.load(inp_ptr, mask=d_mask, other=0.0)

    # Load position encoding
    pos_ptr = position_encoding_ptr + position_idx * stride_pos_p + d_offs * stride_pos_d
    pos_encoding = tl.load(pos_ptr, mask=d_mask, other=0.0)

    # Add position encoding
    output = inp_emb + pos_encoding

    # Store output
    out_ptr = output_ptr + token_id * stride_n + d_offs * stride_d
    tl.store(out_ptr, output, mask=d_mask)


def add_position_encoding(
    input_embeddings: torch.Tensor,
    position_encoding: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add position encoding to input embeddings.

    Args:
        input_embeddings: Input embeddings of shape [num_tokens, hidden_dim]
        position_encoding: Position encoding of shape [max_seq_len, hidden_dim]
        position_ids: Optional position IDs of shape [num_tokens].
                     If None, uses range(num_tokens).

    Returns:
        Output embeddings of shape [num_tokens, hidden_dim]
    """
    num_tokens, hidden_dim = input_embeddings.shape

    # Create position IDs if not provided
    if position_ids is None:
        position_ids = torch.arange(num_tokens, device=input_embeddings.device, dtype=torch.long)

    # Output buffer
    output = torch.empty_like(input_embeddings)

    # Launch kernel
    BLOCK_SIZE_D = min(triton.next_power_of_2(hidden_dim), 512)
    grid = (num_tokens, triton.cdiv(hidden_dim, BLOCK_SIZE_D))

    _add_position_encoding_kernel[grid](
        output,
        input_embeddings,
        position_encoding,
        position_ids,
        num_tokens,
        hidden_dim,
        input_embeddings.stride(0),
        input_embeddings.stride(1),
        position_encoding.stride(0),
        position_encoding.stride(1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return output
