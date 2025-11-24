# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _apply_rotary_impl(x_l, x_h, cos_l, cos_h, sin_l, sin_h):
    """Apply rotary positional embedding implementation."""
    # x_l, x_h: [BLOCK, BLOCK_N]
    # cos_l, cos_h, sin_l, sin_h: [BLOCK, BLOCK_N]

    # qe_l = q_l * cos_l - q_h * sin_l
    # qe_h = q_h * cos_h + q_l * sin_h

    # triton 3.4 would do fma 3 times to perform the above computation,
    # which causes higher numerical error. So we manually expand the
    # computation to avoid fma.
    x_l_new0 = x_l * cos_l + 0
    x_l_new1 = x_h * sin_l + 0
    x_h_new0 = x_h * cos_h + 0
    x_h_new1 = x_l * sin_h + 0
    return x_l_new0 - x_l_new1, x_h_new0 + x_h_new1


@triton.jit(do_not_specialize=('seq_len', ))
def apply_rotary_pos_emb_qk_kernel(
    Q,
    K,
    COS,
    SIN,
    Q_EMB,
    K_EMB,
    seq_len,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_qes: tl.constexpr,
    stride_qeh: tl.constexpr,
    stride_qed: tl.constexpr,
    stride_kes: tl.constexpr,
    stride_keh: tl.constexpr,
    stride_ked: tl.constexpr,
    half_size: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply rotary on key AND query kernel."""
    head_id = tl.program_id(0)
    seq_block_id = tl.program_id(1)

    offs_s = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    mask_s = offs_s < seq_len

    feat_size = half_size * 2
    offs_d = tl.arange(0, BLOCK_N)
    mask_d = offs_d < half_size

    feat_offset_l = offs_d
    feat_offset_h = feat_offset_l + half_size

    # [BLOCK, BLOCK_N] mask
    seq_mask = mask_s[:, None] & mask_d[None, :]

    # cos/sin index: [seq, dim]
    cs_offset_l = offs_s[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = offs_s[:, None] * feat_size + feat_offset_h[None, :]

    q_elem_type = Q.dtype.element_ty
    cos_l = tl.astype(tl.load(COS + cs_offset_l, mask=seq_mask, other=0.0), q_elem_type)
    cos_h = tl.astype(tl.load(COS + cs_offset_h, mask=seq_mask, other=0.0), q_elem_type)
    sin_l = tl.astype(tl.load(SIN + cs_offset_l, mask=seq_mask, other=0.0), q_elem_type)
    sin_h = tl.astype(tl.load(SIN + cs_offset_h, mask=seq_mask, other=0.0), q_elem_type)

    if head_id < BLOCK_QH:
        # Q path
        q_ptr = Q + offs_s[:, None] * stride_qs
        qe_ptr = Q_EMB + offs_s[:, None] * stride_qes

        ql_ptrs = q_ptr + feat_offset_l[None, :] * stride_qd + head_id * stride_qh
        qh_ptrs = q_ptr + feat_offset_h[None, :] * stride_qd + head_id * stride_qh

        qel_ptrs = qe_ptr + feat_offset_l[None, :] * stride_qed + head_id * stride_qeh
        qeh_ptrs = qe_ptr + feat_offset_h[None, :] * stride_qed + head_id * stride_qeh

        q_l = tl.load(ql_ptrs, mask=seq_mask, other=0.0)
        q_h = tl.load(qh_ptrs, mask=seq_mask, other=0.0)

        qe_l, qe_h = _apply_rotary_impl(q_l, q_h, cos_l, cos_h, sin_l, sin_h)

        tl.store(qel_ptrs, qe_l, mask=seq_mask)
        tl.store(qeh_ptrs, qe_h, mask=seq_mask)
    else:
        # K path
        k_head_id = head_id - BLOCK_QH

        k_ptr = K + offs_s[:, None] * stride_ks
        ke_ptr = K_EMB + offs_s[:, None] * stride_kes

        kl_ptrs = k_ptr + feat_offset_l[None, :] * stride_kd + k_head_id * stride_kh
        kh_ptrs = k_ptr + feat_offset_h[None, :] * stride_kd + k_head_id * stride_kh

        kel_ptrs = ke_ptr + feat_offset_l[None, :] * stride_ked + k_head_id * stride_keh
        keh_ptrs = ke_ptr + feat_offset_h[None, :] * stride_ked + k_head_id * stride_keh

        k_l = tl.load(kl_ptrs, mask=seq_mask, other=0.0)
        k_h = tl.load(kh_ptrs, mask=seq_mask, other=0.0)

        ke_l, ke_h = _apply_rotary_impl(k_l, k_h, cos_l, cos_h, sin_l, sin_h)

        tl.store(kel_ptrs, ke_l, mask=seq_mask)
        tl.store(keh_ptrs, ke_h, mask=seq_mask)


def apply_rotary_pos_emb(q: Tensor,
                         k: Tensor,
                         cos: Tensor,
                         sin: Tensor,
                         q_embed: Tensor = None,
                         k_embed: Tensor = None):
    """Apply rotary positional embedding on query and key.

    Args:
        q (Tensor): Query state. Expected shape: [..., seq_len, num_heads, head_dim]
        k (Tensor): Key state. Expected shape: [..., seq_len, num_heads, head_dim]
        cos (Tensor): cosine matrix (seq_len, dim). Must be contiguous.
        sin (Tensor): sine matrix (seq_len, dim). Must be contiguous.
        q_embed (Tensor): output q, can be same as q
        k_embed (Tensor): output k, can be same as k

    Returns:
        Tuple[Tensor, Tensor]: Embedded query and key.
    """
    # Ensure cos/sin are contiguous for safe flatten indexing
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    
    if cos.device != q.device:
        cos = cos.to(device=q.device)
    if sin.device != q.device:
        sin = sin.to(device=q.device)

    if q_embed is None:
        q_embed = torch.empty_like(q)
    if k_embed is None:
        k_embed = torch.empty_like(k)

    seq_len = cos.numel() // cos.size(-1)

    if q.size(-1) == cos.size(-1):
        half_size = q.size(-1) // 2
    elif q.size(-1) > cos.size(-1):
        # only do rope with rope_dim size
        half_size = cos.size(-1) // 2
    else:
        raise ValueError('Not support head_dim < rope_dim, '
                         f'but given head_dim={q.size(-1)} '
                         f'rope_dim={cos.size(-1)}')
    # Use half_size directly instead of next_power_of_2 to avoid wasting registers
    BLOCK_N = half_size
    num_heads_q = q.size(-2)
    num_heads_k = k.size(-2)
    num_warps = 2
    num_stages = 1

    # compute best BLOCK size
    num_threads = num_warps * 32
    elem_size = q.dtype.itemsize
    elem_per_ldgv4 = 16 // elem_size
    BLOCK = num_threads * elem_per_ldgv4 // BLOCK_N
    BLOCK = max(1, BLOCK)

    grid = (
        num_heads_q + num_heads_k,
        triton.cdiv(seq_len, BLOCK),
    )
    apply_rotary_pos_emb_qk_kernel[grid](q,
                                         k,
                                         cos,
                                         sin,
                                         q_embed,
                                         k_embed,
                                         seq_len=seq_len,
                                         stride_qs=q.stride(-3),
                                         stride_qh=q.stride(-2),
                                         stride_qd=q.stride(-1),
                                         stride_ks=k.stride(-3),
                                         stride_kh=k.stride(-2),
                                         stride_kd=k.stride(-1),
                                         stride_qes=q_embed.stride(-3),
                                         stride_qeh=q_embed.stride(-2),
                                         stride_qed=q_embed.stride(-1),
                                         stride_kes=k_embed.stride(-3),
                                         stride_keh=k_embed.stride(-2),
                                         stride_ked=k_embed.stride(-1),
                                         half_size=half_size,
                                         BLOCK=BLOCK,
                                         BLOCK_QH=num_heads_q,
                                         BLOCK_N=BLOCK_N,
                                         num_warps=num_warps,
                                         num_stages=num_stages)

    return q_embed, k_embed
