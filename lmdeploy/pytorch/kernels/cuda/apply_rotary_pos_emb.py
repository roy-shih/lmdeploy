# Copyright (c) UnieAI.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _apply_rotary_impl(x_l, x_h, cos_l, cos_h, sin_l, sin_h):
    """
    基本 RoPE 計算：
      y_l = x_l * cos_l - x_h * sin_l
      y_h = x_h * cos_h + x_l * sin_h
    """
    t0 = x_l * cos_l
    t1 = x_h * sin_l
    t2 = x_h * cos_h
    t3 = x_l * sin_h
    y_l = t0 - t1
    y_h = t2 + t3
    return y_l, y_h


@triton.jit
def apply_rotary_pos_emb_qk_kernel(
    Q,
    K,
    COS,
    SIN,
    Q_EMB,
    K_EMB,
    seq_len,
    stride_qs, stride_qh, stride_qd,
    stride_ks, stride_kh, stride_kd,
    stride_qes, stride_qeh, stride_qed,
    stride_kes, stride_keh, stride_ked,
    half_size: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_QH: tl.constexpr,
):
    """
    Q/K 上套用 RoPE。假設最後兩維是 [num_heads, head_dim]。
    COS/SIN layout: [seq_len, rope_dim]
    """
    head_id = tl.program_id(0)
    seq_block = tl.program_id(1)

    # seq offset
    offs_s = seq_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < seq_len

    # dim offset (只處理前 half_size，即一半維度)
    offs_d = tl.arange(0, BLOCK_N)
    mask_d = offs_d < half_size

    feat_offset_l = offs_d
    feat_offset_h = feat_offset_l + half_size

    seq_mask = mask_s[:, None] & mask_d[None, :]

    feat_size = half_size * 2

    # cos/sin index: [seq, dim]
    cs_offset_l = offs_s[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = offs_s[:, None] * feat_size + feat_offset_h[None, :]

    q_dtype = Q.dtype.element_ty

    cos_l = tl.load(COS + cs_offset_l, mask=seq_mask, other=0.0)
    cos_h = tl.load(COS + cs_offset_h, mask=seq_mask, other=0.0)
    sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask, other=0.0)
    sin_h = tl.load(SIN + cs_offset_h, mask=seq_mask, other=0.0)

    cos_l = tl.astype(cos_l, q_dtype)
    cos_h = tl.astype(cos_h, q_dtype)
    sin_l = tl.astype(sin_l, q_dtype)
    sin_h = tl.astype(sin_h, q_dtype)

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


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed: Tensor = None,
    k_embed: Tensor = None,
):
    """
    ROCm 友善版 RoPE；使用預先算好的 cos/sin。
    預期 q,k shape: [..., seq_len, num_heads, head_dim]
    cos,sin shape: [seq_len, rope_dim]
    """
    device = q.device
    if cos.device != device:
        cos = cos.to(device=device)
    if sin.device != device:
        sin = sin.to(device=device)

    cos = cos.contiguous()
    sin = sin.contiguous()

    if q_embed is None:
        q_embed = torch.empty_like(q)
    if k_embed is None:
        k_embed = torch.empty_like(k)

    # 假設最後三維是 [seq, heads, dim]
    seq_len = cos.shape[0]
    rope_dim = cos.shape[1]

    head_dim = q.size(-1)
    if head_dim == rope_dim:
        half_size = head_dim // 2
    elif head_dim > rope_dim:
        half_size = rope_dim // 2
    else:
        raise ValueError(
            f"Not support head_dim < rope_dim, got head_dim={head_dim}, rope_dim={rope_dim}"
        )

    # reshape 成 [B, seq_len, num_heads, head_dim]
    q_shape = q.shape
    k_shape = k.shape
    q = q.view(-1, q_shape[-3], q_shape[-2], q_shape[-1])
    k = k.view(-1, k_shape[-3], k_shape[-2], k_shape[-1])
    q_embed = q_embed.view_as(q)
    k_embed = k_embed.view_as(k)

    B = q.shape[0]
    S = q.shape[1]
    H_q = q.shape[2]
    H_k = k.shape[2]
    D = q.shape[3]

    assert S == seq_len, "seq_len mismatch between q and cos"

    # strides
    stride_qs = q.stride(-3)
    stride_qh = q.stride(-2)
    stride_qd = q.stride(-1)

    stride_ks = k.stride(-3)
    stride_kh = k.stride(-2)
    stride_kd = k.stride(-1)

    stride_qes = q_embed.stride(-3)
    stride_qeh = q_embed.stride(-2)
    stride_qed = q_embed.stride(-1)

    stride_kes = k_embed.stride(-3)
    stride_keh = k_embed.stride(-2)
    stride_ked = k_embed.stride(-1)

    # BLOCK 設計：針對 ROCm，BLOCK_N 不用 next_power_of_2，直接用 half_size，BLOCK_S 用 16/32
    BLOCK_N = half_size
    BLOCK_S = 16

    grid = (
        H_q + H_k,           # head 維度（先 Q 後 K）
        triton.cdiv(S, BLOCK_S) * B,  # seq block * batch
    )

    apply_rotary_pos_emb_qk_kernel[grid](
        q,
        k,
        cos,
        sin,
        q_embed,
        k_embed,
        seq_len=S,
        stride_qs=stride_qs,
        stride_qh=stride_qh,
        stride_qd=stride_qd,
        stride_ks=stride_ks,
        stride_kh=stride_kh,
        stride_kd=stride_kd,
        stride_qes=stride_qes,
        stride_qeh=stride_qeh,
        stride_qed=stride_qed,
        stride_kes=stride_kes,
        stride_keh=stride_keh,
        stride_ked=stride_ked,
        half_size=half_size,
        BLOCK_S=BLOCK_S,
        BLOCK_N=BLOCK_N,
        BLOCK_QH=H_q,
        num_warps=2,
    )

    # reshape 回原始形狀
    q_embed = q_embed.view(q_shape)
    k_embed = k_embed.view(k_shape)
    return q_embed, k_embed
