import math

import torch
import triton
import triton.language as tl

from rollout.constant import PREFILL_BLOCK_M, DECODE_SPLITS


@triton.jit
def _rope_q(q1, q2, cq, sq):
    q1f = q1.to(tl.float32)
    q2f = q2.to(tl.float32)
    o1 = (q1f * cq - q2f * sq).to(q1.dtype)
    o2 = (q2f * cq + q1f * sq).to(q1.dtype)
    return o1, o2


@triton.jit
def _rope_k(k1, k2, ck, sk):
    k1f = k1.to(tl.float32)
    k2f = k2.to(tl.float32)
    k1o = (k1f * ck - k2f * sk).to(k1.dtype)
    k2o = (k2f * ck + k1f * sk).to(k1.dtype)
    return k1o, k2o


@triton.jit
def _rope_qk_dot(q1o, q2o, qp, k1, k2, kp, ck, sk):
    k1o, k2o = _rope_k(k1, k2, ck, sk)
    return tl.dot(q1o, k1o.T) + tl.dot(q2o, k2o.T) + tl.dot(qp, kp.T)


@triton.jit
def _unrope(d1o, d2o, c, s):


    d1of = d1o.to(tl.float32)
    d2of = d2o.to(tl.float32)
    d1 = d1of * c + d2of * s
    d2 = d2of * c - d1of * s
    return d1, d2


@triton.jit
def _prefill_kernel(
        Q, Out, Lse,
        Kpool, Vpool,
        PageTable, MaskTable, CuPages, CuQ, Qpos, PagePos,
        RopeCos, RopeSin,
        softmax_scale,
        stride_qm, stride_qh,
        stride_om, stride_oh,
        stride_kvn, stride_kvh,
        seqlen_q,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        HALF: tl.constexpr, ROT: tl.constexpr, PASS: tl.constexpr, PASS_PAD: tl.constexpr,
        GROUP_SIZE: tl.constexpr, PAGE: tl.constexpr,
        RETURN_LSE: tl.constexpr,
):
    blk = tl.program_id(0)
    req = tl.program_id(1)
    off_h = tl.program_id(2)
    off_kv_h = off_h // GROUP_SIZE

    q_start = tl.load(CuQ + req)
    Lq = tl.load(CuQ + req + 1) - q_start
    q_blk_off = blk * BLOCK_M
    if q_blk_off >= Lq:
        return

    p0 = tl.load(CuPages + req)
    p1 = tl.load(CuPages + req + 1)
    q_base_pos = tl.load(Qpos + req)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF)
    offs_p = tl.arange(0, PASS_PAD)
    p_mask = offs_p < PASS
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_local = q_blk_off + offs_m
    q_rows = q_start + q_local
    qm_mask = q_local < Lq

    qbase = off_h * stride_qh + q_rows[:, None] * stride_qm
    q1 = tl.load(Q + qbase + offs_h[None, :], mask=qm_mask[:, None], other=0.0)
    q2 = tl.load(Q + qbase + (HALF + offs_h)[None, :], mask=qm_mask[:, None], other=0.0)
    qp = tl.load(Q + qbase + (ROT + offs_p)[None, :], mask=qm_mask[:, None] & p_mask[None, :], other=0.0)

    pos_q = q_base_pos + q_local
    cq = tl.load(RopeCos + pos_q[:, None] * HALF + offs_h[None, :], mask=qm_mask[:, None], other=0.0).to(tl.float32)
    sq = tl.load(RopeSin + pos_q[:, None] * HALF + offs_h[None, :], mask=qm_mask[:, None], other=0.0).to(tl.float32)
    q1o, q2o = _rope_q(q1, q2, cq, sq)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)


    q_max_pos = q_base_pos + q_blk_off + BLOCK_M - 1
    for pi in range(p0, p1):
        pos_base = tl.load(PagePos + pi)
        if pos_base <= q_max_pos:


            pg = tl.load(PageTable + pi).to(tl.int64)
            valid = tl.load(MaskTable + pi).to(tl.int32)
            tok0 = pg * PAGE
            for sub in tl.static_range(0, PAGE, BLOCK_N):
                toks = sub + offs_n
                n_mask = toks < valid
                krow = (tok0 + toks)[:, None] * stride_kvn + off_kv_h * stride_kvh
                k1 = tl.load(Kpool + krow + offs_h[None, :], mask=n_mask[:, None], other=0.0)
                k2 = tl.load(Kpool + krow + (HALF + offs_h)[None, :], mask=n_mask[:, None], other=0.0)
                kp = tl.load(Kpool + krow + (ROT + offs_p)[None, :], mask=n_mask[:, None] & p_mask[None, :], other=0.0)
                v = tl.load(Vpool + krow + offs_d[None, :], mask=n_mask[:, None], other=0.0)

                pos_k = pos_base + toks
                ck = tl.load(RopeCos + pos_k[:, None] * HALF + offs_h[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
                sk = tl.load(RopeSin + pos_k[:, None] * HALF + offs_h[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)

                qk = _rope_qk_dot(q1o, q2o, qp, k1, k2, kp, ck, sk)
                causal = pos_q[:, None] >= pos_k[None, :]
                qk = tl.where(causal & n_mask[None, :], qk, float("-inf"))

                m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, m_i)
                p = tl.exp(qk * softmax_scale - m_ij[:, None])
                l_ij = tl.sum(p, 1)
                acc_o = acc_o * tl.exp(m_i - m_ij)[:, None]
                acc_o += tl.dot(p.to(v.dtype), v)
                m_i = m_ij
                lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

    acc_o = acc_o * tl.exp(m_i - lse_i)[:, None]
    tl.store(Out + off_h * stride_oh + q_rows[:, None] * stride_om + offs_d[None, :],
             acc_o, mask=qm_mask[:, None])
    if RETURN_LSE:

        tl.store(Lse + off_h * seqlen_q + q_rows, lse_i, mask=qm_mask)


def _prefill_geom(q, k_pool, cos, num_kv_heads):
    S, nheads, d = q.shape
    HALF = cos.shape[1]
    ROT = 2 * HALF
    P = k_pool.shape[1]
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)

    assert d == BLOCK_HEADDIM, f"head_dim={d}  must be  2  power of "
    PASS = BLOCK_HEADDIM - ROT
    PASS_PAD = max(triton.next_power_of_2(PASS), 16)
    GROUP_SIZE = nheads // num_kv_heads
    return S, nheads, d, HALF, ROT, P, BLOCK_HEADDIM, PASS, PASS_PAD, GROUP_SIZE


def _prefill_forward(q, page_table, mask_table, cu_pages, cu_q, qpos, page_pos, k_pool, v_pool,
                     cos, sin, num_kv_heads, num_seg, softmax_scale, return_lse):

    q = q if q.stride(-1) == 1 else q.contiguous()
    S, nheads, d, HALF, ROT, P, BLOCK_HEADDIM, PASS, PASS_PAD, GROUP_SIZE = \
        _prefill_geom(q, k_pool, cos, num_kv_heads)
    BLOCK_M = PREFILL_BLOCK_M


    max_blocks = max(1, (S + BLOCK_M - 1) // BLOCK_M)
    kf = k_pool.view(-1, num_kv_heads, d)
    vf = v_pool.view(-1, num_kv_heads, d)
    o = torch.empty_like(q)
    lse = torch.empty(nheads, S, dtype=torch.float32, device=q.device) if return_lse \
        else torch.empty(1, dtype=torch.float32, device=q.device)
    _prefill_kernel[(max_blocks, num_seg, nheads)](
        q, o, lse, kf, vf,
        page_table, mask_table, cu_pages, cu_q, qpos, page_pos,
        cos, sin,
        softmax_scale,
        q.stride(0), q.stride(1),
        o.stride(0), o.stride(1),
        kf.stride(0), kf.stride(1),
        S,
        BLOCK_M=BLOCK_M, BLOCK_N=min(64, P),
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        HALF=HALF, ROT=ROT, PASS=PASS, PASS_PAD=PASS_PAD,
        GROUP_SIZE=GROUP_SIZE, PAGE=P,
        RETURN_LSE=return_lse,
        num_warps=4 if d <= 64 else 8, num_stages=1,
    )
    return (o, lse) if return_lse else o


def flash_prefill(q, page_table, mask_table, cu_pages, cu_q, qpos, page_pos, k_pool, v_pool,
                  cos, sin, num_kv_heads, num_seg, softmax_scale=None):
    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
    return _prefill_forward(q, page_table, mask_table, cu_pages, cu_q, qpos, page_pos,
                            k_pool, v_pool, cos, sin, num_kv_heads, num_seg,
                            softmax_scale, return_lse=False)


@triton.jit
def _prefill_bwd_preprocess(
        Out, DO, Delta,
        CuQ,
        stride_om, stride_oh,
        stride_dom, stride_doh,
        seqlen_q,
        BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
):
    blk = tl.program_id(0)
    req = tl.program_id(1)
    off_h = tl.program_id(2)

    q_start = tl.load(CuQ + req)
    Lq = tl.load(CuQ + req + 1) - q_start
    q_blk_off = blk * BLOCK_M
    if q_blk_off >= Lq:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_local = q_blk_off + offs_m
    q_rows = q_start + q_local
    m_mask = q_local < Lq

    o = tl.load(Out + off_h * stride_oh + q_rows[:, None] * stride_om + offs_d[None, :],
                mask=m_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + off_h * stride_doh + q_rows[:, None] * stride_dom + offs_d[None, :],
                 mask=m_mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_h * seqlen_q + q_rows, delta, mask=m_mask)


@triton.jit
def _prefill_bwd_kernel(
        Q, DO, DQ, DKpool, DVpool,
        Kpool, Vpool,
        PageTable, MaskTable, CuPages, CuQ, Qpos, PagePos,
        RopeCos, RopeSin,
        LSE, Delta,
        softmax_scale,
        stride_qm, stride_qh,
        stride_dom, stride_doh,
        stride_kvn, stride_kvh,
        seqlen_q,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        HALF: tl.constexpr, ROT: tl.constexpr, PASS: tl.constexpr, PASS_PAD: tl.constexpr,
        GROUP_SIZE: tl.constexpr, PAGE: tl.constexpr,
):
    blk = tl.program_id(0)
    req = tl.program_id(1)
    off_h = tl.program_id(2)
    off_kv_h = off_h // GROUP_SIZE

    q_start = tl.load(CuQ + req)
    Lq = tl.load(CuQ + req + 1) - q_start
    q_blk_off = blk * BLOCK_M
    if q_blk_off >= Lq:
        return

    p0 = tl.load(CuPages + req)
    p1 = tl.load(CuPages + req + 1)
    q_base_pos = tl.load(Qpos + req)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF)
    offs_p = tl.arange(0, PASS_PAD)
    p_mask = offs_p < PASS
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_local = q_blk_off + offs_m
    q_rows = q_start + q_local
    qm_mask = q_local < Lq

    qbase = off_h * stride_qh + q_rows[:, None] * stride_qm
    q1 = tl.load(Q + qbase + offs_h[None, :], mask=qm_mask[:, None], other=0.0)
    q2 = tl.load(Q + qbase + (HALF + offs_h)[None, :], mask=qm_mask[:, None], other=0.0)
    qp = tl.load(Q + qbase + (ROT + offs_p)[None, :], mask=qm_mask[:, None] & p_mask[None, :], other=0.0)

    pos_q = q_base_pos + q_local
    cq = tl.load(RopeCos + pos_q[:, None] * HALF + offs_h[None, :], mask=qm_mask[:, None], other=0.0).to(tl.float32)
    sq = tl.load(RopeSin + pos_q[:, None] * HALF + offs_h[None, :], mask=qm_mask[:, None], other=0.0).to(tl.float32)
    q1o, q2o = _rope_q(q1, q2, cq, sq)

    do = tl.load(DO + off_h * stride_doh + q_rows[:, None] * stride_dom + offs_d[None, :],
                 mask=qm_mask[:, None], other=0.0)
    lse_i = tl.load(LSE + off_h * seqlen_q + q_rows, mask=qm_mask, other=0.0)
    Di = tl.load(Delta + off_h * seqlen_q + q_rows, mask=qm_mask, other=0.0)

    dq1o = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    dq2o = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    dqp = tl.zeros([BLOCK_M, PASS_PAD], dtype=tl.float32)

    q_max_pos = q_base_pos + q_blk_off + BLOCK_M - 1
    for pi in range(p0, p1):
        pos_base = tl.load(PagePos + pi)
        if pos_base <= q_max_pos:
            pg = tl.load(PageTable + pi).to(tl.int64)
            valid = tl.load(MaskTable + pi).to(tl.int32)
            tok0 = pg * PAGE
            for sub in tl.static_range(0, PAGE, BLOCK_N):
                toks = sub + offs_n
                n_mask = toks < valid
                krow = (tok0 + toks)[:, None] * stride_kvn + off_kv_h * stride_kvh
                k1 = tl.load(Kpool + krow + offs_h[None, :], mask=n_mask[:, None], other=0.0)
                k2 = tl.load(Kpool + krow + (HALF + offs_h)[None, :], mask=n_mask[:, None], other=0.0)
                kp = tl.load(Kpool + krow + (ROT + offs_p)[None, :], mask=n_mask[:, None] & p_mask[None, :], other=0.0)
                v = tl.load(Vpool + krow + offs_d[None, :], mask=n_mask[:, None], other=0.0)

                pos_k = pos_base + toks
                ck = tl.load(RopeCos + pos_k[:, None] * HALF + offs_h[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
                sk = tl.load(RopeSin + pos_k[:, None] * HALF + offs_h[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
                k1o, k2o = _rope_k(k1, k2, ck, sk)

                qk = tl.dot(q1o, k1o.T) + tl.dot(q2o, k2o.T) + tl.dot(qp, kp.T)
                causal = pos_q[:, None] >= pos_k[None, :]
                valid_ij = causal & n_mask[None, :] & qm_mask[:, None]
                p = tl.where(valid_ij, tl.exp(qk * softmax_scale - lse_i[:, None]), 0.0)

                dp = tl.dot(do, v.T)
                ds = (p * (dp - Di[:, None]) * softmax_scale).to(k1o.dtype)


                tl.atomic_add(DVpool + krow + offs_d[None, :], tl.dot(p.to(do.dtype).T, do),
                              mask=n_mask[:, None], sem='relaxed')
                dk1o = tl.dot(ds.T, q1o)
                dk2o = tl.dot(ds.T, q2o)
                dkp = tl.dot(ds.T, qp)
                dk1, dk2 = _unrope(dk1o, dk2o, ck, sk)
                tl.atomic_add(DKpool + krow + offs_h[None, :], dk1, mask=n_mask[:, None], sem='relaxed')
                tl.atomic_add(DKpool + krow + (HALF + offs_h)[None, :], dk2, mask=n_mask[:, None], sem='relaxed')
                tl.atomic_add(DKpool + krow + (ROT + offs_p)[None, :], dkp,
                              mask=n_mask[:, None] & p_mask[None, :], sem='relaxed')


                dq1o += tl.dot(ds, k1o)
                dq2o += tl.dot(ds, k2o)
                dqp += tl.dot(ds, kp)

    dq1, dq2 = _unrope(dq1o, dq2o, cq, sq)
    dqbase = off_h * stride_qh + q_rows[:, None] * stride_qm
    tl.store(DQ + dqbase + offs_h[None, :], dq1, mask=qm_mask[:, None])
    tl.store(DQ + dqbase + (HALF + offs_h)[None, :], dq2, mask=qm_mask[:, None])
    tl.store(DQ + dqbase + (ROT + offs_p)[None, :], dqp, mask=qm_mask[:, None] & p_mask[None, :])


def _prefill_backward(do, q, o, lse, page_table, mask_table, cu_pages, cu_q, qpos, page_pos,
                      k_pool, v_pool, cos, sin, num_kv_heads, num_seg, softmax_scale):
    do = do if do.stride(-1) == 1 else do.contiguous()
    S, nheads, d, HALF, ROT, P, BLOCK_HEADDIM, PASS, PASS_PAD, GROUP_SIZE = \
        _prefill_geom(q, k_pool, cos, num_kv_heads)


    BLOCK_M = min(PREFILL_BLOCK_M, 64)
    max_blocks = max(1, (S + BLOCK_M - 1) // BLOCK_M)

    kf = k_pool.view(-1, num_kv_heads, d)
    vf = v_pool.view(-1, num_kv_heads, d)
    dq = torch.zeros_like(q)

    dkf = torch.zeros_like(kf, dtype=torch.float32)
    dvf = torch.zeros_like(vf, dtype=torch.float32)
    delta = torch.empty(nheads, S, dtype=torch.float32, device=q.device)

    grid = (max_blocks, num_seg, nheads)
    _prefill_bwd_preprocess[grid](
        o, do, delta, cu_q,
        o.stride(0), o.stride(1),
        do.stride(0), do.stride(1),
        S,
        BLOCK_M=BLOCK_M, BLOCK_HEADDIM=BLOCK_HEADDIM,
    )
    _prefill_bwd_kernel[grid](
        q, do, dq, dkf, dvf, kf, vf,
        page_table, mask_table, cu_pages, cu_q, qpos, page_pos,
        cos, sin,
        lse, delta,
        softmax_scale,
        q.stride(0), q.stride(1),
        do.stride(0), do.stride(1),
        kf.stride(0), kf.stride(1),
        S,
        BLOCK_M=BLOCK_M, BLOCK_N=min(64, P),
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        HALF=HALF, ROT=ROT, PASS=PASS, PASS_PAD=PASS_PAD,
        GROUP_SIZE=GROUP_SIZE, PAGE=P,
        num_warps=4 if d <= 64 else 8, num_stages=1,
    )
    dk_pool = dkf.view_as(k_pool).to(k_pool.dtype)
    dv_pool = dvf.view_as(v_pool).to(v_pool.dtype)
    return dq, dk_pool, dv_pool


class _FlashPrefill(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_pool, v_pool, page_table, mask_table, cu_pages, cu_q, qpos, page_pos,
                cos, sin, num_kv_heads, num_seg, softmax_scale):
        q = q if q.stride(-1) == 1 else q.contiguous()
        softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        o, lse = _prefill_forward(q, page_table, mask_table, cu_pages, cu_q, qpos, page_pos,
                                  k_pool, v_pool, cos, sin, num_kv_heads, num_seg,
                                  softmax_scale, return_lse=True)
        ctx.save_for_backward(q, o, lse, k_pool, v_pool, page_table, mask_table, cu_pages,
                              cu_q, qpos, page_pos, cos, sin)
        ctx.num_kv_heads = num_kv_heads
        ctx.num_seg = num_seg
        ctx.softmax_scale = softmax_scale
        return o

    @staticmethod
    def backward(ctx, do):
        (q, o, lse, k_pool, v_pool, page_table, mask_table, cu_pages,
         cu_q, qpos, page_pos, cos, sin) = ctx.saved_tensors
        dq, dk_pool, dv_pool = _prefill_backward(
            do, q, o, lse, page_table, mask_table, cu_pages, cu_q, qpos, page_pos,
            k_pool, v_pool, cos, sin, ctx.num_kv_heads, ctx.num_seg, ctx.softmax_scale)
        return (dq, dk_pool, dv_pool, None, None, None, None, None, None,
                None, None, None, None, None)


def flash_prefill_func(q, k_pool, v_pool, page_table, mask_table, cu_pages, cu_q, qpos, page_pos,
                       cos, sin, num_kv_heads, num_seg, softmax_scale=None):

    return _FlashPrefill.apply(q, k_pool, v_pool, page_table, mask_table, cu_pages, cu_q,
                               qpos, page_pos, cos, sin, num_kv_heads, num_seg, softmax_scale)


@triton.jit
def _decode_split_kernel(
        Q, Partial, Mpart, Lpart,
        Kpool, Vpool,
        PageTable, MaskTable, CuPages, Qpos, PagePos,
        RopeCos, RopeSin,
        softmax_scale,
        stride_qn, stride_qh,
        stride_pn, stride_ph, stride_ps,
        stride_mn, stride_mh,
        stride_kvn, stride_kvh,
        num_splits,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        HALF: tl.constexpr, ROT: tl.constexpr, PASS: tl.constexpr, PASS_PAD: tl.constexpr,
        GROUP_SIZE: tl.constexpr, PAGE: tl.constexpr,
):
    bid = tl.program_id(0)
    off_kv_h = tl.program_id(1)
    sid = tl.program_id(2)

    p0 = tl.load(CuPages + bid)
    p1 = tl.load(CuPages + bid + 1)
    npages = p1 - p0
    chunk = (npages + num_splits - 1) // num_splits
    sp0 = sid * chunk
    sp1 = tl.minimum((sid + 1) * chunk, npages)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF)
    offs_p = tl.arange(0, PASS_PAD)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    p_mask = offs_p < PASS
    m_mask = offs_m < GROUP_SIZE
    h = off_kv_h * GROUP_SIZE + offs_m

    qbase = bid * stride_qn + h[:, None] * stride_qh
    q1 = tl.load(Q + qbase + offs_h[None, :], mask=m_mask[:, None], other=0.0)
    q2 = tl.load(Q + qbase + (HALF + offs_h)[None, :], mask=m_mask[:, None], other=0.0)
    qp = tl.load(Q + qbase + (ROT + offs_p)[None, :], mask=m_mask[:, None] & p_mask[None, :], other=0.0)

    pos_q = tl.load(Qpos + bid) + offs_m * 0
    cq = tl.load(RopeCos + pos_q[:, None] * HALF + offs_h[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
    sq = tl.load(RopeSin + pos_q[:, None] * HALF + offs_h[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
    q1o, q2o = _rope_q(q1, q2, cq, sq)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)


    for j in range(sp0, sp1):
        valid = tl.load(MaskTable + p0 + j).to(tl.int32)
        pg = tl.load(PageTable + p0 + j).to(tl.int64)
        pos_base = tl.load(PagePos + p0 + j)
        tok0 = pg * PAGE
        for sub in tl.static_range(0, PAGE, BLOCK_N):
            toks = sub + offs_n
            n_mask = toks < valid
            krow = (tok0 + toks)[:, None] * stride_kvn + off_kv_h * stride_kvh
            k1 = tl.load(Kpool + krow + offs_h[None, :], mask=n_mask[:, None], other=0.0)
            k2 = tl.load(Kpool + krow + (HALF + offs_h)[None, :], mask=n_mask[:, None], other=0.0)
            kp = tl.load(Kpool + krow + (ROT + offs_p)[None, :], mask=n_mask[:, None] & p_mask[None, :], other=0.0)
            v = tl.load(Vpool + krow + offs_d[None, :], mask=n_mask[:, None], other=0.0)

            pos_k = pos_base + toks
            ck = tl.load(RopeCos + pos_k[:, None] * HALF + offs_h[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
            sk = tl.load(RopeSin + pos_k[:, None] * HALF + offs_h[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)

            qk = _rope_qk_dot(q1o, q2o, qp, k1, k2, kp, ck, sk)
            qk = tl.where(n_mask[None, :], qk * softmax_scale, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    po = bid * stride_pn + h[:, None] * stride_ph + sid * stride_ps + offs_d[None, :]
    tl.store(Partial + po, acc, mask=m_mask[:, None])
    mo = bid * stride_mn + h * stride_mh + sid
    tl.store(Mpart + mo, m_i, mask=m_mask)
    tl.store(Lpart + mo, l_i, mask=m_mask)


@triton.jit
def _decode_combine(
        Partial, Mpart, Lpart, Out,
        stride_pn, stride_ph, stride_ps,
        stride_mn, stride_mh,
        stride_on, stride_oh,
        BLOCK_HEADDIM: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    bid = tl.program_id(0)
    off_h = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_s = tl.arange(0, NUM_SPLITS)

    mo = bid * stride_mn + off_h * stride_mh + offs_s
    m = tl.load(Mpart + mo)
    l = tl.load(Lpart + mo)
    m_g = tl.max(m)
    scale = tl.exp(m - m_g)
    l_g = tl.sum(l * scale)

    po = bid * stride_pn + off_h * stride_ph + offs_s[:, None] * stride_ps + offs_d[None, :]
    acc = tl.load(Partial + po)
    acc = tl.sum(acc * scale[:, None], axis=0) / l_g
    tl.store(Out + bid * stride_on + off_h * stride_oh + offs_d, acc)


def _pick_num_splits(N, kv_heads, device, ns_max=64):
    n_sm = torch.cuda.get_device_properties(device).multi_processor_count
    ns = min(max(DECODE_SPLITS, n_sm // (N * kv_heads)), ns_max)
    return 1 << (ns.bit_length() - 1)


def flash_decode(q, page_table, mask_table, cu_pages, qpos, page_pos, k_pool, v_pool,
                 cos, sin, num_kv_heads, num_splits, softmax_scale=None,
                 partial=None, m_part=None, l_part=None, out=None):

    q = q if q.stride(-1) == 1 else q.contiguous()
    N, nheads, d = q.shape
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    HALF = cos.shape[1]
    ROT = 2 * HALF
    P = k_pool.shape[1]
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    assert d == BLOCK_HEADDIM, f"head_dim={d}  must be  2  power of "
    PASS = BLOCK_HEADDIM - ROT
    PASS_PAD = max(triton.next_power_of_2(PASS), 16)
    GROUP_SIZE = nheads // num_kv_heads
    BLOCK_M = max(16, triton.next_power_of_2(GROUP_SIZE))
    BLOCK_N = P

    kf = k_pool.view(-1, num_kv_heads, d)
    vf = v_pool.view(-1, num_kv_heads, d)
    o = out if out is not None else torch.empty_like(q)
    if partial is None:
        partial = torch.empty(N, nheads, num_splits, d, dtype=torch.float32, device=q.device)
        m_part = torch.empty(N, nheads, num_splits, dtype=torch.float32, device=q.device)
        l_part = torch.empty(N, nheads, num_splits, dtype=torch.float32, device=q.device)

    _decode_split_kernel[(N, num_kv_heads, num_splits)](
        q, partial, m_part, l_part,
        kf, vf,
        page_table, mask_table, cu_pages, qpos, page_pos,
        cos, sin,
        softmax_scale,
        q.stride(0), q.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2),
        m_part.stride(0), m_part.stride(1),
        kf.stride(0), kf.stride(1),
        num_splits,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        HALF=HALF, ROT=ROT, PASS=PASS, PASS_PAD=PASS_PAD,
        GROUP_SIZE=GROUP_SIZE, PAGE=P,
        num_warps=4, num_stages=3,
    )
    _decode_combine[(N, nheads)](
        partial, m_part, l_part, o,
        partial.stride(0), partial.stride(1), partial.stride(2),
        m_part.stride(0), m_part.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_HEADDIM=BLOCK_HEADDIM, NUM_SPLITS=num_splits, num_warps=4,
    )
    return o
