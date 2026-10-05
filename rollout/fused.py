import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd(X, W, Y, stride_x, N, eps, OFFSET: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    X += row * stride_x
    Y += row * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * (OFFSET + w)
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_gated_fwd(X, W, G, Y, stride_x, stride_g, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    X += row * stride_x
    G += row * stride_g
    Y += row * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd) * w * (g * tl.sigmoid(g))
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


def _rows(x, N):

    xf = x.reshape(-1, N)
    if xf.stride(1) != 1:
        xf = xf.contiguous()
    return xf


def _nwarps(BLOCK_N):
    return min(max(BLOCK_N // 256, 1), 16)


def rms_norm(x, weight, eps, offset=1.0):

    N = x.shape[-1]
    xf = _rows(x, N)
    y = torch.empty(xf.shape, dtype=x.dtype, device=x.device)
    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_fwd[(xf.shape[0],)](xf, weight, y, xf.stride(0), N, eps,
                                  OFFSET=offset, BLOCK_N=BLOCK_N, num_warps=_nwarps(BLOCK_N))
    return y.view(x.shape)


@triton.jit
def _rope_fwd(X, COS, SIN, Y, stride_xr, stride_xh, stride_cr, nheads,
              HD: tl.constexpr, ROT: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    xp = X + r * stride_xr + h * stride_xh
    yp = Y + (r * nheads + h) * HD
    cols = tl.arange(0, BLOCK)

    pm = (cols >= ROT) & (cols < HD)
    tl.store(yp + cols, tl.load(xp + cols, mask=pm, other=0.0), mask=pm)

    i = tl.arange(0, HALF)
    c1 = tl.load(COS + r * stride_cr + i).to(tl.float32)
    s1 = tl.load(SIN + r * stride_cr + i).to(tl.float32)
    c2 = tl.load(COS + r * stride_cr + HALF + i).to(tl.float32)
    s2 = tl.load(SIN + r * stride_cr + HALF + i).to(tl.float32)
    x1 = tl.load(xp + i).to(tl.float32)
    x2 = tl.load(xp + HALF + i).to(tl.float32)
    o1 = x1 * c1 - x2 * s1
    o2 = x2 * c2 + x1 * s2
    tl.store(yp + i, o1.to(Y.dtype.element_ty))
    tl.store(yp + HALF + i, o2.to(Y.dtype.element_ty))


def apply_rope_fused(q, k, cos, sin):
    hd = q.shape[-1]
    rot = cos.shape[-1]
    half = rot // 2
    rows = q.shape[0] * q.shape[1]
    cosf = cos.reshape(rows, rot)
    sinf = sin.reshape(rows, rot)
    BLOCK = triton.next_power_of_2(hd)
    outs = []
    for t in (q, k):
        nh = t.shape[2]
        tf = t.reshape(rows, nh, hd)
        y = torch.empty(rows, nh, hd, dtype=t.dtype, device=t.device)
        _rope_fwd[(rows, nh)](tf, cosf, sinf, y, tf.stride(0), tf.stride(1), cosf.stride(0),
                              nh, HD=hd, ROT=rot, HALF=half, BLOCK=BLOCK, num_warps=4)
        outs.append(y.view(t.shape))
    return outs[0], outs[1]


@triton.jit
def _silu_and_mul_fwd(GU, Y, stride_gu, N, BLOCK_N: tl.constexpr):

    row = tl.program_id(0).to(tl.int64)
    GU += row * stride_gu
    Y += row * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    g = tl.load(GU + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + N + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g * tl.sigmoid(g)) * u
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


def silu_and_mul(gu):

    N = gu.shape[-1] // 2
    guf = gu.reshape(-1, 2 * N)
    if guf.stride(1) != 1:
        guf = guf.contiguous()
    y = torch.empty(guf.shape[0], N, dtype=gu.dtype, device=gu.device)
    BLOCK_N = triton.next_power_of_2(N)
    _silu_and_mul_fwd[(guf.shape[0],)](guf, y, guf.stride(0), N,
                                       BLOCK_N=BLOCK_N, num_warps=_nwarps(BLOCK_N))
    return y.view(*gu.shape[:-1], N)


def rms_norm_gated(x, weight, gate, eps):

    N = x.shape[-1]
    xf = _rows(x, N)
    gf = _rows(gate, N)
    y = torch.empty(xf.shape, dtype=x.dtype, device=x.device)
    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_gated_fwd[(xf.shape[0],)](xf, weight, gf, y, xf.stride(0), gf.stride(0), N, eps,
                                        BLOCK_N=BLOCK_N, num_warps=_nwarps(BLOCK_N))
    return y.view(x.shape)
