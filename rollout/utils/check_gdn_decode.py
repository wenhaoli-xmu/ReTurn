import argparse

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update

from rollout.cache import LinearStateCache

try:
    from causal_conv1d import causal_conv1d_update
except ImportError:
    causal_conv1d_update = None

conv_update = causal_conv1d_update or torch_causal_conv1d_update


NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_C = KEY_DIM * 2 + VALUE_DIM


def split_heads(y):

    B, L, _ = y.shape
    q, k, v = torch.split(y, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    return (q.reshape(B, L, NUM_K_HEADS, HEAD_K_DIM),
            k.reshape(B, L, NUM_K_HEADS, HEAD_K_DIM),
            v.reshape(B, L, NUM_V_HEADS, HEAD_V_DIM))


def make_seqs(lens, K, dtype, device, seed):

    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(CONV_C, 1, K, generator=gen) / K ** 0.5).to(device, dtype)
    b = (torch.randn(CONV_C, generator=gen) * 0.1).to(device, dtype)
    seqs = []
    for L in lens:
        x = torch.randn(CONV_C, L, generator=gen).to(device, dtype)
        g = (-F.softplus(torch.randn(L, NUM_V_HEADS, generator=gen)) * 0.5).to(device)
        beta = torch.sigmoid(torch.randn(L, NUM_V_HEADS, generator=gen)).to(device, dtype)
        seqs.append({"x": x, "g": g, "beta": beta})
    return seqs, w, b


def run_train(seq, w, b, K):
    x, L = seq["x"], seq["x"].shape[1]
    y = F.silu(F.conv1d(x[None], w, b, padding=K - 1, groups=CONV_C)[0, :, :L])
    q, k, v = split_heads(y.transpose(0, 1)[None])
    rep = NUM_V_HEADS // NUM_K_HEADS
    out, state = chunk_gated_delta_rule(
        q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2), v,
        g=seq["g"][None], beta=seq["beta"][None],
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    return out[0], state[0]


def prefill_chunk(x_seg, hist, g_seg, beta_seg, init, w, b, K):

    comb = torch.cat([hist, x_seg], dim=1)
    y = F.silu(F.conv1d(comb[None], w, b, groups=CONV_C)[0])
    q, k, v = split_heads(y.transpose(0, 1)[None])
    rep = NUM_V_HEADS // NUM_K_HEADS
    out, state = chunk_gated_delta_rule(
        q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2), v,
        g=g_seg[None], beta=beta_seg[None],
        initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=True)
    return out[0], state[0], comb[:, -K:].contiguous()


def decode_step(x_t, conv_state, rec_state, g_t, beta_t, w2, b):

    y = conv_update(x_t, conv_state, w2, b, "silu").transpose(1, 2)
    q, k, v = split_heads(y)
    out, last = fused_recurrent_gated_delta_rule(
        q, k, v, g=g_t, beta=beta_t, initial_state=rec_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    return out, last


def _stack_step(seqs, order, pos, key, dtype=None):
    t = torch.stack([seqs[i][key][:, pos[i]:pos[i] + 1] if key == "x"
                     else seqs[i][key][pos[i]:pos[i] + 1] for i in order])
    return t


def run_rollout_manual(seqs, phases, w, b, K):
    N, w2 = len(seqs), w.squeeze(1)
    P1, D1, P2, D2 = phases
    outs = [[] for _ in range(N)]
    zeros_hist = w.new_zeros(CONV_C, K - 1)

    sts, wins = [], []
    for i, s in enumerate(seqs):
        o, st, win = prefill_chunk(s["x"][:, :P1[i]], zeros_hist, s["g"][:P1[i]],
                                   s["beta"][:P1[i]], None, w, b, K)
        outs[i].append(o)
        sts.append(st)
        wins.append(win)
    conv, rec = torch.stack(wins), torch.stack(sts)

    def decode_round(start, D):
        nonlocal rec
        for t in range(D):
            pos = [start[i] + t for i in range(N)]
            out, rec = decode_step(_stack_step(seqs, range(N), pos, "x"), conv, rec,
                                   _stack_step(seqs, range(N), pos, "g"),
                                   _stack_step(seqs, range(N), pos, "beta"), w2, b)
            for i in range(N):
                outs[i].append(out[i])

    decode_round([P1[i] for i in range(N)], D1)

    for i, s in enumerate(seqs):
        lo = P1[i] + D1
        o, st, win = prefill_chunk(s["x"][:, lo:lo + P2[i]], conv[i, :, 1:], s["g"][lo:lo + P2[i]],
                                   s["beta"][lo:lo + P2[i]], rec[i:i + 1], w, b, K)
        outs[i].append(o)
        conv[i], rec[i] = win, st

    decode_round([P1[i] + D1 + P2[i] for i in range(N)], D2)
    return [torch.cat(o) for o in outs], conv.clone(), rec.clone()


def run_rollout_cache(seqs, phases, w, b, K, device):
    N, w2 = len(seqs), w.squeeze(1)
    P1, D1, P2, D2 = phases
    outs = [[] for _ in range(N)]
    zeros_hist = w.new_zeros(CONV_C, K - 1)
    dims = (NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, CONV_C, K)
    cache = LinearStateCache(device, N, dims)
    if w.dtype != cache.g_conv.dtype:
        cache.g_conv = torch.zeros_like(cache.g_conv, dtype=w.dtype)
    sids = list(range(N))
    for s in sids:
        cache.alloc(s)

    def decode_round(start, D):
        order = list(sids)
        cache.fill(order)
        for t in range(D):
            if t == D // 2:
                order = order[1:] + order[:1]
                cache.fill(order)
            conv_prev, rec_prev = cache.gather()
            pos = {i: start[i] + t for i in range(N)}
            posl = [pos[i] for i in order]
            out, last = decode_step(
                torch.stack([seqs[i]["x"][:, pos[i]:pos[i] + 1] for i in order]),
                conv_prev, rec_prev,
                torch.stack([seqs[i]["g"][pos[i]:pos[i] + 1] for i in order]),
                torch.stack([seqs[i]["beta"][pos[i]:pos[i] + 1] for i in order]), w2, b)
            cache.scatter(conv_prev, last)
            for j, i in enumerate(order):
                outs[i].append(out[j])
        for s in sids:
            cache.persist(s)


    sts, wins = [], []
    for i, s in enumerate(seqs):
        o, st, win = prefill_chunk(s["x"][:, :P1[i]], zeros_hist, s["g"][:P1[i]],
                                   s["beta"][:P1[i]], None, w, b, K)
        outs[i].append(o)
        sts.append(st)
        wins.append(win)
    cache.scatter_prefill(sids, torch.stack(wins), torch.stack(sts))

    decode_round([P1[i] for i in range(N)], D1)


    hist, init = cache.prefill_state(sids, seqs[0]["x"])
    sts, wins = [], []
    for i, s in enumerate(seqs):
        lo = P1[i] + D1
        o, st, win = prefill_chunk(s["x"][:, lo:lo + P2[i]], hist[i], s["g"][lo:lo + P2[i]],
                                   s["beta"][lo:lo + P2[i]], init[i:i + 1], w, b, K)
        outs[i].append(o)
        sts.append(st)
        wins.append(win)
    cache.scatter_prefill(sids, torch.stack(wins), torch.stack(sts))

    decode_round([P1[i] + D1 + P2[i] for i in range(N)], D2)

    conv = torch.stack([cache.conv[s] for s in sids])
    rec = torch.stack([cache.rec[s] for s in sids])
    return [torch.cat(o) for o in outs], conv, rec


def _err(a, b):
    a, b = a.float(), b.float()
    m = (a - b).abs().max().item()
    return m, (a - b).abs().mean().item(), m / b.abs().max().clamp_min(1e-6).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--p1", type=str, default="", help=" Comma-separated initial  prefill  length ")
    ap.add_argument("--p2", type=str, default="", help=" Comma-separated continuation  prefill  length ")
    ap.add_argument("--d1", type=int, default=192)
    ap.add_argument("--d2", type=int, default=192)
    ap.add_argument("--kernel-size", type=int, default=4)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    K, N = args.kernel_size, args.batch
    gen = torch.Generator().manual_seed(args.seed)
    P1 = [int(x) for x in args.p1.split(",")] if args.p1 else \
        torch.randint(256, 1025, (N,), generator=gen).tolist()
    P2 = [int(x) for x in args.p2.split(",")] if args.p2 else \
        torch.randint(64, 257, (N,), generator=gen).tolist()
    N = len(P1)
    lens = [P1[i] + args.d1 + P2[i] + args.d2 for i in range(N)]
    phases = (P1, args.d1, P2, args.d2)

    print(f" Device ={torch.cuda.get_device_name()}  dtype={args.dtype}  K={K}  seed={args.seed}")
    print(f"conv  update : {'causal_conv1d(CUDA)' if causal_conv1d_update else 'torch fallback(HF)'}")
    print(f" Timeline : P1={P1} + decode {args.d1} + P2={P2} + decode {args.d2}  ( total length ={lens})")

    seqs, w, b = make_seqs(lens, K, dtype, device, args.seed)
    train = [run_train(s, w, b, K) for s in seqs]
    m_out, m_conv, m_rec = run_rollout_manual(seqs, phases, w, b, K)
    c_out, c_conv, c_rec = run_rollout_cache(seqs, phases, w, b, K, device)


    plumb = max(max((c_out[i] - m_out[i]).abs().max().item() for i in range(N)),
                (c_conv - m_conv).abs().max().item(), (c_rec.float() - m_rec.float()).abs().max().item())
    print(f"\n[1] LinearStateCache  state transfer  (fill/gather/scatter/persist/prefill_state) vs  manual per-thread state :")
    print(f"    max abs diff = {plumb:.3e}  {'✅ bit-exact' if plumb == 0 else '❌  state transfer is lossy !'}")


    atol, rtol = (2e-2, 3e-2) if dtype == torch.bfloat16 else (5e-3, 5e-3)
    print(f"\n[2] ROLLOUT(recurrent  per  token) vs TRAIN(chunk  single pass ), atol={atol} rtol={rtol}:")
    print(f"{'seq':>4} | {'prefill max':>11} | {'dec max':>9} {'dec mean':>9} {'dec rel':>8} "
          f"{' first → last  step':>16} | {'state max':>9} {'state rel':>9} | {'close':>5}")
    print("-" * 108)
    all_ok = True
    for i in range(N):
        t_out, t_state = train[i]
        r_out = c_out[i]
        dec = torch.cat([torch.arange(P1[i], P1[i] + args.d1),
                         torch.arange(lens[i] - args.d2, lens[i])]).to(device)
        pre = torch.cat([torch.arange(0, P1[i]),
                         torch.arange(P1[i] + args.d1, P1[i] + args.d1 + P2[i])]).to(device)
        pm, _, _ = _err(r_out[pre], t_out[pre])
        dm, dmean, drel = _err(r_out[dec], t_out[dec])
        e0, _, _ = _err(r_out[P1[i]], t_out[P1[i]])
        e1, _, _ = _err(r_out[lens[i] - 1], t_out[lens[i] - 1])
        sm, _, srel = _err(c_rec[i], t_state)
        ok = (torch.allclose(r_out[dec].float(), t_out[dec].float(), atol=atol, rtol=rtol)
              and torch.allclose(c_rec[i].float(), t_state.float(), atol=atol, rtol=rtol))
        all_ok &= ok
        print(f"{i:>4} | {pm:>11.3e} | {dm:>9.3e} {dmean:>9.3e} {drel:>8.2%} "
              f"{e0:>7.1e}→{e1:>7.1e} | {sm:>9.3e} {srel:>9.2%} | {'✅' if ok else '❌':>4}")
    print("-" * 108)
    print(f" Result :  state transfer  {' lossless ' if plumb == 0 else ' lossy '};decode kernel gap "
          f"{'✅  within tolerance ' if all_ok else '❌  outside tolerance , see table above '}")


if __name__ == "__main__":
    main()
