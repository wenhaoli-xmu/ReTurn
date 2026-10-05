import argparse
import math

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule


NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128


def make_inputs(lens, dtype, device, seed):

    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    seqs = []
    for L in lens:

        q = torch.randn(1, L, NUM_V_HEADS, HEAD_K_DIM, generator=g_cpu, dtype=torch.float32)
        k = torch.randn(1, L, NUM_V_HEADS, HEAD_K_DIM, generator=g_cpu, dtype=torch.float32)
        v = torch.randn(1, L, NUM_V_HEADS, HEAD_V_DIM, generator=g_cpu, dtype=torch.float32)

        g = -F.softplus(torch.randn(1, L, NUM_V_HEADS, generator=g_cpu, dtype=torch.float32)) * 0.5
        beta = torch.sigmoid(torch.randn(1, L, NUM_V_HEADS, generator=g_cpu, dtype=torch.float32))
        cast = lambda t: t.to(device=device, dtype=dtype)
        seqs.append((cast(q), cast(k), cast(v), g.to(device), beta.to(device)))
    return seqs


def run_eager(seqs):

    outs, states = [], []
    for q, k, v, g, beta in seqs:
        o, s = chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        outs.append(o)
        states.append(s)
    return outs, states


def pack(seqs, device):

    q = torch.cat([s[0] for s in seqs], dim=1)
    k = torch.cat([s[1] for s in seqs], dim=1)
    v = torch.cat([s[2] for s in seqs], dim=1)
    g = torch.cat([s[3] for s in seqs], dim=1)
    beta = torch.cat([s[4] for s in seqs], dim=1)
    lens = [s[0].shape[1] for s in seqs]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                      dtype=torch.int32, device=device)
    return q, k, v, g, beta, cu


def run_varlen(packed):

    q, k, v, g, beta, cu = packed
    o, state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu)


    cul = cu.tolist()
    outs = [o[:, cul[i]:cul[i + 1]] for i in range(len(cul) - 1)]
    return outs, state


def _err(a, b):
    a, b = a.float(), b.float()
    denom = b.abs().max().clamp_min(1e-6)
    return (a - b).abs().max().item(), (a - b).abs().mean().item(), ((a - b).abs().max() / denom).item()


def compare(eager, varlen, lens, dtype):
    e_out, e_state = eager
    v_out, v_state = varlen

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 3e-2 if dtype == torch.bfloat16 else 1e-4

    print(f"\n{'seq':>4} {'len':>7} | {'out max':>10} {'out mean':>10} {'out rel':>9} | "
          f"{'state max':>10} {'state rel':>9} | {'out close':>9} {'st close':>9}")
    print("-" * 92)
    all_ok = True
    for i, L in enumerate(lens):
        om, omean, orel = _err(v_out[i], e_out[i])
        sm, _, srel = _err(v_state[i:i + 1], e_state[i])
        oc = torch.allclose(v_out[i].float(), e_out[i].float(), atol=atol, rtol=rtol)
        sc = torch.allclose(v_state[i:i + 1].float(), e_state[i].float(), atol=atol, rtol=rtol)
        all_ok &= oc and sc
        print(f"{i:>4} {L:>7} | {om:>10.3e} {omean:>10.3e} {orel:>9.2%} | "
              f"{sm:>10.3e} {srel:>9.2%} | {str(oc):>9} {str(sc):>9}")
    print("-" * 92)
    print(f" Accuracy result : {'✅  aligned  (all close within tol)' if all_ok else '❌  mismatch , see table above '}  "
          f"(atol={atol}, rtol={rtol})")
    return all_ok


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8, help=" Sequence count (--lens  uses random lengths when unspecified )")
    ap.add_argument("--lens", type=str, default="", help=" Comma-separated sequence lengths , overrides  --batch")
    ap.add_argument("--min-len", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.lens:
        lens = [int(x) for x in args.lens.split(",")]
    else:
        gcpu = torch.Generator().manual_seed(args.seed)
        lens = torch.randint(args.min_len, args.max_len + 1, (args.batch,), generator=gcpu).tolist()

    total = sum(lens)
    print(f" Device ={torch.cuda.get_device_name()}  dtype={args.dtype}  seed={args.seed}")
    print(f" Sequence count ={len(lens)}   total  token={total}   length ={lens}")
    print(f"GDN: V_heads={NUM_V_HEADS} K_heads={NUM_K_HEADS} head_dim={HEAD_K_DIM} scale={1/math.sqrt(HEAD_K_DIM):.5f}")

    seqs = make_inputs(lens, dtype, device, args.seed)
    packed = pack(seqs, device)

    eager = run_eager(seqs)
    varlen = run_varlen(packed)
    ok = compare(eager, varlen, lens, dtype)

    t_eager = bench(lambda: run_eager(seqs), args.iters, args.warmup)
    t_varlen = bench(lambda: run_varlen(packed), args.iters, args.warmup)
    print(f"\n Performance ({args.iters} iters  mean ,packing/cat  excluded ):")
    print(f"  EAGER ( per sequence  batch=1)   : {t_eager:8.3f} ms")
    print(f"  VARLEN (cu_seqlens  once ): {t_varlen:8.3f} ms")
    print(f"   speedup  eager/varlen     : {t_eager / t_varlen:6.2f}×")
    print(f"\n Summary :  accuracy {' aligned ' if ok else ' mismatch '},varlen {' faster ' if t_varlen < t_eager else ' slower '} "
          f"({t_eager / t_varlen:.2f}×)")


if __name__ == "__main__":
    main()
