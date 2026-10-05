import time
import torch

from rollout.cache import ParaPageCache
from rollout.attention import flash_para_page_decode


def bench_one(N, ctx=16384, P=64, H=16, Hkv=4, D=256, iters=50, dev="cuda:0"):
    scale = 1.0 / (D ** 0.5)
    cache = ParaPageCache(max_tokens=P * ((ctx // P + 2) * N),
                          num_heads=Hkv, head_dim=D, device=dev)
    for r in range(N):
        cache.alloc(r)
        k = torch.randn(ctx, Hkv, D, device=dev, dtype=torch.bfloat16)
        cache.insert_paragraph(r, k, k)
    cache.activate(list(range(N)))

    q = torch.randn(N, H, D, device=dev, dtype=torch.bfloat16)

    for _ in range(10):
        o = flash_para_page_decode(q, cache, scale)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        o = flash_para_page_decode(q, cache, scale)
    torch.cuda.synchronize(dev)
    dt = (time.perf_counter() - t0) / iters


    kv_bytes = N * ctx * Hkv * D * 2 * 2
    return dt, kv_bytes


def main():
    dev = "cuda:0"
    print(f"{'N':>4} {'ms':>9} {'KV GB/s':>9}")
    for N in (1, 4, 16, 64, 256):
        dt, kvb = bench_one(N, dev=dev)
        print(f"{N:>4} {dt*1e3:>9.3f} {kvb/dt/1e9:>9.1f}")


if __name__ == "__main__":
    main()
