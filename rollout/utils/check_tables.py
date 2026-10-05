import random

import torch

from rollout.cache import KVCache
from rollout.constant import PAGE_SIZE

FAST = [0, 0]


def ref(kv, sids, prefill):

    pg_all, mk_all, pp_all, cu_p, cu_q, qpos = [], [], [], [0], [0], []
    tot_q = 0
    for sid in sids:
        pg, nt = kv.pages[sid], kv.ntok[sid]
        pg_all += list(pg)
        mk_all += list(nt)
        acc = 0
        for c in nt:
            pp_all.append(acc)
            acc += c
        cu_p.append(len(pg_all))
        sl = kv.seqlen_map[sid]
        if prefill:
            qpos.append(sl - kv._req_q[sid])
            tot_q += kv._req_q[sid]
            cu_q.append(tot_q)
        else:
            qpos.append(sl - 1)
    return pg_all, mk_all, pp_all, cu_p, cu_q, qpos, (tot_q if prefill else len(sids))


def cmp(kv, sids, prefill, tag):
    e_pg, e_mk, e_pp, e_cp, e_cq, e_qp, e_tq = ref(kv, sids, prefill)
    was_fast = kv._tkey == (kv._tv, tuple(sids))
    n, tp, tq = kv.build_tables(sids, prefill)
    FAST[0] += was_fast
    FAST[1] += 1
    got = {
        "page_table": kv.page_table[:tp].tolist(),
        "mask_table": kv.mask_table[:tp].tolist(),
        "page_pos": kv.page_pos[:tp].tolist(),
        "cu_pages": kv.cu_pages[:n + 1].tolist(),
        "qpos": kv.qpos[:n].tolist(),
    }
    exp = {"page_table": e_pg, "mask_table": e_mk, "page_pos": e_pp,
           "cu_pages": e_cp, "qpos": e_qp}
    if prefill:
        got["cu_q"], exp["cu_q"] = kv.cu_q[:n + 1].tolist(), e_cq
    for k in exp:
        assert got[k] == exp[k], (
            f"[{tag}] {k}  mismatch （{' fast ' if was_fast else ' slow '} path ，sids={sids}）\n"
            f"  got={got[k][:24]}...\n  exp={exp[k][:24]}...")
    assert (n, tp, tq) == (len(sids), len(e_pg), e_tq), f"[{tag}]  return value mismatch "


def append(kv, sid, L, new_para=False):
    kv.plan_reset()
    kv.plan_append(sid, L, new_para=new_para)
    kv.flush_wpos()


def make_variant(kv, sid, pid, ntok):

    tail = kv.cut(sid, pid)
    append(kv, sid, ntok, new_para=True)
    kv.alt[sid][pid] = kv.paste(sid, tail)


def main():
    random.seed(0)
    dev = "cuda:0"
    kv = KVCache(1, 1 << 18, 1, 8, 8, device=dev)
    sids = ["a", "b", "c"]
    for s in sids:
        kv.alloc(s)
        append(kv, s, random.randint(300, 900), new_para=True)
        append(kv, s, random.randint(300, 900), new_para=True)
        append(kv, s, random.randint(300, 900), new_para=True)
    kv.set_req_q({s: 1 for s in sids})
    cmp(kv, sids, False, " initial ")


    for t in range(PAGE_SIZE * 2 + 5):
        for s in sids:
            append(kv, s, 1)
        cmp(kv, sids, False, f"decode t={t}")


    for sub in ([sids[0]], sids[::-1], sids[1:], sids):
        cmp(kv, sub, False, f"batch={sub}")
        cmp(kv, sub, False, f"batch={sub}  duplicate ")


    kv.set_req_q({"a": 700, "b": 5, "c": 1})
    append(kv, "a", 700, new_para=True)
    append(kv, "b", 5)
    append(kv, "c", 1)
    cmp(kv, sids, True, "prefill")
    kv.set_req_q({s: 1 for s in sids})


    make_variant(kv, "a", 1, 120)
    for turn in range(3):
        kv.swap("a", 1)
        cmp(kv, sids, False, f"swap #{turn + 1}")


    append(kv, "b", 200, new_para=True)
    cmp(kv, sids, False, " before pop ")
    kv.pop_paragraph("b")
    cmp(kv, sids, False, " after pop ")


    kv.free("c")
    cmp(kv, ["a", "b"], False, "free")
    kv.alloc("c")
    append(kv, "c", 400, new_para=True)
    cmp(kv, sids, False, "alloc  reuse ")


    ops = ["dec", "dec", "dec", "para", "swap", "pop", "batch"]
    for i in range(300):
        op = random.choice(ops)
        if op == "dec":
            for s in sids:
                append(kv, s, 1)
        elif op == "para":
            append(kv, random.choice(sids), random.randint(1, 200), new_para=True)
        elif op == "swap":
            kv.swap("a", 1)
        elif op == "pop":
            s = random.choice(sids)
            pid = kv.para_id[s][-1]
            if pid not in kv.alt[s] and len(kv.para_id[s]) > 1:
                kv.pop_paragraph(s)
        sub = random.sample(sids, random.randint(1, len(sids)))
        cmp(kv, sub, False, f" random  {i} {op}")

    assert FAST[0] > FAST[1] // 4, f" fast path call count  {FAST[0]}/{FAST[1]}  times ， comparison did not cover it "
    print(f"build_tables  reference comparison passed ：{FAST[1]}  table builds ， fast path calls  {FAST[0]}  times ")


if __name__ == "__main__":
    main()
