import json
from dataclasses import dataclass
from pathlib import Path


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


@dataclass
class Unit:
    ids: list
    begin: int
    ctx: list


def units(row, fold=True):
    paras = row["paras"]
    ends = [p["start"] for p in paras[1:]] + [len(row["tokens"])]
    table, pid_to_unit, folded = [], {}, set()

    for p, end in zip(paras, ends):
        if fold:
            for pid in p.get("swap") or []:
                assert pid in pid_to_unit, (
                    f"{p['kind']}@{p['start']} swap  references nonexistent  pid={pid}")
                if pid in folded:
                    folded.remove(pid)
                else:
                    folded.add(pid)

        if p["kind"] == "probe":
            continue

        pid = p.get("pid")
        assert pid is not None, f"{p['kind']}@{p['start']}  missing  pid"
        assert pid not in pid_to_unit, f" duplicate  pid={pid}"
        ctx = [index for old_pid, index in pid_to_unit.items()
               if not fold or old_pid not in folded]
        pid_to_unit[pid] = len(table)
        table.append(Unit(row["tokens"][p["start"]:end], p["start"], ctx))


    assert table, " trajectory has no formal （ not  probe） units "
    if fold:
        expected = {p["pid"] for p in paras
                    if p.get("pid") is not None and p.get("folded", False)}
        assert folded == expected, (
            f"swap  replay final state mismatch : replay={sorted(folded)}, trace={sorted(expected)}")
    return table


def assistant_tokens_by_begin(row):
    paras = row["paras"]
    ends = [p["start"] for p in paras[1:]] + [len(row["tokens"])]
    result = {}
    for para, end in zip(paras, ends):
        if para["kind"] != "assistant":
            continue
        tokens = [position for position in range(para["start"], end)
                  if row["loss_mask"][position]]
        if tokens:
            result[para["start"]] = tokens
    return result


def assistant_tokens(row):
    return [position for tokens in assistant_tokens_by_begin(row).values()
            for position in tokens]


def predictor_rows(tokens, begin):
    return [p - begin - 1 for p in tokens]


def max_pos(rows):
    return max(sum(len(unit.ids) for unit in units(row, fold=False)) for row in rows) + 8


def validate(row, teacher=False):
    n = len(row["tokens"])
    for key in ("loss_mask", "rollout_logprobs"):
        assert len(row[key]) == n, (key, len(row[key]), n)
    tokens = assistant_tokens(row)
    assert tokens, " trajectory has no formal  assistant  sampled  token"
    if teacher:
        for key in ("teacher_topk_ids", "teacher_topk_logprobs",
                    "teacher_token_logprobs"):
            assert len(row[key]) == n, (key, len(row[key]), n)
        for p in tokens:
            k = len(row["teacher_topk_ids"][p])
            assert k and len(row["teacher_topk_logprobs"][p]) == k, p
    return row
