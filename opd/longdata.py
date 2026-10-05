import json
import random
import re
from dataclasses import dataclass

import pyarrow.parquet as pq

START = "You are given a long document, split into chunks. Read it and answer the question at the end."

QA_PROMPT = ("Answer the question based on the document above. Only give me the "
             "answer and do not output any other words.\n\nQuestion: {}\nAnswer:")

_CHUNK_RE = re.compile(r"<CHUNK_(\d+)>(.*?)</CHUNK_\1>", re.DOTALL)


LME_START = ("Below is your chat history with the user, split into sessions. The date of each "
             "session is written at its beginning. Read it and answer the question at the end.")
LME_QA_PROMPT = ("Based on the chat history above, write an answer in the form of a short phrase "
                 "for the following question. Answer with exact words from the history whenever "
                 "possible.\n\nQuestion: {} Short answer:")
LME_DROP_TYPE = "single-session-preference"
LME_EVAL_SEED = 20260822
LME_EVAL_LIMIT = 200


@dataclass
class Doc:
    doc_id: str
    sessions: list
    qa: str
    start: str = START


def split_chunks(content):
    a = content.index("Document:\n") + len("Document:\n")
    b = content.index("\n\nQuestion: ", a)
    out = [(int(k), t.strip()) for k, t in _CHUNK_RE.findall(content[a:b])]
    assert [k for k, _ in out] == list(range(len(out))), "chunk  indices are not consecutive "
    return out


def _lme_session_block(date, turns):
    out = "DATE: " + date + "\nCONVERSATION:\n"
    for turn in turns:
        out += turn["role"] + ' said, "' + turn["content"].strip() + '"\n'
    return out


def _load_longmemeval_train(path):
    with open(path) as f:
        data = json.load(f)
    rows = [d for d in data if d["question_type"] != LME_DROP_TYPE
            and not d["question_id"].endswith("_abs")]
    shuffled = list(rows)
    random.Random(LME_EVAL_SEED).shuffle(shuffled)
    eval_ids = {d["question_id"] for d in shuffled[:LME_EVAL_LIMIT]}
    train_rows = [d for d in rows if d["question_id"] not in eval_ids]
    assert len(rows) == len(eval_ids) + len(train_rows), "LongMemEval split overlaps"
    assert train_rows, "LongMemEval training split is empty"

    out = []
    for d in train_rows:
        sessions = [(k, date, _lme_session_block(date, turns))
                    for k, (date, turns) in enumerate(
                        zip(d["haystack_dates"], d["haystack_sessions"]))]
        out.append(Doc(
            doc_id=f"lme-{d['question_id']}",
            sessions=sessions,
            qa=LME_QA_PROMPT.format(d["question"].strip()),
            start=LME_START))
    return out


def _load_locomo_train(path):
    from agent.locomo.config import LOCOMO_CONFIGS
    from agent.locomo.data import load as load_locomo

    samples = load_locomo(path, LOCOMO_CONFIGS["cat5_seed"])
    assert samples, "LoCoMo training split is empty"
    return [Doc(
        doc_id=f"locomo-{sample.conv_id}-{sample.qa_index}",
        sessions=sample.sessions,
        qa=sample.qa,
        start=sample.start,
    ) for sample in samples]


def load(path):
    if str(path).endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if data and "conversation" in data[0]:
            return _load_locomo_train(path)
        return _load_longmemeval_train(path)

    out = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=64, columns=["prompt", "extra_info"]):
        for row in batch.to_pylist():
            chunks = split_chunks(row["prompt"][-1]["content"])
            out.append(Doc(
                doc_id=f"rlvr-{len(out)}",
                sessions=[(k, "", text) for k, text in chunks],
                qa=QA_PROMPT.format(row["extra_info"]["question"].strip())))
    return out


def prompt_ids(doc, tok):
    text = "<|im_start|>system\n" + doc.start + "<|im_end|>\n"
    return tok(text, add_special_tokens=False)["input_ids"]
